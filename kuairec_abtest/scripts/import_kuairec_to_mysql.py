"""
文件用途：将 KuaiRec 真实 CSV 数据分块导入 MySQL，支持 dry-run、核心表导入和大表单独导入。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import BIGINT, FLOAT, INTEGER, TEXT, VARCHAR


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from db_config import ensure_database_exists, get_engine, test_connection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_ROOT = PROJECT_ROOT / "data"
DEFAULT_TABLES = [
    "small_matrix",
    "user_features",
    "item_daily_features",
    "item_categories",
]

TABLE_CONFIG = {
    "small_matrix": {
        "file_name": "small_matrix.csv",
        "table_name": "kuairec_small_matrix",
        "indexes": [
            ("idx_kuairec_small_user_id", "user_id"),
            ("idx_kuairec_small_video_id", "video_id"),
            ("idx_kuairec_small_timestamp", "timestamp"),
        ],
    },
    "big_matrix": {
        "file_name": "big_matrix.csv",
        "table_name": "kuairec_big_matrix",
        "indexes": [
            ("idx_kuairec_big_user_id", "user_id"),
            ("idx_kuairec_big_video_id", "video_id"),
            ("idx_kuairec_big_timestamp", "timestamp"),
        ],
    },
    "user_features": {
        "file_name": "user_features.csv",
        "table_name": "kuairec_user_features",
        "indexes": [
            ("idx_kuairec_user_features_user_id", "user_id"),
            ("idx_kuairec_user_active_degree", "user_active_degree"),
        ],
    },
    "item_daily_features": {
        "file_name": "item_daily_features.csv",
        "table_name": "kuairec_item_daily_features",
        "indexes": [
            ("idx_kuairec_item_daily_video_id", "video_id"),
            ("idx_kuairec_item_daily_date", "date"),
            ("idx_kuairec_item_daily_author_id", "author_id"),
        ],
    },
    "item_categories": {
        "file_name": "item_categories.csv",
        "table_name": "kuairec_item_categories",
        "indexes": [
            ("idx_kuairec_item_categories_video_id", "video_id"),
        ],
    },
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="导入 KuaiRec 数据到 MySQL。")
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_SEARCH_ROOT),
        help="KuaiRec 数据搜索根目录，默认会在项目 data/ 目录下递归查找。",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=DEFAULT_TABLES,
        help=(
            "待导入的数据表别名，支持：small_matrix、big_matrix、user_features、"
            "item_daily_features、item_categories，或使用 all 导入全部。"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="pandas 读取 CSV 的分块大小，默认 50000。",
    )
    parser.add_argument(
        "--insert-batch-size",
        type=int,
        default=1000,
        help="to_sql 每批写入 MySQL 的记录数，默认 1000。",
    )
    parser.add_argument(
        "--if-exists",
        choices=["replace", "append", "fail"],
        default="replace",
        help="目标表已存在时的处理方式，默认 replace。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查数据文件与导入计划，不连接数据库。",
    )
    return parser.parse_args()


def resolve_table_keys(selected_tables: list[str]) -> list[str]:
    """解析用户选择的导入表别名。"""
    try:
        normalized_tables = [table_name.strip() for table_name in selected_tables]
        if "all" in normalized_tables:
            return list(TABLE_CONFIG.keys())

        invalid_tables = [
            table_name
            for table_name in normalized_tables
            if table_name not in TABLE_CONFIG
        ]
        if invalid_tables:
            raise ValueError(f"不支持的数据表别名：{invalid_tables}")
        return normalized_tables
    except Exception as exc:
        raise ValueError(f"解析导入表失败，原因：{exc}") from exc


def find_dataset_root(search_root: Path) -> Path:
    """定位 KuaiRec 解压后的真实 CSV 所在目录。"""
    try:
        if (search_root / "small_matrix.csv").exists():
            return search_root

        matches = sorted(search_root.rglob("small_matrix.csv"))
        if not matches:
            raise FileNotFoundError(
                f"未在 {search_root} 下找到 small_matrix.csv，请先下载并解压 KuaiRec 数据。"
            )
        return matches[0].parent
    except Exception as exc:
        raise FileNotFoundError(f"定位 KuaiRec 数据目录失败，原因：{exc}") from exc


def build_sql_dtype_map(table_key: str) -> dict[str, object]:
    """为不同 KuaiRec 表构建 SQLAlchemy 字段类型映射。"""
    try:
        if table_key in {"small_matrix", "big_matrix"}:
            return {
                "user_id": BIGINT(),
                "video_id": BIGINT(),
                "play_duration": BIGINT(),
                "video_duration": BIGINT(),
                "time": VARCHAR(length=32),
                "date": FLOAT(),
                "timestamp": FLOAT(),
                "watch_ratio": FLOAT(),
            }

        if table_key == "user_features":
            dtype_map: dict[str, object] = {
                "user_id": BIGINT(),
                "user_active_degree": VARCHAR(length=64),
                "is_lowactive_period": INTEGER(),
                "is_live_streamer": INTEGER(),
                "is_video_author": INTEGER(),
                "follow_user_num": BIGINT(),
                "follow_user_num_range": VARCHAR(length=64),
                "fans_user_num": BIGINT(),
                "fans_user_num_range": VARCHAR(length=64),
                "friend_user_num": BIGINT(),
                "friend_user_num_range": VARCHAR(length=64),
                "register_days": BIGINT(),
                "register_days_range": VARCHAR(length=64),
            }
            for feat_index in range(18):
                dtype_map[f"onehot_feat{feat_index}"] = INTEGER()
            return dtype_map

        if table_key == "item_daily_features":
            return {
                "video_id": BIGINT(),
                "date": BIGINT(),
                "author_id": BIGINT(),
                "video_type": VARCHAR(length=64),
                "upload_dt": VARCHAR(length=64),
                "upload_type": VARCHAR(length=64),
                "visible_status": VARCHAR(length=64),
                "video_duration": FLOAT(),
                "video_width": BIGINT(),
                "video_height": BIGINT(),
                "music_id": BIGINT(),
                "video_tag_id": BIGINT(),
                "video_tag_name": VARCHAR(length=255),
                "show_cnt": BIGINT(),
                "show_user_num": BIGINT(),
                "play_cnt": BIGINT(),
                "play_user_num": BIGINT(),
                "play_duration": BIGINT(),
                "complete_play_cnt": BIGINT(),
                "complete_play_user_num": BIGINT(),
                "valid_play_cnt": BIGINT(),
                "valid_play_user_num": BIGINT(),
                "long_time_play_cnt": BIGINT(),
                "long_time_play_user_num": BIGINT(),
                "short_time_play_cnt": BIGINT(),
                "short_time_play_user_num": BIGINT(),
                "play_progress": FLOAT(),
                "comment_stay_duration": BIGINT(),
                "like_cnt": BIGINT(),
                "like_user_num": BIGINT(),
                "click_like_cnt": BIGINT(),
                "double_click_cnt": BIGINT(),
                "cancel_like_cnt": BIGINT(),
                "cancel_like_user_num": BIGINT(),
                "comment_cnt": BIGINT(),
                "comment_user_num": BIGINT(),
                "direct_comment_cnt": BIGINT(),
                "reply_comment_cnt": BIGINT(),
                "delete_comment_cnt": BIGINT(),
                "delete_comment_user_num": BIGINT(),
                "comment_like_cnt": BIGINT(),
                "comment_like_user_num": BIGINT(),
                "follow_cnt": BIGINT(),
                "follow_user_num": BIGINT(),
                "cancel_follow_cnt": BIGINT(),
                "cancel_follow_user_num": BIGINT(),
                "share_cnt": BIGINT(),
                "share_user_num": BIGINT(),
                "download_cnt": BIGINT(),
                "download_user_num": BIGINT(),
                "report_cnt": BIGINT(),
                "report_user_num": BIGINT(),
                "reduce_similar_cnt": BIGINT(),
                "reduce_similar_user_num": BIGINT(),
                "collect_cnt": FLOAT(),
                "collect_user_num": FLOAT(),
                "cancel_collect_cnt": FLOAT(),
                "cancel_collect_user_num": FLOAT(),
            }

        if table_key == "item_categories":
            return {
                "video_id": BIGINT(),
                "feat": TEXT(),
            }

        raise ValueError(f"暂不支持的表别名：{table_key}")
    except Exception as exc:
        raise ValueError(f"构建字段类型映射失败，原因：{exc}") from exc


def get_file_size_mb(file_path: Path) -> float:
    """返回文件大小，单位 MB。"""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except Exception:
        return math.nan


def print_import_plan(dataset_root: Path, table_keys: list[str]) -> None:
    """打印导入计划。"""
    print(f"定位到 KuaiRec 数据目录：{dataset_root}")
    for table_key in table_keys:
        file_name = TABLE_CONFIG[table_key]["file_name"]
        table_name = TABLE_CONFIG[table_key]["table_name"]
        file_path = dataset_root / file_name
        file_size_mb = get_file_size_mb(file_path)
        print(
            f"[计划导入] {table_key} -> {table_name} | "
            f"文件：{file_path} | 大小：{file_size_mb:.2f} MB"
        )


def create_indexes(engine, table_key: str) -> None:
    """为已导入的表补充常用索引。"""
    try:
        table_name = TABLE_CONFIG[table_key]["table_name"]
        indexes = TABLE_CONFIG[table_key]["indexes"]
        with engine.begin() as connection:
            existing_indexes = {
                row[2]
                for row in connection.execute(text(f"SHOW INDEX FROM `{table_name}`"))
            }
            for index_name, column_name in indexes:
                if index_name in existing_indexes:
                    continue
                connection.execute(
                    text(
                        f"CREATE INDEX `{index_name}` "
                        f"ON `{table_name}` (`{column_name}`)"
                    )
                )
                print(f"{table_name} 已创建索引：{index_name}")
    except SQLAlchemyError as exc:
        print(f"索引创建失败，原因：{exc}")
    except Exception as exc:
        print(f"索引创建失败，原因：{exc}")


def query_row_count(engine, table_name: str) -> int | None:
    """查询导入后的表行数。"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
            return int(result.scalar_one())
    except Exception as exc:
        print(f"查询表行数失败，原因：{exc}")
        return None


def import_single_table(
    engine,
    dataset_root: Path,
    table_key: str,
    read_chunk_size: int,
    insert_batch_size: int,
    if_exists: str,
) -> int:
    """分块导入单张 KuaiRec CSV 到 MySQL。"""
    try:
        config = TABLE_CONFIG[table_key]
        file_path = dataset_root / config["file_name"]
        table_name = config["table_name"]

        if not file_path.exists():
            raise FileNotFoundError(f"未找到数据文件：{file_path}")

        total_rows = 0
        write_mode = if_exists
        dtype_map = build_sql_dtype_map(table_key)

        for chunk_index, chunk_df in enumerate(
            pd.read_csv(file_path, chunksize=read_chunk_size, low_memory=False),
            start=1,
        ):
            chunk_df.columns = [column_name.strip() for column_name in chunk_df.columns]
            chunk_df = chunk_df.where(pd.notna(chunk_df), None)

            chunk_df.to_sql(
                name=table_name,
                con=engine,
                if_exists=write_mode,
                index=False,
                chunksize=insert_batch_size,
                method="multi",
                dtype=dtype_map,
            )

            total_rows += len(chunk_df)
            write_mode = "append"
            print(
                f"{table_name} 导入完成第 {chunk_index} 个分块，"
                f"当前累计 {total_rows} 行。"
            )

        create_indexes(engine, table_key)
        return total_rows
    except SQLAlchemyError as exc:
        raise RuntimeError(f"{table_key} 导入失败，数据库异常原因：{exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"{table_key} 导入失败，原因：{exc}") from exc


def main() -> int:
    """执行导入主流程。"""
    try:
        args = parse_args()
        dataset_root = find_dataset_root(Path(args.dataset_root).expanduser())
        table_keys = resolve_table_keys(args.tables)
        print_import_plan(dataset_root, table_keys)

        if args.dry_run:
            print("dry-run 完成：已定位数据文件，未连接数据库。")
            return 0

        if not ensure_database_exists():
            return 1

        if not test_connection():
            return 1

        engine = get_engine()
        summary: list[dict[str, object]] = []

        for table_key in table_keys:
            imported_rows = import_single_table(
                engine=engine,
                dataset_root=dataset_root,
                table_key=table_key,
                read_chunk_size=args.chunk_size,
                insert_batch_size=args.insert_batch_size,
                if_exists=args.if_exists,
            )
            table_name = TABLE_CONFIG[table_key]["table_name"]
            row_count = query_row_count(engine, table_name)
            summary.append(
                {
                    "table_key": table_key,
                    "table_name": table_name,
                    "imported_rows": imported_rows,
                    "mysql_row_count": row_count,
                }
            )

        print("全部导入完成，结果摘要如下：")
        for item in summary:
            print(item)
        return 0
    except Exception as exc:
        print(f"导入流程失败，原因：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
