"""
文件用途：检查 KuaiRec 原始表是否已成功导入 MySQL，并输出各表的行数与样例记录。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from data_loader import load_data


TABLE_NAME_MAP = {
    "small_matrix": "kuairec_small_matrix",
    "big_matrix": "kuairec_big_matrix",
    "user_features": "kuairec_user_features",
    "item_daily_features": "kuairec_item_daily_features",
    "item_categories": "kuairec_item_categories",
}


def validate_table_name(table_name: str) -> str:
    """校验表名格式，避免拼接 SQL 时出现异常。"""
    if not re.fullmatch(r"[A-Za-z0-9_]+", table_name):
        raise ValueError("表名仅允许字母、数字和下划线。")
    return table_name


def check_table(table_name: str) -> dict[str, object]:
    """检查单张表的行数与样例记录。"""
    try:
        safe_table_name = validate_table_name(table_name)
        exists_query = """
        SELECT COUNT(*) AS table_count
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_name = :table_name
        """
        count_query = f"SELECT COUNT(*) AS row_count FROM {safe_table_name}"
        preview_query = f"SELECT * FROM {safe_table_name} LIMIT 5"

        exists_df = load_data(exists_query, params={"table_name": safe_table_name})
        if exists_df.empty or int(exists_df.loc[0, "table_count"]) == 0:
            return {
                "table_name": safe_table_name,
                "exists": False,
                "row_count": None,
                "preview_columns": [],
                "note": "该表当前未导入；如果需要完整行为数据，可单独导入对应表。",
            }

        count_df = load_data(count_query)
        preview_df = load_data(preview_query)

        if count_df.empty:
            return {
                "table_name": safe_table_name,
                "exists": False,
                "row_count": None,
                "preview_columns": [],
            }

        row_count = int(count_df.loc[0, "row_count"])
        preview_columns = preview_df.columns.tolist() if not preview_df.empty else []
        return {
            "table_name": safe_table_name,
            "exists": True,
            "row_count": row_count,
            "preview_columns": preview_columns,
        }
    except Exception as exc:
        return {
            "table_name": table_name,
            "exists": False,
            "row_count": None,
            "preview_columns": [],
            "error": str(exc),
        }


def check_mysql_import() -> dict[str, object]:
    """汇总检查 KuaiRec 各原始表的导入状态。"""
    try:
        summary = {
            alias: check_table(table_name)
            for alias, table_name in TABLE_NAME_MAP.items()
        }
        return summary
    except Exception as exc:
        return {"error": f"MySQL 导入检查失败，原因：{exc}"}


if __name__ == "__main__":
    result = check_mysql_import()
    print(result)
