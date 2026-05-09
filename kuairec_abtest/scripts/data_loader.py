"""
文件用途：从 MySQL 数据库读取实验数据，并以 pandas DataFrame 形式返回。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from db_config import get_engine


KUAIREC_TABLE_MAP = {
    "small_matrix": "kuairec_small_matrix",
    "big_matrix": "kuairec_big_matrix",
    "user_features": "kuairec_user_features",
    "item_daily_features": "kuairec_item_daily_features",
    "item_categories": "kuairec_item_categories",
}


def load_data(
    query: str,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """执行 SQL 查询并返回 DataFrame。"""
    try:
        engine = get_engine()
        with engine.connect() as connection:
            dataframe = pd.read_sql(text(query), connection, params=params)
        print(f"数据读取成功，返回 {len(dataframe)} 行。")
        return dataframe
    except SQLAlchemyError as exc:
        print(f"数据读取失败，数据库异常原因：{exc}")
        return pd.DataFrame()
    except Exception as exc:
        print(f"数据读取失败，原因：{exc}")
        return pd.DataFrame()


def load_table(table_name: str, limit: int | None = None) -> pd.DataFrame:
    """读取整张表或指定条数样本。"""
    try:
        if not re.fullmatch(r"[A-Za-z0-9_]+", table_name):
            raise ValueError("table_name 仅允许字母、数字和下划线。")

        query = f"SELECT * FROM {table_name}"
        if limit is not None:
            query += " LIMIT :limit_value"
            return load_data(query, params={"limit_value": limit})
        return load_data(query)
    except Exception as exc:
        print(f"表数据读取失败，原因：{exc}")
        return pd.DataFrame()


def load_kuairec_table(dataset_name: str, limit: int | None = None) -> pd.DataFrame:
    """按 KuaiRec 数据集别名读取已导入的 MySQL 表。"""
    try:
        table_name = KUAIREC_TABLE_MAP.get(dataset_name, dataset_name)
        return load_table(table_name, limit=limit)
    except Exception as exc:
        print(f"KuaiRec 表读取失败，原因：{exc}")
        return pd.DataFrame()


if __name__ == "__main__":
    sample_query = """
    SELECT
        user_id,
        video_id,
        play_duration,
        watch_ratio
    FROM kuairec_small_matrix
    LIMIT 10
    """
    result = load_data(sample_query)
    print(result.head())
