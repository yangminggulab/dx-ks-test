"""
文件用途：检查 KuaiRec 真实数据文件是否已下载并解压，并验证关键 CSV 文件可被 pandas 正常读取。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

EXPECTED_FILES = [
    "small_matrix.csv",
    "big_matrix.csv",
    "user_features.csv",
    "item_daily_features.csv",
    "item_categories.csv",
]


def find_file(file_name: str) -> Path | None:
    """在 data 目录下递归查找目标文件。"""
    try:
        matches = list(DATA_DIR.rglob(file_name))
        return matches[0] if matches else None
    except Exception as exc:
        print(f"查找文件失败，原因：{exc}")
        return None


def inspect_csv(file_path: Path, nrows: int = 5) -> dict[str, object]:
    """读取 CSV 的前几行并返回基础信息。"""
    try:
        dataframe = pd.read_csv(file_path, nrows=nrows)
        return {
            "file_path": str(file_path),
            "exists": True,
            "shape_preview": dataframe.shape,
            "columns": dataframe.columns.tolist(),
            "readable": True,
        }
    except Exception as exc:
        return {
            "file_path": str(file_path),
            "exists": True,
            "shape_preview": None,
            "columns": [],
            "readable": False,
            "error": str(exc),
        }


def check_kuairec_data() -> dict[str, object]:
    """检查 KuaiRec 数据集的关键文件状态。"""
    try:
        results: dict[str, object] = {"data_dir": str(DATA_DIR), "files": {}}

        for file_name in EXPECTED_FILES:
            file_path = find_file(file_name)
            if file_path is None:
                results["files"][file_name] = {
                    "exists": False,
                    "readable": False,
                    "error": "未找到该文件。",
                }
                continue

            results["files"][file_name] = inspect_csv(file_path)

        return results
    except Exception as exc:
        return {
            "data_dir": str(DATA_DIR),
            "files": {},
            "error": f"检查 KuaiRec 数据失败，原因：{exc}",
        }


if __name__ == "__main__":
    print(check_kuairec_data())
