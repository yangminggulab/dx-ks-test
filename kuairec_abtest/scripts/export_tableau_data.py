"""
文件用途：将第一版 AB Test 的输出结果整理为 Tableau 更易接入的 CSV 数据集，并生成基础字段说明。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from ab_test import t_test
from experiment_config import get_abtest_v1_spec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLEAU_OUTPUT_DIR = OUTPUT_DIR / "tableau"
EXPERIMENT_SPEC = get_abtest_v1_spec()


def load_csv(file_path: Path) -> pd.DataFrame:
    """读取 CSV 文件并返回 DataFrame。"""
    try:
        dataframe = pd.read_csv(file_path)
        print(f"读取成功：{file_path}")
        return dataframe
    except Exception as exc:
        print(f"读取失败，原因：{exc}")
        return pd.DataFrame()


def save_csv(dataframe: pd.DataFrame, file_path: Path) -> str | None:
    """保存 DataFrame 到 CSV 文件。"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(file_path, index=False, encoding="utf-8")
        print(f"保存成功：{file_path}")
        return str(file_path)
    except Exception as exc:
        print(f"保存失败，原因：{exc}")
        return None


def build_kpi_cards(
    group_summary_df: pd.DataFrame,
    user_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """生成适合 Tableau KPI 卡片使用的指标表。"""
    try:
        control_row = group_summary_df[group_summary_df["group_name"] == "control"].iloc[0]
        treatment_row = group_summary_df[group_summary_df["group_name"] == "treatment"].iloc[0]

        metric_name_map = {
            "user_cnt": ("用户数", "User Count"),
            "avg_play_cnt": ("人均播放次数", "Average Play Count"),
            "avg_completion_rate": ("人均完播率", "Average Completion Rate"),
            "avg_watch_ratio": ("平均观看比", "Average Watch Ratio"),
            "avg_play_duration": ("平均播放时长", "Average Play Duration"),
            "user_share": ("用户占比", "User Share"),
        }

        t_test_config = {
            "avg_completion_rate": ("completion_rate", "用户级完播率 t-test"),
            "avg_watch_ratio": ("avg_watch_ratio", "用户级观看比 t-test"),
            "avg_play_duration": ("avg_play_duration", "用户级播放时长 t-test"),
        }

        control_user_df = user_metrics_df[user_metrics_df["group_name"] == "control"]
        treatment_user_df = user_metrics_df[user_metrics_df["group_name"] == "treatment"]

        rows: list[dict[str, Any]] = []
        for metric_id, (metric_name_cn, metric_name_en) in metric_name_map.items():
            control_value = float(control_row[metric_id])
            treatment_value = float(treatment_row[metric_id])
            abs_diff = treatment_value - control_value
            relative_diff_pct = (
                abs_diff / control_value if control_value not in [0, 0.0] else None
            )

            p_value = None
            is_significant = None
            test_note = None
            if metric_id in t_test_config:
                column_name, test_note = t_test_config[metric_id]
                test_result = t_test(
                    control_user_df[column_name],
                    treatment_user_df[column_name],
                )
                p_value = test_result.get("p_value")
                is_significant = test_result.get("is_significant")

            rows.append(
                {
                    "metric_id": metric_id,
                    "metric_name_cn": metric_name_cn,
                    "metric_name_en": metric_name_en,
                    "control_value": control_value,
                    "treatment_value": treatment_value,
                    "abs_diff": abs_diff,
                    "relative_diff_pct": relative_diff_pct,
                    "better_group": (
                        "treatment"
                        if abs_diff > 0
                        else "control"
                        if abs_diff < 0
                        else "tie"
                    ),
                    "p_value": p_value,
                    "is_significant": is_significant,
                    "test_note": test_note,
                }
            )

        return pd.DataFrame(rows)
    except Exception as exc:
        print(f"构建 KPI 卡片数据失败，原因：{exc}")
        return pd.DataFrame()


def build_group_metrics_long(group_summary_df: pd.DataFrame) -> pd.DataFrame:
    """生成分组长表，便于 Tableau 统一做柱状图和筛选。"""
    try:
        metric_cn_map = {
            "user_cnt": "用户数",
            "avg_play_cnt": "人均播放次数",
            "avg_completion_rate": "人均完播率",
            "avg_watch_ratio": "平均观看比",
            "avg_play_duration": "平均播放时长",
            "user_share": "用户占比",
        }
        long_df = group_summary_df.melt(
            id_vars=["group_name"],
            value_vars=list(metric_cn_map.keys()),
            var_name="metric_id",
            value_name="metric_value",
        )
        long_df["metric_name_cn"] = long_df["metric_id"].map(metric_cn_map)
        return long_df
    except Exception as exc:
        print(f"构建分组长表失败，原因：{exc}")
        return pd.DataFrame()


def build_segment_metrics_long(segment_summary_df: pd.DataFrame) -> pd.DataFrame:
    """将分层宽表整理为 Tableau 友好的长表。"""
    try:
        rows: list[dict[str, Any]] = []
        for _, row in segment_summary_df.iterrows():
            for group_name in ["control", "treatment"]:
                rows.append(
                    {
                        "user_active_degree": row.get("user_active_degree"),
                        "group_name": group_name,
                        "user_cnt": row.get(f"user_cnt_{group_name}"),
                        "avg_completion_rate": row.get(
                            f"avg_completion_rate_{group_name}"
                        ),
                        "avg_watch_ratio": row.get(f"avg_watch_ratio_{group_name}"),
                        "completion_rate_diff": row.get("completion_rate_diff"),
                        "total_user_cnt": row.get("total_user_cnt"),
                    }
                )
        return pd.DataFrame(rows)
    except Exception as exc:
        print(f"构建分层长表失败，原因：{exc}")
        return pd.DataFrame()


def build_user_distribution(user_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """整理用户级分布数据，便于 Tableau 做直方图与箱线图。"""
    try:
        distribution_df = user_metrics_df.copy()
        distribution_df["completion_rate_bucket"] = pd.cut(
            distribution_df["completion_rate"],
            bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            include_lowest=True,
            right=True,
        ).astype(str)
        distribution_df["watch_ratio_bucket"] = pd.cut(
            distribution_df["avg_watch_ratio"],
            bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            include_lowest=True,
            right=True,
        ).astype(str)
        return distribution_df
    except Exception as exc:
        print(f"构建用户分布数据失败，原因：{exc}")
        return pd.DataFrame()


def write_manifest(manifest: dict[str, Any], file_path: Path) -> str | None:
    """写出 Tableau 数据说明清单。"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"清单已保存：{file_path}")
        return str(file_path)
    except Exception as exc:
        print(f"写入清单失败，原因：{exc}")
        return None


def export_tableau_data() -> dict[str, Any]:
    """导出 Tableau 所需的 CSV 数据集。"""
    try:
        group_summary_df = load_csv(OUTPUT_DIR / "abtest_v1_group_summary.csv")
        segment_summary_df = load_csv(OUTPUT_DIR / "abtest_v1_segment_summary.csv")
        user_metrics_df = load_csv(OUTPUT_DIR / "abtest_v1_user_metrics.csv")

        if group_summary_df.empty or segment_summary_df.empty or user_metrics_df.empty:
            raise ValueError(
                "第一版 AB Test 输出文件缺失或为空，请先运行 run_first_abtest.py。"
            )

        kpi_cards_df = build_kpi_cards(group_summary_df, user_metrics_df)
        group_metrics_long_df = build_group_metrics_long(group_summary_df)
        segment_metrics_long_df = build_segment_metrics_long(segment_summary_df)
        user_distribution_df = build_user_distribution(user_metrics_df)

        outputs = {
            "kpi_cards_path": save_csv(
                kpi_cards_df,
                TABLEAU_OUTPUT_DIR / "tableau_kpi_cards.csv",
            ),
            "group_metrics_long_path": save_csv(
                group_metrics_long_df,
                TABLEAU_OUTPUT_DIR / "tableau_group_metrics_long.csv",
            ),
            "segment_metrics_long_path": save_csv(
                segment_metrics_long_df,
                TABLEAU_OUTPUT_DIR / "tableau_segment_metrics_long.csv",
            ),
            "user_distribution_path": save_csv(
                user_distribution_df,
                TABLEAU_OUTPUT_DIR / "tableau_user_distribution.csv",
            ),
        }

        manifest = {
            "purpose": "KuaiRec AB Test Tableau 数据层",
            "experiment_id": EXPERIMENT_SPEC.experiment_id,
            "experiment_version": EXPERIMENT_SPEC.version,
            "data_sources": {
                "tableau_kpi_cards.csv": "用于 KPI 卡片和指标摘要展示。",
                "tableau_group_metrics_long.csv": "用于实验组与对照组指标对比柱状图。",
                "tableau_segment_metrics_long.csv": "用于用户活跃度分层对比图。",
                "tableau_user_distribution.csv": "用于完播率分布图、箱线图和用户级筛选分析。",
            },
            "depends_on": {
                "group_summary": str(OUTPUT_DIR / "abtest_v1_group_summary.csv"),
                "segment_summary": str(OUTPUT_DIR / "abtest_v1_segment_summary.csv"),
                "user_metrics": str(OUTPUT_DIR / "abtest_v1_user_metrics.csv"),
                "design_json": str(OUTPUT_DIR / "abtest_v1_design.json"),
                "run_manifest": str(OUTPUT_DIR / "abtest_v1_run_manifest.json"),
            },
            "paths": outputs,
        }
        outputs["manifest_path"] = write_manifest(
            manifest,
            TABLEAU_OUTPUT_DIR / "tableau_manifest.json",
        )
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
        return outputs
    except Exception as exc:
        error_result = {"error": f"导出 Tableau 数据失败，原因：{exc}"}
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        return error_result


if __name__ == "__main__":
    export_tableau_data()
