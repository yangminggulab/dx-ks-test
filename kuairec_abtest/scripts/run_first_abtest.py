"""
文件用途：基于已导入 MySQL 的 KuaiRec 数据运行第一版离线 AB Test，输出指标结果、分层结果、图表与 Markdown 报告。
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

from ab_test import chi_square_test, t_test
from data_loader import load_data
from db_config import test_connection
from experiment_config import get_abtest_v1_spec, write_experiment_design_files
from visualization import (
    plot_completion_rate_distribution,
    plot_group_mean_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
EXPERIMENT_SPEC = get_abtest_v1_spec()

USER_METRICS_QUERY = """
WITH user_metrics AS (
    SELECT
        CASE
            WHEN MOD(CRC32(CAST(s.user_id AS CHAR)), 2) = 0 THEN 'control'
            ELSE 'treatment'
        END AS group_name,
        s.user_id,
        SUM(CASE WHEN s.play_duration > 0 THEN 1 ELSE 0 END) AS play_cnt,
        SUM(
            CASE
                WHEN s.play_duration >= s.video_duration AND s.video_duration > 0 THEN 1
                ELSE 0
            END
        ) AS complete_cnt,
        AVG(LEAST(GREATEST(s.watch_ratio, 0), 1)) AS avg_watch_ratio,
        AVG(s.play_duration) AS avg_play_duration,
        MAX(u.user_active_degree) AS user_active_degree
    FROM kuairec_small_matrix AS s
    LEFT JOIN kuairec_user_features AS u
        ON s.user_id = u.user_id
    GROUP BY group_name, s.user_id
)
SELECT
    group_name,
    user_id,
    user_active_degree,
    play_cnt,
    complete_cnt,
    complete_cnt / NULLIF(play_cnt, 0) AS completion_rate,
    avg_watch_ratio,
    avg_play_duration
FROM user_metrics
WHERE play_cnt > 0
"""

EXPOSURE_SUMMARY_QUERY = """
SELECT
    CASE
        WHEN MOD(CRC32(CAST(user_id AS CHAR)), 2) = 0 THEN 'control'
        ELSE 'treatment'
    END AS group_name,
    SUM(
        CASE
            WHEN play_duration > 0
             AND play_duration >= video_duration
             AND video_duration > 0 THEN 1
            ELSE 0
        END
    ) AS complete_play_cnt,
    SUM(
        CASE
            WHEN play_duration > 0
             AND NOT (play_duration >= video_duration AND video_duration > 0) THEN 1
            ELSE 0
        END
    ) AS incomplete_play_cnt
FROM kuairec_small_matrix
GROUP BY group_name
ORDER BY group_name
"""


def fetch_user_metrics() -> pd.DataFrame:
    """读取第一版 AB Test 所需的用户级指标。"""
    try:
        dataframe = load_data(USER_METRICS_QUERY)
        if dataframe.empty:
            raise ValueError("未读取到用户级指标数据。")
        dataframe["user_active_degree"] = dataframe["user_active_degree"].fillna("UNKNOWN")
        return dataframe
    except Exception as exc:
        print(f"读取用户级指标失败，原因：{exc}")
        return pd.DataFrame()


def fetch_exposure_summary() -> pd.DataFrame:
    """读取曝光级完播列联表。"""
    try:
        dataframe = load_data(EXPOSURE_SUMMARY_QUERY)
        if dataframe.empty:
            raise ValueError("未读取到曝光级汇总数据。")
        return dataframe
    except Exception as exc:
        print(f"读取曝光级汇总失败，原因：{exc}")
        return pd.DataFrame()


def build_group_summary(user_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """生成分组汇总结果。"""
    try:
        summary_df = (
            user_metrics_df.groupby("group_name", as_index=False)
            .agg(
                user_cnt=("user_id", "nunique"),
                avg_play_cnt=("play_cnt", "mean"),
                avg_completion_rate=("completion_rate", "mean"),
                avg_watch_ratio=("avg_watch_ratio", "mean"),
                avg_play_duration=("avg_play_duration", "mean"),
            )
            .sort_values(by="group_name")
        )
        summary_df["user_share"] = summary_df["user_cnt"] / summary_df["user_cnt"].sum()
        return summary_df
    except Exception as exc:
        print(f"构建分组汇总失败，原因：{exc}")
        return pd.DataFrame()


def build_segment_summary(user_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """按用户活跃度输出分层汇总结果。"""
    try:
        segment_df = (
            user_metrics_df.groupby(["user_active_degree", "group_name"], as_index=False)
            .agg(
                user_cnt=("user_id", "nunique"),
                avg_completion_rate=("completion_rate", "mean"),
                avg_watch_ratio=("avg_watch_ratio", "mean"),
            )
        )

        pivot_df = (
            segment_df.pivot(
                index="user_active_degree",
                columns="group_name",
                values=["user_cnt", "avg_completion_rate", "avg_watch_ratio"],
            )
            .sort_index(axis=1)
            .reset_index()
        )
        pivot_df.columns = [
            "_".join([str(level) for level in column if str(level) != ""]).strip("_")
            for column in pivot_df.columns.to_flat_index()
        ]

        if {
            "avg_completion_rate_control",
            "avg_completion_rate_treatment",
        }.issubset(pivot_df.columns):
            pivot_df["completion_rate_diff"] = (
                pivot_df["avg_completion_rate_treatment"]
                - pivot_df["avg_completion_rate_control"]
            )

        if {"user_cnt_control", "user_cnt_treatment"}.issubset(pivot_df.columns):
            pivot_df["total_user_cnt"] = (
                pivot_df["user_cnt_control"].fillna(0)
                + pivot_df["user_cnt_treatment"].fillna(0)
            )

        if "total_user_cnt" in pivot_df.columns:
            pivot_df = pivot_df.sort_values(
                by=["total_user_cnt", "user_active_degree"],
                ascending=[False, True],
            )

        return pivot_df
    except Exception as exc:
        print(f"构建分层汇总失败，原因：{exc}")
        return pd.DataFrame()


def write_report(
    group_summary_df: pd.DataFrame,
    segment_summary_df: pd.DataFrame,
    exposure_summary_df: pd.DataFrame,
    t_test_result: dict[str, Any],
    chi_square_result: dict[str, Any],
    distribution_plot_path: str | None,
    mean_plot_path: str | None,
) -> str | None:
    """生成第一版 AB Test 的 Markdown 报告。"""
    try:
        report_path = OUTPUT_DIR / "abtest_v1_report.md"

        control_row = group_summary_df[group_summary_df["group_name"] == "control"].iloc[0]
        treatment_row = group_summary_df[group_summary_df["group_name"] == "treatment"].iloc[0]
        completion_rate_lift = (
            treatment_row["avg_completion_rate"] - control_row["avg_completion_rate"]
        )

        top_segments = segment_summary_df.head(5).copy()

        def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
            """将 DataFrame 转为简单 Markdown 表格，避免额外依赖。"""
            if dataframe.empty:
                return "| no_data |\n| --- |\n| empty |"

            display_df = dataframe.copy()
            for column_name in display_df.columns:
                if pd.api.types.is_numeric_dtype(display_df[column_name]):
                    display_df[column_name] = display_df[column_name].map(
                        lambda value: (
                            f"{value:.4f}" if isinstance(value, float) else str(value)
                        )
                    )
                else:
                    display_df[column_name] = display_df[column_name].astype(str)

            header = "| " + " | ".join(display_df.columns.tolist()) + " |"
            separator = "| " + " | ".join(["---"] * len(display_df.columns)) + " |"
            rows = [
                "| " + " | ".join(row_values) + " |"
                for row_values in display_df.values.tolist()
            ]
            return "\n".join([header, separator, *rows])

        report_lines = [
            "# 第一版 KuaiRec AB Test 报告",
            "",
            "用途说明：本报告由 `run_first_abtest.py` 自动生成，基于已导入 MySQL 的 KuaiRec 数据进行离线 AB Test 第一版分析。",
            "",
            "## 分析口径",
            "",
            "- 数据源：`kuairec_small_matrix` 与 `kuairec_user_features`。",
            "- 分流方式：按 `CRC32(user_id) % 2` 稳定切分为 `control` 与 `treatment`。",
            "- 完播定义：`play_duration >= video_duration` 且 `video_duration > 0`。",
            "- 用户级指标：对每个用户计算完播率、平均观看比、平均播放时长，再做 t-test。",
            "- 曝光级指标：对播放曝光的完播 / 未完播列联表做卡方检验。",
            "",
            "## 核心结果",
            "",
            f"- 对照组用户数：{int(control_row['user_cnt'])}",
            f"- 实验组用户数：{int(treatment_row['user_cnt'])}",
            f"- 对照组人均完播率：{control_row['avg_completion_rate']:.4f}",
            f"- 实验组人均完播率：{treatment_row['avg_completion_rate']:.4f}",
            f"- 完播率差值（实验组 - 对照组）：{completion_rate_lift:.4f}",
            f"- 用户级 t-test p-value：{t_test_result['p_value']:.6f}",
            f"- 用户级 t-test 结论：{'显著' if t_test_result['is_significant'] else '不显著'}",
            f"- 曝光级卡方检验 p-value：{chi_square_result['p_value']:.6e}",
            f"- 曝光级卡方检验结论：{'显著' if chi_square_result['is_significant'] else '不显著'}",
            "",
            "## 结果解读",
            "",
            "- 第一版结果显示：实验组在用户级完播率上略低于对照组，且 t-test 未达到显著水平。",
            "- 曝光级卡方检验非常显著，说明在超大样本下，极小差异也可能被检出。",
            "- 面试表达时建议强调：应优先结合用户级口径和业务显著性，不只依赖曝光级 p-value。",
            "",
            "## 曝光级列联表",
            "",
            dataframe_to_markdown(exposure_summary_df),
            "",
            "## 用户活跃度分层（Top 5）",
            "",
            dataframe_to_markdown(top_segments),
            "",
            "## 输出文件",
            "",
            f"- 用户级明细：`{OUTPUT_DIR / 'abtest_v1_user_metrics.csv'}`",
            f"- 分组汇总：`{OUTPUT_DIR / 'abtest_v1_group_summary.csv'}`",
            f"- 分层汇总：`{OUTPUT_DIR / 'abtest_v1_segment_summary.csv'}`",
        ]

        if distribution_plot_path:
            report_lines.append(f"- 完播率分布图：`{distribution_plot_path}`")
        if mean_plot_path:
            report_lines.append(f"- 均值对比图：`{mean_plot_path}`")

        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        print(f"分析报告已保存至：{report_path}")
        return str(report_path)
    except Exception as exc:
        print(f"写入分析报告失败，原因：{exc}")
        return None


def run_first_abtest(alpha: float = 0.05) -> dict[str, Any]:
    """执行第一版离线 AB Test，并输出结果摘要。"""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        design_outputs = write_experiment_design_files(EXPERIMENT_SPEC, OUTPUT_DIR)

        if not test_connection():
            raise ConnectionError("数据库连接失败，无法执行 AB Test。")

        user_metrics_df = fetch_user_metrics()
        exposure_summary_df = fetch_exposure_summary()

        if user_metrics_df.empty or exposure_summary_df.empty:
            raise ValueError("核心分析数据为空，无法继续分析。")

        group_summary_df = build_group_summary(user_metrics_df)
        segment_summary_df = build_segment_summary(user_metrics_df)

        control_values = user_metrics_df.loc[
            user_metrics_df["group_name"] == "control", "completion_rate"
        ]
        treatment_values = user_metrics_df.loc[
            user_metrics_df["group_name"] == "treatment", "completion_rate"
        ]
        t_test_result = t_test(control_values, treatment_values, alpha=alpha)

        contingency_df = exposure_summary_df[
            ["complete_play_cnt", "incomplete_play_cnt"]
        ]
        chi_square_result = chi_square_test(contingency_df, alpha=alpha)

        user_metrics_path = OUTPUT_DIR / "abtest_v1_user_metrics.csv"
        group_summary_path = OUTPUT_DIR / "abtest_v1_group_summary.csv"
        segment_summary_path = OUTPUT_DIR / "abtest_v1_segment_summary.csv"
        exposure_summary_path = OUTPUT_DIR / "abtest_v1_exposure_summary.csv"

        user_metrics_df.to_csv(user_metrics_path, index=False, encoding="utf-8")
        group_summary_df.to_csv(group_summary_path, index=False, encoding="utf-8")
        segment_summary_df.to_csv(segment_summary_path, index=False, encoding="utf-8")
        exposure_summary_df.to_csv(exposure_summary_path, index=False, encoding="utf-8")

        distribution_plot_path = plot_completion_rate_distribution(
            user_metrics_df,
            output_path=str(OUTPUT_DIR / "abtest_v1_completion_distribution.png"),
        )
        mean_plot_path = plot_group_mean_comparison(
            user_metrics_df,
            metric_col="completion_rate",
            output_path=str(OUTPUT_DIR / "abtest_v1_completion_mean.png"),
            title="Mean Completion Rate by Group",
            ylabel="Mean Completion Rate",
        )

        report_path = write_report(
            group_summary_df=group_summary_df,
            segment_summary_df=segment_summary_df,
            exposure_summary_df=exposure_summary_df,
            t_test_result=t_test_result,
            chi_square_result=chi_square_result,
            distribution_plot_path=distribution_plot_path,
            mean_plot_path=mean_plot_path,
        )

        run_manifest_path = OUTPUT_DIR / "abtest_v1_run_manifest.json"
        result = {
            "experiment_id": EXPERIMENT_SPEC.experiment_id,
            "experiment_version": EXPERIMENT_SPEC.version,
            "group_summary_path": str(group_summary_path),
            "segment_summary_path": str(segment_summary_path),
            "user_metrics_path": str(user_metrics_path),
            "exposure_summary_path": str(exposure_summary_path),
            "distribution_plot_path": distribution_plot_path,
            "mean_plot_path": mean_plot_path,
            "report_path": report_path,
            "design_outputs": design_outputs,
            "t_test_result": t_test_result,
            "chi_square_result": chi_square_result,
        }
        run_manifest_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["run_manifest_path"] = str(run_manifest_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except Exception as exc:
        error_result = {"error": f"第一版 AB Test 执行失败，原因：{exc}"}
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        return error_result


if __name__ == "__main__":
    run_first_abtest()
