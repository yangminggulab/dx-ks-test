"""
文件用途：绘制实验组与对照组的核心指标对比图，支持完播率分布图和均值对比图。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def plot_completion_rate_distribution(
    dataframe: pd.DataFrame,
    group_col: str = "group_name",
    completion_rate_col: str = "completion_rate",
    output_path: str | None = None,
    title: str = "Completion Rate Distribution by Group",
    xlabel: str = "Completion Rate",
    ylabel: str = "Density",
) -> str | None:
    """绘制实验组与对照组的完播率分布图。"""
    try:
        required_columns = {group_col, completion_rate_col}
        if not required_columns.issubset(dataframe.columns):
            missing_columns = required_columns - set(dataframe.columns)
            raise ValueError(f"缺少必要字段：{sorted(missing_columns)}")

        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=dataframe,
            x=completion_rate_col,
            hue=group_col,
            bins=20,
            kde=True,
            stat="density",
            common_norm=False,
            element="step",
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        if output_path:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"完播率分布图已保存至：{save_path}")
            plt.close()
            return str(save_path)

        plt.close()
        return None
    except Exception as exc:
        print(f"绘制完播率分布图失败，原因：{exc}")
        plt.close()
        return None


def plot_group_mean_comparison(
    dataframe: pd.DataFrame,
    group_col: str = "group_name",
    metric_col: str = "completion_rate",
    output_path: str | None = None,
    title: str | None = None,
    xlabel: str = "Group",
    ylabel: str = "Mean",
) -> str | None:
    """绘制实验组与对照组的指标均值对比图。"""
    try:
        required_columns = {group_col, metric_col}
        if not required_columns.issubset(dataframe.columns):
            missing_columns = required_columns - set(dataframe.columns)
            raise ValueError(f"缺少必要字段：{sorted(missing_columns)}")

        mean_df = (
            dataframe.groupby(group_col, as_index=False)[metric_col]
            .mean()
            .sort_values(by=group_col)
        )

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=mean_df,
            x=group_col,
            y=metric_col,
            hue=group_col,
            palette="Set2",
            dodge=False,
            legend=False,
        )
        plt.title(title or f"Mean {metric_col} by Group")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        if output_path:
            save_path = Path(output_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"均值对比图已保存至：{save_path}")
            plt.close()
            return str(save_path)

        plt.close()
        return None
    except Exception as exc:
        print(f"绘制均值对比图失败，原因：{exc}")
        plt.close()
        return None


if __name__ == "__main__":
    demo_df = pd.DataFrame(
        {
            "group_name": ["control", "control", "control", "treatment", "treatment", "treatment"],
            "completion_rate": [0.36, 0.41, 0.39, 0.44, 0.46, 0.43],
        }
    )
    plot_completion_rate_distribution(
        demo_df,
        output_path="/Users/liubike/Desktop/快手test/kuairec_abtest/output/demo_completion_distribution.png",
    )
    plot_group_mean_comparison(
        demo_df,
        output_path="/Users/liubike/Desktop/快手test/kuairec_abtest/output/demo_mean_comparison.png",
    )
