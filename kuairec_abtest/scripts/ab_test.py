"""
文件用途：提供 AB Test 常用统计检验函数，包括连续变量的 t 检验和离散变量的卡方检验。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind


def t_test(
    control_values: list[float] | np.ndarray | pd.Series,
    treatment_values: list[float] | np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """对连续变量进行双样本 t 检验。"""
    try:
        control_array = pd.Series(control_values, dtype="float64").dropna()
        treatment_array = pd.Series(treatment_values, dtype="float64").dropna()

        if control_array.empty or treatment_array.empty:
            raise ValueError("输入样本为空，无法进行 t 检验。")

        statistic, p_value = ttest_ind(
            control_array,
            treatment_array,
            equal_var=False,
            nan_policy="omit",
        )
        is_significant = bool(p_value < alpha)
        result = {
            "test_name": "independent_t_test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": alpha,
            "is_significant": is_significant,
        }
        print(
            f"t 检验完成：p-value={result['p_value']:.6f}，"
            f"{'差异显著' if is_significant else '差异不显著'}。"
        )
        return result
    except Exception as exc:
        print(f"t 检验失败，原因：{exc}")
        return {
            "test_name": "independent_t_test",
            "statistic": None,
            "p_value": None,
            "alpha": alpha,
            "is_significant": None,
            "error": str(exc),
        }


def chi_square_test(
    contingency_table: pd.DataFrame | np.ndarray | list[list[int]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """对离散变量的列联表进行卡方检验。"""
    try:
        table = pd.DataFrame(contingency_table)
        if table.empty:
            raise ValueError("列联表为空，无法进行卡方检验。")

        chi2_stat, p_value, dof, expected = chi2_contingency(table)
        is_significant = bool(p_value < alpha)
        result = {
            "test_name": "chi_square_test",
            "chi2_stat": float(chi2_stat),
            "p_value": float(p_value),
            "dof": int(dof),
            "alpha": alpha,
            "is_significant": is_significant,
            "expected_freq": expected.tolist(),
        }
        print(
            f"卡方检验完成：p-value={result['p_value']:.6f}，"
            f"{'差异显著' if is_significant else '差异不显著'}。"
        )
        return result
    except Exception as exc:
        print(f"卡方检验失败，原因：{exc}")
        return {
            "test_name": "chi_square_test",
            "chi2_stat": None,
            "p_value": None,
            "dof": None,
            "alpha": alpha,
            "is_significant": None,
            "expected_freq": None,
            "error": str(exc),
        }


if __name__ == "__main__":
    control_sample = [0.42, 0.38, 0.45, 0.40, 0.41]
    treatment_sample = [0.47, 0.43, 0.48, 0.46, 0.44]
    t_test(control_sample, treatment_sample)

    sample_table = [[120, 80], [140, 60]]
    chi_square_test(sample_table)
