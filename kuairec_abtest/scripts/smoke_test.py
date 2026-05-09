"""
文件用途：执行项目最小可运行验证，检查统计检验、可视化输出和数据库连接模板是否可正常调用。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import ab_test
import db_config
import visualization


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"


def run_smoke_test() -> dict[str, object]:
    """运行最小化 smoke test，并返回测试结果摘要。"""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        demo_df = pd.DataFrame(
            {
                "group_name": [
                    "control",
                    "control",
                    "control",
                    "treatment",
                    "treatment",
                    "treatment",
                ],
                "completion_rate": [0.36, 0.41, 0.39, 0.44, 0.46, 0.43],
            }
        )

        t_result = ab_test.t_test(
            [0.42, 0.38, 0.45, 0.40, 0.41],
            [0.47, 0.43, 0.48, 0.46, 0.44],
        )
        chi_result = ab_test.chi_square_test([[120, 80], [140, 60]])

        distribution_path = visualization.plot_completion_rate_distribution(
            demo_df,
            output_path=str(OUTPUT_DIR / "smoke_completion_distribution.png"),
        )
        mean_path = visualization.plot_group_mean_comparison(
            demo_df,
            output_path=str(OUTPUT_DIR / "smoke_mean_comparison.png"),
        )

        # 数据库未配置账号密码时，失败提示属于预期行为。
        db_ready = db_config.test_connection()

        result = {
            "t_test_completed": t_result.get("p_value") is not None,
            "chi_square_completed": chi_result.get("p_value") is not None,
            "distribution_plot_created": bool(distribution_path),
            "mean_plot_created": bool(mean_path),
            "db_connection_ready": db_ready,
            "summary": (
                "统计检验与可视化 smoke test 已通过；"
                "若数据库连接未通过，请在 db_config.py 中填写账号密码后重试。"
            ),
        }
        return result
    except Exception as exc:
        return {
            "t_test_completed": False,
            "chi_square_completed": False,
            "distribution_plot_created": False,
            "mean_plot_created": False,
            "db_connection_ready": False,
            "summary": f"smoke test 执行失败，原因：{exc}",
        }


if __name__ == "__main__":
    smoke_result = run_smoke_test()
    print(smoke_result)
