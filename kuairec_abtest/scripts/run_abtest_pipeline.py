"""
文件用途：一键执行第一版 AB Test 的设计文档生成、分析、看板数据导出。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from experiment_config import get_abtest_v1_spec, write_experiment_design_files
from export_tableau_data import export_tableau_data
from run_first_abtest import run_first_abtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="运行第一版 KuaiRec AB Test 全流程。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="覆盖默认显著性水平，默认使用实验方案中的 0.05。",
    )
    parser.add_argument(
        "--skip-tableau",
        action="store_true",
        help="只跑分析结果，不导出 Tableau CSV。",
    )
    parser.add_argument(
        "--design-only",
        action="store_true",
        help="只生成实验设计文档，不执行分析和 Tableau 导出。",
    )
    return parser.parse_args()


def run_pipeline(
    alpha: float | None = None,
    skip_tableau: bool = False,
    design_only: bool = False,
) -> dict[str, Any]:
    """执行第一版 AB Test 全流程。"""
    spec = get_abtest_v1_spec()
    design_outputs = write_experiment_design_files(spec, OUTPUT_DIR)

    result: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "version": spec.version,
        "design_outputs": design_outputs,
    }

    if design_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    analysis_result = run_first_abtest(alpha=alpha or spec.default_alpha)
    result["analysis_result"] = analysis_result

    if "error" not in analysis_result and not skip_tableau:
        result["tableau_result"] = export_tableau_data()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    arguments = parse_args()
    run_pipeline(
        alpha=arguments.alpha,
        skip_tableau=arguments.skip_tableau,
        design_only=arguments.design_only,
    )
