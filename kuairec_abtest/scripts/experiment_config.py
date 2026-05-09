"""
文件用途：统一维护 AB Test 实验方案配置，并输出可复用的设计文档。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricDefinition:
    """定义实验指标。"""

    metric_id: str
    metric_name_cn: str
    metric_name_en: str
    role: str
    level: str
    description: str


@dataclass(frozen=True)
class StatisticalTestDefinition:
    """定义统计检验方案。"""

    test_id: str
    display_name: str
    level: str
    target_metric: str
    alpha: float
    purpose: str


@dataclass(frozen=True)
class SplitDefinition:
    """定义实验分流方案。"""

    unit: str
    method: str
    sql_rule: str
    groups: tuple[str, ...]
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class OutputDefinition:
    """定义实验产出物。"""

    file_name: str
    description: str


@dataclass(frozen=True)
class ExperimentSpec:
    """定义一个可执行的实验方案。"""

    experiment_id: str
    version: str
    title: str
    objective: str
    null_hypothesis: str
    alternative_hypothesis: str
    default_alpha: float
    output_prefix: str
    split_definition: SplitDefinition
    core_metrics: tuple[MetricDefinition, ...]
    supporting_metrics: tuple[MetricDefinition, ...]
    segment_dimensions: tuple[str, ...]
    statistical_tests: tuple[StatisticalTestDefinition, ...]
    decision_rules: tuple[str, ...]
    outputs: tuple[OutputDefinition, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """将实验配置转换为可序列化字典。"""
        return asdict(self)


def get_abtest_v1_spec() -> ExperimentSpec:
    """返回第一版离线 KuaiRec AB Test 方案。"""
    return ExperimentSpec(
        experiment_id="kuairec_abtest_v1",
        version="v1",
        title="KuaiRec 第一版离线 AB Test",
        objective="验证推荐策略调整后，是否提升用户级完播率，并输出可复用的实验分析闭环。",
        null_hypothesis="实验组与对照组在用户级完播率上不存在显著差异。",
        alternative_hypothesis="实验组与对照组在用户级完播率上存在显著差异。",
        default_alpha=0.05,
        output_prefix="abtest_v1",
        split_definition=SplitDefinition(
            unit="user_id",
            method="CRC32(user_id) % 2",
            sql_rule=(
                "CASE WHEN MOD(CRC32(CAST(user_id AS CHAR)), 2) = 0 "
                "THEN 'control' ELSE 'treatment' END"
            ),
            groups=("control", "treatment"),
            rationale=(
                "保证同一用户稳定落在同一组，避免跨组污染。",
                "规则简单、可复现，适合离线模拟和面试展示。",
            ),
        ),
        core_metrics=(
            MetricDefinition(
                metric_id="avg_completion_rate",
                metric_name_cn="人均完播率",
                metric_name_en="Average Completion Rate",
                role="core",
                level="user",
                description="按用户先计算完播率，再对组内用户取平均，是第一版最核心指标。",
            ),
        ),
        supporting_metrics=(
            MetricDefinition(
                metric_id="avg_watch_ratio",
                metric_name_cn="平均观看比",
                metric_name_en="Average Watch Ratio",
                role="supporting",
                level="user",
                description="辅助观察推荐策略是否改变用户观看深度。",
            ),
            MetricDefinition(
                metric_id="avg_play_duration",
                metric_name_cn="平均播放时长",
                metric_name_en="Average Play Duration",
                role="supporting",
                level="user",
                description="辅助判断策略是否带来观看时长变化。",
            ),
            MetricDefinition(
                metric_id="user_share",
                metric_name_cn="用户占比",
                metric_name_en="User Share",
                role="guardrail",
                level="group",
                description="用于确认分流后的组间样本占比是否大致均衡。",
            ),
        ),
        segment_dimensions=("user_active_degree",),
        statistical_tests=(
            StatisticalTestDefinition(
                test_id="user_completion_rate_ttest",
                display_name="用户级完播率 t-test",
                level="user",
                target_metric="completion_rate",
                alpha=0.05,
                purpose="判断实验组与对照组在人均完播率上是否存在显著差异。",
            ),
            StatisticalTestDefinition(
                test_id="user_watch_ratio_ttest",
                display_name="用户级观看比 t-test",
                level="user",
                target_metric="avg_watch_ratio",
                alpha=0.05,
                purpose="辅助判断推荐策略是否影响观看深度。",
            ),
            StatisticalTestDefinition(
                test_id="user_play_duration_ttest",
                display_name="用户级播放时长 t-test",
                level="user",
                target_metric="avg_play_duration",
                alpha=0.05,
                purpose="辅助观察策略变化是否影响观看时长。",
            ),
            StatisticalTestDefinition(
                test_id="exposure_completion_chi_square",
                display_name="曝光级完播列联表卡方检验",
                level="exposure",
                target_metric="complete_play_cnt_vs_incomplete_play_cnt",
                alpha=0.05,
                purpose="补充观察超大样本下曝光级差异，但不单独作为上线依据。",
            ),
        ),
        decision_rules=(
            "优先看用户级核心指标，而不是只看曝光级显著性。",
            "p-value < 0.05 只是统计标准，还要同时判断方向是否正确。",
            "差异即使显著，也需要结合业务意义评估是否值得上线。",
            "若关键分层用户出现明显受损，应谨慎解释实验结果。",
        ),
        outputs=(
            OutputDefinition("abtest_v1_design.json", "机器可读的第一版实验设计文件。"),
            OutputDefinition("abtest_v1_design.md", "面向汇报和复盘的实验设计说明。"),
            OutputDefinition("abtest_v1_user_metrics.csv", "用户级实验明细。"),
            OutputDefinition("abtest_v1_group_summary.csv", "实验组与对照组汇总指标。"),
            OutputDefinition("abtest_v1_segment_summary.csv", "按用户活跃度的分层汇总。"),
            OutputDefinition("abtest_v1_exposure_summary.csv", "曝光级完播 / 未完播列联表。"),
            OutputDefinition("abtest_v1_completion_distribution.png", "用户级完播率分布图。"),
            OutputDefinition("abtest_v1_completion_mean.png", "实验组与对照组均值对比图。"),
            OutputDefinition("abtest_v1_report.md", "第一版实验分析报告。"),
            OutputDefinition("abtest_v1_run_manifest.json", "本次实验运行产物清单。"),
            OutputDefinition("tableau/tableau_manifest.json", "Tableau 数据层清单。"),
        ),
        notes=(
            "第一版重点是搭建最小可运行闭环，不直接等价于真实生产实验平台。",
            "后续版本可以继续扩展为 A/B/C/D 多策略、样本量评估和护栏指标体系。",
        ),
    )


def build_design_markdown(spec: ExperimentSpec) -> str:
    """将实验配置渲染为 Markdown 设计文档。"""
    core_metric_lines = [
        f"- `{metric.metric_id}` / {metric.metric_name_cn}：{metric.description}"
        for metric in spec.core_metrics
    ]
    supporting_metric_lines = [
        f"- `{metric.metric_id}` / {metric.metric_name_cn}：{metric.description}"
        for metric in spec.supporting_metrics
    ]
    test_lines = [
        (
            f"- `{test.display_name}`：作用于 `{test.target_metric}`，"
            f"alpha={test.alpha:.2f}，用途：{test.purpose}"
        )
        for test in spec.statistical_tests
    ]
    output_lines = [
        f"- `{output.file_name}`：{output.description}" for output in spec.outputs
    ]
    note_lines = [f"- {note}" for note in spec.notes]

    return "\n".join(
        [
            f"# {spec.title} 设计文档",
            "",
            f"- 实验 ID：`{spec.experiment_id}`",
            f"- 版本：`{spec.version}`",
            f"- 默认显著性水平：`{spec.default_alpha:.2f}`",
            "",
            "## 实验目标",
            "",
            f"- {spec.objective}",
            "",
            "## 假设",
            "",
            f"- 原假设 H0：{spec.null_hypothesis}",
            f"- 备择假设 H1：{spec.alternative_hypothesis}",
            "",
            "## 分流设计",
            "",
            f"- 实验单位：`{spec.split_definition.unit}`",
            f"- 分流方法：`{spec.split_definition.method}`",
            f"- SQL 规则：`{spec.split_definition.sql_rule}`",
            f"- 分组：{', '.join(spec.split_definition.groups)}",
            *[f"- {reason}" for reason in spec.split_definition.rationale],
            "",
            "## 指标设计",
            "",
            "### 核心指标",
            "",
            *core_metric_lines,
            "",
            "### 辅助与护栏指标",
            "",
            *supporting_metric_lines,
            "",
            "### 分层维度",
            "",
            *[f"- `{dimension}`" for dimension in spec.segment_dimensions],
            "",
            "## 统计检验",
            "",
            *test_lines,
            "",
            "## 判定原则",
            "",
            *[f"- {rule}" for rule in spec.decision_rules],
            "",
            "## 预期输出文件",
            "",
            *output_lines,
            "",
            "## 备注",
            "",
            *note_lines,
            "",
        ]
    )


def write_experiment_design_files(
    spec: ExperimentSpec,
    output_dir: Path,
) -> dict[str, str]:
    """写出实验设计的 JSON 和 Markdown 文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{spec.output_prefix}_design.json"
    markdown_path = output_dir / f"{spec.output_prefix}_design.md"

    json_path.write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(build_design_markdown(spec), encoding="utf-8")

    print(f"实验设计 JSON 已保存至：{json_path}")
    print(f"实验设计 Markdown 已保存至：{markdown_path}")

    return {
        "design_json_path": str(json_path),
        "design_markdown_path": str(markdown_path),
    }
