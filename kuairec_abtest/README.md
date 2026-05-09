# KuaiRec AB Test 数据分析项目

用途说明：本项目用于搭建一个模拟短视频平台推荐策略 AB Test 的数据分析工程骨架，便于后续开展数据抽取、指标分析、显著性检验与结果可视化。

## 环境要求

- Python 3.10+
- UTF-8 编码

## 安装依赖

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest
pip install -r requirements.txt
```

## 目录结构与用途

```text
kuairec_abtest/
├── data/
├── docs/
├── notebooks/
├── output/
├── scripts/
├── sql/
├── README.md
└── requirements.txt
```

- `data/`：存放原始数据、抽样数据和中间结果数据。
- `docs/`：存放项目背景、分析规范、指标定义等文档。
- `notebooks/`：存放 Jupyter Notebook，便于进行探索性分析与结果展示。
- `output/`：存放图表、分析结果导出文件和临时产物。
- `scripts/`：存放数据库连接、数据读取、统计检验、可视化等 Python 脚本。
- `sql/`：存放建表、分组、指标统计、A/A Test 等 SQL 模板。
- `README.md`：项目总览与使用说明。
- `requirements.txt`：Python 依赖清单。

## 当前阶段任务

1. 建立 MySQL 连接模板。
2. 准备 SQL 分析模板。
3. 准备统计检验与可视化脚本。
4. 为后续 AB Test 业务分析提供可复用的工程基础。

## 最小运行测试

如果当前验收目标只是确认项目骨架可运行，可以执行以下命令：

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp python3 scripts/smoke_test.py
```

预期结果：

- t 检验与卡方检验能够正常输出结果。
- `output/` 目录下生成两张示例图片。
- 若尚未填写数据库账号密码，数据库连接会打印失败原因，这属于模板阶段的正常现象。

## KuaiRec 真实数据检查

当 `KuaiRec` 数据下载并解压到 `data/` 目录后，可以执行以下命令验证关键 CSV 是否可读取：

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest
python3 scripts/check_kuairec_data.py
```

该脚本会检查：

- `small_matrix.csv`
- `big_matrix.csv`
- `user_features.csv`
- `item_daily_features.csv`
- `item_categories.csv`

## KuaiRec 导入 MySQL

如果你希望直接使用 Python 从 MySQL 读取 KuaiRec 数据，建议按以下顺序执行：

1. 在 [db_config.py](/Users/liubike/Desktop/快手test/kuairec_abtest/scripts/db_config.py) 中填写 `DB_HOST`、`DB_PORT`、`DB_NAME`、`DB_USER`、`DB_PASSWORD`。
2. 先运行 dry-run，确认脚本能够定位到真实数据文件：

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest
python3 scripts/import_kuairec_to_mysql.py --dry-run
```

3. 先导入核心表，跑通最小链路：

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest
python3 scripts/import_kuairec_to_mysql.py
```

4. 如果需要完整行为量级，再额外导入 `big_matrix`：

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest
python3 scripts/import_kuairec_to_mysql.py --tables big_matrix --chunk-size 20000
```

5. 导入完成后检查各表行数：

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest
python3 scripts/check_mysql_import.py
```

默认导入的核心表包括：

- `kuairec_small_matrix`
- `kuairec_user_features`
- `kuairec_item_daily_features`
- `kuairec_item_categories`

可选大表：

- `kuairec_big_matrix`

如果你更习惯在 `MySQL Workbench` 里先手动建表，再导入数据，可以参考：

- [05_kuairec_raw_tables.sql](/Users/liubike/Desktop/快手test/kuairec_abtest/sql/05_kuairec_raw_tables.sql)

## 第一版 AB Test

如果你希望直接运行一版基于真实 KuaiRec 数据的离线 AB Test，可以执行：

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp python3 scripts/run_first_abtest.py
```

该脚本会自动完成：

- 从 MySQL 读取 `kuairec_small_matrix` 与 `kuairec_user_features`
- 按 `user_id` 做稳定分流，构造 `control` / `treatment`
- 生成用户级完播率、曝光级列联表、显著性检验与分层结果
- 输出图表、CSV 文件和 Markdown 报告

主要输出文件位于 `output/`：

- `abtest_v1_report.md`
- `abtest_v1_group_summary.csv`
- `abtest_v1_segment_summary.csv`
- `abtest_v1_completion_distribution.png`
- `abtest_v1_completion_mean.png`

## 第一版 AB Test 设计思路

这一版的目标不是直接替业务做最终上线决策，而是先搭出一条可复用的 AB Test 基础链路：能稳定分流、能算核心指标、能做显著性检验、能输出看板数据。

### 1. 实验目标

- 模拟验证“推荐策略调整后，是否能提升用户级完播率”。
- 为后续更复杂的推荐策略实验提供最小可复用模板。

### 2. 实验假设

- 原假设 `H0`：实验组与对照组在用户级完播率上没有显著差异。
- 备择假设 `H1`：实验组与对照组在用户级完播率上存在显著差异。

这一版先用“双侧检验”判断是否有差异，后续如果策略方向非常明确，可以再讨论是否切换为单侧检验。

### 3. 分流设计

- 分流单位：`user_id`
- 分流方式：`CRC32(user_id) % 2`
- 分组结果：
  - `control`
  - `treatment`

这样设计的原因：

- 用用户级分流，比曝光级分流更符合推荐策略实验的真实口径。
- 同一用户稳定落在同一组，能避免用户跨组污染。
- 规则简单，可复现，方便离线模拟与面试展示。

### 4. 指标设计

核心指标：

- `avg_completion_rate`

原因：

- 完播率和短视频推荐效果高度相关。
- 用户级完播率比曝光级完播率更能反映真实用户体验。

辅助指标：

- `avg_watch_ratio`
- `avg_play_duration`

原因：

- 避免只看完播率造成片面判断。
- 如果完播率变化不显著，辅助指标仍能帮助判断策略是否影响观看深度。

分层观察指标：

- `user_active_degree`

原因：

- 不同活跃层级用户对策略的反应可能不同。
- 即使总体结果一般，分层结果也可能提供策略优化方向。

### 5. 统计检验设计

用户级检验：

- 对 `completion_rate`、`avg_watch_ratio`、`avg_play_duration` 做 `t-test`

曝光级检验：

- 对完播 / 未完播列联表做 `chi-square test`

设计原则：

- 用户级检验优先用于判断策略是否真正改善用户体验。
- 曝光级检验只作为补充，不单独作为上线依据。

### 6. 第一版判定标准

这一版先采用最基础的判断框架：

- 统计显著性：`p-value < 0.05`
- 方向正确：实验组核心指标优于对照组
- 业务可解释：差值不能只“显著但几乎没有业务意义”
- 分层无明显异常：不能出现核心用户群明显受损

### 7. 当前结果应如何理解

从当前输出看：

- 实验组用户级完播率略低于对照组
- 用户级 `t-test` 不显著
- 曝光级卡方检验显著

这说明：

- 当前第一版策略没有证明自己在用户级完播率上更优
- 超大样本下，曝光级非常容易检出微小差异
- 后续汇报应优先强调“用户级口径 + 业务意义”，而不是只强调曝光级显著

### 8. 为什么这只是第一版

第一版设计的重点是“先跑通”，不是“一次定终局”。

后续第二版、第三版可以继续迭代这些点：

- 是否增加护栏指标
- 是否增加实验周期与样本量评估
- 是否进一步按用户、内容、时段做分层
- 是否把不同推荐策略抽象成 `A/B/C/D` 可切换方案
- 是否把上线判定标准写得更业务化

如果把这个项目用于面试或汇报，可以把第一版定义为：

- “搭建 AB Test 最小闭环”
- “验证核心指标口径”
- “为后续策略实验提供模板”

## Tableau 数据准备

如果你想把当前 AB Test 结果接入 Tableau，可以执行：

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest
python3 scripts/export_tableau_data.py
```

该脚本会在 `output/tableau/` 下生成 Tableau 友好的 CSV：

- `tableau_kpi_cards.csv`
- `tableau_group_metrics_long.csv`
- `tableau_segment_metrics_long.csv`
- `tableau_user_distribution.csv`

接入说明见：

- [tableau_setup.md](/Users/liubike/Desktop/快手test/kuairec_abtest/docs/tableau_setup.md)
