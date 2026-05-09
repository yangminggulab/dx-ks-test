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
