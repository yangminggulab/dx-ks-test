# Tableau 配置说明

用途说明：本文件用于说明如何将当前 KuaiRec AB Test 项目输出快速接入 Tableau，并搭建一页可用于展示与汇报的 Dashboard。

## 1. 适用场景

当你已经完成 SQL 取数、MySQL 入库、Python 统计检验，并生成了第一版 AB Test 结果后，可以使用 Tableau 将结果统一展示为一页看板，提升结果查看与汇报效率。

## 2. 推荐连接的数据文件

建议优先连接以下四个文件：

- `/Users/liubike/Desktop/快手test/kuairec_abtest/output/tableau/tableau_kpi_cards.csv`
- `/Users/liubike/Desktop/快手test/kuairec_abtest/output/tableau/tableau_group_metrics_long.csv`
- `/Users/liubike/Desktop/快手test/kuairec_abtest/output/tableau/tableau_segment_metrics_long.csv`
- `/Users/liubike/Desktop/快手test/kuairec_abtest/output/tableau/tableau_user_distribution.csv`

## 3. 各文件用途

### `tableau_kpi_cards.csv`

适合做：

- KPI 卡片
- 实验组 vs 对照组指标摘要
- 差值与显著性说明

关键字段：

- `metric_name_cn`
- `control_value`
- `treatment_value`
- `abs_diff`
- `relative_diff_pct`
- `p_value`
- `is_significant`

### `tableau_group_metrics_long.csv`

适合做：

- 实验组 vs 对照组柱状图
- 多指标切换图

关键字段：

- `group_name`
- `metric_id`
- `metric_name_cn`
- `metric_value`

### `tableau_segment_metrics_long.csv`

适合做：

- 用户活跃度分层对比图
- 分层条形图
- 分层热力图

关键字段：

- `user_active_degree`
- `group_name`
- `user_cnt`
- `avg_completion_rate`
- `avg_watch_ratio`
- `completion_rate_diff`

### `tableau_user_distribution.csv`

适合做：

- 完播率分布图
- 箱线图
- 用户级筛选分析

关键字段：

- `group_name`
- `user_id`
- `user_active_degree`
- `completion_rate`
- `avg_watch_ratio`
- `avg_play_duration`
- `completion_rate_bucket`
- `watch_ratio_bucket`

## 4. 推荐 Dashboard 结构

建议做成 1 页 Dashboard，布局如下：

1. 顶部：KPI 卡片区  
建议放：
- 对照组人均完播率
- 实验组人均完播率
- 完播率差值
- 用户级 t-test 是否显著

2. 中间左侧：总体指标对比柱状图  
数据源：`tableau_group_metrics_long.csv`

3. 中间右侧：用户活跃度分层对比图  
数据源：`tableau_segment_metrics_long.csv`

4. 底部：用户级完播率分布图  
数据源：`tableau_user_distribution.csv`

## 5. 推荐筛选器

建议加以下筛选器：

- `group_name`
- `user_active_degree`
- `metric_name_cn`

## 6. 使用建议

- Tableau 主要负责展示，不建议在 Tableau 内重做核心统计检验。
- 核心显著性检验仍以 Python 脚本输出结果为准。
- 面试展示时，可以将 Tableau 看板作为“结果沉淀与业务展示层”来讲，而不是分析逻辑本身。

## 7. 数据更新方式

如果第一版 AB Test 结果更新，请先重新运行：

```bash
cd /Users/liubike/Desktop/快手test/kuairec_abtest
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp python3 scripts/run_first_abtest.py
python3 scripts/export_tableau_data.py
```

然后在 Tableau 中刷新数据源即可。

## 8. Tableau 实操步骤

下面这套流程适合直接在 `Tableau Desktop` 或 `Tableau Public` 里手工搭 1 页 Dashboard。

### 第一步：连接数据源

建议不要把 4 张 CSV 强行做 Join，因为它们的统计粒度不同：

- `tableau_kpi_cards.csv`：指标级
- `tableau_group_metrics_long.csv`：组别 x 指标级
- `tableau_segment_metrics_long.csv`：分层 x 组别级
- `tableau_user_distribution.csv`：用户级

推荐做法：

1. 打开 Tableau。
2. 选择 `连接 -> 文本文件`。
3. 先连接 `tableau_kpi_cards.csv`。
4. 在左上角 `数据源` 区域继续添加另外 3 张 CSV，作为独立数据源存在。

### 第二步：做 4 张 KPI 卡片

数据源：`tableau_kpi_cards.csv`

建议围绕 `metric_name_cn = 人均完播率` 做 4 张卡片：

1. `KPI_对照组完播率`
字段拖法：
- 将 `metric_name_cn` 拖到筛选器，只保留 `人均完播率`
- 将 `control_value` 拖到 `文本`
- 数字格式改为 `百分比`

2. `KPI_实验组完播率`
字段拖法：
- 同样筛选 `metric_name_cn = 人均完播率`
- 将 `treatment_value` 拖到 `文本`
- 数字格式改为 `百分比`

3. `KPI_完播率差值`
字段拖法：
- 同样筛选 `metric_name_cn = 人均完播率`
- 将 `abs_diff` 拖到 `文本`
- 数字格式改为 `百分比`
- 可以给 `better_group` 上色，下降显示红色、上升显示绿色

4. `KPI_t-test结论`
字段拖法：
- 同样筛选 `metric_name_cn = 人均完播率`
- 将 `p_value` 拖到 `文本`
- 将 `is_significant` 拖到 `颜色` 或 `文本`

如果你想让展示更像汇报面板，可以把第 4 张卡片的标题命名为：

- `用户级 t-test p-value`
- 或 `用户级显著性结论`

### 第三步：做总体指标对比图

数据源：`tableau_group_metrics_long.csv`

建议图表类型：并列柱状图

字段拖法：

- `metric_name_cn` -> 列
- `metric_value` -> 行
- `group_name` -> 颜色

建议设置：

- 把 `metric_name_cn` 放到筛选器中，默认保留：
  - `用户数`
  - `人均播放次数`
  - `人均完播率`
  - `平均观看比`
  - `平均播放时长`
- 如果你觉得不同指标量纲差太大，可以只保留一个指标做切换展示。

### 第四步：做活跃度分层对比图

数据源：`tableau_segment_metrics_long.csv`

建议图表类型：横向并列条形图

字段拖法：

- `user_active_degree` -> 行
- `avg_completion_rate` -> 列
- `group_name` -> 颜色

可选增强：

- 将 `user_cnt` 拖到 `标签`
- 将 `completion_rate_diff` 拖到 `工具提示`
- 按 `total_user_cnt` 降序排序，优先展示用户量大的分层

如果你想强调“实验组比对照组高还是低”，也可以再做一张只看 `completion_rate_diff` 的条形图。

### 第五步：做用户级完播率分布图

数据源：`tableau_user_distribution.csv`

推荐两种做法，先做第 1 种最稳：

做法 1：分桶柱状图

- `completion_rate_bucket` -> 列
- `Number of Records` 或 `CNT(user_id)` -> 行
- `group_name` -> 颜色

做法 2：箱线图

- `group_name` -> 列
- `completion_rate` -> 行
- 在 `显示我` 里切换为箱线图

如果面试更偏业务表达，优先用分桶柱状图；如果更偏分析表达，可以保留箱线图。

### 第六步：拼 Dashboard

建议新建 1 个 Dashboard，尺寸可先用 `1200 x 900`。

布局建议：

- 顶部横向放 4 张 KPI 卡片
- 中间左侧放总体指标对比图
- 中间右侧放活跃度分层对比图
- 底部放用户级完播率分布图

标题可以直接写：

`KuaiRec 第一版 AB Test Dashboard`

副标题可以写：

`基于 user_id 稳定分流，核心观察指标为用户级完播率`

### 第七步：加筛选器

建议加入这些筛选器，并显示在右侧：

- `group_name`
- `user_active_degree`
- `metric_name_cn`

注意：

- `metric_name_cn` 主要作用于总体指标图
- `user_active_degree` 主要作用于分层图和分布图
- 不同数据源的筛选器不一定能自动联动，属于正常现象

### 第八步：格式优化

建议统一以下格式：

- 完播率、观看比：显示为百分比
- 用户数、播放次数：显示为整数
- p-value：保留 3 到 6 位小数

颜色建议：

- `control`：蓝色
- `treatment`：橙色

### 第九步：常见坑

1. 不要把 4 张 CSV 直接做成一张大宽表，否则很容易重复统计。
2. 不要在 Tableau 里重算显著性，显著性结果以 Python 输出为准。
3. 如果图表为空，先检查当前筛选器是否把 `metric_name_cn` 或 `group_name` 全部过滤掉了。
