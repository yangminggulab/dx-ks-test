# 项目变更日志

## 2026-05-11

### SVD 推荐系统 + 可视化 + 优化模型

**svd_recommender.py（新增）**
- 用 `big_matrix.csv`（12.5M 行，训练集）训练截断 SVD，生成个性化推荐列表。
- 分块读取（500K 行/块），限定到 `small_matrix` 有的视频，与 AB Test 评估框架对接。
- 不重建完整稠密矩阵，直接在因子空间计算 top-K 推荐，内存安全。
- 修复了初版错误（原版误用 `small_matrix` 作训练集，改为正确使用 `big_matrix`）。

**机器学习结果可视化/svd_visualization.py（新增）**
- 生成 6 张可解释性分析图（2×3 布局）：
  ① 奇异值衰减曲线 + 累计解释方差（肘部法找最优 k）
  ② 用户 Embedding 2D 投影（PCA，按活跃度着色）
  ③ 视频 Embedding 2D 投影（PCA，按播放热度着色）
  ④ 推荐个性化热图（用户间 Jaccard 相似度，验证个性化生效）
  ⑤ 热门偏差分析（推荐频率 vs 视频热度相关性）
  ⑥ 推荐置信度分布（高/低交互用户对比）
- 图片默认保存到 `机器学习结果可视化/` 文件夹。

**mf_optimized.py（新增）**
- 优化一：Biased MF（Funk SVD + SGD）
  加入全局均值 μ、用户偏置 b_u、视频偏置 b_i，分离系统性偏差与真实偏好。
  向量化 mini-batch SGD，实测 RMSE 比基础 SVD 降低 ~6%。
- 优化二：iALS（Implicit ALS，Hu et al. 2008）
  把 watch_ratio 转为置信度 c = 1 + 40·r，0 不再是负反馈而是低置信度偏好。
  高效 ALS 更新：预计算 VᵀV，每用户仅对已见物品加修正，不遍历全矩阵。
- 对比管道 `run_comparison_pipeline`：依次跑 SVD / BiasedMF / iALS，输出 RMSE 对比表。

**export_tableau_data.py（修改）**
- 自动优先读取 v3 实验结果（`abtest_v3_*.csv`），v3 不存在时回退到 v1。
- 处理 v3 列名差异（`completion_rate_on_recommended` → `completion_rate`）。
- `TABLEAU_OUTPUT_DIR` 改为 `tableau与数据源/`，与 TWB 文件读取路径对齐。

**tableau与数据源/（新增文件夹）**
- 将 `ks-tableau.twb` 和 4 个数据源 CSV 统一放入同一文件夹。
- 更新 TWB 内部的 `directory` 路径，修复 Tableau 无法刷新数据的问题。
- Tableau 数据已更新为 v3 结果（完播率 ~86%，修复了原先显示 33% 的问题）。

## 2026-05-09

- 将第一版 AB Test 方案工程化为可复用模板，而不只是一组分散脚本。
- 新增统一实验配置脚本，集中维护目标、假设、分流、指标、检验与输出定义。
- 新增一键运行入口，可自动生成实验设计文档、跑分析并导出 Tableau 数据。
- 第一版分析脚本新增曝光级汇总 CSV 与运行清单产出。
- Tableau 导出清单补充实验版本与依赖文件说明。
- 忽略 Tableau 自动恢复临时文件，避免干扰后续 Git 提交。
