# 项目变更日志

## 2026-05-26（二）

### CL4SRec 150 epoch 重跑结论 + README 实验结果更新

**实验结论（CL4SRec）**
- 扩大至 150 epoch，best checkpoint 在 epoch 93（val_loss=0.469），HR@50=0.0216，低于 50 epoch 的 0.0267。
- 根因：InfoNCE 对比学习 loss 与 HR@50 在 KuaiRec 2.0 上不相关；矩阵密度 15% 使稀疏数据增强收益有限。
- 结论：CL4SRec 在此数据集上不优于 SideInfo，属合理实验结果，无需继续调参。

**README.md（更新）**
- 删除"已知历史实验结果（Step3 基线）"旧节，替换为完整"实验结果"节。
- 新增全链路指标汇总表（6 个模型）、显著性检验表、三条关键发现说明。

**服务器维护**
- 禁用 Windows 休眠：`powercfg /h off` + `hibernate-timeout-ac/dc = 0`，彻底解决训练中途机器入睡问题。
- `sync_and_run.sh` 默认端口由 2222 改为 22（Windows 原生 OpenSSH）。

## 2026-05-26

### 代码质量修复（3 处）+ TODO 更新（idle sleep 根本原因及修复方案）

**scripts/eval_recommenders.py（修复）**
- `n_users` 字段从 `len(common_users)` 改为 `len(hit_rates)`。
- 原来的写法把"有推荐但推荐列表为空（`not recs` 触发 continue）"的用户也计入 n_users，导致分母偏大、指标被低估。现在 n_users 与实际参与均值计算的用户数严格一致。

**scripts/svd_recommender.py（修复）**
- 删除 `compute_rmse` 里的死代码：第一行 `pred_vals = (U[cx.row] @ np.diag(sigma) @ Vt)[:, cx.col]` 及紧随其后的"更高效"注释，两者均被第二次赋值立刻覆盖，从未参与计算。

**scripts/bert4rec.py（修复）**
- 结果字典字段名 `"bpr_loss"` 改为 `"ce_loss"`。BERT4Rec 使用的是 CrossEntropy loss（Masked Item Prediction），原名与实际 loss 类型不符，易误导后续分析。

**TODO.md（更新）**
- 记录 Windows GPU 服务器 idle sleep 根本原因：`DailyWake_0900` 唤醒后约 19 分钟因无人操作触发电源计划 idle sleep 计时器重新入睡，与盖子/休眠设置无关。
- 补充一次性修复命令（`powercfg /change standby-timeout-ac 0`），待下次 SSH 连通后执行。
- 更正连接命令：用户名 `thisi`，走 `ssh win-local`，旧的 `thisislbk@:2222` WSL-sshd 方案已废弃。

## 2026-05-24

### 修复 LLMRec 实验链路 + 断点恢复补推理 + README/启动脚本对齐

**scripts/llm_rec.py（修改）**
- 将 LLMRec 从“写成 CL4SRec 下一步、实际却走 SASRec 召回”修正为真正的 `CL4SRec + LLM 精排`。
- 新增 `recall_model` 参数，当前支持 `sasrec / cl4srec`，默认仍保留兼容值，但主实验入口统一切到 `cl4srec`。
- Ollama 不可用时，不再模糊描述为“随机重排”或让人误解成完整前沿实验，而是明确 fallback 到召回模型原始排序。
- 结果里额外记录 `_llm_used`、`_recall_model`、`_recall_model_name`，供主调度器判断实验 E 是否真的跑到了 LLM。

**scripts/run_all_experiments.py（修改）**
- 修正实验 E 的实现：`llmrec` 分支现在显式使用 `recall_model="cl4srec"`。
- `--models` 语义收口为“只训练指定模型”，不再偷偷把 `wbpr` 强行加回训练队列。
- 新增推荐列表恢复逻辑：若 `results/*_recs.pkl` 丢失但 checkpoint 还在，会自动从 checkpoint 补推理，再继续评估/显著性检验。
- 若 LLMRec 本轮未实际调用 Ollama，则跳过实验 E 的显著性结论，避免把 fallback 结果误当成 LLM 提升。
- 结果 JSON 新增 `llm_used / recall_model / recall_model_name` 字段，便于断点恢复和事后核查。

**scripts/eval_advanced.py（修改）**
- 对齐主流程，LLMRec 的高级评估入口默认也改为 `CL4SRec` 召回，避免两个入口行为不一致。

**scripts/sasrec.py / bert4rec.py / sideinfo_rec.py / cl4srec.py / two_tower.py（修改）**
- 新增“仅推理模式保护”：
  当 `n_epochs <= 0` 且不存在可恢复 checkpoint 时，直接报错，不再拿随机初始化权重生成推荐。
- 这避免了断点恢复场景里最隐蔽的一类伪结果。

**scripts/sync_and_run.sh（修改）**
- 去掉仓库里的个人服务器地址和本机脚本绝对路径，改为使用 `KUAIREC_SERVER / KUAIREC_PORT / KUAIREC_REMOTE_DIR` 环境变量注入。
- `--status` 输出里把 `0.0000` 这类合法指标也正确显示出来，不再因为 Python truthy 判断把它误打成 `pending`。

**README.md（修改）**
- 将 LLMRec 描述修正为 `CL4SRec 召回 + LLM 精排`，并明确写出 fallback 只保留结果、不输出实验 E 显著性结论。
- 更新本地/远程启动说明，使 `--models`、断点恢复补推理、环境变量式服务器配置与真实代码一致。
- 移除 README 中的个人 SSH 地址和本机绝对路径，降低仓库外发时的信息泄露风险。

## 2026-05-19

### 新增 SASRec 序列推荐模型 + 实验二对比框架

**scripts/sasrec.py（新增）**
- 实现 SASRec（Self-Attentive Sequential Recommendation）召回模型。
- 用户历史视频序列 → Transformer（因果自注意力 + 位置编码）→ 当前兴趣向量。
- 训练方式：WBPR loss（与 Two-Tower 保持一致，公平对比）；Early Stopping（patience=5）；每 epoch 保存 `_latest.pt` / `_best.pt`，支持中断续训。
- 推理：预计算全量视频归一化 embedding（同双塔离线索引逻辑），用户序列最后位置向量做内积检索 top-K。
- 没有序列（交互 < 2 条）的用户给空推荐，保证 user_ids 全覆盖，不影响评估框架。

**scripts/eval_recommenders.py（修改）**
- 新增 `run_comparison_v2`：TwoTower-WBPR（对照）vs SASRec（实验）。
- `include_bpr=True` 可随时将 TwoTower-BPR 重新加入三模型对比，接口保留，默认不运行。
- CLI 新增 `--experiment v1/v2`（默认 v2）和 `--include-bpr` 开关。
- 实验一（`run_comparison`，BPR vs WBPR）代码完整保留，不受影响。

**实验设计**：
- 对照组：TwoTower-WBPR（实验一冠军）
- 实验组：SASRec
- 假设：序列 Transformer 捕捉时序依赖，Hit Rate / NDCG / avg_watch_ratio 显著优于静态双塔

## 2026-05-18

### Two-Tower Early Stopping + 为 Transformer 做准备

**two_tower.py（修改）**
- 新增 Early Stopping 机制，解决 20 epoch 过拟合问题（3 epoch 后 val_loss 即开始反弹）。
- 交互数据以固定 seed=42 切 90% train / 10% val，保证 BPR 和 WBPR 用相同切分便于对比。
- 每 epoch 计算验证集 BPR/WBPR loss（torch.no_grad，无梯度，开销约 20%）。
- val_loss 改善时：在内存保留最佳权重（state_dict clone），同时写 `_best.pt` 到磁盘。
- 连续 `patience`（默认 5）轮无改善则停止，推理时自动恢复最佳权重，不用过拟合的最终 epoch。
- 中断恢复 checkpoint（`_latest.pt`）额外存储 `best_val_loss` 和 `patience_counter`，续训可正确接续早停状态。
- 新增 CLI 参数 `--patience`，`--n-epochs` 默认改为 50（最大轮数，早停会提前退出）。

**eval_recommenders.py（修改）**
- `run_comparison` 新增 `patience` 参数（默认 5），透传给双塔训练流程。
- `--n-epochs` CLI 默认值改为 50。
- `--patience` CLI 参数新增。

**背景**：20 epoch run（2026-05-17）结果显示明显过拟合：BPR Hit Rate 0.0283→0.0055，
WBPR Hit Rate 0.0573→0.0008。Early Stopping 是此次修复的核心动作。
下一步：重跑对比实验（删除旧 checkpoint），再考虑引入 SASRec（Transformer 架构）。

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
