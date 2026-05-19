# AB Test 实验报告

> 每次实验追加一节，保留历史记录，不覆盖前次内容。

---

## 实验一：TwoTower-BPR vs TwoTower-WBPR（2026-05-18）

### 1. 实验背景与假设

**背景**：双塔（Two-Tower）模型是工业界主流的召回架构（YouTube / 快手 / 抖音均在用）。
训练方式是 BPR（贝叶斯个性化排序）：对每条正样本随机采一个负样本，
loss = -log σ(pos_score − neg_score)，只要正样本排在负样本前面即可。

**问题**：普通 BPR 把 watch_ratio=0.1（划走）和 watch_ratio=1.0（完播）等权重对待，
忽略了正反馈强度的差异。

**假设**：用 watch_ratio 对正样本加权（WBPR），让完播样本贡献更大梯度，
可以让模型更专注学"真正喜欢"而非"随便扫了一眼"，从而提升推荐质量。

---

### 2. 实验设计

| 项目 | 内容 |
|---|---|
| 对照组 | TwoTower-BPR（普通 BPR，等权重） |
| 实验组 | TwoTower-WBPR（watch_ratio 加权 BPR） |
| 训练集 | big_matrix（10.2M 条交互，7,176 用户 × 9,381 视频） |
| 评估集 | small_matrix（密集矩阵，1,411 用户全量覆盖，作答案本） |
| 评估指标 | Hit Rate@50 / avg_watch_ratio@50 / NDCG@50 |
| 统计检验 | 双样本 Welch t-test（逐用户指标），α=0.05 |

**模型架构**（两组完全相同，只有 loss 不同）：
- 用户塔：user_id emb(32) + active_degree emb(4) + 数值特征(3) → MLP(39→128→64) → L2 归一化
- 视频塔：video_id emb(32) + 类别 multi-hot(31) + 时长(1) → MLP(64→128→64) → L2 归一化
- 打分：用户向量 · 视频向量（内积，等价于余弦相似度）

---

### 3. 训练细节

- 设备：Apple MPS（M 系列 GPU）
- 优化器：Adam，lr=0.001，batch_size=4096
- 早停：90% 训练 / 10% 验证切分（固定 seed=42），patience=5
- Checkpoint：每 epoch 保存 `_latest.pt`（续训用），val_loss 改善时保存 `_best.pt`（推理用）

**踩过的坑（值得记住）**：

1. **过拟合**：第一次跑 20 epoch 没有 Early Stopping，BPR Hit Rate 从 epoch 3 的 0.0283 退化到 0.0055，WBPR 从 0.0573 退化到 0.0008。WBPR 过拟合更严重，因为高权重样本被反复记忆。加了 Early Stopping 后完全解决。

2. **MPS 不支持 float64**：WBPR 的权重 `w_b` 如果用 `float()` 转换会升为 float64，MPS 报错。改为直接用 `np.float32` 标量传入就好。

3. **日志空文件**：后台运行时 `python3 script.py > log` 日志是空的，因为 Python 默认 stdout 缓冲。加 `-u` 参数（`python3 -u script.py > log`）解决。

4. **两个模型写同一个文件**：BPR 和 WBPR 都往同一路径写推荐 CSV，第二个会覆盖第一个。解决方案：eval_recommenders.py 不让单个模型写文件，统一在比较完后保存。

---

### 4. 实验结果

| 模型 | 用户数 | Hit Rate@50 | avg_watch_ratio@50 | NDCG@50 |
|---|---|---|---|---|
| TwoTower-BPR（对照） | 1,411 | 0.0060 | 0.0050 | 0.0008 |
| TwoTower-WBPR（实验） | 1,411 | 0.0209 | 0.0172 | 0.0028 |
| **Lift** | — | **+248.6%** | **+246.2%** | **+253.2%** |

**统计显著性（Welch t-test，α=0.05）**：

| 指标 | p-value | 结论 |
|---|---|---|
| Hit Rate | < 0.0001 | 显著 ✓ |
| avg_watch_ratio | < 0.0001 | 显著 ✓ |
| NDCG | < 0.0001 | 显著 ✓ |

---

### 5. 结论

**假设成立**：WBPR 在全部三个指标上均显著优于 BPR，提升幅度约 2.5 倍，p-value 均 < 0.0001。

用 watch_ratio 加权 BPR loss 是一个低成本、高收益的改进——
架构完全不变，只改了一行 loss 公式，就让模型更准确地捕捉"真正喜欢"的信号。

---

### 6. 本次实验心得

- **Early Stopping 是必须的**，不是可选的。没有它，任何实验结论都不可信，因为最终 epoch 不一定是最好的。

- **val_loss 比 train_loss 更重要**。train_loss 一直在下降给人一种"模型在进步"的错觉，实际上 val_loss 反弹才是真实信号。

- **WBPR 比 BPR 更容易过拟合**，因为高权重样本被重复强化，模型更快"记住"训练集。Early Stopping 对 WBPR 尤其关键。

- **Checkpoint 设计要考虑续训场景**：保存 `patience_counter` 和 `best_val_loss` 到 checkpoint，否则中断后续训的早停状态会丢失。

- **统计显著性和业务显著性都要看**：p < 0.0001 说明差异不是偶然，+249% lift 说明差异足够大有业务价值。只看其中一个都不完整。

---

---

## 实验二：TwoTower-WBPR vs SASRec（2026-05-19）

### 1. 实验背景与假设

**背景**：实验一证明 WBPR 显著优于 BPR（+249% lift），WBPR 成为新的对照基线。
Two-Tower 的本质局限是把用户历史压缩成一个**静态向量**，无法感知观看顺序。

**假设**：SASRec 用 Transformer 对用户历史序列建模，捕捉时序依赖
（"先看 A 再看 B 更可能看 C"），可进一步提升推荐质量。

---

### 2. 实验设计

| 项目 | 内容 |
|---|---|
| 对照组 | TwoTower-WBPR（实验一冠军） |
| 实验组 | SASRec（序列 Transformer） |
| 训练集 | big_matrix（10.2M 条交互，7,176 用户 × 9,381 视频） |
| 评估集 | small_matrix（1,411 用户全量覆盖） |
| 评估指标 | Hit Rate@50 / avg_watch_ratio@50 / NDCG@50 |
| 统计检验 | 双样本 Welch t-test，α=0.05 |

**SASRec 架构**：
- item embedding(64) + 位置编码(64) → Transformer(2层, 2头) → 最后位置向量 → top-K 检索
- 训练：WBPR loss（与对照组相同，公平对比）；Early Stopping patience=5

**计算设备**：RTX 4060 Laptop（CUDA，8GB 显存）

---

### 3. 训练细节

- 优化器：Adam，lr=0.001，batch_size=2048
- 早停：90% 用户训练 / 10% 用户验证切分（seed=42），patience=5
- 实际训练：epoch 41 触发 Early Stopping（最佳 epoch 36，val_loss=0.030）
- 每 epoch 约 0.3s（序列样本少：6,459 训练 / 717 验证）

**踩过的坑：**

1. **val_loss = nan**：左 padding 序列 + 因果 mask + padding mask 三叠，
   PAD 位置在 causal mask 下无法 attend 到任何 token，Transformer softmax 输出 nan。
   修复：去掉 `src_key_padding_mask`，只保留因果 mask（因果 mask 已足够，padding mask 冗余且有害）。

2. **item index 0 被误当 PAD**：item_index 里第一个视频的索引是 0，
   代码用 `pad_seq[pad_seq != 0] += 1` 做偏移，会跳过 index=0 的视频，让它被当成 PAD。
   修复：赋值时直接 `+1`，保证所有真实视频都偏移到 1..n_items。

---

### 4. 实验结果

| 模型 | 用户数 | Hit Rate@50 | avg_watch_ratio@50 | NDCG@50 |
|---|---|---|---|---|
| TwoTower-WBPR（对照） | 1,411 | 0.0198 | 0.0198 | 0.0054 |
| SASRec（实验） | 1,411 | 0.0786 | 0.0780 | 0.0106 |
| **Lift** | — | **+296%** | **+294%** | **+97%** |

**统计显著性（Welch t-test，α=0.05）**：

| 指标 | p-value | 结论 |
|---|---|---|
| Hit Rate | < 0.0001 | 显著 ✓ |
| avg_watch_ratio | < 0.0001 | 显著 ✓ |
| NDCG | < 0.0001 | 显著 ✓ |

---

### 5. 结论

**假设成立**：SASRec 在 Hit Rate 和 avg_watch_ratio 上相对 WBPR 提升约 **3 倍**，
NDCG 提升约 **1 倍**，p-value 全部 < 0.0001。

序列时序信息对推荐质量有显著价值——相同训练数据下，
只需把用户历史当成序列而非静态集合，召回效果就翻了 3 倍。

---

### 6. 本次实验心得

- **序列建模 vs 静态 embedding 的差距是真实的**。KuaiRec 没有时间戳，
  SASRec 用的是"隐式序列"（原始数据顺序），已经能比双塔好 3 倍，
  真实场景有时间戳后效果应该更强。

- **val_loss = nan 是高危症状**，不要忽视。本次 nan 导致 Early Stopping 完全失效，
  模型只跑了 5 epoch 就停了，修复后正常跑到 epoch 41，结论一致但指标更可信。

- **mask 组合要谨慎**：因果 mask + padding mask 在某些边界条件下会产生全屏蔽行，
  PyTorch 不会自动报错，只会静默输出 nan。

- **两台机器的配合流程跑通了**：Mac 写代码 → rsync 同步 → RTX 4060 跑训练 →
  Mac 监控日志。4060 每 epoch 约 0.3s（SASRec）/ 27s（WBPR），比 M2 MPS 快约 3-4 倍。

---

<!-- 下次实验从这里追加 -->
