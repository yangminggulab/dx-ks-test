# KuaiRec 推荐系统演进项目

基于快手 KuaiRec 2.0 真实数据集，从零搭建完整的推荐系统技术演进链路，覆盖 AB Test 工程基础、传统协同过滤、深度序列推荐到 LLM 精排前沿方案。

---

## 项目结构

```text
kuairec_abtest/
├── data/KuaiRec 2.0/data/      原始数据（big_matrix, small_matrix, 特征表）
├── output/
│   ├── checkpoints/            模型 checkpoint（每 epoch 保存，断点续训）
│   ├── results/                每个模型的评估指标 JSON + 推荐列表 pkl
│   └── experiment.log          训练全程日志
├── scripts/                    所有 Python 脚本（见下方说明）
└── docs/                       设计文档与变更日志
```

---

## 推荐系统演进路线

```
AB Test 工程  →  SVD  →  TwoTower-BPR  →  TwoTower-WBPR  →  SASRec  →  BERT4Rec  →  SideInfo  →  CL4SRec  →  LLMRec
    (基础)       (协同)      (Step1)            (Step3)          (Step4)     (Step5)      (Step6)     (Step7)     (前沿)
```

### 模型说明

| 模型 | 文件 | 核心创新点 | 对照实验 |
|---|---|---|---|
| TwoTower-WBPR | `two_tower.py` | watch_ratio 加权 BPR | Step3 基线 |
| SASRec | `sasrec.py` | 因果自注意力序列推荐 | **实验A**：WBPR → SASRec |
| BERT4Rec | `bert4rec.py` | 双向注意力 + Masked Item Prediction | **实验B**：SASRec → BERT4Rec |
| SideInfo-SASRec | `sideinfo_rec.py` | ID + 视频类别/时长特征融合 | **实验C**：BERT4Rec → SideInfo |
| CL4SRec | `cl4srec.py` | 对比学习（InfoNCE）+ 三种数据增强 | **实验D**：SideInfo → CL4SRec |
| LLMRec | `llm_rec.py` | CL4SRec 召回 + 本地 LLM 精排 | **实验E**：CL4SRec → LLMRec |

### 评估指标

所有模型在 `small_matrix.csv`（密集答案本，~1000用户 × 3700视频）上离线评估：

- **Hit Rate@50**：推荐列表中用户真实看过的比例
- **avg_watch_ratio@50**：推荐视频的平均完播率
- **NDCG@50**：命中+排名综合指标（排名靠前的命中贡献更多）

每对相邻模型做 Welch t-test（逐用户指标），判断提升是否统计显著。

---

## 快速开始

### 环境要求

```bash
Python 3.10+
PyTorch 2.x（CUDA 12.x 推荐）
pip install pandas numpy scipy scikit-learn torch
```

### 在 GPU 服务器上训练（推荐）

**Mac 端一键操作**（代码同步 + 远程启动，断 SSH 不影响训练）：

```bash
cd kuairec_abtest/scripts

export KUAIREC_SERVER='user@your-host'
# 可选：
# export KUAIREC_PORT='2222'
# export KUAIREC_REMOTE_DIR='~/kuairec_abtest'

bash sync_and_run.sh           # 同步代码 + 启动全部实验（不含 LLM）
bash sync_and_run.sh --status  # 查看进度 + 已完成模型指标
bash sync_and_run.sh --log     # 实时追看训练日志（Ctrl+C 停止查看）
bash sync_and_run.sh --kill    # 停止训练（checkpoint 已保存，可续训）
```

### 在本地直接训练

```bash
cd kuairec_abtest/scripts

# 全流程（训练 Step4→5→6→7；Step3 基线从历史结果读取）
python run_all_experiments.py

# 含 LLM 精排（CL4SRec 召回 + LLM 重排，需先启动 Ollama）
python run_all_experiments.py --with-llm

# 只训练指定模型（不会自动补跑 WBPR）
python run_all_experiments.py --models sasrec bert4rec

# 快速验证（10 epoch）
python run_all_experiments.py --n-epochs 10 --patience 3

# 强制重跑（忽略已有 checkpoint）
python run_all_experiments.py --force
```

### 断点续训

训练中途断连或手动停止后，重新执行相同命令即可自动恢复：

- 已完成模型：从 `results/*.json` 跳过，推荐列表从 `results/*.pkl` 加载
- 未完成模型：从 `checkpoints/*_latest.pt` 恢复到中断的 epoch
- 若 `recs.pkl` 丢失但 checkpoint 仍在：会自动补推理，再继续显著性检验

---

## 脚本说明

### 核心模型

| 脚本 | 用途 |
|---|---|
| `sasrec.py` | SASRec 序列推荐（Step4） |
| `bert4rec.py` | BERT4Rec 双向序列推荐（Step5） |
| `sideinfo_rec.py` | SASRec + 视频侧特征（Step6） |
| `cl4srec.py` | CL4SRec 对比学习（Step7） |
| `llm_rec.py` | LLM 精排前沿方案 |
| `two_tower.py` | TwoTower 双塔召回（Step3 基线） |
| `svd_recommender.py` | SVD 协同过滤 |

### 评估与实验

| 脚本 | 用途 |
|---|---|
| `run_all_experiments.py` | **主入口**：全流程训练+评估+显著性检验 |
| `eval_advanced.py` | 单次多模型对比评估 |
| `eval_recommenders.py` | 离线评估核心函数（load_ground_truth / evaluate） |
| `ab_test.py` | Welch t-test 工具函数 |

### 运维脚本

| 脚本 | 用途 |
|---|---|
| `sync_and_run.sh` | Mac 端一键同步+启动（SSH → WSL2） |
| `run_server.sh` | WSL2 服务器端启动脚本 |

### AB Test 工程基础

| 脚本 | 用途 |
|---|---|
| `run_first_abtest.py` | 第一版 AB Test（基于 MySQL） |
| `run_abtest_pipeline.py` | AB Test 一键流水线 |
| `export_tableau_data.py` | 导出 Tableau 看板数据 |
| `import_kuairec_to_mysql.py` | KuaiRec 数据导入 MySQL |

---

## 输出文件说明

训练完成后，`output/` 目录结构如下：

```text
output/
├── checkpoints/
│   ├── sasrec_best.pt          各模型最佳权重
│   ├── sasrec_latest.pt        各模型最新权重（用于续训）
│   ├── bert4rec_best.pt
│   └── ...
├── results/
│   ├── sasrec_result.json      各模型评估指标
│   ├── sasrec_recs.pkl         各模型推荐列表（断连续跑显著性检验用）
│   ├── bert4rec_result.json
│   ├── ...
│   └── significance_tests.json 各实验 t-test 结果
├── all_models_comparison.csv   全模型汇总对比表
└── experiment.log              训练全程日志
```

---

## LLM 精排说明

LLMRec 使用本地 Ollama 运行，无需 API key：

```bash
# 服务器上安装并启动
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
ollama serve &

# 然后启动实验
python run_all_experiments.py --with-llm
```

当前实现是 **CL4SRec 召回 + LLM 精排**。

若 Ollama 未启动，LLMRec 会 fallback 为 `CL4SRec` 原始召回结果，流水线不报错；
但这种情况下**不会输出实验 E 的显著性结论**，避免把“没跑到 LLM”误写成前沿实验结果。

---

## 已知历史实验结果（Step3 基线）

| 模型 | Hit Rate@50 | avg_watch_ratio | NDCG@50 |
|---|---|---|---|
| TwoTower-BPR | 0.0060 | 0.0050 | 0.0008 |
| TwoTower-WBPR | 0.0210 | 0.0172 | 0.0028 |

WBPR 相对 BPR 提升约 +249%（NDCG），Step4 起以 WBPR 为基线做链式对比。

---

## 待办事项

### 立即（下次开机后）

- [ ] 开 Windows GPU 服务器，运行 `KUAIREC_SERVER="thisislbk@192.168.1.18" bash sync_and_run.sh` 同步最新代码并启动全量训练
- [ ] 训练结束后查看 CL4SRec 日志：找最后一个 ★ 出现在第几 epoch，判断 early stopping 是否太早触发

### 代码质量（中优先级）

- [ ] `eval_recommenders.py`：返回的 `n_users` 字段包含空推荐用户，与实际参与统计的用户数不一致，建议改为 `len(hit_rates)`
- [ ] `svd_recommender.py:264`：`compute_rmse` 里第一行 `pred_vals` 赋值是死代码，被第二次赋值立刻覆盖，删掉即可

### 低优先级

- [ ] 各模型的 `_get_device()`、`_build_user_sequences()`、checkpoint 保存/恢复逻辑高度重复，可抽到共享工具模块
- [ ] `bert4rec.py`：结果字典字段名 `"bpr_loss"` 实际是 CrossEntropy loss，名字误导，改为 `"ce_loss"` 更准确

---

## 变更日志

详见 [docs/change_log.md](kuairec_abtest/docs/change_log.md)
