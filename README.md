# KuaiRec 推荐系统演进项目

基于快手 KuaiRec 2.0 真实数据集，从零搭建完整的推荐系统技术演进链路，覆盖 AB Test 工程基础、传统协同过滤、深度序列推荐，全程以 A/B Test 验证各阶段提升的统计显著性。

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
AB Test 工程  →  SVD  →  TwoTower-BPR  →  TwoTower-WBPR  →  SASRec  →  BERT4Rec  →  SideInfo  →  CL4SRec
    (基础)       (协同)      (Step1)            (Step3)          (Step4)     (Step5)      (Step6)     (Step7)
```

### 模型说明

| 模型 | 文件 | 核心创新点 | 对照实验 |
|---|---|---|---|
| TwoTower-WBPR | `models/two_tower.py` | watch_ratio 加权 BPR | Step3 基线 |
| SASRec | `models/sasrec.py` | 因果自注意力序列推荐 | **实验A**：WBPR → SASRec |
| BERT4Rec | `models/bert4rec.py` | 双向注意力 + Masked Item Prediction | **实验B**：SASRec → BERT4Rec |
| SideInfo-SASRec | `models/sideinfo.py` | ID + 视频类别/时长特征融合 | **实验C**：BERT4Rec → SideInfo |
| CL4SRec | `models/cl4srec.py` | 对比学习（InfoNCE）+ 三种数据增强 | **实验D**：SideInfo → CL4SRec |

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

实验在配备 RTX 4060 的 Linux 服务器上训练，通过 `sync_and_run.sh` 实现 Mac 本地一键同步代码、远程启动训练、nohup 保活（断开 SSH 不影响进度）。

### 训练

```bash
cd kuairec_abtest/scripts

# 全流程（训练 Step4→5→6→7；Step3 基线从历史结果读取）
python run_experiments.py

# 只训练指定模型
python run_experiments.py --models sasrec bert4rec

# 快速验证（10 epoch）
python run_experiments.py --n-epochs 10 --patience 3

# 强制重跑（忽略已有 checkpoint）
python run_experiments.py --force
```

### 断点续训

训练中途断连或手动停止后，重新执行相同命令即可自动恢复：

- 已完成模型：从 `results/*.json` 跳过，推荐列表从 `results/*.pkl` 加载
- 未完成模型：从 `checkpoints/*_latest.pt` 恢复到中断的 epoch
- 若 `recs.pkl` 丢失但 checkpoint 仍在：会自动补推理，再继续显著性检验

---

## 脚本说明

### 目录结构（重构后）

```text
scripts/
├── models/                         模型包（重构新增）
│   ├── __init__.py                 导出所有模型类
│   ├── base.py                     抽象基类 + 共享工具（ModelData, BaseRecommender）
│   ├── kuairec_loader.py           统一数据加载（只读一次）
│   ├── sasrec.py                   SASRec 模型类
│   ├── bert4rec.py                 BERT4Rec 模型类
│   ├── sideinfo.py                 SideInfo-SASRec 模型类
│   ├── cl4srec.py                  CL4SRec 模型类
│   └── two_tower.py                TwoTower 模型类（BPR / WBPR）
├── run_experiments.py              新主程序（干净版，~150行纯调度逻辑）
├── run_all_experiments.py          旧主程序（保留，向后兼容）
├── eval_recommenders.py            离线评估核心函数（不动）
└── ab_test.py                      Welch t-test 工具函数（不动）
```

### 新主程序：`run_experiments.py`

```bash
# 全流程（训练 Step4→5→6→7；Step3 基线从历史结果读取）
python run_experiments.py

# 只训练指定模型
python run_experiments.py --models sasrec bert4rec

# 快速验证（10 epoch）
python run_experiments.py --n-epochs 10 --patience 3

# 强制重跑（忽略已有 checkpoint）
python run_experiments.py --force

# 自定义 top-K
python run_experiments.py --top-k 20
```

**新版优势**：数据只加载一次（`load_model_data()`），所有模型共享同一份 `ModelData`，
避免重复 I/O。模型类继承 `BaseRecommender`，接口统一：`model.train()` + `model.recommend()`。

### 兼容旧接口：`run_all_experiments.py`

旧主程序保留不动，接口与 `run_experiments.py` 兼容，可继续使用：

```bash
python run_all_experiments.py              # 默认：跑全部
python run_all_experiments.py --models sasrec bert4rec
```

### 其他核心文件

| 脚本 | 用途 |
|---|---|
| `svd_recommender.py` | SVD 协同过滤 + 数据加载工具（被 models/ 依赖） |
| `eval_advanced.py` | 单次多模型对比评估 |

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

## 实验结果

### 全链路指标汇总

| 模型 | Hit Rate@50 | avg_watch_ratio@50 | NDCG@50 | vs 上一步 |
|---|---|---|---|---|
| TwoTower-BPR（参考） | 0.0060 | 0.0050 | 0.0008 | — |
| TwoTower-WBPR（Step3 基线） | 0.0210 | 0.0172 | 0.0028 | +249% |
| SASRec（Step4） | 0.0795 | 0.0755 | 0.0165 | **+490%** |
| BERT4Rec（Step5） | 0.0046 | 0.0056 | 0.0008 | -95.2% |
| SideInfo-SASRec（Step6） | **0.1286** | **0.1370** | **0.0256** | **+3100%** |
| CL4SRec（Step7） | 0.0216 | 0.0246 | 0.0035 | -86.3% |

### 显著性检验（Welch t-test）

| 实验 | 对比 | p-value | 结论 |
|---|---|---|---|
| 实验B | SASRec → BERT4Rec | <0.001 | 显著下降 |
| 实验C | BERT4Rec → SideInfo | <0.001 | 显著提升 |

### 关键发现

**BERT4Rec 崩塌**：Hit Rate 从 0.0795 跌至 0.0046（-95.2%）。双向注意力 + Masked Item Prediction 在此数据集上严重过拟合，主因是序列较短（中位数约 5 步）、掩码训练信号稀疏。

**SideInfo 最优**：引入视频类别和时长特征后，NDCG 从 0.0008 跳至 0.0256（+3100%），是全链路最大单步提升。特征融合在短视频场景收益显著。

**CL4SRec 未超越 SideInfo**：150 epoch 训练后 best checkpoint 在 epoch 93，HR@50=0.0216，低于 50 epoch 的 0.0267。InfoNCE 对比学习 loss 与 HR@50 在此数据集上不相关；KuaiRec 2.0 高密度交互（矩阵密度 15%）使对比学习的稀疏数据增强收益有限。

---

## 变更日志

详见 [docs/change_log.md](kuairec_abtest/docs/change_log.md)
