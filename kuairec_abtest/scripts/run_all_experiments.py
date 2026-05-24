"""
文件用途：Step4-7 + LLM 全流程实验总调度器（断连安全版）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  实验设计（对照组 → 实验组，每轮验证一个技术增量）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  链式对比：每一步只和上一步比，验证该技术增量的净收益。

  实验 A  TwoTower-WBPR  →  SASRec        （Step3 → Step4）
          问题：静态用户向量 vs 序列 Transformer，序列信息有多大价值？
          对照组：TwoTower-WBPR（已有结果，从上次实验加载）
          实验组：SASRec（因果自注意力序列推荐）

  实验 B  SASRec  →  BERT4Rec              （Step4 → Step5）
          问题：单向（因果）vs 双向注意力，哪个序列建模更充分？
          对照组：SASRec（只能看历史）
          实验组：BERT4Rec（上下文双向建模 + Masked Item Prediction）

  实验 C  BERT4Rec  →  SideInfo-SASRec    （Step5 → Step6）
          问题：在序列模型上加入视频内容特征，能否带来额外收益？
          对照组：BERT4Rec（纯 ID 序列）
          实验组：SideInfo-SASRec（ID + 视频类别/时长特征融合）

  实验 D  SideInfo-SASRec  →  CL4SRec     （Step6 → Step7）
          问题：自监督对比学习能否进一步提升序列表示的鲁棒性？
          对照组：SideInfo-SASRec（监督学习）
          实验组：CL4SRec（WBPR + InfoNCE 对比学习）

  实验 E  CL4SRec  →  LLMRec              （Step7 → 前沿）
          问题：LLM 精排能否在最强深度模型上进一步提升最终效果？
          对照组：CL4SRec（当前最强深度模型）
          实验组：LLMRec（CL4SRec 召回 + LLM 精排）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  断连安全机制（三重保险）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. 逐 epoch checkpoint：每轮训练结束保存 latest.pt + best.pt
     → 断连后从中断 epoch 续训，不损失任何进度
  2. 逐模型 result JSON：每个模型训练+推理完成后保存 results/xxx.json
     → 重启脚本时自动跳过已完成模型
  3. nohup 日志：所有输出写入 experiment.log
     → 断连后 tail -f experiment.log 查看实时进度

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  在 GPU 服务器上的标准启动方式（选一种）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  方式一：nohup（推荐，断 SSH 不影响）
    nohup python run_all_experiments.py > experiment.log 2>&1 &
    echo $!   # 记住 PID 方便 kill
    tail -f experiment.log   # 查看实时日志（Ctrl+C 只是停止看，不影响训练）

  方式二：screen（可随时重新 attach）
    screen -S exp
    python run_all_experiments.py
    # Ctrl+A D 脱离；screen -r exp 重新进入

  方式三：tmux（同 screen，更现代）
    tmux new -s exp
    python run_all_experiments.py
    # Ctrl+B D 脱离；tmux attach -t exp 重新进入

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLI 参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python run_all_experiments.py              # 默认：跑全部，跳过 LLM
  python run_all_experiments.py --with-llm   # 包含 LLM 精排实验
  python run_all_experiments.py --models sasrec bert4rec  # 只跑指定
  python run_all_experiments.py --force      # 忽略已有结果重新跑
  python run_all_experiments.py --n-epochs 20 --patience 3  # 快速验证
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ── 路径设置 ──────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

OUTPUT_DIR  = PROJECT_DIR / "output"
CKPT_DIR    = OUTPUT_DIR / "checkpoints"
RESULT_DIR  = OUTPUT_DIR / "results"
LOG_PATH    = OUTPUT_DIR / "experiment.log"

# ── 实验配置 ──────────────────────────────────────────────────────────
#
# 运行顺序：SASRec 先跑（是三个实验的对照组），结果可复用
# 每个模型只训练一次，多个实验引用同一份结果
#
# 训练顺序（wbpr 不在此列，它的结果直接从上次实验加载）
MODEL_ORDER = ["sasrec", "bert4rec", "sideinfo", "cl4srec", "llmrec"]

# 实验对照定义：(对照组模型key, 实验组模型key, 实验ID, 实验说明)
# 链式结构：每步只和上一步比，验证单个技术增量的净收益
EXPERIMENTS = [
    ("wbpr",     "sasrec",    "A", "静态双塔 vs 序列Transformer（Step3 → Step4）"),
    ("sasrec",   "bert4rec",  "B", "单向 vs 双向注意力（Step4 → Step5）"),
    ("bert4rec", "sideinfo",  "C", "纯ID序列 vs +视频侧特征（Step5 → Step6）"),
    ("sideinfo", "cl4srec",   "D", "监督学习 vs +对比学习（Step6 → Step7）"),
    ("cl4srec",  "llmrec",    "E", "深度模型 vs LLM精排（Step7 → 前沿）"),
]

DISPLAY_NAME = {
    "wbpr":     "TwoTower-WBPR (Step3)",
    "sasrec":   "SASRec        (Step4)",
    "bert4rec": "BERT4Rec      (Step5)",
    "sideinfo": "SideInfo      (Step6)",
    "cl4srec":  "CL4SRec       (Step7)",
    "llmrec":   "LLMRec        (前沿)",
}

# WBPR 历史指标（上次实验已完成，直接硬编码结果）
# 来源：output/two_tower_comparison.csv
WBPR_KNOWN_METRICS = {
    "model_key":        "wbpr",
    "model_name":       "TwoTower-WBPR",
    "train_min":        0.0,   # 历史结果，不重新计时
    "n_users":          1411,
    "hit_rate":         0.02095,
    "avg_watch_ratio":  0.01724,
    "ndcg":             0.00279,
}


# ══════════════════════════════════════════════════════════════════════
# 日志：同时打印到 stdout 和 log 文件
# ══════════════════════════════════════════════════════════════════════

class _Tee:
    """将 stdout 同时写入日志文件。"""
    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout

    def write(self, msg: str) -> int:
        self._stdout.write(msg)
        self._file.write(msg)
        return len(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def fileno(self):
        return self._stdout.fileno()


def _setup_logging(log_path: Path):
    sys.stdout = _Tee(log_path)
    sys.stderr = sys.stdout


# ══════════════════════════════════════════════════════════════════════
# 结果持久化（逐模型 JSON + 推荐列表 pickle）
# ══════════════════════════════════════════════════════════════════════

def _result_path(model_key: str) -> Path:
    return RESULT_DIR / f"{model_key}_result.json"

def _recs_path(model_key: str) -> Path:
    """推荐列表持久化路径（pickle，用于重启后的显著性检验）。"""
    return RESULT_DIR / f"{model_key}_recs.pkl"


def _load_result(model_key: str) -> dict | None:
    p = _result_path(model_key)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_recs(model_key: str, recs: dict) -> None:
    """把 {uid: [vid,...]} 存成 pickle，重启后可直接加载做显著性检验。"""
    import pickle
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_recs_path(model_key), "wb") as f:
        pickle.dump(recs, f, protocol=4)


def _load_recs(model_key: str) -> dict | None:
    """从 pickle 加载推荐列表，不存在则返回 None。"""
    import pickle
    p = _recs_path(model_key)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save_result(model_key: str, metrics: dict, train_sec: float) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "model_key":  model_key,
        "model_name": metrics.get("model", model_key),
        "train_sec":  round(train_sec, 1),
        "train_min":  round(train_sec / 60, 2),
        "n_users":    metrics.get("n_users", 0),
        "n_items":    metrics.get("n_items", 0),
        "bpr_loss":   metrics.get("bpr_loss", 0.0),
        # eval 指标在后面填充
        "hit_rate":         None,
        "avg_watch_ratio":  None,
        "ndcg":             None,
    }
    with open(_result_path(model_key), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _update_result_metrics(model_key: str, eval_metrics: dict) -> None:
    p = _result_path(model_key)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    data.update({
        "hit_rate":        eval_metrics.get("hit_rate"),
        "avg_watch_ratio": eval_metrics.get("avg_watch_ratio"),
        "ndcg":            eval_metrics.get("ndcg"),
        "n_eval_users":    eval_metrics.get("n_users"),
    })
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════
# 模型训练入口
# ══════════════════════════════════════════════════════════════════════

def _train_model(
    model_key: str,
    n_epochs: int,
    top_k: int,
    patience: int,
) -> dict:
    """训练单个模型，返回 pipeline 的完整 result dict（含 recommendations）。"""
    kwargs: dict[str, Any] = dict(
        n_epochs=n_epochs,
        top_k=top_k,
        patience=patience,
        output_dir=OUTPUT_DIR,
        checkpoint_dir=CKPT_DIR,
    )

    if model_key == "wbpr":
        # TwoTower-WBPR：checkpoint 已存在则跳过训练直接推理
        from two_tower import run_two_tower_pipeline
        return run_two_tower_pipeline(**kwargs, weighted=True)

    elif model_key == "sasrec":
        from sasrec import run_sasrec_pipeline
        return run_sasrec_pipeline(**kwargs)

    elif model_key == "bert4rec":
        from bert4rec import run_bert4rec_pipeline
        return run_bert4rec_pipeline(**kwargs)

    elif model_key == "sideinfo":
        from sideinfo_rec import run_sideinfo_pipeline
        return run_sideinfo_pipeline(**kwargs)

    elif model_key == "cl4srec":
        from cl4srec import run_cl4srec_pipeline
        return run_cl4srec_pipeline(**kwargs)

    elif model_key == "llmrec":
        from llm_rec import run_llm_rec_pipeline
        return run_llm_rec_pipeline(
            recall_k=top_k,
            top_k=top_k,
            n_epochs=n_epochs,
            patience=patience,
            output_dir=OUTPUT_DIR,
            checkpoint_dir=CKPT_DIR,
        )

    else:
        raise ValueError(f"未知模型：{model_key}")


# ══════════════════════════════════════════════════════════════════════
# 统计显著性（逐实验）
# ══════════════════════════════════════════════════════════════════════

def _significance_report(
    ctrl_key: str,
    exp_key:  str,
    ctrl_recs: dict,
    exp_recs:  dict,
    ground_truth: dict,
    top_k: int,
    exp_id: str,
    description: str,
) -> dict:
    from eval_recommenders import evaluate
    from ab_test import t_test

    m_ctrl = evaluate(ctrl_recs, ground_truth, top_k=top_k, return_per_user=True)
    m_exp  = evaluate(exp_recs,  ground_truth, top_k=top_k, return_per_user=True)
    pu_ctrl = m_ctrl["_per_user"]
    pu_exp  = m_exp["_per_user"]

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  实验 {exp_id}  {description}")
    print(f"  对照组：{DISPLAY_NAME[ctrl_key]}   实验组：{DISPLAY_NAME[exp_key]}")
    print(sep)

    results = {}
    for metric, key in [
        ("Hit Rate@K",      "hit_rates"),
        ("avg_watch_ratio", "avg_wrs"),
        ("NDCG@K",          "ndcgs"),
    ]:
        ctrl_mean = sum(pu_ctrl[key]) / len(pu_ctrl[key]) if pu_ctrl[key] else 0
        exp_mean  = sum(pu_exp[key])  / len(pu_exp[key])  if pu_exp[key]  else 0
        lift      = (exp_mean - ctrl_mean) / (ctrl_mean + 1e-9)
        res       = t_test(pu_ctrl[key], pu_exp[key])
        sig       = "显著 ✓" if res["is_significant"] else "不显著 ✗"
        print(
            f"  {metric:20s}  ctrl={ctrl_mean:.4f}  exp={exp_mean:.4f}"
            f"  lift={lift:+.2%}  p={res['p_value']:.4f}  {sig}"
        )
        results[key] = {
            "ctrl": ctrl_mean, "exp": exp_mean,
            "lift": lift, "p_value": res["p_value"],
            "significant": res["is_significant"],
        }
    print(sep)
    return results


# ══════════════════════════════════════════════════════════════════════
# 最终汇总表
# ══════════════════════════════════════════════════════════════════════

FULL_ORDER = ["wbpr", "sasrec", "bert4rec", "sideinfo", "cl4srec", "llmrec"]


def _print_final_table(all_metrics: dict[str, dict]) -> None:
    """打印所有模型的最终指标对比表，按 Step3→4→5→6→7→前沿 排列。"""
    header = f"{'模型':<30} {'HR@K':>8} {'WR@K':>8} {'NDCG@K':>8} {'vs上一步':>10} {'训练时长':>10}"
    sep    = "─" * 76
    print(f"\n{'═'*76}")
    print("  最终汇总：推荐系统演进链路（Step3 → Step4 → Step5 → Step6 → Step7 → 前沿）")
    print(f"{'═'*76}")
    print(header)
    print(sep)

    prev_ndcg = None
    for key in FULL_ORDER:
        if key not in all_metrics:
            continue
        m    = all_metrics[key]
        hr   = m.get("hit_rate")        or 0.0
        wr   = m.get("avg_watch_ratio") or 0.0
        ndcg = m.get("ndcg")            or 0.0
        mins = m.get("train_min")       or 0.0

        # vs 上一步的 NDCG 提升（链式对比）
        if prev_ndcg is not None and prev_ndcg > 0:
            vs_prev = f"{(ndcg - prev_ndcg)/prev_ndcg:+.1%}"
        else:
            vs_prev = "—"
        prev_ndcg = ndcg

        name = DISPLAY_NAME.get(key, key)
        print(
            f"  {name:<28} {hr:>7.4f}  {wr:>7.4f}  "
            f"{ndcg:>7.4f}  {vs_prev:>9}  {mins:>6.1f}min"
        )
    print(f"{'═'*76}\n")


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Steps 4-7 + LLM 全流程实验（断连安全版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models",    nargs="+", choices=MODEL_ORDER, default=None,
                        help="只跑指定模型（默认全部，不含 LLM）")
    parser.add_argument("--with-llm",  action="store_true",
                        help="包含 LLMRec 实验（需要 Ollama 在服务器上运行）")
    parser.add_argument("--force",     action="store_true",
                        help="强制重新训练（忽略已有 result JSON）")
    parser.add_argument("--n-epochs",  type=int, default=50)
    parser.add_argument("--patience",  type=int, default=5)
    parser.add_argument("--top-k",     type=int, default=50)
    args = parser.parse_args()

    # ── 创建目录、启动日志 ────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_logging(LOG_PATH)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"  实验启动：{ts}")
    print(f"  参数：n_epochs={args.n_epochs}  patience={args.patience}  top_k={args.top_k}")
    print(f"  日志：{LOG_PATH}")
    print(f"  checkpoint 目录：{CKPT_DIR}")
    print(f"  结果目录：{RESULT_DIR}")
    print(f"{'='*70}\n")

    # ── 确定要跑的模型 ────────────────────────────────────────────────
    if args.models:
        run_keys = [k for k in args.models if k != "wbpr"]
    else:
        run_keys = [k for k in MODEL_ORDER if k != "llmrec"] + (["llmrec"] if args.with_llm else [])

    # wbpr 始终加在最前面（实验 A 的对照组），让显著性检验能拿到推荐列表
    run_keys = ["wbpr"] + [k for k in run_keys if k != "wbpr"]

    print(f"  待训练模型：{run_keys}\n")

    # ── 逐模型训练 ────────────────────────────────────────────────────
    # 内存中保留 recommendations 供后续实验对比（不写大文件）
    all_recs:    dict[str, dict]  = {}   # model_key -> {uid: [vid,...]}
    all_metrics: dict[str, dict]  = {}   # model_key -> result JSON dict

    # 先加载已有结果（用于跳过）；wbpr 直接注入历史指标
    all_metrics["wbpr"] = WBPR_KNOWN_METRICS
    for key in MODEL_ORDER:
        saved = _load_result(key)
        if saved:
            all_metrics[key] = saved

    for model_key in run_keys:
        display = DISPLAY_NAME.get(model_key, model_key)
        saved   = _load_result(model_key)

        if saved and not args.force:
            print(f"[SKIP] {display} — 已有结果（{saved.get('train_min',0):.1f}min），跳过训练。")
            all_metrics[model_key] = saved
            # 尝试从磁盘加载推荐列表（显著性检验需要）
            cached_recs = _load_recs(model_key)
            if cached_recs is not None:
                all_recs[model_key] = cached_recs
                print(f"       推荐列表已从缓存加载（{len(cached_recs):,} 用户）。\n")
            else:
                print(f"       推荐列表缓存不存在，显著性检验时将重新推理。\n")
            continue

        print(f"\n{'━'*70}")
        print(f"  开始训练：{display}")
        print(f"  时间：{time.strftime('%H:%M:%S')}")
        print(f"{'━'*70}")

        t0 = time.time()
        try:
            result = _train_model(model_key, args.n_epochs, args.top_k, args.patience)
            elapsed = time.time() - t0

            # 保存推荐列表：内存 + 磁盘（断连重启后可直接加载）
            all_recs[model_key] = result["recommendations"]
            _save_recs(model_key, result["recommendations"])

            # 持久化基础指标
            _save_result(model_key, result, elapsed)
            all_metrics[model_key] = _load_result(model_key)

            print(f"\n[OK] {display} 训练完成，耗时 {elapsed/60:.1f} 分钟。")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n[ERROR] {display} 训练失败（{elapsed/60:.1f}min）：{e}")
            traceback.print_exc()
            print("  → 跳过该模型，继续下一个。\n")
            continue

    # ── 离线评估（加载 small_matrix 答案本）────────────────────────────
    print(f"\n{'━'*70}")
    print("  离线评估：加载 small_matrix 答案本……")
    print(f"{'━'*70}")

    try:
        from eval_recommenders import load_ground_truth, evaluate
        ground_truth = load_ground_truth()
    except Exception as e:
        print(f"[ERROR] 无法加载 ground truth：{e}")
        print("  评估跳过，结果文件仍已保存。")
        ground_truth = None

    if ground_truth:
        for model_key in run_keys:
            # 如果内存里有推荐列表（刚训练完的），直接用；
            # 否则需要重新推理（跳过了训练但要评估）
            if model_key not in all_recs:
                saved = _load_result(model_key)
                if saved and saved.get("hit_rate") is not None:
                    print(f"[SKIP] {DISPLAY_NAME[model_key]} 评估结果已存在，跳过。")
                    continue
                # 重新推理（0 epoch训练，直接加载 best checkpoint 推理）
                print(f"[INFO] {DISPLAY_NAME[model_key]} 无内存推荐列表，重新推理……")
                try:
                    result = _train_model(model_key, 0, args.top_k, args.patience)
                    all_recs[model_key] = result["recommendations"]
                except Exception as e:
                    print(f"[ERROR] 重新推理失败：{e}")
                    continue

            recs    = all_recs[model_key]
            metrics = evaluate(recs, ground_truth, top_k=args.top_k)
            _update_result_metrics(model_key, metrics)
            all_metrics[model_key] = _load_result(model_key)
            print(
                f"  {DISPLAY_NAME[model_key]:<26}  "
                f"HR={metrics['hit_rate']:.4f}  "
                f"WR={metrics['avg_watch_ratio']:.4f}  "
                f"NDCG={metrics['ndcg']:.4f}"
            )

    # ── 逐实验显著性检验 ──────────────────────────────────────────────
    if ground_truth and all_recs:
        print(f"\n{'━'*70}")
        print("  统计显著性检验（各实验对照组 vs 实验组）")
        print(f"{'━'*70}")

        sig_results: dict[str, dict] = {}
        for ctrl_key, exp_key, exp_id, description in EXPERIMENTS:
            if ctrl_key not in all_recs or exp_key not in all_recs:
                print(f"\n[SKIP] 实验 {exp_id}：{ctrl_key} 或 {exp_key} 的推荐列表不在内存中。")
                continue
            sig = _significance_report(
                ctrl_key=ctrl_key,
                exp_key=exp_key,
                ctrl_recs=all_recs[ctrl_key],
                exp_recs=all_recs[exp_key],
                ground_truth=ground_truth,
                top_k=args.top_k,
                exp_id=exp_id,
                description=description,
            )
            sig_results[f"exp_{exp_id}"] = sig

        # 保存显著性检验结果
        sig_path = RESULT_DIR / "significance_tests.json"
        with open(sig_path, "w", encoding="utf-8") as f:
            json.dump(sig_results, f, ensure_ascii=False, indent=2)
        print(f"\n  显著性检验结果已保存：{sig_path}")

    # ── 最终汇总 ──────────────────────────────────────────────────────
    _print_final_table(all_metrics)

    # 导出汇总 CSV
    try:
        import pandas as pd
        rows = []
        for key in FULL_ORDER:
            if key not in all_metrics:
                continue
            m = all_metrics[key]
            rows.append({
                "model":           DISPLAY_NAME.get(key, key),
                "hit_rate":        m.get("hit_rate"),
                "avg_watch_ratio": m.get("avg_watch_ratio"),
                "ndcg":            m.get("ndcg"),
                "train_min":       m.get("train_min"),
                "n_users":         m.get("n_users"),
            })
        if rows:
            df = pd.DataFrame(rows)
            csv_path = OUTPUT_DIR / "all_models_comparison.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"  汇总 CSV 已保存：{csv_path}")
    except Exception as e:
        print(f"  [WARN] 导出 CSV 失败：{e}")

    print(f"\n{'='*70}")
    print(f"  全部实验完成！时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  结果目录：{RESULT_DIR}")
    print(f"  日志文件：{LOG_PATH}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
