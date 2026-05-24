"""
文件用途：Steps 4-7 + LLM 前沿路线的统一对比评估框架。

【模型演进路线】
  Step 4: SASRec        （因果自注意力序列推荐）
  Step 5: BERT4Rec      （双向自注意力 + Masked Item Prediction）
  Step 6: SideInfo-SASRec（序列推荐 + 视频内容特征融合）
  Step 7: CL4SRec       （对比学习增强序列推荐）
  前沿:   LLMRec        （SASRec 召回 + LLM 精排）

【评估指标（与 eval_recommenders.py 相同）】
  Hit Rate@K / avg_watch_ratio@K / NDCG@K，在 small_matrix 上评估。

【用法示例】
  # 单模型跑通
  python eval_advanced.py --models sasrec bert4rec

  # 全量对比（全部 5 个模型，默认）
  python eval_advanced.py

  # 跳过 LLM（Ollama 未安装时）
  python eval_advanced.py --skip-llm

  # 配合 checkpoint 恢复（中断后续训）
  python eval_advanced.py --checkpoint-dir output/checkpoints
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from eval_recommenders import evaluate, load_ground_truth, print_comparison
from ab_test import t_test


# ══════════════════════════════════════════════════════════════════════
# 模型注册表
# ══════════════════════════════════════════════════════════════════════

def _run_sasrec(kwargs: dict) -> dict:
    from sasrec import run_sasrec_pipeline
    return run_sasrec_pipeline(**kwargs)

def _run_bert4rec(kwargs: dict) -> dict:
    from bert4rec import run_bert4rec_pipeline
    return run_bert4rec_pipeline(**kwargs)

def _run_sideinfo(kwargs: dict) -> dict:
    from sideinfo_rec import run_sideinfo_pipeline
    return run_sideinfo_pipeline(**kwargs)

def _run_cl4srec(kwargs: dict) -> dict:
    from cl4srec import run_cl4srec_pipeline
    return run_cl4srec_pipeline(**kwargs)

def _run_llmrec(kwargs: dict) -> dict:
    from llm_rec import run_llm_rec_pipeline
    # llm_rec 接口参数名略有不同
    llm_kwargs = {k: v for k, v in kwargs.items() if k in (
        "recall_k", "top_k", "n_epochs", "patience",
        "eligible_video_ids", "output_dir", "checkpoint_dir", "_test_df",
        "llm_model", "ollama_url", "max_hist",
    )}
    llm_kwargs.setdefault("recall_k", kwargs.get("top_k", 50))
    return run_llm_rec_pipeline(**llm_kwargs)


MODEL_REGISTRY = {
    "sasrec":    _run_sasrec,
    "bert4rec":  _run_bert4rec,
    "sideinfo":  _run_sideinfo,
    "cl4srec":   _run_cl4srec,
    "llmrec":    _run_llmrec,
}

MODEL_DISPLAY = {
    "sasrec":   "Step4  SASRec",
    "bert4rec": "Step5  BERT4Rec",
    "sideinfo": "Step6  SideInfo-SASRec",
    "cl4srec":  "Step7  CL4SRec",
    "llmrec":   "Front  LLMRec",
}


# ══════════════════════════════════════════════════════════════════════
# 统计显著性：相邻模型两两 t-test
# ══════════════════════════════════════════════════════════════════════

def _run_significance_tests(
    model_results: list[dict],
    ground_truth: dict,
    top_k: int,
) -> None:
    """对相邻两个模型做 Welch t-test，打印显著性结论。"""
    if len(model_results) < 2:
        return

    per_user_metrics = []
    for r in model_results:
        m = evaluate(r["recommendations"], ground_truth, top_k=top_k, return_per_user=True)
        per_user_metrics.append((r["model"], m["_per_user"]))

    print("\n── 统计显著性检验（相邻模型 Welch t-test，α=0.05）────────────────────")
    for i in range(len(per_user_metrics) - 1):
        name_a, pu_a = per_user_metrics[i]
        name_b, pu_b = per_user_metrics[i + 1]
        print(f"\n  {name_a}  vs  {name_b}")
        for metric, key in [
            ("Hit Rate@K",      "hit_rates"),
            ("avg_watch_ratio", "avg_wrs"),
            ("NDCG@K",          "ndcgs"),
        ]:
            res = t_test(pu_a[key], pu_b[key])
            sig = "显著 ✓" if res["is_significant"] else "不显著 ✗"
            print(f"    {metric:20s}  p={res['p_value']:.4f}  {sig}")
    print("────────────────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════
# 对比表输出（含训练时间列）
# ══════════════════════════════════════════════════════════════════════

def _compare_and_print(
    model_results: list[dict],
    ground_truth: dict,
    top_k: int,
    train_times: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for r in model_results:
        name = r["model"]
        metrics = evaluate(r["recommendations"], ground_truth, top_k=top_k)
        rows.append({
            "model":           name,
            "n_users":         metrics["n_users"],
            "hit_rate":        metrics["hit_rate"],
            "avg_watch_ratio": metrics["avg_watch_ratio"],
            "ndcg":            metrics["ndcg"],
            "train_min":       round(train_times.get(name, 0) / 60, 1),
        })

    df = pd.DataFrame(rows)
    baseline_hr   = df.loc[0, "hit_rate"]
    baseline_wr   = df.loc[0, "avg_watch_ratio"]
    baseline_ndcg = df.loc[0, "ndcg"]
    df["hr_lift"]   = (df["hit_rate"]        - baseline_hr)   / (baseline_hr   + 1e-9)
    df["wr_lift"]   = (df["avg_watch_ratio"]  - baseline_wr)   / (baseline_wr   + 1e-9)
    df["ndcg_lift"] = (df["ndcg"]             - baseline_ndcg) / (baseline_ndcg + 1e-9)

    fmt = {
        "n_users":         lambda x: f"{int(x):,}",
        "hit_rate":        lambda x: f"{x:.4f}",
        "avg_watch_ratio": lambda x: f"{x:.4f}",
        "ndcg":            lambda x: f"{x:.4f}",
        "train_min":       lambda x: f"{x:.1f}min",
        "hr_lift":         lambda x: f"{x:+.2%}",
        "wr_lift":         lambda x: f"{x:+.2%}",
        "ndcg_lift":       lambda x: f"{x:+.2%}",
    }
    display = df.copy()
    for col, fn in fmt.items():
        if col in display.columns:
            display[col] = display[col].apply(fn)

    print("\n" + "═" * 100)
    print("  推荐系统演进对比（Step 4 → Step 5 → Step 6 → Step 7 → 前沿）")
    print("  基线 = 第一行（SASRec）")
    print("═" * 100)
    print(display.to_string(index=False))
    print("═" * 100)
    print("注：指标在 small_matrix 上评估；lift = 相对 SASRec 的提升幅度。\n")

    return df


# ══════════════════════════════════════════════════════════════════════
# 主对比流程
# ══════════════════════════════════════════════════════════════════════

def run_advanced_comparison(
    models: list[str] | None = None,
    n_epochs: int = 50,
    top_k: int = 50,
    patience: int = 5,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    skip_llm: bool = False,
) -> pd.DataFrame:
    """
    训练并评估指定模型列表，打印对比表 + 统计显著性。

    models: 列表，可选 ["sasrec", "bert4rec", "sideinfo", "cl4srec", "llmrec"]
            默认全跑（skip_llm=True 时排除 llmrec）。
    """
    default_order = ["sasrec", "bert4rec", "sideinfo", "cl4srec", "llmrec"]
    if models is None:
        models = [m for m in default_order if not (skip_llm and m == "llmrec")]
    else:
        models = [m.lower() for m in models if m.lower() in MODEL_REGISTRY]
        if skip_llm and "llmrec" in models:
            models.remove("llmrec")

    if not models:
        raise ValueError("没有有效的模型名称。可选：" + ", ".join(MODEL_REGISTRY))

    ckpt_root = (Path(checkpoint_dir) if checkpoint_dir else
                 (Path(output_dir) / "checkpoints" if output_dir else Path("output/checkpoints")))

    shared_kwargs: dict[str, Any] = dict(
        n_epochs=n_epochs,
        top_k=top_k,
        patience=patience,
        eligible_video_ids=eligible_video_ids,
        output_dir=None,
        checkpoint_dir=ckpt_root,
    )

    model_results: list[dict] = []
    train_times:   dict[str, float] = {}

    for model_key in models:
        display = MODEL_DISPLAY.get(model_key, model_key)
        print("\n" + "═" * 70)
        print(f"  训练模型：{display}")
        print("═" * 70)

        runner = MODEL_REGISTRY[model_key]
        t0 = time.time()
        result = runner(shared_kwargs)
        elapsed = time.time() - t0

        train_times[result["model"]] = elapsed
        model_results.append(result)
        print(f"  [{display}] 完成，耗时 {elapsed/60:.1f} 分钟。")

    print("\n[Eval] 加载 small_matrix 答案本……")
    ground_truth = load_ground_truth()

    df_cmp = _compare_and_print(model_results, ground_truth, top_k, train_times)
    _run_significance_tests(model_results, ground_truth, top_k)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_path = out / "advanced_comparison.csv"
        df_cmp.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"[Eval] 对比结果已保存：{save_path}")

    return df_cmp


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Steps 4-7 + LLM 推荐系统演进对比评估框架。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python eval_advanced.py                          # 全量跑（含 LLMRec）
  python eval_advanced.py --skip-llm               # 跳过 LLMRec
  python eval_advanced.py --models sasrec bert4rec  # 只跑指定模型
  python eval_advanced.py --n-epochs 30 --patience 3  # 快速验证
        """,
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=None,
        help="要运行的模型（默认全部）",
    )
    parser.add_argument("--n-epochs",       type=int, default=50)
    parser.add_argument("--top-k",          type=int, default=50)
    parser.add_argument("--patience",       type=int, default=5)
    parser.add_argument("--skip-llm",       action="store_true",
                        help="跳过 LLMRec（Ollama 未安装时用）")
    parser.add_argument("--output-dir",     type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )
    ckpt = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    run_advanced_comparison(
        models=args.models,
        n_epochs=args.n_epochs,
        top_k=args.top_k,
        patience=args.patience,
        output_dir=out,
        checkpoint_dir=ckpt,
        skip_llm=args.skip_llm,
    )
