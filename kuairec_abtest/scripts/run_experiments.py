"""
run_experiments.py — 推荐系统实验主程序（干净版）

用法:
  python run_experiments.py                          # 跑全部 4 个模型
  python run_experiments.py --models sasrec bert4rec # 只跑指定模型
  python run_experiments.py --force                  # 忽略已有结果重新跑
  python run_experiments.py --n-epochs 10 --top-k 20 --patience 3
"""
from __future__ import annotations

import argparse, json, pickle, sys, time, traceback
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

OUTPUT_DIR = PROJECT_DIR / "output"
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
RESULT_DIR = OUTPUT_DIR / "results"

MODEL_ORDER = ["sasrec", "bert4rec", "sideinfo", "cl4srec"]

EXPERIMENTS = [
    ("wbpr",     "sasrec",   "A", "静态双塔 vs 序列Transformer"),
    ("sasrec",   "bert4rec", "B", "单向 vs 双向注意力"),
    ("bert4rec", "sideinfo", "C", "纯ID序列 vs +视频侧特征"),
    ("sideinfo", "cl4srec",  "D", "监督学习 vs +对比学习"),
]

DISPLAY_NAME = {
    "wbpr": "TwoTower-WBPR (Step3)", "sasrec": "SASRec (Step4)",
    "bert4rec": "BERT4Rec (Step5)",  "sideinfo": "SideInfo (Step6)",
    "cl4srec": "CL4SRec (Step7)",
}

WBPR_KNOWN = {
    "model_key": "wbpr", "hit_rate": 0.02095, "avg_watch_ratio": 0.01724,
    "ndcg": 0.00279, "train_min": 0.0,
}


# ── 持久化 ────────────────────────────────────────────────────────────

def _rpath(k): return RESULT_DIR / f"{k}_result.json"
def _ppath(k): return RESULT_DIR / f"{k}_recs.pkl"

def _load_result(k):
    p = _rpath(k)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

def _save_result(k, n_users, n_items, elapsed):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _rpath(k).write_text(json.dumps({"model_key": k, "n_users": n_users, "n_items": n_items,
        "train_sec": round(elapsed, 1), "train_min": round(elapsed / 60, 2)},
        ensure_ascii=False, indent=2), encoding="utf-8")

def _save_recs(k, recs):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ppath(k), "wb") as f: pickle.dump(recs, f, protocol=4)

def _load_recs(k):
    p = _ppath(k)
    if not p.exists(): return None
    with open(p, "rb") as f: return pickle.load(f)


# ── 模型工厂 ──────────────────────────────────────────────────────────

def _build_model(key, data):
    from models import SASRec, BERT4Rec, SideInfoRec, CL4SRec
    m = {"sasrec": SASRec, "bert4rec": BERT4Rec, "sideinfo": SideInfoRec, "cl4srec": CL4SRec}
    if key not in m: raise ValueError(f"未知模型：{key}")
    return m[key](data, OUTPUT_DIR, CKPT_DIR)


# ── 评估 + 显著性检验 ─────────────────────────────────────────────────

def _evaluate_all(all_recs, ground_truth, top_k, run_keys):
    from eval_recommenders import evaluate
    from ab_test import t_test
    all_metrics = {}
    for key in run_keys:
        if key not in all_recs: continue
        m = evaluate(all_recs[key], ground_truth, top_k=top_k)
        all_metrics[key] = m
        saved = _load_result(key) or {}
        saved.update({"hit_rate": m["hit_rate"], "avg_watch_ratio": m["avg_watch_ratio"],
                      "ndcg": m["ndcg"]})
        _rpath(key).write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  {DISPLAY_NAME.get(key, key):<26}  HR={m['hit_rate']:.4f}  WR={m['avg_watch_ratio']:.4f}  NDCG={m['ndcg']:.4f}")

    for ctrl_k, exp_k, exp_id, desc in EXPERIMENTS:
        if exp_k not in run_keys or not all_recs.get(ctrl_k) or not all_recs.get(exp_k): continue
        mc = evaluate(all_recs[ctrl_k], ground_truth, top_k=top_k, return_per_user=True)
        me = evaluate(all_recs[exp_k],  ground_truth, top_k=top_k, return_per_user=True)
        print(f"\n{'─'*70}\n  实验 {exp_id}  {desc}\n{'─'*70}")
        for label, key in [("Hit Rate@K", "hit_rates"), ("avg_watch_ratio", "avg_wrs"), ("NDCG@K", "ndcgs")]:
            c = mc["_per_user"][key]; e = me["_per_user"][key]
            cm = sum(c)/len(c) if c else 0; em = sum(e)/len(e) if e else 0
            res = t_test(c, e)
            sig = "显著 ✓" if res["is_significant"] else "不显著 ✗"
            print(f"  {label:20s}  ctrl={cm:.4f}  exp={em:.4f}  lift={(em-cm)/(cm+1e-9):+.2%}  p={res['p_value']:.4f}  {sig}")
    return all_metrics


# ── 汇总表 ────────────────────────────────────────────────────────────

def _print_summary(all_metrics):
    print(f"\n{'='*76}\n  最终汇总：演进链路（Step3 → Step4 → Step5 → Step6 → Step7）\n{'='*76}")
    print(f"{'模型':<30} {'HR@K':>8} {'WR@K':>8} {'NDCG@K':>8} {'vs上一步':>10} {'训练时长':>10}\n{'─'*76}")
    prev = None
    for k in ["wbpr"] + MODEL_ORDER:
        if k not in all_metrics: continue
        m = all_metrics[k]
        hr, wr, ndcg, mins = (m.get(f) or 0.0 for f in ["hit_rate", "avg_watch_ratio", "ndcg", "train_min"])
        vs = f"{(ndcg - prev) / prev:+.1%}" if prev else "—"
        prev = ndcg
        print(f"  {DISPLAY_NAME.get(k,k):<28} {hr:>7.4f}  {wr:>7.4f}  {ndcg:>7.4f}  {vs:>9}  {mins:>6.1f}min")
    print(f"{'='*76}\n")


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="推荐系统实验主程序")
    p.add_argument("--models",   nargs="+", choices=MODEL_ORDER)
    p.add_argument("--force",    action="store_true")
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--top-k",    type=int, default=50)
    args = p.parse_args()

    for d in [OUTPUT_DIR, CKPT_DIR, RESULT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    run_keys = args.models or MODEL_ORDER
    print(f"\n{'='*70}\n  实验启动：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  参数：n_epochs={args.n_epochs}  patience={args.patience}  top_k={args.top_k}")
    print(f"  待训练模型：{run_keys}\n")

    # 1. 加载数据（只加载一次）
    from models.kuairec_loader import load_model_data
    data = load_model_data()

    # 2. 预加载已有缓存
    all_recs    = {k: v for k in MODEL_ORDER if (v := _load_recs(k)) is not None}
    all_metrics = {"wbpr": WBPR_KNOWN,
                   **{k: v for k in MODEL_ORDER if (v := _load_result(k)) is not None}}
    # 加入 wbpr 历史推荐（如果存在）
    wbpr_recs = _load_recs("wbpr")
    if wbpr_recs: all_recs["wbpr"] = wbpr_recs

    # 3. 逐模型训练
    for key in run_keys:
        saved = _load_result(key)
        if saved and not args.force:
            print(f"[SKIP] {DISPLAY_NAME.get(key, key)} — 已有结果（{saved.get('train_min',0):.1f}min），跳过。")
            all_metrics[key] = saved
            continue

        print(f"\n{'━'*70}\n  开始训练：{DISPLAY_NAME.get(key, key)}  {time.strftime('%H:%M:%S')}\n{'━'*70}")
        t0 = time.time()
        try:
            model = _build_model(key, data)
            model.train(n_epochs=args.n_epochs, patience=args.patience)
            recs    = model.recommend(top_k=args.top_k)
            elapsed = time.time() - t0
            all_recs[key] = recs
            _save_recs(key, recs)
            _save_result(key, len(data.user_ids), len(data.item_ids), elapsed)
            all_metrics[key] = _load_result(key)
            print(f"\n[OK] {DISPLAY_NAME.get(key, key)} 完成，耗时 {elapsed/60:.1f} 分钟。")
        except Exception as e:
            print(f"\n[ERROR] {DISPLAY_NAME.get(key, key)} 失败：{e}")
            traceback.print_exc()

    # 4. 离线评估 + 显著性检验
    print(f"\n{'━'*70}\n  离线评估……\n{'━'*70}")
    eval_m = _evaluate_all(all_recs, data.ground_truth, args.top_k, run_keys)
    all_metrics.update(eval_m)

    # 5. 汇总
    _print_summary(all_metrics)

    try:
        import pandas as pd
        rows = [{"model": DISPLAY_NAME.get(k, k), **{f: all_metrics[k].get(f)
                 for f in ["hit_rate", "avg_watch_ratio", "ndcg", "train_min"]}}
                for k in ["wbpr"] + MODEL_ORDER if k in all_metrics]
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "all_models_comparison.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"  [WARN] 导出 CSV 失败：{e}")

    print(f"  全部实验完成！{time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
