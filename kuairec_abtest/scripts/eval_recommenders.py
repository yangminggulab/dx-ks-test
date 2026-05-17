"""
文件用途：离线评估推荐模型，不依赖 MySQL，直接读 small_matrix CSV。

【评估逻辑】
  big_matrix → 训练模型 → 每人 top-K 推荐列表
                                    ↓
  small_matrix（答案本）→ 查推荐列表里的视频真实完播率
                                    ↓
  计算指标：Hit Rate@K / avg_watch_ratio@K / NDCG@K

【为什么用 small_matrix 做答案本？】
  small_matrix 是密集矩阵（~1000 用户 × ~3700 视频全量覆盖），
  每个用户几乎看过所有视频，watch_ratio 有可信的真实值。
  big_matrix 是稀疏的（大部分 user-item 对没有记录），
  用它做答案会把"没记录"误判为"不喜欢"。

【三个指标的含义】
  Hit Rate@K    推荐的 K 个视频里，用户在 small_matrix 里真实看过的比例
                越高 = 推荐的视频用户真的会看
  avg_watch_ratio@K  推荐视频的平均完播率（只计在 small_matrix 里有记录的）
                越高 = 推荐的视频用户不只是看了还看完了
  NDCG@K        Normalized Discounted Cumulative Gain，排名靠前的命中贡献更多
                综合衡量"命中了多少"和"命中的排在前面吗"
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))


def _find_small_matrix_csv() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "KuaiRec 2.0" / "data" / "small_matrix.csv",
        Path("/Users/liubike/Desktop/快手test/kuairec_abtest/data/KuaiRec 2.0/data/small_matrix.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_ground_truth(min_watch_ratio: float = 0.0) -> dict[Any, dict[Any, float]]:
    """
    从 small_matrix.csv 加载答案本。

    Returns:
        {user_id: {video_id: watch_ratio}}
        只保留 watch_ratio >= min_watch_ratio 的记录（可以过滤掉几乎没看的）
    """
    path = _find_small_matrix_csv()
    if not path.exists():
        raise FileNotFoundError(f"small_matrix.csv 不存在：{path}")

    df = pd.read_csv(path, usecols=["user_id", "video_id", "watch_ratio"])
    df["watch_ratio"] = pd.to_numeric(df["watch_ratio"], errors="coerce").fillna(0.0).clip(0.0)

    if min_watch_ratio > 0:
        df = df[df["watch_ratio"] >= min_watch_ratio]

    gt: dict[Any, dict[Any, float]] = {}
    for uid, grp in df.groupby("user_id"):
        gt[uid] = dict(zip(grp["video_id"], grp["watch_ratio"]))

    print(f"[Eval] 答案本加载完成：{len(gt):,} 用户，"
          f"共 {df.shape[0]:,} 条 (user, video) 记录。")
    return gt


def evaluate(
    recommendations: dict[Any, list[Any]],
    ground_truth: dict[Any, dict[Any, float]],
    top_k: int = 50,
) -> dict[str, float]:
    """
    计算推荐列表在 ground_truth 上的离线指标。

    只评估 recommendations 和 ground_truth 都有的用户（交集）。

    Returns:
        {hit_rate, avg_watch_ratio, ndcg, n_users}
    """
    common_users = set(recommendations.keys()) & set(ground_truth.keys())
    if not common_users:
        print("[Eval] 警告：推荐用户和答案本没有交集，无法评估。")
        return {"hit_rate": 0.0, "avg_watch_ratio": 0.0, "ndcg": 0.0, "n_users": 0}

    hit_rates, avg_wrs, ndcgs = [], [], []

    for uid in common_users:
        recs = recommendations[uid][:top_k]
        user_gt = ground_truth[uid]

        if not recs:
            continue

        # Hit Rate@K：推荐中有多少个视频用户真的看了
        hits = [1 if vid in user_gt else 0 for vid in recs]
        hit_rates.append(sum(hits) / len(recs))

        # avg_watch_ratio@K：推荐视频的平均完播率（未看过计 0）
        avg_wrs.append(sum(user_gt.get(vid, 0.0) for vid in recs) / len(recs))

        # NDCG@K：排名越靠前的命中贡献越大（log2 折扣）
        dcg  = sum(user_gt.get(vid, 0.0) / np.log2(i + 2) for i, vid in enumerate(recs))
        # 理想情况：把该用户所有真实 watch_ratio 从高到低排列
        ideal = sorted(user_gt.values(), reverse=True)[:len(recs)]
        idcg  = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "n_users":          len(common_users),
        "hit_rate":         float(np.mean(hit_rates)),
        "avg_watch_ratio":  float(np.mean(avg_wrs)),
        "ndcg":             float(np.mean(ndcgs)),
    }


def compare_models(
    model_results: list[dict[str, Any]],
    ground_truth: dict[Any, dict[Any, float]],
    top_k: int = 50,
) -> pd.DataFrame:
    """
    批量评估多个模型，返回对比 DataFrame。

    model_results 是 run_two_tower_pipeline / run_svd_pipeline 的返回值列表。
    """
    rows = []
    for r in model_results:
        model_name = r.get("model", "unknown")
        print(f"\n[Eval] 评估模型：{model_name} ...")
        metrics = evaluate(r["recommendations"], ground_truth, top_k=top_k)
        rows.append({"model": model_name, **metrics})

    df = pd.DataFrame(rows)

    # 相对提升（以第一个模型为基线）
    baseline_hr = df.loc[0, "hit_rate"]
    baseline_wr = df.loc[0, "avg_watch_ratio"]
    baseline_ndcg = df.loc[0, "ndcg"]
    df["hit_rate_lift"]        = (df["hit_rate"] - baseline_hr) / (baseline_hr + 1e-9)
    df["avg_watch_ratio_lift"] = (df["avg_watch_ratio"] - baseline_wr) / (baseline_wr + 1e-9)
    df["ndcg_lift"]            = (df["ndcg"] - baseline_ndcg) / (baseline_ndcg + 1e-9)

    return df


def print_comparison(df: pd.DataFrame) -> None:
    """格式化打印对比表。"""
    fmt = {
        "n_users":           lambda x: f"{int(x):,}",
        "hit_rate":          lambda x: f"{x:.4f}",
        "avg_watch_ratio":   lambda x: f"{x:.4f}",
        "ndcg":              lambda x: f"{x:.4f}",
        "hit_rate_lift":     lambda x: f"{x:+.2%}",
        "avg_watch_ratio_lift": lambda x: f"{x:+.2%}",
        "ndcg_lift":         lambda x: f"{x:+.2%}",
    }
    display = df.copy()
    for col, fn in fmt.items():
        if col in display.columns:
            display[col] = display[col].apply(fn)

    print("\n" + "═" * 80)
    print("  离线评估对比（基线 = 第一行）")
    print("═" * 80)
    print(display.to_string(index=False))
    print("═" * 80)
    print("注：Hit Rate / avg_watch_ratio / NDCG 均在 small_matrix 上评估。")
    print("    lift 为相对第一个模型的提升幅度（正值 = 更好）。\n")


# ══════════════════════════════════════════════════════════════════════
# 主对比流程：TwoTower-BPR vs TwoTower-WBPR
# ══════════════════════════════════════════════════════════════════════

def run_comparison(
    n_epochs: int = 20,
    top_k: int = 50,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    训练 BPR 和 WBPR 两个双塔模型，在 small_matrix 上对比评估。
    """
    from two_tower import run_two_tower_pipeline

    shared = dict(
        n_epochs=n_epochs, top_k=top_k,
        eligible_video_ids=eligible_video_ids,
        output_dir=output_dir, _test_df=_test_df,
    )

    print("\n" + "═" * 60)
    print("  第一轮：TwoTower-BPR（对照，普通 BPR）")
    print("═" * 60)
    r_bpr = run_two_tower_pipeline(**shared, weighted=False)

    print("\n" + "═" * 60)
    print("  第二轮：TwoTower-WBPR（实验，watch_ratio 加权）")
    print("═" * 60)
    r_wbpr = run_two_tower_pipeline(**shared, weighted=True)

    gt = load_ground_truth()
    df_cmp = compare_models([r_bpr, r_wbpr], gt, top_k=top_k)
    print_comparison(df_cmp)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_cmp.to_csv(output_dir / "two_tower_comparison.csv", index=False, encoding="utf-8-sig")
        print(f"[Eval] 对比结果已保存：{output_dir / 'two_tower_comparison.csv'}")

    return df_cmp


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="对比 TwoTower-BPR vs TwoTower-WBPR。")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--top-k",    type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )

    run_comparison(n_epochs=args.n_epochs, top_k=args.top_k, output_dir=out)
