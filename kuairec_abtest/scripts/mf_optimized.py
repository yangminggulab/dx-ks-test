"""
文件用途：在基础 SVD 上加入两个关键优化。

优化一：Biased MF（Funk SVD）
  预测 = μ + b_用户 + b_视频 + U[用户] · V[视频]
  用 SGD 学习偏置项，分离"这个视频天然受欢迎"和"这个用户真的喜欢"。

优化二：iALS（Implicit ALS，Hu et al. 2008）
  把 watch_ratio 转成置信度 c = 1 + α·watch_ratio
  用交替最小二乘求解，0 不再是负反馈，而是低置信度的"偏好=1"。

两个算法都与 svd_recommender.py 保持相同的返回格式，
可直接接入 svd_visualization.py 做对比可视化。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from svd_recommender import (
    build_sparse_matrix,
    generate_personalized_recommendations,
    load_big_matrix_interactions,
    recommendations_to_dataframe,
)


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _compute_rmse_on_observed(
    u_arr: np.ndarray,
    i_arr: np.ndarray,
    r_arr: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    mu: float = 0.0,
    b_u: np.ndarray | None = None,
    b_i: np.ndarray | None = None,
) -> float:
    """在已有交互位置计算 RMSE。"""
    preds = (U[u_arr] * V[i_arr]).sum(axis=1) + mu
    if b_u is not None:
        preds += b_u[u_arr]
    if b_i is not None:
        preds += b_i[i_arr]
    return float(np.sqrt(np.mean((r_arr - preds) ** 2)))


def _top_k_from_factors(
    U: np.ndarray,
    V: np.ndarray,
    original_matrix: csr_matrix,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    top_k: int,
    mu: float = 0.0,
    b_u: np.ndarray | None = None,
    b_i: np.ndarray | None = None,
) -> dict[Any, list[Any]]:
    """
    用因子矩阵为所有用户生成 top-K 推荐（过滤已看）。
    支持偏置项（b_u / b_i），不存在时退化为基础点积。
    """
    print(f"[推荐] 为 {len(user_ids):,} 位用户生成 top-{top_k} 推荐……")
    seen = original_matrix.tolil()
    # 预加全局偏置到物品偏置，减少循环内计算量
    item_bias = (b_i if b_i is not None else np.zeros(V.shape[0])) + mu

    recommendations: dict[Any, list[Any]] = {}
    for idx, uid in enumerate(user_ids):
        scores = V @ U[idx] + item_bias
        if b_u is not None:
            scores = scores + b_u[idx]

        seen_cols = seen.rows[idx]
        if seen_cols:
            scores[seen_cols] = -np.inf

        if top_k >= len(scores):
            top_idx = np.argsort(scores)[::-1]
        else:
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        recommendations[uid] = item_ids[top_idx].tolist()

    print("[推荐] 完成。")
    return recommendations


# ══════════════════════════════════════════════════════════════════════
# 优化一：Biased MF（Funk SVD + SGD）
#
# 核心改进：
#   基础 SVD 直接分解 watch_ratio 矩阵，隐因子里混入了用户/视频的
#   系统性偏差。加入偏置项后，U·V 只需要学"纯粹的偏好残差"。
#
# 数学：
#   r̂_ui = μ + b_u + b_i + U[u] · V[i]
#   损失  = Σ(r_ui - r̂_ui)² + λ(‖U‖² + ‖V‖² + b_u² + b_i²)
#   更新  = SGD，mini-batch 向量化
# ══════════════════════════════════════════════════════════════════════

def run_biased_mf_pipeline(
    n_factors: int = 50,
    n_epochs: int = 20,
    lr: float = 0.005,
    reg: float = 0.02,
    top_k: int = 50,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Biased MF（Funk SVD）完整流程。

    超参数参考：
      n_factors=50, n_epochs=20, lr=0.005, reg=0.02
      （来自 Netflix Prize 优胜方案经验值，KuaiRec 无需大幅调整）
    """
    df = _test_df if _test_df is not None else load_big_matrix_interactions(eligible_video_ids)
    matrix, user_ids, item_ids = build_sparse_matrix(df)

    # —— 准备整数索引数组（向量化 SGD 的关键）——
    u_index = {uid: i for i, uid in enumerate(user_ids)}
    i_index = {iid: i for i, iid in enumerate(item_ids)}
    u_arr = df["user_id"].map(u_index).values.astype(np.int32)
    i_arr = df["video_id"].map(i_index).values.astype(np.int32)
    r_arr = df["watch_ratio"].values.astype(np.float64)

    n_users, n_items = len(user_ids), len(item_ids)
    rng = np.random.default_rng(42)

    # —— 初始化参数 ——
    mu   = float(r_arr.mean())
    b_u  = np.zeros(n_users)
    b_i  = np.zeros(n_items)
    U    = rng.normal(0, 0.1, (n_users, n_factors))
    V    = rng.normal(0, 0.1, (n_items, n_factors))

    print(
        f"\n[BiasedMF] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，"
        f"k={n_factors}，epochs={n_epochs}，lr={lr}，reg={reg}"
    )

    # —— mini-batch SGD ——
    batch = 50_000
    n_obs = len(r_arr)

    for epoch in range(n_epochs):
        t0 = time.time()
        perm = rng.permutation(n_obs)
        u_p, i_p, r_p = u_arr[perm], i_arr[perm], r_arr[perm]

        for start in range(0, n_obs, batch):
            ub = u_p[start:start + batch]
            ib = i_p[start:start + batch]
            rb = r_p[start:start + batch]

            # 预测 & 误差
            pred  = mu + b_u[ub] + b_i[ib] + (U[ub] * V[ib]).sum(axis=1)
            err   = rb - pred                          # (batch,)

            # 更新偏置
            np.add.at(b_u, ub, lr * (err - reg * b_u[ub]))
            np.add.at(b_i, ib, lr * (err - reg * b_i[ib]))

            # 更新隐因子（先缓存 U[ub] 避免写后读冲突）
            U_old = U[ub].copy()
            np.add.at(U, ub, lr * (err[:, None] * V[ib]  - reg * U[ub]))
            np.add.at(V, ib, lr * (err[:, None] * U_old  - reg * V[ib]))

        rmse = _compute_rmse_on_observed(u_arr, i_arr, r_arr, U, V, mu, b_u, b_i)
        print(f"  epoch {epoch+1:>2}/{n_epochs}  RMSE={rmse:.6f}  ({time.time()-t0:.1f}s)")

    final_rmse = _compute_rmse_on_observed(u_arr, i_arr, r_arr, U, V, mu, b_u, b_i)
    print(f"[BiasedMF] 训练完成，最终 RMSE={final_rmse:.6f}")

    recs = _top_k_from_factors(U, V, matrix, user_ids, item_ids, top_k, mu, b_u, b_i)
    rec_df = recommendations_to_dataframe(recs)

    result: dict[str, Any] = {
        "model":    "BiasedMF",
        "n_users":  n_users,
        "n_items":  n_items,
        "n_factors": n_factors,
        "top_k":    top_k,
        "rmse":     final_rmse,
        "mu":       mu,
        "recommendations":    recs,
        "recommendations_df": rec_df,
        # 供可视化模块使用（Vt = V.T 与 SVD 格式对齐）
        "_U":        U,
        "_Vt":       V.T,
        "_matrix":   matrix,
        "_user_ids": user_ids,
        "_item_ids": item_ids,
        "_interaction_df": df,
        "singular_values": [],   # BiasedMF 无奇异值
        "b_u": b_u,
        "b_i": b_i,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_df.to_csv(output_dir / f"biased_mf_top{top_k}_recommendations.csv",
                      index=False, encoding="utf-8-sig")

    return result


# ══════════════════════════════════════════════════════════════════════
# 优化二：iALS（Implicit ALS，Hu et al. 2008）
#
# 核心改进：
#   基础 SVD 把 watch_ratio=0 当"不喜欢"，但 0 只是"没看过"。
#   iALS 把所有 (user, item) 对的偏好都设为 1（p=1），
#   只用 watch_ratio 控制这条数据的置信度（高 watch_ratio → 更可信）。
#
# 数学：
#   confidence: c_ui = 1 + α * watch_ratio  （已看：高置信；未看：置信=1）
#   preference: p_ui = 1 if 看过 else 0     （二值）
#   损失 = Σ_u,i c_ui(p_ui - U[u]·V[i])² + λ(‖U‖²+‖V‖²)
#
# 高效 ALS 更新（不显式遍历未见 item）：
#   固定 V，更新 U[u]：
#     A_u = V^TV + λI + Σ_{i∈I_u}(c_ui-1)·V[i]V[i]^T
#     b_u = Σ_{i∈I_u} c_ui · V[i]
#     U[u] = A_u^{-1} b_u
#   V^TV 代表"所有物品置信=1"的贡献，只对已见物品加修正项。
# ══════════════════════════════════════════════════════════════════════

def run_ials_pipeline(
    n_factors: int = 50,
    n_iters: int = 15,
    reg: float = 0.01,
    alpha: float = 40.0,
    top_k: int = 50,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    iALS 完整流程。

    超参数参考：
      n_factors=50, n_iters=15, reg=0.01, alpha=40
      （alpha=40 来自 Hu et al. 原论文，是最常用的起始值）
    """
    df = _test_df if _test_df is not None else load_big_matrix_interactions(eligible_video_ids)
    matrix, user_ids, item_ids = build_sparse_matrix(df)

    n_users, n_items = len(user_ids), len(item_ids)
    k = n_factors
    rng = np.random.default_rng(42)

    # —— 初始化因子矩阵 ——
    U = rng.normal(0, 0.1, (n_users, k)).astype(np.float64)
    V = rng.normal(0, 0.1, (n_items, k)).astype(np.float64)

    # —— 预处理：按用户/物品分组索引（加速 ALS 内层循环）——
    u_index = {uid: i for i, uid in enumerate(user_ids)}
    i_index = {iid: i for i, iid in enumerate(item_ids)}
    u_arr = df["user_id"].map(u_index).values.astype(np.int32)
    i_arr = df["video_id"].map(i_index).values.astype(np.int32)
    r_arr = df["watch_ratio"].values.astype(np.float64)
    c_arr = 1.0 + alpha * r_arr                         # 置信度

    # 按用户分组：user_groups[u] = (item_indices, confidences)
    user_groups: list[tuple[np.ndarray, np.ndarray]] = [
        (np.array([], dtype=np.int32), np.array([], dtype=np.float64))
    ] * n_users
    _tmp_u: dict[int, list] = {i: [] for i in range(n_users)}
    _tmp_c: dict[int, list] = {i: [] for i in range(n_users)}
    for u, i, c in zip(u_arr, i_arr, c_arr):
        _tmp_u[u].append(i)
        _tmp_c[u].append(c)
    for u in range(n_users):
        user_groups[u] = (np.array(_tmp_u[u], np.int32),
                          np.array(_tmp_c[u], np.float64))

    # 按物品分组
    item_groups: list[tuple[np.ndarray, np.ndarray]] = [
        (np.array([], dtype=np.int32), np.array([], dtype=np.float64))
    ] * n_items
    _tmp_ui: dict[int, list] = {i: [] for i in range(n_items)}
    _tmp_ic: dict[int, list] = {i: [] for i in range(n_items)}
    for u, i, c in zip(u_arr, i_arr, c_arr):
        _tmp_ui[i].append(u)
        _tmp_ic[i].append(c)
    for i in range(n_items):
        item_groups[i] = (np.array(_tmp_ui[i], np.int32),
                          np.array(_tmp_ic[i], np.float64))

    lambda_I = reg * np.eye(k)

    print(
        f"\n[iALS] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，"
        f"k={k}，iters={n_iters}，reg={reg}，alpha={alpha}"
    )

    for it in range(n_iters):
        t0 = time.time()

        # —— 固定 V，更新所有用户向量 ——
        VtV = V.T @ V                                   # (k, k)，只算一次
        for u in range(n_users):
            i_obs, c_obs = user_groups[u]
            if len(i_obs) == 0:
                continue
            V_u = V[i_obs]                              # (|I_u|, k)
            # A_u = V^TV + λI + Σ(c-1)·V[i]V[i]^T
            A_u = VtV + lambda_I + (V_u * (c_obs - 1)[:, None]).T @ V_u
            # b_u = Σ c_ui·V[i]  （p_ui=1，所以直接乘置信度）
            b_u = V_u.T @ c_obs
            U[u] = np.linalg.solve(A_u, b_u)

        # —— 固定 U，更新所有物品向量 ——
        UtU = U.T @ U                                   # (k, k)
        for i in range(n_items):
            u_obs, c_obs = item_groups[i]
            if len(u_obs) == 0:
                continue
            U_i = U[u_obs]
            A_i = UtU + lambda_I + (U_i * (c_obs - 1)[:, None]).T @ U_i
            b_i = U_i.T @ c_obs
            V[i] = np.linalg.solve(A_i, b_i)

        rmse = _compute_rmse_on_observed(u_arr, i_arr, r_arr, U, V)
        print(f"  iter {it+1:>2}/{n_iters}  RMSE={rmse:.6f}  ({time.time()-t0:.1f}s)")

    final_rmse = _compute_rmse_on_observed(u_arr, i_arr, r_arr, U, V)
    print(f"[iALS] 训练完成，最终 RMSE={final_rmse:.6f}")

    recs = _top_k_from_factors(U, V, matrix, user_ids, item_ids, top_k)
    rec_df = recommendations_to_dataframe(recs)

    result: dict[str, Any] = {
        "model":    "iALS",
        "n_users":  n_users,
        "n_items":  n_items,
        "n_factors": k,
        "top_k":    top_k,
        "rmse":     final_rmse,
        "recommendations":    recs,
        "recommendations_df": rec_df,
        "_U":        U,
        "_Vt":       V.T,
        "_matrix":   matrix,
        "_user_ids": user_ids,
        "_item_ids": item_ids,
        "_interaction_df": df,
        "singular_values": [],
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_df.to_csv(output_dir / f"ials_top{top_k}_recommendations.csv",
                      index=False, encoding="utf-8-sig")

    return result


# ══════════════════════════════════════════════════════════════════════
# 对比管道：SVD vs BiasedMF vs iALS
# ══════════════════════════════════════════════════════════════════════

def run_comparison_pipeline(
    n_factors: int = 50,
    top_k: int = 50,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    依次运行三个模型，打印 RMSE 对比表。

    Returns:
        comparison_df  含 model / rmse / rmse_lift_vs_svd 三列的 DataFrame
    """
    from svd_recommender import run_svd_pipeline

    shared = dict(
        n_factors=n_factors, top_k=top_k,
        eligible_video_ids=eligible_video_ids,
        output_dir=output_dir, _test_df=_test_df
    )

    print("\n" + "═" * 55)
    print("  三模型对比：SVD → BiasedMF → iALS")
    print("═" * 55)

    r_svd  = run_svd_pipeline(**shared)
    r_bmf  = run_biased_mf_pipeline(**{k: v for k, v in shared.items()
                                       if k != "n_factors"},
                                    n_factors=n_factors)
    r_ials = run_ials_pipeline(**{k: v for k, v in shared.items()
                                  if k != "n_factors"},
                               n_factors=n_factors)

    rows = [
        {"model": "SVD（基线）",  "rmse": r_svd["rmse"]},
        {"model": "BiasedMF",     "rmse": r_bmf["rmse"]},
        {"model": "iALS",         "rmse": r_ials["rmse"]},
    ]
    df_cmp = pd.DataFrame(rows)
    svd_rmse = df_cmp.loc[df_cmp["model"] == "SVD（基线）", "rmse"].iloc[0]
    df_cmp["rmse_lift_vs_svd"] = (svd_rmse - df_cmp["rmse"]) / svd_rmse

    print("\n" + "═" * 55)
    print(df_cmp.to_string(index=False,
          float_format=lambda x: f"{x:.6f}" if abs(x) < 10 else f"{x:.2%}"))
    print("═" * 55)
    print("注：rmse_lift_vs_svd > 0 表示比 SVD 基线更好（RMSE 更低）")
    print("    iALS 的 RMSE 不可与 SVD/BiasedMF 直接比较：")
    print("    SVD/BiasedMF 优化目标 = 重建 watch_ratio（连续值）")
    print("    iALS      优化目标 = 重建 preference=1/0（二值）")
    print("    iALS 应通过 AB Test 完播率 或 AUC 评估推荐质量。\n")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_cmp.to_csv(output_dir / "mf_comparison.csv", index=False, encoding="utf-8-sig")

    return df_cmp


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行优化后的矩阵分解模型。")
    parser.add_argument("--model", choices=["biased_mf", "ials", "compare"],
                        default="compare", help="运行哪个模型（默认 compare 跑全部对比）")
    parser.add_argument("--n-factors", type=int, default=50)
    parser.add_argument("--top-k",     type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )

    if args.model == "biased_mf":
        run_biased_mf_pipeline(n_factors=args.n_factors, top_k=args.top_k, output_dir=out)
    elif args.model == "ials":
        run_ials_pipeline(n_factors=args.n_factors, top_k=args.top_k, output_dir=out)
    else:
        run_comparison_pipeline(n_factors=args.n_factors, top_k=args.top_k, output_dir=out)
