"""
文件用途：用 big_matrix（训练集）训练 SVD 矩阵分解，为每位用户生成个性化推荐列表。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【设计逻辑 —— 与 v3 策略的核心区别】

v3 策略 A / B（全局排序）：
  用 big_matrix 计算每个视频的 avg_watch_ratio 或 avg_completion_rate
  → 对所有用户推荐同一份 top-N 视频列表
  → 策略差异来自"排序信号不同"，但推荐列表对用户无差别化

SVD（策略 C，个性化推荐）：
  用 big_matrix 训练截断 SVD → 学习用户和视频的隐语义因子
  → 为每位用户生成不同的个性化 top-K 视频列表
  → 策略差异来自"个性化 vs 全局排序"，更接近真实推荐系统

【评估方式（与 v3 相同）】
  big_matrix（训练）→ 学习用户 / 视频 embedding
      ↓
  生成个性化推荐列表（每人 top-K 个视频）
      ↓
  small_matrix（答案本）→ 查用户对推荐视频的真实完播率
      ↓
  与策略 A / B 做 AB Test

【内存安全设计】
  big_matrix 有约 12.5M 行。全量稀疏矩阵可在内存中处理，
  但绝不重建完整稠密矩阵。所有 top-K 推荐均在因子空间直接计算：
    score(user_i, item_j) = U[i] · (sigma * Vt)[:, j]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))


# ── big_matrix CSV 路径（与 abtest_v3_strategies.py 保持相同的查找逻辑）──
def _find_big_matrix_csv() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "KuaiRec 2.0" / "data" / "big_matrix.csv",
        Path("/Users/liubike/Desktop/快手test/kuairec_abtest/data/KuaiRec 2.0/data/big_matrix.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


BIG_MATRIX_CSV = _find_big_matrix_csv()

# 过滤门槛：用户或视频至少有多少条交互记录才纳入矩阵
# 太少会导致 embedding 噪声大，也大幅减少矩阵维度
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 10

# 默认超参数
DEFAULT_N_FACTORS = 50
DEFAULT_TOP_K = 50
CHUNK_SIZE = 500_000


# ────────────────────────────── 数据加载 ──────────────────────────────

def load_big_matrix_interactions(
    eligible_video_ids: set | None = None,
) -> pd.DataFrame:
    """
    分块读取 big_matrix.csv，返回含 (user_id, video_id, watch_ratio) 的 DataFrame。

    只保留：
      - play_duration > 0（有实际播放行为）
      - eligible_video_ids 内的视频（通常是 small_matrix 的视频集合）

    参数 eligible_video_ids：
      big_matrix 里排名最高的视频，往往在 small_matrix 里没有记录，
      无法完成离线评估。必须先限定到 small_matrix 有的视频再训练。
    """
    if not BIG_MATRIX_CSV.exists():
        raise FileNotFoundError(
            f"big_matrix.csv 不存在：{BIG_MATRIX_CSV}\n"
            "请确认 KuaiRec 数据已下载并放在 data/KuaiRec 2.0/data/ 目录下。"
        )

    print(f"\n[SVD] 分块读取 big_matrix（块大小 {CHUNK_SIZE:,} 行）……")
    needed_cols = ["user_id", "video_id", "watch_ratio", "play_duration"]
    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(BIG_MATRIX_CSV, usecols=needed_cols, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)
        chunk = chunk[chunk["play_duration"] > 0].copy()
        if eligible_video_ids is not None:
            chunk = chunk[chunk["video_id"].isin(eligible_video_ids)]
        chunk["watch_ratio"] = (
            pd.to_numeric(chunk["watch_ratio"], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        chunks.append(chunk[["user_id", "video_id", "watch_ratio"]])

    df = pd.concat(chunks, ignore_index=True)
    print(
        f"[SVD] big_matrix 读取完成：共 {total_rows:,} 行原始数据，"
        f"过滤后保留 {len(df):,} 条有效交互。"
    )

    # 去重：同一 (user, item) 取最大 watch_ratio（防止多次播放导致矩阵值偏移）
    df = df.groupby(["user_id", "video_id"], as_index=False)["watch_ratio"].max()

    # 过滤低频用户和低频视频（减噪 + 减少矩阵维度）
    item_counts = df["video_id"].value_counts()
    user_counts = df["user_id"].value_counts()
    eligible_items = set(item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index)
    eligible_users = set(user_counts[user_counts >= MIN_USER_INTERACTIONS].index)
    before = len(df)
    df = df[df["video_id"].isin(eligible_items) & df["user_id"].isin(eligible_users)]
    print(
        f"[SVD] 低频过滤后：{before:,} → {len(df):,} 条，"
        f"用户 {len(eligible_users):,} 人，视频 {len(eligible_items):,} 个。"
    )
    return df


# ────────────────────────────── 矩阵构建 ──────────────────────────────

def build_sparse_matrix(
    df: pd.DataFrame,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    把 DataFrame 转换为稀疏 CSR 矩阵。

    Returns:
        matrix   - shape (n_users, n_items)，值为 watch_ratio
        user_ids - 行索引 → user_id 的映射数组
        item_ids - 列索引 → video_id 的映射数组
    """
    user_ids = df["user_id"].unique()
    item_ids = df["video_id"].unique()
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    row = df["user_id"].map(user_index).values
    col = df["video_id"].map(item_index).values
    data = df["watch_ratio"].values.astype(np.float32)

    matrix = csr_matrix((data, (row, col)), shape=(len(user_ids), len(item_ids)))
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print(
        f"[SVD] 稀疏矩阵构建完成：{matrix.shape[0]:,} 用户 × {matrix.shape[1]:,} 视频，"
        f"密度 {density:.6%}，非零元素 {matrix.nnz:,}。"
    )
    return matrix, user_ids, item_ids


# ────────────────────────────── SVD 分解 ──────────────────────────────

def train_svd(
    matrix: csr_matrix,
    n_factors: int = DEFAULT_N_FACTORS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    截断 SVD（Truncated SVD）分解稀疏 user-item 矩阵。

    使用 scipy ARPACK 求解器，不需要稠密化矩阵，适合 10M+ 级别的稀疏数据。

    Returns:
        U     - 用户隐因子，shape (n_users, k)
        sigma - 奇异值（降序），shape (k,)
        Vt    - 视频隐因子转置，shape (k, n_items)
    """
    k = min(n_factors, min(matrix.shape) - 1)
    print(f"\n[SVD] 开始截断 SVD 分解，k={k}，矩阵大小 {matrix.shape}……")

    U, sigma, Vt = svds(matrix.astype(np.float64), k=k)

    # svds 返回升序，转为降序
    order = np.argsort(sigma)[::-1]
    U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]

    explained = (sigma ** 2).sum() / (np.array(matrix.data, dtype=np.float64) ** 2).sum()
    print(
        f"[SVD] 分解完成：top-{k} 因子解释方差比 {explained:.4%}，"
        f"最大奇异值 {sigma[0]:.4f}，最小奇异值 {sigma[-1]:.4f}。"
    )
    return U, sigma, Vt


# ────────────────────────────── 个性化推荐 ────────────────────────────

def generate_personalized_recommendations(
    U: np.ndarray,
    sigma: np.ndarray,
    Vt: np.ndarray,
    original_matrix: csr_matrix,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    exclude_seen: bool = True,
) -> dict[Any, list[Any]]:
    """
    为每位用户生成个性化 top-K 推荐列表。

    核心计算（不重建稠密矩阵）：
      score(user_i) = U[i] · diag(sigma) · Vt
                    = U[i] @ (sigma[:, None] * Vt).T
    这里 sigma_Vt = sigma[:, None] * Vt，shape (k, n_items)，只算一次。

    Returns:
        {user_id: [video_id_1, ..., video_id_k]}  每位用户的推荐列表（按预测分降序）
    """
    print(f"\n[SVD] 为 {len(user_ids):,} 位用户生成个性化 top-{top_k} 推荐……")

    # 预计算：sigma · Vt，shape (k, n_items)
    sigma_Vt = sigma[:, None] * Vt

    # 转换原始矩阵为 lil 格式，方便按行查已看视频
    seen = original_matrix.tolil()

    recommendations: dict[Any, list[Any]] = {}
    for i, uid in enumerate(user_ids):
        # 个性化评分向量，shape (n_items,)
        scores = U[i] @ sigma_Vt

        if exclude_seen:
            seen_cols = seen.rows[i]
            if seen_cols:
                scores[seen_cols] = -np.inf

        # 取 top-K（argpartition 比全排序快）
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        recommendations[uid] = item_ids[top_indices].tolist()

    print(f"[SVD] 推荐列表生成完成，共 {len(recommendations):,} 位用户。")
    return recommendations


# ────────────────────────────── RMSE 评估 ────────────────────────────

def compute_rmse(
    original_matrix: csr_matrix,
    U: np.ndarray,
    sigma: np.ndarray,
    Vt: np.ndarray,
) -> float:
    """在已有交互位置上评估重建 RMSE，衡量分解质量。"""
    cx = original_matrix.tocoo()
    pred_vals = (U[cx.row] @ np.diag(sigma) @ Vt)[:, cx.col]
    # 注意：上面按行批量计算，但这样会做大量无用乘法；
    # 更高效：逐元素取 U[i] @ diag(sigma) @ Vt[:, j]
    pred_vals = np.array([
        float(U[r] @ (sigma * Vt[:, c]))
        for r, c in zip(cx.row, cx.col)
    ])
    true_vals = np.array(cx.data, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((true_vals - pred_vals) ** 2)))
    print(f"[SVD] RMSE（已交互位置）= {rmse:.6f}")
    return rmse


# ────────────────────────────── 推荐列表转 DataFrame ─────────────────

def recommendations_to_dataframe(
    recommendations: dict[Any, list[Any]],
) -> pd.DataFrame:
    """将 {user_id: [video_id, ...]} 转换为 DataFrame，列：user_id / video_id / rank。"""
    records = [
        {"user_id": uid, "video_id": vid, "rank": rank}
        for uid, vids in recommendations.items()
        for rank, vid in enumerate(vids, start=1)
    ]
    return pd.DataFrame(records, columns=["user_id", "video_id", "rank"])


# ────────────────────────────── 主流程 ───────────────────────────────

def run_svd_pipeline(
    n_factors: int = DEFAULT_N_FACTORS,
    top_k: int = DEFAULT_TOP_K,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    完整 SVD 推荐流程。

    Args:
        n_factors          SVD 隐因子维度
        top_k              每位用户推荐视频数
        eligible_video_ids 只在这些视频里生成推荐（应传入 small_matrix 的视频集合）
        output_dir         输出目录，None 则不写文件
        _test_df           测试用途：直接传入 DataFrame，跳过 CSV 读取

    Returns:
        包含 rmse / recommendations_df / 模型元信息的字典
    """
    # 1. 加载 big_matrix
    if _test_df is not None:
        df = _test_df
    else:
        df = load_big_matrix_interactions(eligible_video_ids=eligible_video_ids)

    # 2. 构建稀疏矩阵
    matrix, user_ids, item_ids = build_sparse_matrix(df)

    # 3. SVD 分解
    U, sigma, Vt = train_svd(matrix, n_factors=n_factors)

    # 4. RMSE 评估
    rmse = compute_rmse(matrix, U, sigma, Vt)

    # 5. 生成个性化推荐
    recommendations = generate_personalized_recommendations(
        U, sigma, Vt, matrix, user_ids, item_ids, top_k=top_k
    )
    rec_df = recommendations_to_dataframe(recommendations)

    result: dict[str, Any] = {
        "n_users": len(user_ids),
        "n_items": len(item_ids),
        "n_factors": int(sigma.shape[0]),
        "top_k": top_k,
        "rmse": rmse,
        "recommendations": recommendations,       # dict 格式，供 AB Test 管道调用
        "recommendations_df": rec_df,             # DataFrame 格式，供 CSV 导出
        "singular_values": sigma.tolist(),
        # 供可视化模块使用的模型矩阵
        "_U": U,
        "_Vt": Vt,
        "_matrix": matrix,
        "_user_ids": user_ids,
        "_item_ids": item_ids,
        "_interaction_df": df,
    }

    # 6. 写出
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_path = output_dir / f"svd_top{top_k}_recommendations.csv"
        rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
        print(f"[SVD] 推荐列表已保存：{rec_path}")
        result["output_path"] = str(rec_path)

    return result


# ────────────────────────────── CLI ──────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "用 big_matrix 训练 SVD 矩阵分解，为每位用户生成个性化推荐列表。\n"
            "推荐列表可用于 AB Test v4（SVD 个性化 vs v3 全局排序策略）。"
        )
    )
    parser.add_argument("--n-factors", type=int, default=DEFAULT_N_FACTORS,
                        help=f"SVD 隐因子维度，默认 {DEFAULT_N_FACTORS}。")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"每位用户推荐视频数，默认 {DEFAULT_TOP_K}。")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，不指定则默认写到 kuairec_abtest/output/。")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )

    # 从 small_matrix 获取可评估的视频集合（需要 MySQL 连接）
    # 如果没有 DB，去掉 eligible_video_ids 参数可跳过此过滤
    try:
        from data_loader import load_data
        sm_video_df = load_data("SELECT DISTINCT video_id FROM kuairec_small_matrix")
        eligible = set(sm_video_df["video_id"].tolist()) if not sm_video_df.empty else None
        print(f"[SVD] small_matrix 可评估视频：{len(eligible):,} 个" if eligible else "")
    except Exception:
        eligible = None
        print("[SVD] 无法连接 MySQL，不限定视频范围（eligible_video_ids=None）。")

    result = run_svd_pipeline(
        n_factors=args.n_factors,
        top_k=args.top_k,
        eligible_video_ids=eligible,
        output_dir=output_dir,
    )

    print(
        f"\n[SVD 推荐摘要]\n"
        f"  训练数据 ：big_matrix\n"
        f"  用户数   ：{result['n_users']:,}\n"
        f"  视频数   ：{result['n_items']:,}\n"
        f"  隐因子   ：{result['n_factors']}\n"
        f"  RMSE     ：{result['rmse']:.6f}\n"
        f"  推荐条数 ：{len(result['recommendations_df']):,}\n"
        f"  前 5 条  ：\n{result['recommendations_df'].head(5).to_string(index=False)}"
    )
