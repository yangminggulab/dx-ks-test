"""
文件用途：基于 SVD 矩阵分解实现协同过滤推荐，输出用户 Top-K 视频推荐列表。

流程：
  1. 从 MySQL 加载 user-item 交互数据（watch_ratio 作为隐式评分信号）
  2. 构建稀疏用户-物品矩阵
  3. 截断 SVD（Truncated SVD）分解，提取 n_factors 个隐语义因子
  4. 重建预测评分矩阵，过滤已交互 item，为每位用户生成 Top-K 推荐
  5. 计算 RMSE 评估分解质量，输出推荐列表 CSV
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

from data_loader import load_data


# ────────────────────────────── 数据加载 ──────────────────────────────

def load_interaction_data(limit: int | None = None) -> pd.DataFrame:
    """从 MySQL 加载 user-item 交互数据，返回含 user_id / video_id / watch_ratio 的 DataFrame。"""
    query = """
        SELECT user_id, video_id, watch_ratio
        FROM kuairec_small_matrix
        WHERE watch_ratio IS NOT NULL
    """
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    df = load_data(query)
    if df.empty:
        raise RuntimeError("数据库返回空结果，请检查 MySQL 连接和表是否存在。")
    df["watch_ratio"] = pd.to_numeric(df["watch_ratio"], errors="coerce").fillna(0.0)
    df["watch_ratio"] = df["watch_ratio"].clip(lower=0.0)
    return df


# ────────────────────────────── 矩阵构建 ──────────────────────────────

def build_interaction_matrix(
    df: pd.DataFrame,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    将 DataFrame 转换为稀疏 user-item 矩阵。

    Returns:
        matrix   - CSR 格式稀疏矩阵，shape (n_users, n_items)
        user_ids - 行索引对应的 user_id 数组
        item_ids - 列索引对应的 video_id 数组
    """
    user_ids = df["user_id"].unique()
    item_ids = df["video_id"].unique()

    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    row = df["user_id"].map(user_index).values
    col = df["video_id"].map(item_index).values
    data = df["watch_ratio"].values.astype(np.float32)

    matrix = csr_matrix(
        (data, (row, col)),
        shape=(len(user_ids), len(item_ids)),
    )
    print(
        f"用户-物品矩阵构建完成：{matrix.shape[0]} 用户 × {matrix.shape[1]} 视频，"
        f"稀疏度 {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}。"
    )
    return matrix, user_ids, item_ids


# ────────────────────────────── SVD 分解 ──────────────────────────────

def train_svd(
    matrix: csr_matrix,
    n_factors: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对稀疏矩阵执行截断 SVD（Truncated SVD）。

    参数 n_factors 控制保留的奇异值个数（即隐语义因子维度）。
    使用 ARPACK 求解器，适合大规模稀疏矩阵。

    Returns:
        U     - 用户隐因子矩阵，shape (n_users, n_factors)
        sigma - 奇异值向量，shape (n_factors,)
        Vt    - 物品隐因子矩阵转置，shape (n_factors, n_items)
    """
    n_factors = min(n_factors, min(matrix.shape) - 1)
    print(f"开始 SVD 分解，隐因子维度 k={n_factors}……")
    U, sigma, Vt = svds(matrix.astype(np.float64), k=n_factors)

    # svds 返回的奇异值升序排列，统一转为降序
    order = np.argsort(sigma)[::-1]
    U, sigma, Vt = U[:, order], sigma[order], Vt[order, :]

    explained = (sigma**2).sum() / (matrix.data**2).sum()
    print(
        f"SVD 分解完成：top-{n_factors} 奇异值解释方差比例 {explained:.4%}，"
        f"最大奇异值 {sigma[0]:.4f}，最小奇异值 {sigma[-1]:.4f}。"
    )
    return U, sigma, Vt


# ────────────────────────────── 预测与推荐 ────────────────────────────

def reconstruct_scores(
    U: np.ndarray,
    sigma: np.ndarray,
    Vt: np.ndarray,
) -> np.ndarray:
    """用 U·Σ·Vt 重建完整预测评分矩阵，shape (n_users, n_items)。"""
    return U @ np.diag(sigma) @ Vt


def get_top_k_recommendations(
    predicted: np.ndarray,
    original_matrix: csr_matrix,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    top_k: int = 10,
    exclude_seen: bool = True,
) -> pd.DataFrame:
    """
    为所有用户生成 Top-K 推荐列表。

    Args:
        predicted       - 重建的预测评分矩阵
        original_matrix - 原始稀疏矩阵（用于过滤已交互 item）
        user_ids        - 行索引对应的 user_id
        item_ids        - 列索引对应的 video_id
        top_k           - 每位用户推荐视频数量
        exclude_seen    - 是否过滤已有交互的视频

    Returns:
        DataFrame with columns: user_id, video_id, predicted_score, rank
    """
    print(f"正在为 {len(user_ids)} 位用户生成 Top-{top_k} 推荐……")
    records: list[dict[str, Any]] = []

    # 转稠密布尔掩码（seen[i, j]=True 表示用户 i 已看过视频 j）
    seen_mask = original_matrix.toarray() > 0 if exclude_seen else None

    for user_idx in range(len(user_ids)):
        scores = predicted[user_idx].copy()

        if exclude_seen and seen_mask is not None:
            scores[seen_mask[user_idx]] = -np.inf

        top_item_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_item_indices = top_item_indices[np.argsort(scores[top_item_indices])[::-1]]

        uid = user_ids[user_idx]
        for rank, item_idx in enumerate(top_item_indices, start=1):
            records.append(
                {
                    "user_id": uid,
                    "video_id": item_ids[item_idx],
                    "predicted_score": float(scores[item_idx]),
                    "rank": rank,
                }
            )

    rec_df = pd.DataFrame(records, columns=["user_id", "video_id", "predicted_score", "rank"])
    print(f"推荐列表生成完成，共 {len(rec_df)} 条记录。")
    return rec_df


# ────────────────────────────── 评估指标 ──────────────────────────────

def compute_rmse(
    original_matrix: csr_matrix,
    predicted: np.ndarray,
) -> float:
    """在已有交互位置上计算 RMSE（均方根误差），衡量分解重建质量。"""
    cx = original_matrix.tocoo()
    true_vals = np.array(cx.data, dtype=np.float64)
    pred_vals = predicted[cx.row, cx.col]
    rmse = float(np.sqrt(np.mean((true_vals - pred_vals) ** 2)))
    print(f"RMSE（已交互位置）= {rmse:.6f}")
    return rmse


# ────────────────────────────── 主流程 ───────────────────────────────

def run_svd_pipeline(
    n_factors: int = 50,
    top_k: int = 10,
    data_limit: int | None = None,
    output_dir: Path | None = None,
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    执行完整 SVD 推荐流程。

    Args:
        n_factors  - 隐因子维度
        top_k      - 每位用户推荐视频数
        data_limit - 读取数据行数上限（None 表示全量）
        output_dir - 输出目录，None 则不写文件
        df         - 直接传入 DataFrame（用于测试，跳过 MySQL 读取）

    Returns:
        包含 rmse、推荐 DataFrame、输出路径等信息的字典
    """
    # 1. 加载数据
    if df is None:
        df = load_interaction_data(limit=data_limit)

    # 2. 构建矩阵
    matrix, user_ids, item_ids = build_interaction_matrix(df)

    # 3. SVD 分解
    U, sigma, Vt = train_svd(matrix, n_factors=n_factors)

    # 4. 重建预测评分
    predicted = reconstruct_scores(U, sigma, Vt)

    # 5. 评估
    rmse = compute_rmse(matrix, predicted)

    # 6. 生成推荐列表
    rec_df = get_top_k_recommendations(
        predicted, matrix, user_ids, item_ids, top_k=top_k
    )

    result: dict[str, Any] = {
        "n_users": len(user_ids),
        "n_items": len(item_ids),
        "n_factors": n_factors,
        "top_k": top_k,
        "rmse": rmse,
        "recommendations": rec_df,
    }

    # 7. 写出 CSV
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_path = output_dir / f"svd_top{top_k}_recommendations.csv"
        rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
        print(f"推荐列表已保存至：{rec_path}")
        result["output_path"] = str(rec_path)

    return result


# ────────────────────────────── CLI ──────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 SVD 矩阵分解为 KuaiRec 用户生成视频推荐列表。"
    )
    parser.add_argument(
        "--n-factors", type=int, default=50,
        help="隐因子维度（奇异值数量），默认 50。",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="每位用户推荐视频数，默认 10。",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="读取数据行数上限，默认读取全量数据。",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录路径，不指定则只打印结果摘要。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )
    result = run_svd_pipeline(
        n_factors=args.n_factors,
        top_k=args.top_k,
        data_limit=args.limit,
        output_dir=output_dir,
    )
    print(
        f"\n[SVD 推荐摘要]\n"
        f"  用户数：{result['n_users']}\n"
        f"  视频数：{result['n_items']}\n"
        f"  隐因子：{result['n_factors']}\n"
        f"  RMSE  ：{result['rmse']:.6f}\n"
        f"  推荐条数：{len(result['recommendations'])}\n"
        f"  前 5 条：\n{result['recommendations'].head(5).to_string(index=False)}"
    )
