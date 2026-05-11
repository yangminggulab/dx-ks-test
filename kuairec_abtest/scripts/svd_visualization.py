"""
文件用途：SVD 推荐模型可视化与可解释性分析。

生成 6 张图，回答三个核心问题：
  ① 模型学了多少信息？  → 奇异值衰减曲线 + 累计解释方差
  ② 模型学到什么结构？  → 用户 Embedding 2D 投影 + 视频 Embedding 2D 投影
  ③ 推荐质量如何？      → 个性化多样性热图 + 热门偏差分析

用法：
    from svd_recommender import run_svd_pipeline
    from svd_visualization import run_svd_visualization

    result = run_svd_pipeline(...)          # 先跑 SVD
    run_svd_visualization(result, df)       # 再跑可视化
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["font.family"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]


# ─────────────────────────────────────────────────────────────────────
# 图1：奇异值衰减曲线 + 累计解释方差
#
# 读法：
#   左轴（柱）→ 每个因子的奇异值大小，衰减越快说明前几个因子越重要
#   右轴（线）→ 累计解释方差比，看需要多少个因子才能覆盖 80% / 90%
#   "肘部"位置就是最优 k 的参考点
# ─────────────────────────────────────────────────────────────────────
def plot_singular_values(
    sigma: np.ndarray,
    ax: plt.Axes,
) -> None:
    k = len(sigma)
    indices = np.arange(1, k + 1)

    explained = (sigma ** 2) / (sigma ** 2).sum()
    cumulative = np.cumsum(explained)

    ax2 = ax.twinx()

    ax.bar(indices, sigma, color="#4C72B0", alpha=0.7, label="奇异值")
    ax2.plot(indices, cumulative * 100, color="#DD8452", linewidth=2,
             marker="o", markersize=3, label="累计解释方差 %")

    # 标出 80% 和 90% 阈值线
    for thresh, ls in [(80, "--"), (90, ":")]:
        idx = np.searchsorted(cumulative * 100, thresh)
        if idx < k:
            ax2.axhline(thresh, color="gray", linestyle=ls, linewidth=1, alpha=0.6)
            ax2.text(k * 0.98, thresh + 1, f"{thresh}%", ha="right",
                     color="gray", fontsize=8)

    ax.set_xlabel("因子索引 (奇异值排名)")
    ax.set_ylabel("奇异值大小", color="#4C72B0")
    ax2.set_ylabel("累计解释方差 %", color="#DD8452")
    ax.set_title("① 奇异值衰减曲线\n（肘部 = 有效因子数量）")
    ax.tick_params(axis="y", labelcolor="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#DD8452")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)


# ─────────────────────────────────────────────────────────────────────
# 图2：用户 Embedding 2D 投影（PCA）
#
# 读法：
#   每个点是一个用户，位置由其 50 维隐因子向量降到 2D
#   颜色 = 用户活跃度（来自 user_active_degree 列）
#   聚类明显 → 模型学到了用户群体差异
#   活跃度相同的用户聚在一起 → 活跃度是 SVD 捕获的重要特征
# ─────────────────────────────────────────────────────────────────────
def plot_user_embedding(
    U: np.ndarray,
    user_ids: np.ndarray,
    user_meta: pd.DataFrame | None,
    ax: plt.Axes,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(U)
    var_explained = pca.explained_variance_ratio_ * 100

    df_plot = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "user_id": user_ids,
    })

    # 拼入活跃度信息（如果有）
    if user_meta is not None and "user_active_degree" in user_meta.columns:
        df_plot = df_plot.merge(
            user_meta[["user_id", "user_active_degree"]].drop_duplicates("user_id"),
            on="user_id", how="left"
        )
        df_plot["user_active_degree"] = df_plot["user_active_degree"].fillna("UNKNOWN")
        hue_col = "user_active_degree"
    else:
        df_plot["密度区间"] = pd.qcut(
            np.sqrt(coords[:, 0]**2 + coords[:, 1]**2), q=4,
            labels=["核心", "活跃", "普通", "边缘"]
        )
        hue_col = "密度区间"

    sns.scatterplot(
        data=df_plot, x="x", y="y", hue=hue_col,
        alpha=0.5, s=15, ax=ax, legend="brief"
    )
    ax.set_title(
        f"② 用户 Embedding 2D 投影（PCA）\n"
        f"PC1={var_explained[0]:.1f}%  PC2={var_explained[1]:.1f}%"
    )
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
    ax.legend(title=hue_col, fontsize=7, title_fontsize=7,
              markerscale=1.5, loc="best")


# ─────────────────────────────────────────────────────────────────────
# 图3：视频 Embedding 2D 投影（PCA）
#
# 读法：
#   每个点是一个视频，位置由其 50 维因子向量降到 2D
#   颜色 = 该视频在 big_matrix 里的播放热度（log 标准化）
#   右上角的热门视频和左下角的冷门视频是否有明显分离？
#   → 分离明显：SVD 把"流行度"当成了重要的隐性维度
# ─────────────────────────────────────────────────────────────────────
def plot_item_embedding(
    Vt: np.ndarray,
    item_ids: np.ndarray,
    item_play_counts: pd.Series | None,
    ax: plt.Axes,
) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Vt.T)
    var_explained = pca.explained_variance_ratio_ * 100

    df_plot = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "video_id": item_ids,
    })

    if item_play_counts is not None:
        df_plot["play_count"] = df_plot["video_id"].map(item_play_counts).fillna(1)
        df_plot["log_play"] = np.log1p(df_plot["play_count"])
        color_col = "log_play"
        scatter = ax.scatter(
            df_plot["x"], df_plot["y"],
            c=df_plot[color_col], cmap="YlOrRd",
            alpha=0.4, s=8
        )
        plt.colorbar(scatter, ax=ax, label="log(播放次数)")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.3, s=8, color="#4C72B0")

    ax.set_title(
        f"③ 视频 Embedding 2D 投影（PCA）\n"
        f"颜色=播放热度  PC1={var_explained[0]:.1f}%  PC2={var_explained[1]:.1f}%"
    )
    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")


# ─────────────────────────────────────────────────────────────────────
# 图4：个性化程度 —— 用户间推荐重叠度热图
#
# 读法：
#   随机抽 20 个用户，计算每对用户推荐列表的 Jaccard 相似度
#   颜色越深（相似度越高）→ 推荐越雷同，个性化越差
#   对角线必然为 1（自己和自己 100% 重叠）
#   非对角线平均值低 → 模型确实在做个性化
# ─────────────────────────────────────────────────────────────────────
def plot_recommendation_diversity(
    recommendations: dict[Any, list[Any]],
    ax: plt.Axes,
    sample_n: int = 20,
) -> None:
    uids = list(recommendations.keys())
    if len(uids) > sample_n:
        rng = np.random.default_rng(42)
        uids = rng.choice(uids, size=sample_n, replace=False).tolist()

    n = len(uids)
    sim_matrix = np.zeros((n, n))

    for i, u1 in enumerate(uids):
        set1 = set(recommendations[u1])
        for j, u2 in enumerate(uids):
            set2 = set(recommendations[u2])
            union = set1 | set2
            if union:
                sim_matrix[i, j] = len(set1 & set2) / len(union)

    mean_off_diag = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))

    mask = np.zeros_like(sim_matrix, dtype=bool)
    np.fill_diagonal(mask, True)

    sns.heatmap(
        sim_matrix, ax=ax,
        cmap="Blues", vmin=0, vmax=1,
        xticklabels=False, yticklabels=False,
        cbar_kws={"label": "Jaccard 相似度"}
    )
    ax.set_title(
        f"④ 推荐个性化程度（用户间相似度热图）\n"
        f"非对角均值 = {mean_off_diag:.3f}（越低越个性化）"
    )
    ax.set_xlabel("用户（随机抽样 20 人）")
    ax.set_ylabel("用户（随机抽样 20 人）")


# ─────────────────────────────────────────────────────────────────────
# 图5：热门偏差分析
#
# 读法：
#   X轴 = 视频在 big_matrix 里的播放次数（热度）
#   Y轴 = 该视频被推荐给多少用户（推荐频率）
#   正相关强 → 模型在推热门视频（热门偏差严重）
#   较为随机 → 模型真正学到了偏好，不只是推爆款
# ─────────────────────────────────────────────────────────────────────
def plot_popularity_bias(
    recommendations: dict[Any, list[Any]],
    item_play_counts: pd.Series,
    ax: plt.Axes,
) -> None:
    from collections import Counter
    rec_freq = Counter(
        vid for vids in recommendations.values() for vid in vids
    )
    df = pd.DataFrame({
        "video_id": list(rec_freq.keys()),
        "rec_count": list(rec_freq.values()),
    })
    df["play_count"] = df["video_id"].map(item_play_counts).fillna(0)
    df = df[df["play_count"] > 0]

    corr = df["play_count"].corr(df["rec_count"])

    ax.scatter(
        np.log1p(df["play_count"]),
        np.log1p(df["rec_count"]),
        alpha=0.3, s=10, color="#4C72B0"
    )

    # 趋势线
    z = np.polyfit(np.log1p(df["play_count"]), np.log1p(df["rec_count"]), 1)
    p = np.poly1d(z)
    x_line = np.linspace(np.log1p(df["play_count"]).min(),
                         np.log1p(df["play_count"]).max(), 100)
    ax.plot(x_line, p(x_line), color="#DD8452", linewidth=2,
            label=f"趋势线（r={corr:.2f}）")

    ax.set_xlabel("log(视频播放次数) → 热度")
    ax.set_ylabel("log(被推荐次数) → 推荐频率")
    ax.set_title(
        f"⑤ 热门偏差分析\n"
        f"Pearson r={corr:.3f}（越接近 0 偏差越小）"
    )
    ax.legend(fontsize=9)


# ─────────────────────────────────────────────────────────────────────
# 图6：用户推荐分数分布
#
# 读法：
#   每个用户的 top-1 预测分数分布
#   分布集中且偏高 → 模型对推荐有信心
#   分布散乱接近 0 → 模型分辨能力弱，推荐质量不确定
#   同时对比：有交互用户 vs 低交互用户的分数差异
# ─────────────────────────────────────────────────────────────────────
def plot_score_distribution(
    U: np.ndarray,
    sigma: np.ndarray,
    Vt: np.ndarray,
    original_matrix: csr_matrix,
    user_ids: np.ndarray,
    ax: plt.Axes,
    sample_n: int = 500,
) -> None:
    sigma_Vt = sigma[:, None] * Vt
    rng = np.random.default_rng(42)
    indices = rng.choice(len(user_ids), size=min(sample_n, len(user_ids)), replace=False)

    top1_scores, interaction_counts = [], []
    for i in indices:
        scores = U[i] @ sigma_Vt
        seen = original_matrix.getrow(i).indices
        scores[seen] = -np.inf
        top1_scores.append(float(scores.max()))
        interaction_counts.append(len(seen))

    df = pd.DataFrame({
        "top1_score": top1_scores,
        "interaction_count": interaction_counts,
    })
    median_interactions = df["interaction_count"].median()
    df["用户类型"] = df["interaction_count"].apply(
        lambda x: f"高交互（≥{int(median_interactions)}次）"
        if x >= median_interactions
        else f"低交互（<{int(median_interactions)}次）"
    )

    sns.kdeplot(
        data=df, x="top1_score", hue="用户类型",
        fill=True, alpha=0.4, ax=ax
    )
    ax.axvline(df["top1_score"].median(), color="gray",
               linestyle="--", linewidth=1.5, label=f"中位数={df['top1_score'].median():.3f}")
    ax.set_xlabel("Top-1 预测评分")
    ax.set_ylabel("密度")
    ax.set_title("⑥ 推荐置信度分布\n（高/低交互用户对比）")
    ax.legend(fontsize=8)


# ─────────────────────────────────────────────────────────────────────
# 主入口：一次生成全部 6 张图
# ─────────────────────────────────────────────────────────────────────

def run_svd_visualization(
    svd_result: dict[str, Any],
    interaction_df: pd.DataFrame,
    output_dir: Path | None = None,
    user_meta_df: pd.DataFrame | None = None,
) -> str | None:
    """
    生成 SVD 可解释性分析图（6 张子图）。

    Args:
        svd_result      run_svd_pipeline() 的返回值
        interaction_df  原始 (user_id, video_id, watch_ratio) DataFrame
        output_dir      保存路径，None 则不保存
        user_meta_df    可选，含 user_id / user_active_degree 的用户信息表

    Returns:
        保存的文件路径（或 None）
    """
    sigma      = np.array(svd_result["singular_values"])
    U          = svd_result.get("_U")
    Vt         = svd_result.get("_Vt")
    matrix     = svd_result.get("_matrix")
    user_ids   = svd_result.get("_user_ids")
    item_ids   = svd_result.get("_item_ids")
    recs       = svd_result["recommendations"]

    if U is None:
        print("[可视化] svd_result 中缺少模型矩阵，请在 run_svd_pipeline 中设置 return_model=True。")
        return None

    # 视频播放次数（用于热门偏差分析）
    item_play_counts = interaction_df.groupby("video_id")["watch_ratio"].count()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("SVD 推荐模型 —— 可视化与可解释性分析", fontsize=15, fontweight="bold", y=1.01)

    plot_singular_values(sigma, axes[0, 0])
    plot_user_embedding(U, user_ids, user_meta_df, axes[0, 1])
    plot_item_embedding(Vt, item_ids, item_play_counts, axes[0, 2])
    plot_recommendation_diversity(recs, axes[1, 0])
    plot_popularity_bias(recs, item_play_counts, axes[1, 1])
    plot_score_distribution(U, sigma, Vt, matrix, user_ids, axes[1, 2])

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "svd_explainability.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[可视化] 已保存：{save_path}")
        plt.close(fig)
        return str(save_path)

    plt.show()
    plt.close(fig)
    return None
