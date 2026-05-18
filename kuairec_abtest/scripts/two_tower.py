"""
文件用途：Two-Tower（双塔）召回模型，与 svd_recommender.py 接口兼容。

工业界最常用的个性化召回架构（YouTube / 快手 / 抖音均在用）：

  用户塔                          视频塔
  user_id emb                     video_id emb
  + user_active_degree emb        + 类别 multi-hot (31维)
  + 数值特征(follow/fans/注册天数)   + 视频时长
        ↓ MLP                           ↓ MLP
    user_emb (64维)  ── 内积 ──  item_emb (64维)
                        ↓
                    预测分数（越高越匹配）

【相比 SVD 的核心区别】
  SVD      只有 ID 信号，新视频（冷启动）无法推荐
  Two-Tower  ID + 内容特征，新视频有特征向量，可以推荐

【训练方式：BPR（贝叶斯个性化排序）】
  对每条正样本 (u, pos_item) 随机采一个负样本 neg_item
  loss = -log σ(score_pos − score_neg)
  含义：只要正样本排在随机负样本前面，不关心具体分值是多少

【推理时为什么两塔要分开？】
  可以把全部视频提前算好向量存起来（离线索引）；
  用户来了只过一次用户塔，再做向量最近邻检索。
  如果两塔耦合（cross-attention），候选集 10M 时完全算不过来。

【PyTorch 新概念（相比 BiasedMF）】
  Dataset.__getitem__   自定义数据集 + 动态负采样
  多子模块组合           一个 Module 内嵌两个子 Module
  F.normalize           L2 归一化，让内积等价于余弦相似度
  预计算批量推理         固定参数后，矩阵乘法一次生成所有推荐
"""
from __future__ import annotations

import ast
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from svd_recommender import (
    build_sparse_matrix,
    load_big_matrix_interactions,
    recommendations_to_dataframe,
)


# ── KuaiRec 数据目录 ──────────────────────────────────────────────────
def _find_data_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "KuaiRec 2.0" / "data",
        Path("/Users/liubike/Desktop/快手test/kuairec_abtest/data/KuaiRec 2.0/data"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


DATA_DIR = _find_data_dir()

N_CATEGORIES    = 31    # item_categories 类别 ID 范围 0~30
N_ACTIVE_DEGREES = 4    # full_active / high_active / middle_active / UNKNOWN
ACTIVE_DEGREE_MAP = {
    "full_active": 0, "high_active": 1, "middle_active": 2, "UNKNOWN": 3,
}

DEFAULT_EMB_DIM  = 32
DEFAULT_HIDDEN   = 128
DEFAULT_OUT_DIM  = 64
DEFAULT_N_EPOCHS = 20
DEFAULT_LR       = 1e-3
DEFAULT_BATCH    = 4096
DEFAULT_TOP_K    = 50


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# 特征工程
# ══════════════════════════════════════════════════════════════════════

def _build_user_features(
    user_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    加载用户特征，按 user_ids 顺序对齐。

    Returns:
        active_arr : (n_users,) int64    user_active_degree 编码（0-3）
        num_arr    : (n_users, 3) float32  [log1p(follow), log1p(fans), log1p(register_days)]
    """
    n_users = len(user_ids)
    active_arr = np.full(n_users, ACTIVE_DEGREE_MAP["UNKNOWN"], dtype=np.int64)
    num_arr    = np.zeros((n_users, 3), dtype=np.float32)

    uf_path = DATA_DIR / "user_features.csv"
    if not uf_path.exists():
        print("[TwoTower] user_features.csv 不存在，使用全零用户特征。")
        return active_arr, num_arr

    uf = pd.read_csv(uf_path).set_index("user_id")
    for idx, uid in enumerate(user_ids):
        if uid not in uf.index:
            continue
        row = uf.loc[uid]
        active_arr[idx] = ACTIVE_DEGREE_MAP.get(str(row["user_active_degree"]), 3)
        num_arr[idx, 0] = np.log1p(max(0.0, float(row["follow_user_num"])))
        num_arr[idx, 1] = np.log1p(max(0.0, float(row["fans_user_num"])))
        num_arr[idx, 2] = np.log1p(max(0.0, float(row["register_days"])))

    print(f"[TwoTower] 用户特征加载完成：{n_users:,} 用户。")
    return active_arr, num_arr


def _build_item_features(
    item_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    加载视频特征，按 item_ids 顺序对齐。

    Returns:
        cat_arr : (n_items, 31) float32  类别 multi-hot
        dur_arr : (n_items, 1)  float32  log1p(video_duration)
    """
    n_items = len(item_ids)
    cat_arr = np.zeros((n_items, N_CATEGORIES), dtype=np.float32)
    dur_arr = np.zeros((n_items, 1), dtype=np.float32)
    item_idx_map = {int(iid): i for i, iid in enumerate(item_ids)}

    ic_path = DATA_DIR / "item_categories.csv"
    if ic_path.exists():
        ic = pd.read_csv(ic_path)
        for _, row in ic.iterrows():
            vid = int(row["video_id"])
            if vid not in item_idx_map:
                continue
            try:
                cats = ast.literal_eval(str(row["feat"]))
                for c in cats:
                    if 0 <= c < N_CATEGORIES:
                        cat_arr[item_idx_map[vid], c] = 1.0
            except Exception:
                pass
    else:
        print("[TwoTower] item_categories.csv 不存在，使用全零类别特征。")

    idf_path = DATA_DIR / "item_daily_features.csv"
    if idf_path.exists():
        idf = (
            pd.read_csv(idf_path, usecols=["video_id", "video_duration"])
            .drop_duplicates("video_id")
            .set_index("video_id")
        )
        for idx, vid in enumerate(item_ids):
            vid_int = int(vid)
            if vid_int in idf.index:
                dur = idf.loc[vid_int, "video_duration"]
                if pd.notna(dur) and dur > 0:
                    dur_arr[idx, 0] = float(np.log1p(dur))
    else:
        print("[TwoTower] item_daily_features.csv 不存在，使用全零时长特征。")

    print(f"[TwoTower] 视频特征加载完成：{n_items:,} 视频，类别 multi-hot(31) + 时长。")
    return cat_arr, dur_arr


# ══════════════════════════════════════════════════════════════════════
# BPR 数据集
# ══════════════════════════════════════════════════════════════════════

class TwoTowerBPRDataset(Dataset):
    """
    WBPR 训练集：每个样本是 (用户, 正样本视频, 负样本视频, watch_ratio 权重)。

    【为什么需要权重？】
    普通 BPR 只区分"看过/没看过"，把 watch_ratio=0.1 和 watch_ratio=1.0 等价。
    但完播（w≈1.0）是强正反馈，划走（w≈0.1）是弱正反馈，信号强度差距很大。
    WBPR loss = -mean(w × log σ(pos_score − neg_score))
    完播样本的梯度贡献更大，模型更专注学"真正喜欢"而非"随便扫了一眼"。
    """

    def __init__(self, u_arr: np.ndarray, i_arr: np.ndarray, r_arr: np.ndarray, n_items: int):
        self.u_arr   = u_arr
        self.i_arr   = i_arr
        self.r_arr   = r_arr   # watch_ratio，范围 [0, 1]
        self.n_items = n_items

    def __len__(self) -> int:
        return len(self.u_arr)

    def __getitem__(self, idx: int) -> tuple[int, int, int, float]:
        u   = int(self.u_arr[idx])
        pos = int(self.i_arr[idx])
        neg = random.randint(0, self.n_items - 1)   # 均匀随机负采样
        w   = self.r_arr[idx]                       # 正样本权重（保持 float32，float() 会升为 float64）
        return u, pos, neg, w


# ══════════════════════════════════════════════════════════════════════
# 用户塔
# ══════════════════════════════════════════════════════════════════════

class UserTower(nn.Module):
    """
    输入：user_id 整数索引 + active_degree 类别 + 3 个数值特征
    输出：64 维 L2 归一化用户向量

    特征拼接（concat）是 Two-Tower 里融合多模态特征的标准做法：
    把不同来源的向量拼到一起，让 MLP 自己学怎么加权组合。
    """

    def __init__(
        self,
        n_users: int,
        emb_dim: int = DEFAULT_EMB_DIM,
        hidden_dim: int = DEFAULT_HIDDEN,
        out_dim: int = DEFAULT_OUT_DIM,
    ):
        super().__init__()
        self.user_emb   = nn.Embedding(n_users, emb_dim)
        self.active_emb = nn.Embedding(N_ACTIVE_DEGREES, 4)  # 4个活跃度类别 → 4维

        in_dim = emb_dim + 4 + 3   # user_emb(32) + active_emb(4) + 数值特征(3)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),   # 比 BatchNorm 对变长 batch 更稳定
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.active_emb.weight, std=0.01)

    def forward(
        self,
        user_idx: torch.Tensor,     # (batch,) LongTensor
        active_idx: torch.Tensor,   # (batch,) LongTensor
        num_feats: torch.Tensor,    # (batch, 3) float32
    ) -> torch.Tensor:
        u = self.user_emb(user_idx)                    # (batch, 32)
        a = self.active_emb(active_idx)                # (batch, 4)
        x = torch.cat([u, a, num_feats], dim=-1)       # (batch, 39)
        out = self.mlp(x)                              # (batch, 64)
        return F.normalize(out, dim=-1)                # L2 归一化：‖out‖₂ = 1


# ══════════════════════════════════════════════════════════════════════
# 视频塔
# ══════════════════════════════════════════════════════════════════════

class ItemTower(nn.Module):
    """
    输入：video_id 整数索引 + 31维类别 multi-hot + 视频时长
    输出：64 维 L2 归一化视频向量

    multi-hot 向量（不同于 one-hot）：一个视频可属于多个类别，
    对应位置全部置 1，让模型知道"这个视频同时是搞笑 + 宠物视频"。
    """

    def __init__(
        self,
        n_items: int,
        emb_dim: int = DEFAULT_EMB_DIM,
        hidden_dim: int = DEFAULT_HIDDEN,
        out_dim: int = DEFAULT_OUT_DIM,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim)

        in_dim = emb_dim + N_CATEGORIES + 1   # item_emb(32) + 类别(31) + 时长(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(
        self,
        item_idx: torch.Tensor,       # (batch,) LongTensor
        cat_multihot: torch.Tensor,   # (batch, 31) float32
        dur_feat: torch.Tensor,       # (batch, 1)  float32
    ) -> torch.Tensor:
        v = self.item_emb(item_idx)                    # (batch, 32)
        x = torch.cat([v, cat_multihot, dur_feat], dim=-1)  # (batch, 64)
        out = self.mlp(x)                              # (batch, 64)
        return F.normalize(out, dim=-1)


# ══════════════════════════════════════════════════════════════════════
# Two-Tower 完整模型
# ══════════════════════════════════════════════════════════════════════

class TwoTowerModel(nn.Module):
    """
    组合用户塔和视频塔，训练时同时更新两侧参数。

    forward 返回内积分数（已 L2 归一化，所以等价于余弦相似度，范围 [-1, 1]）。
    """

    def __init__(self, user_tower: UserTower, item_tower: ItemTower):
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(
        self,
        user_idx: torch.Tensor,
        active_idx: torch.Tensor,
        num_feats: torch.Tensor,
        item_idx: torch.Tensor,
        cat_multihot: torch.Tensor,
        dur_feat: torch.Tensor,
    ) -> torch.Tensor:
        u_emb = self.user_tower(user_idx, active_idx, num_feats)     # (batch, D)
        i_emb = self.item_tower(item_idx, cat_multihot, dur_feat)    # (batch, D)
        return (u_emb * i_emb).sum(dim=-1)                           # (batch,)


# ══════════════════════════════════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════════════════════════════════

def run_two_tower_pipeline(
    emb_dim: int = DEFAULT_EMB_DIM,
    hidden_dim: int = DEFAULT_HIDDEN,
    out_dim: int = DEFAULT_OUT_DIM,
    n_epochs: int = DEFAULT_N_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH,
    top_k: int = DEFAULT_TOP_K,
    weighted: bool = True,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    patience: int = 5,
    val_frac: float = 0.1,
    _test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Two-Tower 完整流程，返回格式与 svd_recommender.run_svd_pipeline 兼容。

    weighted=True      : WBPR loss，用 watch_ratio 加权（完播贡献更大）
    weighted=False     : 普通 BPR loss，所有正样本等权重
    checkpoint_dir     : 每个 epoch 结束后保存 checkpoint；有已有 checkpoint 时自动续训
    """
    device = _get_device()
    print(f"[TwoTower] device = {device}")

    # 1. 加载交互数据
    df = _test_df if _test_df is not None else load_big_matrix_interactions(eligible_video_ids)
    matrix, user_ids, item_ids = build_sparse_matrix(df)
    n_users, n_items = len(user_ids), len(item_ids)

    # 2. 建整数索引
    u_index = {uid: i for i, uid in enumerate(user_ids)}
    i_index = {iid: i for i, iid in enumerate(item_ids)}
    u_arr = df["user_id"].map(u_index).values.astype(np.int64)
    i_arr = df["video_id"].map(i_index).values.astype(np.int64)
    r_arr = df["watch_ratio"].values.astype(np.float32)

    # 3. 加载特征，转为全量 tensor 放到 device（训练时按索引查）
    user_active_np, user_num_np = _build_user_features(user_ids)
    item_cat_np,    item_dur_np = _build_item_features(item_ids)

    user_active_t = torch.from_numpy(user_active_np).to(device)
    user_num_t    = torch.from_numpy(user_num_np).to(device)
    item_cat_t    = torch.from_numpy(item_cat_np).to(device)
    item_dur_t    = torch.from_numpy(item_dur_np).to(device)

    # 4. 切分训练/验证集（Early Stopping 用验证集监控过拟合）
    #    固定 seed 保证每次切分结果相同（方便对比 BPR / WBPR）
    n_total = len(u_arr)
    perm = np.random.default_rng(42).permutation(n_total)
    n_val = max(1, min(500_000, int(n_total * val_frac)))  # 最多 50 万条，至少 1 条
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TwoTowerBPRDataset(u_arr[train_idx], i_arr[train_idx], r_arr[train_idx], n_items)
    val_ds   = TwoTowerBPRDataset(u_arr[val_idx],   i_arr[val_idx],   r_arr[val_idx],   n_items)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. 模型 & 优化器
    user_tower = UserTower(n_users, emb_dim, hidden_dim, out_dim).to(device)
    item_tower = ItemTower(n_items, emb_dim, hidden_dim, out_dim).to(device)
    model = TwoTowerModel(user_tower, item_tower)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6. Checkpoint 恢复
    ckpt_path: Path | None = None
    best_ckpt_path: Path | None = None
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        variant = "wbpr" if weighted else "bpr"
        ckpt_path      = checkpoint_dir / f"two_tower_{variant}_latest.pt"
        best_ckpt_path = checkpoint_dir / f"two_tower_{variant}_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch      = ckpt["epoch"] + 1
            best_val_loss    = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            if patience_counter >= patience:
                print(f"[TwoTower] 从 checkpoint 恢复：Early Stopping 已完成，直接推理（{ckpt_path}）")
                start_epoch = n_epochs  # 跳过训练循环
            else:
                print(f"[TwoTower] 从 checkpoint 恢复：epoch {start_epoch}/{n_epochs}（{ckpt_path}）")

    loss_label = "WBPR-loss" if weighted else "BPR-loss"
    print(
        f"[TwoTower] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，"
        f"emb={emb_dim}，hidden={hidden_dim}，out={out_dim}，"
        f"max_epochs={n_epochs}，patience={patience}，lr={lr}"
    )
    print(f"  训练集 {len(train_idx):,} 条，验证集 {len(val_idx):,} 条")

    # 7. 训练（BPR / WBPR loss）+ Early Stopping
    avg_loss = 0.0
    avg_val_loss = float("inf")
    best_model_state: dict | None = None  # 内存中保存最佳权重（兜底）

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        # ── 训练阶段 ──────────────────────────────────────────
        model.train()
        total_loss, n_seen = 0.0, 0
        for u_b, pos_b, neg_b, w_b in train_loader:
            u_b   = u_b.to(device)
            pos_b = pos_b.to(device)
            neg_b = neg_b.to(device)
            w_b   = w_b.to(device)

            u_emb   = model.user_tower(u_b, user_active_t[u_b], user_num_t[u_b])
            pos_emb = model.item_tower(pos_b, item_cat_t[pos_b], item_dur_t[pos_b])
            neg_emb = model.item_tower(neg_b, item_cat_t[neg_b], item_dur_t[neg_b])

            pos_score = (u_emb * pos_emb).sum(-1)
            neg_score = (u_emb * neg_emb).sum(-1)

            if weighted:
                loss = -(w_b * F.logsigmoid(pos_score - neg_score)).mean()
            else:
                loss = -F.logsigmoid(pos_score - neg_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(u_b)
            n_seen += len(u_b)

        avg_loss = total_loss / n_seen

        # ── 验证阶段（无梯度，快速）────────────────────────────
        model.eval()
        val_total, n_val_seen = 0.0, 0
        with torch.no_grad():
            for u_b, pos_b, neg_b, w_b in val_loader:
                u_b, pos_b, neg_b, w_b = (
                    u_b.to(device), pos_b.to(device),
                    neg_b.to(device), w_b.to(device),
                )
                u_emb   = model.user_tower(u_b, user_active_t[u_b], user_num_t[u_b])
                pos_emb = model.item_tower(pos_b, item_cat_t[pos_b], item_dur_t[pos_b])
                neg_emb = model.item_tower(neg_b, item_cat_t[neg_b], item_dur_t[neg_b])
                pos_score = (u_emb * pos_emb).sum(-1)
                neg_score = (u_emb * neg_emb).sum(-1)
                if weighted:
                    vl = -(w_b * F.logsigmoid(pos_score - neg_score)).mean()
                else:
                    vl = -F.logsigmoid(pos_score - neg_score).mean()
                val_total  += vl.item() * len(u_b)
                n_val_seen += len(u_b)

        avg_val_loss = val_total / n_val_seen
        improved = avg_val_loss < best_val_loss - 1e-6
        star = " ★" if improved else ""

        print(
            f"  epoch {epoch+1:>2}/{n_epochs}  {loss_label}={avg_loss:.6f}"
            f"  val={avg_val_loss:.6f}{star}  ({time.time()-t0:.1f}s)"
        )

        # ── Early Stopping 逻辑 ───────────────────────────────
        if improved:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 内存中保留最佳权重（clone 避免后续 epoch 覆盖）
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            if best_ckpt_path is not None:
                torch.save({
                    "epoch": epoch, "model": best_model_state,
                    "optimizer": optimizer.state_dict(),
                    "avg_loss": avg_loss, "val_loss": avg_val_loss,
                }, best_ckpt_path)
                print(f"  └─ 最佳模型已保存（val_loss={avg_val_loss:.6f}）")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[TwoTower] Early Stopping：连续 {patience} 轮 val_loss 未改善，"
                    f"停止在 epoch {epoch+1}（最佳 epoch {epoch+1-patience}）"
                )
                if ckpt_path is not None:
                    torch.save({
                        "epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "avg_loss": avg_loss, "val_loss": avg_val_loss,
                        "best_val_loss": best_val_loss, "patience_counter": patience_counter,
                    }, ckpt_path)
                break

        # ── 续训 Checkpoint（意外中断恢复用）─────────────────
        if ckpt_path is not None:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_loss": avg_loss, "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss, "patience_counter": patience_counter,
            }, ckpt_path)

    # 8. 恢复最佳权重用于推理（不是最后一轮的过拟合权重）
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[TwoTower] 已加载最佳权重（val_loss={best_val_loss:.6f}）")
    elif best_ckpt_path is not None and best_ckpt_path.exists():
        best = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best["model"])
        print(f"[TwoTower] 已从磁盘加载最佳权重（val_loss={best['val_loss']:.6f}）")

    print(f"[TwoTower] 训练完成，最终 {loss_label}={avg_loss:.6f}，最佳 val_loss={best_val_loss:.6f}")

    # 8. 生成个性化推荐
    print(f"\n[TwoTower] 为 {n_users:,} 位用户生成个性化 top-{top_k} 推荐……")
    model.eval()
    recommendations: dict = {}
    seen = matrix.tolil()

    with torch.no_grad():
        # 预计算全部视频 embedding（分批避免显存 OOM）
        # 这就是双塔推理的核心：视频侧算一次，用户侧算一次，再矩阵乘法
        all_idx  = torch.arange(n_items, device=device)
        emb_list = []
        for s in range(0, n_items, 2048):
            idx_b = all_idx[s:s + 2048]
            emb_list.append(model.item_tower(idx_b, item_cat_t[idx_b], item_dur_t[idx_b]))
        all_item_emb = torch.cat(emb_list, dim=0)  # (n_items, out_dim)

        # 批量用户推理：(USER_BATCH, D) × (D, n_items) → (USER_BATCH, n_items)
        for u_start in range(0, n_users, 256):
            u_end   = min(u_start + 256, n_users)
            u_idx_b = torch.arange(u_start, u_end, device=device)

            u_emb_b = model.user_tower(
                u_idx_b,
                user_active_t[u_idx_b],
                user_num_t[u_idx_b],
            )
            scores_np = (u_emb_b @ all_item_emb.T).cpu().numpy()  # (batch, n_items)

            for local_i, u_global in enumerate(range(u_start, u_end)):
                scores = scores_np[local_i].copy()   # copy：防止修改底层 tensor 内存
                seen_cols = seen.rows[u_global]
                if seen_cols:
                    scores[seen_cols] = -np.inf

                if top_k >= n_items:
                    top_idx = np.argsort(scores)[::-1]
                else:
                    top_idx = np.argpartition(scores, -top_k)[-top_k:]
                    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

                recommendations[user_ids[u_global]] = item_ids[top_idx].tolist()

    print(f"[TwoTower] 推荐生成完成，共 {len(recommendations):,} 位用户。")
    rec_df = recommendations_to_dataframe(recommendations)

    result: dict[str, Any] = {
        "model":    "TwoTower-WBPR" if weighted else "TwoTower-BPR",
        "n_users":  n_users,
        "n_items":  n_items,
        "emb_dim":  emb_dim,
        "out_dim":  out_dim,
        "top_k":    top_k,
        "bpr_loss": avg_loss,
        "rmse":     0.0,   # 排序模型无 RMSE，保留字段维持接口兼容
        "recommendations":    recommendations,
        "recommendations_df": rec_df,
        "_matrix":   matrix,
        "_user_ids": user_ids,
        "_item_ids": item_ids,
        "_interaction_df": df,
        "singular_values": [],
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_path = output_dir / f"two_tower_top{top_k}_recommendations.csv"
        rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
        print(f"[TwoTower] 推荐列表已保存：{rec_path}")
        result["output_path"] = str(rec_path)

    return result


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-Tower 双塔召回模型。")
    parser.add_argument("--emb-dim",    type=int,   default=DEFAULT_EMB_DIM)
    parser.add_argument("--hidden-dim", type=int,   default=DEFAULT_HIDDEN)
    parser.add_argument("--out-dim",    type=int,   default=DEFAULT_OUT_DIM)
    parser.add_argument("--n-epochs",   type=int,   default=50, help="最大训练轮数（Early Stopping 会提前终止）")
    parser.add_argument("--patience",   type=int,   default=5,  help="Early Stopping 容忍 patience 轮 val_loss 不改善")
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--top-k",      type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--no-weight",  action="store_true", help="用普通 BPR（不加权）")
    parser.add_argument("--output-dir", type=str,   default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )

    run_two_tower_pipeline(
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        n_epochs=args.n_epochs,
        patience=args.patience,
        lr=args.lr,
        top_k=args.top_k,
        weighted=not args.no_weight,
        output_dir=out,
    )
