"""
文件用途：CL4SRec（Contrastive Learning for Sequential Recommendation, 2022）。

【核心思想 vs SASRec】
  SASRec 用 WBPR 损失，只有正负样本对的监督信号，数据稀疏时容易过拟合。
  CL4SRec 在 WBPR 之上加了自监督对比学习（Self-Supervised Contrastive Learning）：
    1. 对同一个用户序列做两次随机数据增强，得到两个不同的"视图"
    2. 同一用户的两个视图应互相接近（正对），不同用户的视图应远离（负对）
    3. 用 InfoNCE loss（NT-Xent）来学这个约束
  这相当于"免费的额外正则化"，让序列编码器学到更鲁棒、泛化更好的用户表示。

【三种数据增强操作（论文原版）】
  Crop   随机截取序列的一个子区间（保留局部兴趣）
  Mask   随机遮盖若干位置（训练对噪声的鲁棒性）
  Reorder 随机打乱一个子区间（测试对顺序扰动的鲁棒性）

【损失函数】
  total_loss = WBPR_loss + λ * CL_loss（InfoNCE）
  λ = cl_weight（默认 0.1）

【与 SideInfo-SASRec 的关系】
  CL4SRec 这里基于纯 SASRec（ID-only）实现，专注展示对比学习的增益。
  工业界常见的扩展是 CL4SRec + SideInfo，两者正交可叠加。

【KuaiRec 适配】
  序列构建、WBPR 训练目标、推理逻辑与 SASRec 完全相同。
  返回格式兼容 eval_advanced.py。
"""
from __future__ import annotations

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

# ── 常量 ──────────────────────────────────────────────────────────────
DEFAULT_EMB_DIM    = 64
DEFAULT_N_HEADS    = 2
DEFAULT_N_LAYERS   = 2
DEFAULT_MAX_SEQ    = 50
DEFAULT_DROPOUT    = 0.2
DEFAULT_N_EPOCHS   = 50
DEFAULT_LR         = 1e-3
DEFAULT_BATCH      = 2048
DEFAULT_TOP_K      = 50
DEFAULT_PATIENCE   = 5
DEFAULT_CL_WEIGHT  = 0.1    # λ：对比学习损失的权重
DEFAULT_CROP_RATIO = 0.7    # Crop 保留比例
DEFAULT_MASK_RATIO = 0.2    # Mask 遮盖比例
DEFAULT_REORDER_RATIO = 0.3 # Reorder 打乱比例
DEFAULT_TEMP       = 0.2    # InfoNCE 温度系数


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# 数据增强操作
# ══════════════════════════════════════════════════════════════════════

def _augment_crop(seq: list[int], ratio: float) -> list[int]:
    """随机截取序列的一个连续子区间，保留 ratio 比例。"""
    n = len(seq)
    if n <= 1:
        return seq[:]
    keep = max(1, int(n * ratio))
    start = random.randint(0, n - keep)
    return seq[start:start + keep]


def _augment_mask(seq: list[int], ratio: float, n_items: int) -> list[int]:
    """随机遮盖 ratio 比例的位置（替换为随机 item，避免 0=PAD 混淆）。"""
    seq = seq[:]
    n_mask = max(1, int(len(seq) * ratio))
    positions = random.sample(range(len(seq)), min(n_mask, len(seq)))
    for p in positions:
        seq[p] = random.randint(1, n_items)   # item token: 1..n_items
    return seq


def _augment_reorder(seq: list[int], ratio: float) -> list[int]:
    """随机选一个子区间并打乱顺序。"""
    seq = seq[:]
    n = len(seq)
    if n <= 1:
        return seq
    sub_len = max(2, int(n * ratio))
    start = random.randint(0, n - sub_len)
    sub = seq[start:start + sub_len]
    random.shuffle(sub)
    seq[start:start + sub_len] = sub
    return seq


def _apply_augmentation(seq: list[int], n_items: int,
                         crop_ratio: float, mask_ratio: float, reorder_ratio: float) -> list[int]:
    """随机选择一种增强方式。"""
    op = random.choice(["crop", "mask", "reorder"])
    if op == "crop":
        return _augment_crop(seq, crop_ratio)
    elif op == "mask":
        return _augment_mask(seq, mask_ratio, n_items)
    else:
        return _augment_reorder(seq, reorder_ratio)


def _pad_seq(seq: list[int], max_seq: int) -> np.ndarray:
    """左 padding 到 max_seq，seq 中已含 +1 偏移（0=PAD）。"""
    arr = np.zeros(max_seq, dtype=np.int64)
    s = seq[-max_seq:]
    arr[-len(s):] = s
    return arr


# ══════════════════════════════════════════════════════════════════════
# 数据准备
# ══════════════════════════════════════════════════════════════════════

def _build_user_sequences(
    df: pd.DataFrame,
    item_index: dict,
    max_seq: int,
) -> dict[Any, list[tuple[int, float]]]:
    """返回 {user_id: [(item_idx, watch_ratio), ...]}，长度 <= max_seq。"""
    sequences: dict[Any, list[tuple[int, float]]] = {}
    for uid, grp in df.groupby("user_id"):
        items  = grp["video_id"].map(item_index).values
        ratios = grp["watch_ratio"].values.astype(np.float32)
        valid  = [(int(i), float(r)) for i, r in zip(items, ratios) if pd.notna(i)]
        sequences[uid] = valid[-max_seq:]
    return sequences


# ══════════════════════════════════════════════════════════════════════
# Dataset：WBPR next-item + 两路增强视图（用于 CL）
# ══════════════════════════════════════════════════════════════════════

class CL4SRecDataset(Dataset):
    """
    每个样本返回：
      seq        (max_seq,)  原始序列（WBPR 训练用）
      pos_item   int          下一个正样本
      neg_item   int          随机负样本
      weight     float        WBPR 权重
      aug1       (max_seq,)  增强视图 1（CL 用）
      aug2       (max_seq,)  增强视图 2（CL 用）
    """

    def __init__(
        self,
        sequences: dict[Any, list[tuple[int, float]]],
        n_items: int,
        max_seq: int,
        crop_ratio: float,
        mask_ratio: float,
        reorder_ratio: float,
    ):
        self.max_seq = max_seq
        self.n_items = n_items
        self.crop_ratio    = crop_ratio
        self.mask_ratio    = mask_ratio
        self.reorder_ratio = reorder_ratio

        # 存储：(pad_seq, raw_items_list, pos_item+1, watch_ratio)
        self.samples: list[tuple[np.ndarray, list[int], int, float]] = []

        for uid, seq in sequences.items():
            if len(seq) < 2:
                continue
            input_seq = seq[:-1]
            pos_item, w = seq[-1]

            items_only = [it + 1 for it, _ in input_seq]   # +1 偏移，0=PAD
            pad_seq = _pad_seq(items_only, max_seq)

            self.samples.append((pad_seq, items_only, pos_item + 1, w))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pad_seq, raw_items, pos, w = self.samples[idx]
        neg = random.randint(1, self.n_items)

        # 两路独立随机增强（每次调用都重新随机）
        a1 = _pad_seq(
            _apply_augmentation(raw_items, self.n_items,
                                 self.crop_ratio, self.mask_ratio, self.reorder_ratio),
            self.max_seq,
        )
        a2 = _pad_seq(
            _apply_augmentation(raw_items, self.n_items,
                                 self.crop_ratio, self.mask_ratio, self.reorder_ratio),
            self.max_seq,
        )

        return (
            torch.from_numpy(pad_seq),
            pos,
            neg,
            np.float32(w),
            torch.from_numpy(a1),
            torch.from_numpy(a2),
        )


# ══════════════════════════════════════════════════════════════════════
# SASRec 编码器（CL4SRec 中复用相同结构）
# ══════════════════════════════════════════════════════════════════════

class CL4SRecEncoder(nn.Module):
    """与 SASRec 相同的因果 Transformer 编码器，抽出来方便两次 forward。"""

    def __init__(
        self,
        n_items: int,
        emb_dim: int = DEFAULT_EMB_DIM,
        n_heads: int = DEFAULT_N_HEADS,
        n_layers: int = DEFAULT_N_LAYERS,
        max_seq: int = DEFAULT_MAX_SEQ,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq = max_seq

        self.item_emb    = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb     = nn.Embedding(max_seq, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm    = nn.LayerNorm(emb_dim)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (batch, max_seq) → (batch, max_seq, emb_dim)"""
        B, L = seq.shape
        device = seq.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(seq) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)
        causal = self._causal_mask(L, device)
        x = self.transformer(x, mask=causal)
        return self.out_norm(x)

    def get_last(self, seq: torch.Tensor) -> torch.Tensor:
        """返回 (batch, emb_dim) 序列最后位置向量。"""
        return self.encode(seq)[:, -1, :]

    def get_user_vector(self, seq: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.get_last(seq), dim=-1)

    def get_item_vectors(self, item_idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_emb(item_idx), dim=-1)


# ══════════════════════════════════════════════════════════════════════
# InfoNCE 对比损失
# ══════════════════════════════════════════════════════════════════════

def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
    """
    NT-Xent（InfoNCE）损失。

    z1, z2: (batch, emb_dim) 归一化向量。
    同一 batch 内，(z1[i], z2[i]) 为正对，其余为负对。

    temperature 控制分布的尖锐程度：
      低 temp → 更 hard 的负例学习，但梯度可能爆炸；
      高 temp → 更平滑但正例学习信号弱。论文推荐 0.1-0.5。
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    batch = z1.size(0)

    # 合并两路：(2B, D)，前 B 是 z1，后 B 是 z2
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / temp   # (2B, 2B)

    # 去掉自相似（对角线）
    mask = torch.eye(2 * batch, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # 正对位置：z1[i] 对应 z2[i]，即位置 (i, i+B) 和 (i+B, i)
    labels = torch.arange(batch, device=z.device)
    labels = torch.cat([labels + batch, labels], dim=0)   # (2B,)

    loss = F.cross_entropy(sim, labels)
    return loss


# ══════════════════════════════════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════════════════════════════════

def run_cl4srec_pipeline(
    emb_dim: int = DEFAULT_EMB_DIM,
    n_heads: int = DEFAULT_N_HEADS,
    n_layers: int = DEFAULT_N_LAYERS,
    max_seq: int = DEFAULT_MAX_SEQ,
    dropout: float = DEFAULT_DROPOUT,
    n_epochs: int = DEFAULT_N_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH,
    top_k: int = DEFAULT_TOP_K,
    patience: int = DEFAULT_PATIENCE,
    cl_weight: float = DEFAULT_CL_WEIGHT,
    temp: float = DEFAULT_TEMP,
    crop_ratio: float = DEFAULT_CROP_RATIO,
    mask_ratio: float = DEFAULT_MASK_RATIO,
    reorder_ratio: float = DEFAULT_REORDER_RATIO,
    val_frac: float = 0.1,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """CL4SRec 完整流程，返回格式与 run_sasrec_pipeline 兼容。"""
    device = _get_device()
    print(f"[CL4SRec] device = {device}")

    df = _test_df if _test_df is not None else load_big_matrix_interactions(eligible_video_ids)
    matrix, user_ids, item_ids = build_sparse_matrix(df)
    n_users, n_items = len(user_ids), len(item_ids)

    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    print(f"[CL4SRec] 构建用户行为序列（max_seq={max_seq}）……")
    sequences = _build_user_sequences(df, item_index, max_seq)
    print(f"[CL4SRec] 共 {len(sequences):,} 用户有序列。")

    all_uids = list(sequences.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(all_uids)
    n_val = max(1, int(len(all_uids) * val_frac))
    val_uids   = set(all_uids[:n_val])
    train_uids = set(all_uids[n_val:])

    train_ds = CL4SRecDataset(
        {u: sequences[u] for u in train_uids},
        n_items, max_seq, crop_ratio, mask_ratio, reorder_ratio,
    )
    val_ds = CL4SRecDataset(
        {u: sequences[u] for u in val_uids},
        n_items, max_seq, crop_ratio, mask_ratio, reorder_ratio,
    )

    if len(train_ds) == 0:
        raise ValueError("[CL4SRec] 训练集为空。")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    print(f"[CL4SRec] 训练集 {len(train_ds):,} 样本，验证集 {len(val_ds):,} 样本。")

    model = CL4SRecEncoder(
        n_items=n_items, emb_dim=emb_dim, n_heads=n_heads,
        n_layers=n_layers, max_seq=max_seq, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path: Path | None = None
    best_ckpt_path: Path | None = None
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state: dict | None = None

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path      = checkpoint_dir / "cl4srec_latest.pt"
        best_ckpt_path = checkpoint_dir / "cl4srec_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch      = ckpt["epoch"] + 1
            best_val_loss    = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            if patience_counter >= patience:
                start_epoch = n_epochs
                print(f"[CL4SRec] 从 checkpoint 恢复：Early Stopping 已完成，直接推理。")
            else:
                print(f"[CL4SRec] 从 checkpoint 恢复：epoch {start_epoch}/{n_epochs}")

    if n_epochs <= 0 and not (
        (best_ckpt_path is not None and best_ckpt_path.exists()) or
        (ckpt_path is not None and ckpt_path.exists())
    ):
        raise ValueError("[CL4SRec] 当前是仅推理模式，但找不到可恢复的 checkpoint。")

    print(
        f"[CL4SRec] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，"
        f"emb={emb_dim}，cl_weight={cl_weight}，temp={temp}"
    )

    avg_loss = 0.0
    avg_val_loss = float("inf")

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        model.train()
        total_loss, n_seen = 0.0, 0
        for seq_b, pos_b, neg_b, w_b, aug1_b, aug2_b in train_loader:
            seq_b  = seq_b.to(device)
            pos_b  = pos_b.to(device)
            neg_b  = neg_b.to(device)
            w_b    = w_b.to(device)
            aug1_b = aug1_b.to(device)
            aug2_b = aug2_b.to(device)

            # ── WBPR loss ────────────────────────────────────────────
            last_out = model.get_last(seq_b)                    # (B, D)
            pos_emb  = model.item_emb(pos_b)
            neg_emb  = model.item_emb(neg_b)
            pos_score = (last_out * pos_emb).sum(-1)
            neg_score = (last_out * neg_emb).sum(-1)
            wbpr_loss = -(w_b * F.logsigmoid(pos_score - neg_score)).mean()

            # ── InfoNCE loss（两路增强视图）───────────────────────────
            z1 = model.get_last(aug1_b)
            z2 = model.get_last(aug2_b)
            cl_loss = info_nce_loss(z1, z2, temp)

            loss = wbpr_loss + cl_weight * cl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(seq_b)
            n_seen     += len(seq_b)

        avg_loss = total_loss / n_seen

        model.eval()
        val_total, n_val_seen = 0.0, 0
        with torch.no_grad():
            for seq_b, pos_b, neg_b, w_b, aug1_b, aug2_b in val_loader:
                seq_b  = seq_b.to(device)
                pos_b  = pos_b.to(device)
                neg_b  = neg_b.to(device)
                w_b    = w_b.to(device)
                aug1_b = aug1_b.to(device)
                aug2_b = aug2_b.to(device)

                last_out  = model.get_last(seq_b)
                pos_emb   = model.item_emb(pos_b)
                neg_emb   = model.item_emb(neg_b)
                ps = (last_out * pos_emb).sum(-1)
                ns = (last_out * neg_emb).sum(-1)
                vl_wbpr = -(w_b * F.logsigmoid(ps - ns)).mean()
                vl_cl   = info_nce_loss(model.get_last(aug1_b), model.get_last(aug2_b), temp)
                vl = vl_wbpr + cl_weight * vl_cl

                val_total  += vl.item() * len(seq_b)
                n_val_seen += len(seq_b)

        avg_val_loss = val_total / n_val_seen
        improved = avg_val_loss < best_val_loss - 1e-6
        star = " ★" if improved else ""

        print(
            f"  epoch {epoch+1:>2}/{n_epochs}  loss={avg_loss:.6f}"
            f"  val={avg_val_loss:.6f}{star}  ({time.time()-t0:.1f}s)"
        )

        if improved:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            if best_ckpt_path is not None:
                torch.save({
                    "epoch": epoch, "model": best_model_state,
                    "optimizer": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                }, best_ckpt_path)
                print(f"  └─ 最佳模型已保存（val_loss={avg_val_loss:.6f}）")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[CL4SRec] Early Stopping：连续 {patience} 轮未改善，"
                    f"停止在 epoch {epoch+1}"
                )
                if ckpt_path is not None:
                    torch.save({
                        "epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "avg_loss": avg_loss, "val_loss": avg_val_loss,
                        "best_val_loss": best_val_loss, "patience_counter": patience_counter,
                    }, ckpt_path)
                break

        if ckpt_path is not None:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_loss": avg_loss, "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss, "patience_counter": patience_counter,
            }, ckpt_path)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[CL4SRec] 已加载最佳权重（val_loss={best_val_loss:.6f}）")
    elif best_ckpt_path is not None and best_ckpt_path.exists():
        best = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best["model"])
        print(f"[CL4SRec] 已从磁盘加载最佳权重（val_loss={best['val_loss']:.6f}）")

    print(f"[CL4SRec] 训练完成，最终 loss={avg_loss:.6f}，最佳 val_loss={best_val_loss:.6f}")

    # ── 推理 ──────────────────────────────────────────────────────────
    print(f"\n[CL4SRec] 为 {n_users:,} 位用户生成个性化 top-{top_k} 推荐……")
    model.eval()
    recommendations: dict = {}
    seen = matrix.tolil()

    all_item_idx = torch.arange(1, n_items + 1, device=device)
    item_emb_chunks = []
    with torch.no_grad():
        for s in range(0, n_items, 2048):
            idx_b = all_item_idx[s:s + 2048]
            item_emb_chunks.append(model.get_item_vectors(idx_b))
    all_item_emb = torch.cat(item_emb_chunks, dim=0)   # (n_items, D)

    uid_list = list(sequences.keys())
    with torch.no_grad():
        for batch_start in range(0, len(uid_list), 256):
            batch_uids = uid_list[batch_start:batch_start + 256]
            batch_seqs = []
            for uid in batch_uids:
                seq = sequences[uid]
                pad_seq = np.zeros(max_seq, dtype=np.int64)
                items_only = [it + 1 for it, _ in seq][-max_seq:]
                pad_seq[-len(items_only):] = items_only
                batch_seqs.append(pad_seq)

            seq_t     = torch.from_numpy(np.stack(batch_seqs)).to(device)
            user_vecs = model.get_user_vector(seq_t)                       # (B, D)
            scores_np = (user_vecs @ all_item_emb.T).cpu().numpy()         # (B, n_items)

            for local_i, uid in enumerate(batch_uids):
                u_global  = user_index[uid]
                scores    = scores_np[local_i].copy()
                seen_cols = seen.rows[u_global]
                if seen_cols:
                    scores[seen_cols] = -np.inf

                if top_k >= n_items:
                    top_idx = np.argsort(scores)[::-1]
                else:
                    top_idx = np.argpartition(scores, -top_k)[-top_k:]
                    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

                recommendations[uid] = item_ids[top_idx].tolist()

    for uid in user_ids:
        if uid not in recommendations:
            recommendations[uid] = []

    print(f"[CL4SRec] 推荐生成完成，共 {len(recommendations):,} 位用户。")
    rec_df = recommendations_to_dataframe(recommendations)

    result: dict[str, Any] = {
        "model":    "CL4SRec",
        "n_users":  n_users,
        "n_items":  n_items,
        "emb_dim":  emb_dim,
        "top_k":    top_k,
        "bpr_loss": avg_loss,
        "rmse":     0.0,
        "recommendations":    recommendations,
        "recommendations_df": rec_df,
        "_matrix":            matrix,
        "_user_ids":          user_ids,
        "_item_ids":          item_ids,
        "_interaction_df":    df,
        "singular_values":    [],
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_path = output_dir / f"cl4srec_top{top_k}_recommendations.csv"
        rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
        print(f"[CL4SRec] 推荐列表已保存：{rec_path}")
        result["output_path"] = str(rec_path)

    return result


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CL4SRec：对比学习增强的序列推荐。")
    parser.add_argument("--emb-dim",       type=int,   default=DEFAULT_EMB_DIM)
    parser.add_argument("--n-heads",       type=int,   default=DEFAULT_N_HEADS)
    parser.add_argument("--n-layers",      type=int,   default=DEFAULT_N_LAYERS)
    parser.add_argument("--max-seq",       type=int,   default=DEFAULT_MAX_SEQ)
    parser.add_argument("--dropout",       type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--n-epochs",      type=int,   default=DEFAULT_N_EPOCHS)
    parser.add_argument("--patience",      type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--lr",            type=float, default=DEFAULT_LR)
    parser.add_argument("--top-k",         type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--cl-weight",     type=float, default=DEFAULT_CL_WEIGHT)
    parser.add_argument("--temp",          type=float, default=DEFAULT_TEMP)
    parser.add_argument("--crop-ratio",    type=float, default=DEFAULT_CROP_RATIO)
    parser.add_argument("--mask-ratio",    type=float, default=DEFAULT_MASK_RATIO)
    parser.add_argument("--reorder-ratio", type=float, default=DEFAULT_REORDER_RATIO)
    parser.add_argument("--output-dir",    type=str,   default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )

    run_cl4srec_pipeline(
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq=args.max_seq,
        dropout=args.dropout,
        n_epochs=args.n_epochs,
        patience=args.patience,
        lr=args.lr,
        top_k=args.top_k,
        cl_weight=args.cl_weight,
        temp=args.temp,
        crop_ratio=args.crop_ratio,
        mask_ratio=args.mask_ratio,
        reorder_ratio=args.reorder_ratio,
        output_dir=out,
    )
