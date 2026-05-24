"""
文件用途：SASRec（Self-Attentive Sequential Recommendation）召回模型。

【核心思想】
  Two-Tower 把用户历史压缩成一个静态向量（user embedding），
  SASRec 把历史行为序列当作 Transformer 的输入，每个位置都能
  关注序列里其他位置，捕捉"先看 A 再看 B 更可能看 C"的时序依赖。

  用户历史序列（按时间排列）:
    [v1, v2, v3, ..., vT]  →  Transformer  →  预测下一个视频

【与 Two-Tower 的关键区别】
  Two-Tower   user_id → embedding → 静态向量，不感知观看顺序
  SASRec      [v1..vT] → 自注意力 → 动态序列向量，感知时序

【训练方式：WBPR（与 Two-Tower 保持一致，便于公平对比）】
  对每条 (序列, 下一个正样本) 随机采一个负样本：
  loss = -mean(watch_ratio × log σ(pos_score − neg_score))

【推理】
  取序列最后一个位置的输出向量作为"当前用户兴趣向量"，
  与全量视频 embedding 做内积，top-K 即为推荐结果。
  视频侧 embedding 可提前算好离线存储（与双塔推理逻辑相同）。

【KuaiRec 数据适配说明】
  KuaiRec 的 big_matrix 没有时间戳，用 (user_id, video_id) 自然顺序
  作为"隐式序列"。这是 SASRec 在无时序数据下的常见退化用法，
  仍然比双塔多了"序列位置信息"这一维度的表达能力。
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
DEFAULT_EMB_DIM  = 64     # item embedding 维度（同时也是 Transformer d_model）
DEFAULT_N_HEADS  = 2      # 多头注意力头数
DEFAULT_N_LAYERS = 2      # Transformer 层数
DEFAULT_MAX_SEQ  = 50     # 截断序列长度（取最近 50 条历史）
DEFAULT_DROPOUT  = 0.2
DEFAULT_N_EPOCHS = 50
DEFAULT_LR       = 1e-3
DEFAULT_BATCH    = 2048   # CUDA/MPS 下序列模型推荐值；CPU 可改小
DEFAULT_TOP_K    = 50
DEFAULT_PATIENCE = 5


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# 数据准备：把交互记录转成用户行为序列
# ══════════════════════════════════════════════════════════════════════

def _build_user_sequences(
    df: pd.DataFrame,
    item_index: dict,
    max_seq: int,
) -> dict[Any, list[tuple[int, float]]]:
    """
    把交互 DataFrame 转成每个用户的历史序列。

    Returns:
        {user_id: [(item_idx, watch_ratio), ...]}  按原始顺序，长度 <= max_seq
    """
    sequences: dict[Any, list[tuple[int, float]]] = {}
    for uid, grp in df.groupby("user_id"):
        items = grp["video_id"].map(item_index).values
        ratios = grp["watch_ratio"].values.astype(np.float32)
        valid = [(int(i), float(r)) for i, r in zip(items, ratios) if pd.notna(i)]
        sequences[uid] = valid[-max_seq:]  # 取最近 max_seq 条
    return sequences


# ══════════════════════════════════════════════════════════════════════
# Dataset：序列 + WBPR 负采样
# ══════════════════════════════════════════════════════════════════════

class SASRecDataset(Dataset):
    """
    每个样本：(输入序列, 正样本, 负样本, watch_ratio 权重)

    构造方式：对用户序列 [v0, v1, ..., vT]，
    取 [v0..v_{T-1}] 为输入，v_T 为正样本，随机采一个未见视频为负样本。
    这是"next-item prediction"的标准训练范式。
    """

    def __init__(
        self,
        sequences: dict[Any, list[tuple[int, float]]],
        n_items: int,
        max_seq: int,
        user_index: dict,
    ):
        self.samples: list[tuple[np.ndarray, int, float]] = []
        # (pad_seq, pos_item_idx, watch_ratio)
        for uid, seq in sequences.items():
            if len(seq) < 2:
                continue
            input_seq = seq[:-1]
            pos_item, w = seq[-1]

            # padding 到 max_seq 长度（左 padding 0，0 是 PAD token）
            pad_seq = np.zeros(max_seq, dtype=np.int64)
            items_only = [it for it, _ in input_seq]
            if len(items_only) > max_seq:
                items_only = items_only[-max_seq:]
            # +1 偏移：0 保留给 PAD，所有真实 item idx 从 1 开始
            # 直接在赋值时 +1，避免 item_index=0 的视频被误当 PAD
            pad_seq[-len(items_only):] = np.array(items_only, dtype=np.int64) + 1

            self.samples.append((pad_seq, pos_item + 1, w))  # pos_item 也 +1

        self.n_items = n_items
        self.max_seq = max_seq

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int, float]:
        seq, pos, w = self.samples[idx]
        neg = random.randint(1, self.n_items)  # 1..n_items（0 是 PAD）
        return torch.from_numpy(seq), pos, neg, np.float32(w)


# ══════════════════════════════════════════════════════════════════════
# SASRec 模型
# ══════════════════════════════════════════════════════════════════════

class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation（Wang-Cheng Kang & Julian McAuley, 2018）。

    简化版本：去掉原版的 FFN dropout，保留核心的
    因果自注意力（causal self-attention）+ 位置编码 + LayerNorm。

    embedding(0) 作为 PAD token，推理时被 attention mask 屏蔽。
    """

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

        # n_items + 1：0 是 PAD token
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(emb_dim)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """上三角 mask，确保位置 t 只能看 <= t 的历史（因果性）。"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask  # True = 屏蔽

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (batch, max_seq) LongTensor，0 为 PAD
        返回: (batch, max_seq, emb_dim) 序列每个位置的上下文表示
        """
        B, L = seq.shape
        device = seq.device

        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(seq) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)

        causal = self._causal_mask(L, device)
        # 只用因果 mask，不用 padding mask：
        # 左 padding + 因果 mask + padding mask 三叠会导致 PAD 位置无处 attend → nan
        # 因果 mask 已保证不泄露未来信息，padding mask 在此冗余且有害
        x = self.transformer(x, mask=causal)
        return self.out_norm(x)  # (batch, L, D)

    def forward(
        self,
        seq: torch.Tensor,       # (batch, max_seq)
        pos_items: torch.Tensor, # (batch,)
        neg_items: torch.Tensor, # (batch,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (pos_score, neg_score)，各 (batch,)。
        取序列最后一个有效位置的向量与目标 item 做内积。
        """
        seq_out = self.encode(seq)                  # (batch, L, D)
        last_out = seq_out[:, -1, :]                # (batch, D)：最后位置 = 当前兴趣

        pos_emb = self.item_emb(pos_items)          # (batch, D)
        neg_emb = self.item_emb(neg_items)          # (batch, D)

        pos_score = (last_out * pos_emb).sum(-1)    # (batch,)
        neg_score = (last_out * neg_emb).sum(-1)    # (batch,)
        return pos_score, neg_score

    def get_user_vector(self, seq: torch.Tensor) -> torch.Tensor:
        """推理用：返回 (batch, D) 归一化用户兴趣向量。"""
        seq_out = self.encode(seq)
        last_out = seq_out[:, -1, :]
        return F.normalize(last_out, dim=-1)

    def get_item_vectors(self, item_idx: torch.Tensor) -> torch.Tensor:
        """推理用：返回 (batch, D) 归一化视频向量。"""
        return F.normalize(self.item_emb(item_idx), dim=-1)


# ══════════════════════════════════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════════════════════════════════

def run_sasrec_pipeline(
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
    val_frac: float = 0.1,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    SASRec 完整流程，返回格式与 run_two_tower_pipeline 兼容。
    """
    device = _get_device()
    print(f"[SASRec] device = {device}")

    # 1. 加载交互数据（与 Two-Tower 相同数据源）
    df = _test_df if _test_df is not None else load_big_matrix_interactions(eligible_video_ids)
    matrix, user_ids, item_ids = build_sparse_matrix(df)
    n_users, n_items = len(user_ids), len(item_ids)

    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    # 2. 构建用户行为序列
    print(f"[SASRec] 构建用户行为序列（max_seq={max_seq}）……")
    sequences = _build_user_sequences(df, item_index, max_seq)
    print(f"[SASRec] 共 {len(sequences):,} 用户有序列（>= 2 条交互）。")

    # 3. 切分训练/验证（用户级别切分，不是交互级别）
    all_uids = list(sequences.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(all_uids)
    n_val_users = max(1, int(len(all_uids) * val_frac))
    val_uids   = set(all_uids[:n_val_users])
    train_uids = set(all_uids[n_val_users:])

    train_seqs = {u: sequences[u] for u in train_uids}
    val_seqs   = {u: sequences[u] for u in val_uids}

    train_ds = SASRecDataset(train_seqs, n_items, max_seq, user_index)
    val_ds   = SASRecDataset(val_seqs,   n_items, max_seq, user_index)

    if len(train_ds) == 0:
        raise ValueError("[SASRec] 训练集为空，请检查数据。")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    print(f"[SASRec] 训练集 {len(train_ds):,} 样本，验证集 {len(val_ds):,} 样本。")

    # 4. 模型 & 优化器
    model = SASRec(
        n_items=n_items, emb_dim=emb_dim, n_heads=n_heads,
        n_layers=n_layers, max_seq=max_seq, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. Checkpoint 恢复
    ckpt_path: Path | None = None
    best_ckpt_path: Path | None = None
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state: dict | None = None

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path      = checkpoint_dir / "sasrec_latest.pt"
        best_ckpt_path = checkpoint_dir / "sasrec_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch      = ckpt["epoch"] + 1
            best_val_loss    = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            if patience_counter >= patience:
                print(f"[SASRec] 从 checkpoint 恢复：Early Stopping 已完成，直接推理。")
                start_epoch = n_epochs
            else:
                print(f"[SASRec] 从 checkpoint 恢复：epoch {start_epoch}/{n_epochs}")

    if n_epochs <= 0 and not (
        (best_ckpt_path is not None and best_ckpt_path.exists()) or
        (ckpt_path is not None and ckpt_path.exists())
    ):
        raise ValueError("[SASRec] 当前是仅推理模式，但找不到可恢复的 checkpoint。")

    print(
        f"[SASRec] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，"
        f"emb={emb_dim}，heads={n_heads}，layers={n_layers}，"
        f"max_seq={max_seq}，max_epochs={n_epochs}，patience={patience}"
    )

    # 6. 训练 + Early Stopping
    avg_loss = 0.0
    avg_val_loss = float("inf")

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        # ── 训练 ──────────────────────────────────────────────
        model.train()
        total_loss, n_seen = 0.0, 0
        for seq_b, pos_b, neg_b, w_b in train_loader:
            seq_b = seq_b.to(device)
            pos_b = pos_b.to(device)
            neg_b = neg_b.to(device)
            w_b   = w_b.to(device)

            pos_score, neg_score = model(seq_b, pos_b, neg_b)
            # clamp(min=1e-8)：防止 w_b=0 与极端负分相乘产生 0×(-inf)=NaN
            loss = -(w_b.clamp(min=1e-8) * F.logsigmoid(pos_score - neg_score)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(seq_b)
            n_seen += len(seq_b)

        avg_loss = total_loss / n_seen

        # ── 验证 ──────────────────────────────────────────────
        model.eval()
        val_total, n_val_seen = 0.0, 0
        with torch.no_grad():
            for seq_b, pos_b, neg_b, w_b in val_loader:
                seq_b = seq_b.to(device)
                pos_b = pos_b.to(device)
                neg_b = neg_b.to(device)
                w_b   = w_b.to(device)
                pos_score, neg_score = model(seq_b, pos_b, neg_b)
                vl = -(w_b.clamp(min=1e-8) * F.logsigmoid(pos_score - neg_score)).mean()
                val_total  += vl.item() * len(seq_b)
                n_val_seen += len(seq_b)

        avg_val_loss = val_total / n_val_seen
        improved = avg_val_loss < best_val_loss - 1e-6
        star = " ★" if improved else ""

        print(
            f"  epoch {epoch+1:>2}/{n_epochs}  WBPR-loss={avg_loss:.6f}"
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
                    "avg_loss": avg_loss, "val_loss": avg_val_loss,
                }, best_ckpt_path)
                print(f"  └─ 最佳模型已保存（val_loss={avg_val_loss:.6f}）")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[SASRec] Early Stopping：连续 {patience} 轮 val_loss 未改善，"
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

        if ckpt_path is not None:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "avg_loss": avg_loss, "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss, "patience_counter": patience_counter,
            }, ckpt_path)

    # 7. 恢复最佳权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[SASRec] 已加载最佳权重（val_loss={best_val_loss:.6f}）")
    elif best_ckpt_path is not None and best_ckpt_path.exists():
        best = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        print(f"[SASRec] 已从磁盘加载最佳权重（val_loss={best['val_loss']:.6f}）")

    print(f"[SASRec] 训练完成，最终 loss={avg_loss:.6f}，最佳 val_loss={best_val_loss:.6f}")

    # 8. 生成个性化推荐（所有在 big_matrix 中有记录的用户）
    print(f"\n[SASRec] 为 {n_users:,} 位用户生成个性化 top-{top_k} 推荐……")
    model.eval()
    recommendations: dict = {}
    seen = matrix.tolil()

    # 预计算全量视频归一化 embedding（离线索引，与双塔推理逻辑一致）
    all_item_idx = torch.arange(1, n_items + 1, device=device)
    item_emb_list = []
    with torch.no_grad():
        for s in range(0, n_items, 2048):
            idx_b = all_item_idx[s:s + 2048]
            item_emb_list.append(model.get_item_vectors(idx_b))
    all_item_emb = torch.cat(item_emb_list, dim=0)  # (n_items, D)

    # 为每个用户构建输入序列 tensor，批量推理
    with torch.no_grad():
        uid_list = list(sequences.keys())
        for batch_start in range(0, len(uid_list), 256):
            batch_uids = uid_list[batch_start:batch_start + 256]
            batch_seqs = []
            for uid in batch_uids:
                seq = sequences[uid]
                pad_seq = np.zeros(max_seq, dtype=np.int64)
                items_only = [it for it, _ in seq][-max_seq:]
                pad_seq[-len(items_only):] = np.array(items_only, dtype=np.int64) + 1
                batch_seqs.append(pad_seq)

            seq_t = torch.from_numpy(np.stack(batch_seqs)).to(device)
            user_vecs = model.get_user_vector(seq_t)                     # (B, D)
            scores_np = (user_vecs @ all_item_emb.T).cpu().numpy()       # (B, n_items)

            for local_i, uid in enumerate(batch_uids):
                u_global = user_index[uid]
                scores = scores_np[local_i].copy()
                seen_cols = seen.rows[u_global]
                if seen_cols:
                    scores[seen_cols] = -np.inf

                if top_k >= n_items:
                    top_idx = np.argsort(scores)[::-1]
                else:
                    top_idx = np.argpartition(scores, -top_k)[-top_k:]
                    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

                recommendations[uid] = item_ids[top_idx].tolist()

    # 没有序列的用户（交互 < 2 条）给空推荐，保证 user_ids 全覆盖
    for uid in user_ids:
        if uid not in recommendations:
            recommendations[uid] = []

    print(f"[SASRec] 推荐生成完成，共 {len(recommendations):,} 位用户。")
    rec_df = recommendations_to_dataframe(recommendations)

    result: dict[str, Any] = {
        "model":    "SASRec",
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
        rec_path = output_dir / f"sasrec_top{top_k}_recommendations.csv"
        rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
        print(f"[SASRec] 推荐列表已保存：{rec_path}")
        result["output_path"] = str(rec_path)

    return result


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SASRec 序列推荐模型。")
    parser.add_argument("--emb-dim",   type=int,   default=DEFAULT_EMB_DIM)
    parser.add_argument("--n-heads",   type=int,   default=DEFAULT_N_HEADS)
    parser.add_argument("--n-layers",  type=int,   default=DEFAULT_N_LAYERS)
    parser.add_argument("--max-seq",   type=int,   default=DEFAULT_MAX_SEQ)
    parser.add_argument("--dropout",   type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--n-epochs",  type=int,   default=DEFAULT_N_EPOCHS)
    parser.add_argument("--patience",  type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--lr",        type=float, default=DEFAULT_LR)
    parser.add_argument("--top-k",     type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--output-dir", type=str,  default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )

    run_sasrec_pipeline(
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq=args.max_seq,
        dropout=args.dropout,
        n_epochs=args.n_epochs,
        patience=args.patience,
        lr=args.lr,
        top_k=args.top_k,
        output_dir=out,
    )
