"""
文件用途：BERT4Rec（Bidirectional Encoder Representations from Transformers for Rec）。

【核心思想 vs SASRec】
  SASRec  用因果（单向）自注意力，只能看历史，训练时预测"下一个"。
  BERT4Rec 用双向自注意力，随机遮盖序列中的 item（Masked Item Prediction），
           训练时同时利用上下文（前后都能看），捕捉更丰富的用户兴趣模式。

【Masked Item Prediction（类比 BERT MLM）】
  输入序列 [v1, v2, [MASK], v4, v5]
  目标：预测 [MASK] 位置原本是什么 item
  mask 比例默认 20%（论文推荐值）

【推理时的处理】
  把序列最后一个位置替换成 [MASK]，用模型预测该位置 → top-K 即推荐结果。
  这与训练目标一致（预测遮盖位置），不像 SASRec 那样有训练/推理不一致问题。

【KuaiRec 适配】
  与 SASRec 相同数据准备逻辑（big_matrix，用户行为序列，无时间戳）。
  返回格式与 run_sasrec_pipeline 完全兼容，可直接插入 eval_advanced.py 的对比框架。
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
DEFAULT_EMB_DIM   = 64
DEFAULT_N_HEADS   = 2
DEFAULT_N_LAYERS  = 2
DEFAULT_MAX_SEQ   = 50
DEFAULT_DROPOUT   = 0.2
DEFAULT_MASK_PROB = 0.2     # Masked Item Prediction 的遮盖概率
DEFAULT_N_EPOCHS  = 50
DEFAULT_LR        = 1e-3
DEFAULT_BATCH     = 256    # CrossEntropy logits=(B*50*9383)，2048会OOM，256≈0.5GB安全
DEFAULT_TOP_K     = 50
DEFAULT_PATIENCE  = 5


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# 数据准备
# ══════════════════════════════════════════════════════════════════════

def _build_user_sequences(
    df: pd.DataFrame,
    item_index: dict,
    max_seq: int,
) -> dict[Any, list[int]]:
    """返回 {user_id: [item_idx, ...]} 按原始顺序，长度 <= max_seq。"""
    sequences: dict[Any, list[int]] = {}
    for uid, grp in df.groupby("user_id"):
        items = grp["video_id"].map(item_index).dropna().astype(int).tolist()
        # +1 偏移：0=PAD，1=MASK，真实 item 从 2 开始
        sequences[uid] = [it + 2 for it in items[-max_seq:]]
    return sequences


# ══════════════════════════════════════════════════════════════════════
# Dataset：Masked Item Prediction
# ══════════════════════════════════════════════════════════════════════

MASK_TOKEN = 1   # 0=PAD，1=[MASK]，item_idx+2 = 真实 item


class BERT4RecDataset(Dataset):
    """
    每个样本：(masked_seq, labels)
    labels[i] = 原始 item_idx（如果该位置被遮盖），否则 = 0（忽略）。
    只在被遮盖的位置计算 cross-entropy loss。
    """

    def __init__(
        self,
        sequences: dict[Any, list[int]],
        n_items: int,
        max_seq: int,
        mask_prob: float,
    ):
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.n_items  = n_items
        self.max_seq  = max_seq
        self.mask_prob = mask_prob

        for uid, seq in sequences.items():
            if len(seq) < 2:
                continue
            # 左 padding 到 max_seq
            pad_seq = np.zeros(max_seq, dtype=np.int64)
            s = seq[-max_seq:]
            pad_seq[-len(s):] = s
            self.samples.append(pad_seq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq = self.samples[idx].copy()   # (max_seq,)
        labels = np.zeros_like(seq)      # 0 = 不计 loss

        for i in range(self.max_seq):
            if seq[i] == 0:              # PAD 不遮盖
                continue
            if random.random() < self.mask_prob:
                labels[i] = seq[i]      # 记录原始 item
                seq[i] = MASK_TOKEN     # 替换成 [MASK]

        # 若全部没被遮盖（小序列概率较低），强制遮盖最后一个非 PAD 位置
        if labels.sum() == 0:
            for i in range(self.max_seq - 1, -1, -1):
                if seq[i] != 0:
                    labels[i] = seq[i]
                    seq[i] = MASK_TOKEN
                    break

        return torch.from_numpy(seq), torch.from_numpy(labels)


# ══════════════════════════════════════════════════════════════════════
# BERT4Rec 模型
# ══════════════════════════════════════════════════════════════════════

class BERT4Rec(nn.Module):
    """
    双向 Transformer 序列推荐模型（Sun et al., 2019）。

    token 空间：
      0            → PAD
      1 (MASK_TOKEN)→ [MASK]
      2 ~ n_items+1 → 真实 item（item_index + 2）

    vocab_size = n_items + 2
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
        self.emb_dim  = emb_dim
        self.max_seq  = max_seq
        vocab_size    = n_items + 2    # 0=PAD, 1=MASK, 2..n_items+1=items

        self.item_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq, emb_dim)
        self.emb_norm = nn.LayerNorm(emb_dim)
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
        self.out_norm = nn.LayerNorm(emb_dim)

        # 预测头：隐向量 → vocab 分布（共享 item embedding 权重）
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.head.weight = self.item_emb.weight   # weight tying

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """key_padding_mask: True = 忽略该位置（PAD）。"""
        return seq == 0   # (batch, max_seq)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (batch, max_seq) → (batch, max_seq, emb_dim)"""
        B, L = seq.shape
        device = seq.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.emb_norm(self.item_emb(seq) + self.pos_emb(pos_ids))
        x = self.emb_dropout(x)
        pad_mask = self._pad_mask(seq)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.out_norm(x)   # (batch, L, D)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """返回 logits: (batch, max_seq, vocab_size)"""
        return self.head(self.encode(seq))

    def get_user_vector(self, seq: torch.Tensor) -> torch.Tensor:
        """
        推理用：把序列最后一个位置替换为 [MASK]，取该位置的隐向量。
        返回 (batch, emb_dim) 归一化向量。

        序列为左填充（PAD=0 在左，item 在右），最后一个真实 item
        固定在最右端（位置 max_seq-1），直接在此处放 MASK 即可。
        原代码错误地将 MASK 放在 lengths[b]-1 处（仍在 PAD 区域内）。
        """
        masked = seq.clone()
        # 序列左填充：最后一个真实 item 固定在最右端
        masked[:, -1] = MASK_TOKEN
        out = self.encode(masked)        # (batch, L, D)
        user_vec = out[:, -1]            # 直接取最右位置的输出
        return F.normalize(user_vec, dim=-1)

    def get_item_vectors(self) -> torch.Tensor:
        """
        返回所有真实 item 的归一化 embedding：(n_items, emb_dim)。
        索引 0 对应 item_index=0（embedding 行 2）。
        """
        # item token 2 ~ n_items+1 → 映射回 0-indexed
        n_items = self.item_emb.num_embeddings - 2
        idx = torch.arange(2, n_items + 2, device=self.item_emb.weight.device)
        return F.normalize(self.item_emb(idx), dim=-1)   # (n_items, D)


# ══════════════════════════════════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════════════════════════════════

def run_bert4rec_pipeline(
    emb_dim: int = DEFAULT_EMB_DIM,
    n_heads: int = DEFAULT_N_HEADS,
    n_layers: int = DEFAULT_N_LAYERS,
    max_seq: int = DEFAULT_MAX_SEQ,
    dropout: float = DEFAULT_DROPOUT,
    mask_prob: float = DEFAULT_MASK_PROB,
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
    """BERT4Rec 完整流程，返回格式与 run_sasrec_pipeline 兼容。"""
    device = _get_device()
    print(f"[BERT4Rec] device = {device}")

    df = _test_df if _test_df is not None else load_big_matrix_interactions(eligible_video_ids)
    matrix, user_ids, item_ids = build_sparse_matrix(df)
    n_users, n_items = len(user_ids), len(item_ids)

    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    print(f"[BERT4Rec] 构建用户行为序列（max_seq={max_seq}）……")
    sequences = _build_user_sequences(df, item_index, max_seq)
    print(f"[BERT4Rec] 共 {len(sequences):,} 用户有序列。")

    all_uids = list(sequences.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(all_uids)
    n_val = max(1, int(len(all_uids) * val_frac))
    val_uids   = set(all_uids[:n_val])
    train_uids = set(all_uids[n_val:])

    train_ds = BERT4RecDataset({u: sequences[u] for u in train_uids}, n_items, max_seq, mask_prob)
    val_ds   = BERT4RecDataset({u: sequences[u] for u in val_uids},   n_items, max_seq, mask_prob)

    if len(train_ds) == 0:
        raise ValueError("[BERT4Rec] 训练集为空。")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    print(f"[BERT4Rec] 训练集 {len(train_ds):,} 样本，验证集 {len(val_ds):,} 样本。")

    model = BERT4Rec(
        n_items=n_items, emb_dim=emb_dim, n_heads=n_heads,
        n_layers=n_layers, max_seq=max_seq, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # label=0 的位置忽略

    ckpt_path: Path | None = None
    best_ckpt_path: Path | None = None
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state: dict | None = None

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path      = checkpoint_dir / "bert4rec_latest.pt"
        best_ckpt_path = checkpoint_dir / "bert4rec_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch      = ckpt["epoch"] + 1
            best_val_loss    = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            if patience_counter >= patience:
                start_epoch = n_epochs
                print(f"[BERT4Rec] 从 checkpoint 恢复：Early Stopping 已完成，直接推理。")
            else:
                print(f"[BERT4Rec] 从 checkpoint 恢复：epoch {start_epoch}/{n_epochs}")

    if n_epochs <= 0 and not (
        (best_ckpt_path is not None and best_ckpt_path.exists()) or
        (ckpt_path is not None and ckpt_path.exists())
    ):
        raise ValueError("[BERT4Rec] 当前是仅推理模式，但找不到可恢复的 checkpoint。")

    print(
        f"[BERT4Rec] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，"
        f"emb={emb_dim}，heads={n_heads}，layers={n_layers}，mask_prob={mask_prob}"
    )

    avg_loss = 0.0
    avg_val_loss = float("inf")

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        model.train()
        total_loss, n_seen = 0.0, 0
        for seq_b, label_b in train_loader:
            seq_b   = seq_b.to(device)
            label_b = label_b.to(device)
            logits = model(seq_b)   # (B, L, vocab)
            # reshape 为 (B*L, vocab) 和 (B*L,) 给 cross-entropy
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                label_b.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * seq_b.size(0)
            n_seen     += seq_b.size(0)

        avg_loss = total_loss / n_seen

        model.eval()
        val_total, n_val_seen = 0.0, 0
        with torch.no_grad():
            for seq_b, label_b in val_loader:
                seq_b   = seq_b.to(device)
                label_b = label_b.to(device)
                logits  = model(seq_b)
                vl = criterion(logits.reshape(-1, logits.size(-1)), label_b.reshape(-1))
                val_total  += vl.item() * seq_b.size(0)
                n_val_seen += seq_b.size(0)

        avg_val_loss = val_total / n_val_seen
        improved = avg_val_loss < best_val_loss - 1e-6
        star = " ★" if improved else ""

        print(
            f"  epoch {epoch+1:>2}/{n_epochs}  MIP-loss={avg_loss:.6f}"
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
                    f"[BERT4Rec] Early Stopping：连续 {patience} 轮未改善，"
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
        print(f"[BERT4Rec] 已加载最佳权重（val_loss={best_val_loss:.6f}）")
    elif best_ckpt_path is not None and best_ckpt_path.exists():
        best = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        print(f"[BERT4Rec] 已从磁盘加载最佳权重（val_loss={best['val_loss']:.6f}）")

    print(f"[BERT4Rec] 训练完成，最终 loss={avg_loss:.6f}，最佳 val_loss={best_val_loss:.6f}")

    # ── 推理 ──────────────────────────────────────────────────────────
    print(f"\n[BERT4Rec] 为 {n_users:,} 位用户生成个性化 top-{top_k} 推荐……")
    model.eval()
    recommendations: dict = {}
    seen = matrix.tolil()

    # 预计算全量 item 归一化 embedding（item token 2..n_items+1）
    with torch.no_grad():
        all_item_emb = model.get_item_vectors().to(device)   # (n_items, D)

    uid_list = list(sequences.keys())
    with torch.no_grad():
        for batch_start in range(0, len(uid_list), 256):
            batch_uids = uid_list[batch_start:batch_start + 256]
            batch_seqs = []
            for uid in batch_uids:
                s = sequences[uid][-max_seq:]
                pad_seq = np.zeros(max_seq, dtype=np.int64)
                pad_seq[-len(s):] = s
                batch_seqs.append(pad_seq)

            seq_t    = torch.from_numpy(np.stack(batch_seqs)).to(device)
            user_vecs = model.get_user_vector(seq_t)                       # (B, D)
            scores_np = (user_vecs @ all_item_emb.T).cpu().numpy()         # (B, n_items)

            for local_i, uid in enumerate(batch_uids):
                u_global = user_index[uid]
                scores   = scores_np[local_i].copy()
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

    print(f"[BERT4Rec] 推荐生成完成，共 {len(recommendations):,} 位用户。")
    rec_df = recommendations_to_dataframe(recommendations)

    result: dict[str, Any] = {
        "model":    "BERT4Rec",
        "n_users":  n_users,
        "n_items":  n_items,
        "emb_dim":  emb_dim,
        "top_k":    top_k,
        "ce_loss":  avg_loss,
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
        rec_path = output_dir / f"bert4rec_top{top_k}_recommendations.csv"
        rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
        print(f"[BERT4Rec] 推荐列表已保存：{rec_path}")
        result["output_path"] = str(rec_path)

    return result


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BERT4Rec 双向序列推荐模型。")
    parser.add_argument("--emb-dim",    type=int,   default=DEFAULT_EMB_DIM)
    parser.add_argument("--n-heads",    type=int,   default=DEFAULT_N_HEADS)
    parser.add_argument("--n-layers",   type=int,   default=DEFAULT_N_LAYERS)
    parser.add_argument("--max-seq",    type=int,   default=DEFAULT_MAX_SEQ)
    parser.add_argument("--dropout",    type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--mask-prob",  type=float, default=DEFAULT_MASK_PROB)
    parser.add_argument("--n-epochs",   type=int,   default=DEFAULT_N_EPOCHS)
    parser.add_argument("--patience",   type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--top-k",      type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--output-dir", type=str,   default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )

    run_bert4rec_pipeline(
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq=args.max_seq,
        dropout=args.dropout,
        mask_prob=args.mask_prob,
        n_epochs=args.n_epochs,
        patience=args.patience,
        lr=args.lr,
        top_k=args.top_k,
        output_dir=out,
    )
