"""
models/bert4rec.py — BERT4Rec 推荐模型类。

双向 Transformer + Masked Item Prediction。
checkpoint 文件名与原版一致（bert4rec_latest.pt / bert4rec_best.pt）。
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from models.base import (
    BaseRecommender, ModelData, get_device,
    save_checkpoint, load_checkpoint,
    build_user_sequences_ids_only,
)

DEFAULT_EMB_DIM   = 64
DEFAULT_N_HEADS   = 2
DEFAULT_N_LAYERS  = 2
DEFAULT_MAX_SEQ   = 50
DEFAULT_DROPOUT   = 0.2
DEFAULT_MASK_PROB = 0.2
DEFAULT_LR        = 1e-3
DEFAULT_BATCH     = 256

MASK_TOKEN = 1  # 0=PAD, 1=[MASK], item_idx+2 = 真实 item


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════

class _BERT4RecDataset(Dataset):
    def __init__(self, sequences: dict, n_items: int, max_seq: int, mask_prob: float):
        self.samples = []
        self.n_items  = n_items
        self.max_seq  = max_seq
        self.mask_prob = mask_prob
        for uid, seq in sequences.items():
            if len(seq) < 2:
                continue
            pad_seq = np.zeros(max_seq, dtype=np.int64)
            s = seq[-max_seq:]
            pad_seq[-len(s):] = s
            self.samples.append(pad_seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx].copy()
        labels = np.zeros_like(seq)
        for i in range(self.max_seq):
            if seq[i] == 0:
                continue
            if random.random() < self.mask_prob:
                labels[i] = seq[i]
                seq[i] = MASK_TOKEN
        if labels.sum() == 0:
            for i in range(self.max_seq - 1, -1, -1):
                if seq[i] != 0:
                    labels[i] = seq[i]
                    seq[i] = MASK_TOKEN
                    break
        return torch.from_numpy(seq), torch.from_numpy(labels)


# ══════════════════════════════════════════════════════════════════════
# 神经网络（完整保留，一行不改）
# ══════════════════════════════════════════════════════════════════════

class _BERT4RecNet(nn.Module):
    def __init__(self, n_items, emb_dim=DEFAULT_EMB_DIM, n_heads=DEFAULT_N_HEADS,
                 n_layers=DEFAULT_N_LAYERS, max_seq=DEFAULT_MAX_SEQ, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq = max_seq
        vocab_size   = n_items + 2
        self.item_emb    = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb     = nn.Embedding(max_seq, emb_dim)
        self.emb_norm    = nn.LayerNorm(emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)
        self.head.weight = self.item_emb.weight
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _pad_mask(self, seq):
        return seq == 0

    def encode(self, seq):
        B, L = seq.shape
        device = seq.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.emb_norm(self.item_emb(seq) + self.pos_emb(pos_ids))
        x = self.emb_dropout(x)
        x = self.transformer(x, src_key_padding_mask=self._pad_mask(seq))
        return self.out_norm(x)

    def forward(self, seq):
        return self.head(self.encode(seq))

    def get_user_vector(self, seq):
        masked = seq.clone()
        masked[:, -1] = MASK_TOKEN
        out = self.encode(masked)
        return F.normalize(out[:, -1], dim=-1)

    def get_item_vectors(self):
        n_items = self.item_emb.num_embeddings - 2
        idx = torch.arange(2, n_items + 2, device=self.item_emb.weight.device)
        return F.normalize(self.item_emb(idx), dim=-1)


# ══════════════════════════════════════════════════════════════════════
# BaseRecommender 子类
# ══════════════════════════════════════════════════════════════════════

class BERT4Rec(BaseRecommender):
    """
    BERT4Rec（Bidirectional Transformer for Sequential Recommendation）。

    __init__(data, output_dir, checkpoint_dir=None,
             emb_dim=64, n_heads=2, n_layers=2, max_seq=50,
             dropout=0.2, mask_prob=0.2, lr=1e-3, batch_size=256)
    train(n_epochs=50, patience=5, val_frac=0.1)
    recommend(top_k=50) -> dict[uid, list[vid]]
    """

    def __init__(
        self,
        data: ModelData,
        output_dir: Path,
        checkpoint_dir: Path | None = None,
        emb_dim: int = DEFAULT_EMB_DIM,
        n_heads: int = DEFAULT_N_HEADS,
        n_layers: int = DEFAULT_N_LAYERS,
        max_seq: int = DEFAULT_MAX_SEQ,
        dropout: float = DEFAULT_DROPOUT,
        mask_prob: float = DEFAULT_MASK_PROB,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH,
    ):
        super().__init__(data, output_dir, checkpoint_dir)
        self.emb_dim    = emb_dim
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.max_seq    = max_seq
        self.dropout    = dropout
        self.mask_prob  = mask_prob
        self.lr         = lr
        self.batch_size = batch_size
        self.device = get_device()
        self._net: _BERT4RecNet | None = None

    def train(self, n_epochs: int = 50, patience: int = 5, val_frac: float = 0.1) -> None:
        device = self.device
        print(f"[BERT4Rec] device = {device}")

        item_ids = self.data.item_ids
        user_ids = self.data.user_ids
        n_items  = len(item_ids)
        item_index = {iid: i for i, iid in enumerate(item_ids)}

        # BERT4Rec 使用 offset=2 的 token 序列（0=PAD, 1=MASK）
        sequences = build_user_sequences_ids_only(
            self.data.interaction_df, item_index, self.max_seq, offset=2
        )

        all_uids = list(sequences.keys())
        rng = np.random.default_rng(42)
        rng.shuffle(all_uids)
        n_val = max(1, int(len(all_uids) * val_frac))
        val_uids   = set(all_uids[:n_val])
        train_uids = set(all_uids[n_val:])

        train_ds = _BERT4RecDataset({u: sequences[u] for u in train_uids}, n_items, self.max_seq, self.mask_prob)
        val_ds   = _BERT4RecDataset({u: sequences[u] for u in val_uids},   n_items, self.max_seq, self.mask_prob)
        if len(train_ds) == 0:
            raise ValueError("[BERT4Rec] 训练集为空。")

        pin = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
        print(f"[BERT4Rec] 训练集 {len(train_ds):,} 样本，验证集 {len(val_ds):,} 样本。")

        model     = _BERT4RecNet(n_items, self.emb_dim, self.n_heads, self.n_layers,
                                 self.max_seq, self.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        ckpt_path      = self.checkpoint_dir / "bert4rec_latest.pt"
        best_ckpt_path = self.checkpoint_dir / "bert4rec_best.pt"
        start_epoch = 0; best_val_loss = float("inf"); patience_counter = 0; best_model_state = None

        ckpt = load_checkpoint(ckpt_path, device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch      = ckpt["epoch"] + 1
            best_val_loss    = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            if patience_counter >= patience:
                start_epoch = n_epochs
                print("[BERT4Rec] Early Stopping 已完成，直接推理。")
            else:
                print(f"[BERT4Rec] 从 checkpoint 恢复：epoch {start_epoch}/{n_epochs}")

        print(f"[BERT4Rec] 开始训练：{len(user_ids):,} 用户 × {n_items:,} 视频")

        avg_loss = 0.0
        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            model.train()
            total_loss, n_seen = 0.0, 0
            for seq_b, label_b in train_loader:
                seq_b = seq_b.to(device); label_b = label_b.to(device)
                logits = model(seq_b)
                loss = criterion(logits.reshape(-1, logits.size(-1)), label_b.reshape(-1))
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * seq_b.size(0); n_seen += seq_b.size(0)
            avg_loss = total_loss / n_seen

            model.eval()
            val_total, n_val = 0.0, 0
            with torch.no_grad():
                for seq_b, label_b in val_loader:
                    seq_b = seq_b.to(device); label_b = label_b.to(device)
                    logits = model(seq_b)
                    vl = criterion(logits.reshape(-1, logits.size(-1)), label_b.reshape(-1))
                    val_total += vl.item() * seq_b.size(0); n_val += seq_b.size(0)
            avg_val = val_total / n_val
            improved = avg_val < best_val_loss - 1e-6
            star = " ★" if improved else ""
            print(f"  epoch {epoch+1:>2}/{n_epochs}  MIP-loss={avg_loss:.6f}  val={avg_val:.6f}{star}  ({time.time()-t0:.1f}s)")

            if improved:
                best_val_loss = avg_val; patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                save_checkpoint(best_ckpt_path, {"epoch": epoch, "model": best_model_state,
                    "optimizer": optimizer.state_dict(), "val_loss": avg_val})
                print(f"  └─ 最佳模型已保存（val_loss={avg_val:.6f}）")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[BERT4Rec] Early Stopping：停止在 epoch {epoch+1}")
                    save_checkpoint(ckpt_path, {"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "best_val_loss": best_val_loss,
                        "patience_counter": patience_counter})
                    break

            save_checkpoint(ckpt_path, {"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "best_val_loss": best_val_loss,
                "patience_counter": patience_counter})

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        elif best_ckpt_path.exists():
            best = load_checkpoint(best_ckpt_path, device)
            model.load_state_dict(best["model"])

        self._net = model
        # 保存 sequences 供 recommend 使用（BERT4Rec 用的是不同 offset 的序列）
        self._sequences = sequences
        print(f"[BERT4Rec] 训练完成，最终 loss={avg_loss:.6f}")

    def recommend(self, top_k: int = 50) -> dict:
        device   = self.device
        item_ids = self.data.item_ids
        user_ids = self.data.user_ids
        matrix   = self.data.matrix
        n_items  = len(item_ids)
        max_seq  = self.max_seq

        if self._net is None:
            model = _BERT4RecNet(n_items, self.emb_dim, self.n_heads, self.n_layers,
                                 self.max_seq, self.dropout).to(device)
            best_ckpt = self.checkpoint_dir / "bert4rec_best.pt"
            ckpt_path = self.checkpoint_dir / "bert4rec_latest.pt"
            ckpt = load_checkpoint(best_ckpt, device) or load_checkpoint(ckpt_path, device)
            if ckpt is None:
                raise RuntimeError("[BERT4Rec] 没有 checkpoint，请先调用 train()。")
            model.load_state_dict(ckpt["model"])
            self._net = model

        # rebuild sequences if needed
        if not hasattr(self, "_sequences"):
            item_index = {iid: i for i, iid in enumerate(item_ids)}
            self._sequences = build_user_sequences_ids_only(
                self.data.interaction_df, item_index, max_seq, offset=2
            )

        model     = self._net
        sequences = self._sequences
        model.eval()

        print(f"\n[BERT4Rec] 为 {len(user_ids):,} 位用户生成个性化 top-{top_k} 推荐……")
        recommendations: dict = {}
        seen = matrix.tolil()
        user_index = {uid: i for i, uid in enumerate(user_ids)}

        with torch.no_grad():
            all_item_emb = model.get_item_vectors().to(device)

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
                seq_t     = torch.from_numpy(np.stack(batch_seqs)).to(device)
                user_vecs = model.get_user_vector(seq_t)
                scores_np = (user_vecs @ all_item_emb.T).cpu().numpy()
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

        print(f"[BERT4Rec] 推荐生成完成，共 {len(recommendations):,} 位用户。")
        return recommendations
