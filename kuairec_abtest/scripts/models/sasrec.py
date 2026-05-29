"""
models/sasrec.py — SASRec 推荐模型类（包装原 sasrec.py 的逻辑）。

内部 Transformer 神经网络代码完整保留，只是重新组织为 BaseRecommender 子类。
checkpoint 文件名与原版一致（sasrec_latest.pt / sasrec_best.pt）。
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from models.base import BaseRecommender, ModelData, get_device, save_checkpoint, load_checkpoint

# ── 默认超参 ──────────────────────────────────────────────────────────
DEFAULT_EMB_DIM  = 64
DEFAULT_N_HEADS  = 2
DEFAULT_N_LAYERS = 2
DEFAULT_MAX_SEQ  = 50
DEFAULT_DROPOUT  = 0.2
DEFAULT_LR       = 1e-3
DEFAULT_BATCH    = 2048


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════

class _SASRecDataset(Dataset):
    def __init__(self, sequences: dict, n_items: int, max_seq: int):
        self.samples: list[tuple[np.ndarray, int, float]] = []
        for uid, seq in sequences.items():
            if len(seq) < 2:
                continue
            input_seq = seq[:-1]
            pos_item, w = seq[-1]
            pad_seq = np.zeros(max_seq, dtype=np.int64)
            items_only = [it for it, _ in input_seq][-max_seq:]
            pad_seq[-len(items_only):] = np.array(items_only, dtype=np.int64) + 1
            self.samples.append((pad_seq, pos_item + 1, w))
        self.n_items = n_items
        self.max_seq = max_seq

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        seq, pos, w = self.samples[idx]
        neg = random.randint(1, self.n_items)
        return torch.from_numpy(seq), pos, neg, np.float32(w)


# ══════════════════════════════════════════════════════════════════════
# 神经网络：因果自注意力 Transformer（完整保留，一行不改）
# ══════════════════════════════════════════════════════════════════════

class _SASRecNet(nn.Module):
    def __init__(self, n_items, emb_dim=DEFAULT_EMB_DIM, n_heads=DEFAULT_N_HEADS,
                 n_layers=DEFAULT_N_LAYERS, max_seq=DEFAULT_MAX_SEQ, dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq = max_seq
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(emb_dim)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def encode(self, seq):
        B, L = seq.shape
        device = seq.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(seq) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)
        x = self.transformer(x, mask=self._causal_mask(L, device))
        return self.out_norm(x)

    def forward(self, seq, pos_items, neg_items):
        seq_out  = self.encode(seq)
        last_out = seq_out[:, -1, :]
        pos_emb  = self.item_emb(pos_items)
        neg_emb  = self.item_emb(neg_items)
        return (last_out * pos_emb).sum(-1), (last_out * neg_emb).sum(-1)

    def get_user_vector(self, seq):
        return F.normalize(self.encode(seq)[:, -1, :], dim=-1)

    def get_item_vectors(self, item_idx):
        return F.normalize(self.item_emb(item_idx), dim=-1)


# ══════════════════════════════════════════════════════════════════════
# BaseRecommender 子类
# ══════════════════════════════════════════════════════════════════════

class SASRec(BaseRecommender):
    """
    SASRec（Self-Attentive Sequential Recommendation）。

    __init__(data, output_dir, checkpoint_dir=None,
             emb_dim=64, n_heads=2, n_layers=2, max_seq=50,
             dropout=0.2, lr=1e-3, batch_size=2048)
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
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH,
    ):
        super().__init__(data, output_dir, checkpoint_dir)
        self.emb_dim    = emb_dim
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.max_seq    = max_seq
        self.dropout    = dropout
        self.lr         = lr
        self.batch_size = batch_size

        self.device = get_device()
        self._net: _SASRecNet | None = None
        self._avg_loss = 0.0

    def train(self, n_epochs: int = 50, patience: int = 5, val_frac: float = 0.1) -> None:
        device = self.device
        print(f"[SASRec] device = {device}")

        df       = self.data.interaction_df
        user_ids = self.data.user_ids
        item_ids = self.data.item_ids
        n_users, n_items = len(user_ids), len(item_ids)

        item_index = {iid: i for i, iid in enumerate(item_ids)}
        sequences  = self.data.sequences

        all_uids = list(sequences.keys())
        rng = np.random.default_rng(42)
        rng.shuffle(all_uids)
        n_val = max(1, int(len(all_uids) * val_frac))
        val_uids   = set(all_uids[:n_val])
        train_uids = set(all_uids[n_val:])

        train_ds = _SASRecDataset({u: sequences[u] for u in train_uids}, n_items, self.max_seq)
        val_ds   = _SASRecDataset({u: sequences[u] for u in val_uids},   n_items, self.max_seq)
        if len(train_ds) == 0:
            raise ValueError("[SASRec] 训练集为空。")

        pin = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
        print(f"[SASRec] 训练集 {len(train_ds):,} 样本，验证集 {len(val_ds):,} 样本。")

        model = _SASRecNet(n_items, self.emb_dim, self.n_heads, self.n_layers,
                           self.max_seq, self.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        ckpt_path      = self.checkpoint_dir / "sasrec_latest.pt"
        best_ckpt_path = self.checkpoint_dir / "sasrec_best.pt"
        start_epoch = 0
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        ckpt = load_checkpoint(ckpt_path, device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch      = ckpt["epoch"] + 1
            best_val_loss    = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            if patience_counter >= patience:
                print(f"[SASRec] Early Stopping 已完成，直接推理。")
                start_epoch = n_epochs
            else:
                print(f"[SASRec] 从 checkpoint 恢复：epoch {start_epoch}/{n_epochs}")

        print(f"[SASRec] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，max_epochs={n_epochs}")

        avg_loss = 0.0
        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            model.train()
            total_loss, n_seen = 0.0, 0
            for seq_b, pos_b, neg_b, w_b in train_loader:
                seq_b = seq_b.to(device); pos_b = pos_b.to(device)
                neg_b = neg_b.to(device); w_b   = w_b.to(device)
                pos_s, neg_s = model(seq_b, pos_b, neg_b)
                loss = -(w_b.clamp(min=1e-8) * F.logsigmoid(pos_s - neg_s)).mean()
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(seq_b); n_seen += len(seq_b)
            avg_loss = total_loss / n_seen

            model.eval()
            val_total, n_val = 0.0, 0
            with torch.no_grad():
                for seq_b, pos_b, neg_b, w_b in val_loader:
                    seq_b = seq_b.to(device); pos_b = pos_b.to(device)
                    neg_b = neg_b.to(device); w_b   = w_b.to(device)
                    ps, ns = model(seq_b, pos_b, neg_b)
                    vl = -(w_b.clamp(min=1e-8) * F.logsigmoid(ps - ns)).mean()
                    val_total += vl.item() * len(seq_b); n_val += len(seq_b)
            avg_val = val_total / n_val
            improved = avg_val < best_val_loss - 1e-6
            star = " ★" if improved else ""
            print(f"  epoch {epoch+1:>2}/{n_epochs}  WBPR-loss={avg_loss:.6f}  val={avg_val:.6f}{star}  ({time.time()-t0:.1f}s)")

            if improved:
                best_val_loss = avg_val; patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                save_checkpoint(best_ckpt_path, {"epoch": epoch, "model": best_model_state,
                    "optimizer": optimizer.state_dict(), "val_loss": avg_val})
                print(f"  └─ 最佳模型已保存（val_loss={avg_val:.6f}）")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[SASRec] Early Stopping：停止在 epoch {epoch+1}")
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

        self._net      = model
        self._avg_loss = avg_loss
        print(f"[SASRec] 训练完成，最终 loss={avg_loss:.6f}")

    def recommend(self, top_k: int = 50) -> dict:
        if self._net is None:
            # 尝试从 checkpoint 加载
            device = self.device
            item_ids = self.data.item_ids
            n_items  = len(item_ids)
            model = _SASRecNet(n_items, self.emb_dim, self.n_heads, self.n_layers,
                               self.max_seq, self.dropout).to(device)
            best_ckpt_path = self.checkpoint_dir / "sasrec_best.pt"
            ckpt_path      = self.checkpoint_dir / "sasrec_latest.pt"
            ckpt = load_checkpoint(best_ckpt_path, device) or load_checkpoint(ckpt_path, device)
            if ckpt is None:
                raise RuntimeError("[SASRec] 没有 checkpoint，请先调用 train()。")
            model.load_state_dict(ckpt["model"])
            self._net = model

        device    = self.device
        model     = self._net
        item_ids  = self.data.item_ids
        user_ids  = self.data.user_ids
        sequences = self.data.sequences
        matrix    = self.data.matrix
        n_items   = len(item_ids)
        max_seq   = self.max_seq

        print(f"\n[SASRec] 为 {len(user_ids):,} 位用户生成个性化 top-{top_k} 推荐……")
        model.eval()
        recommendations: dict = {}
        seen = matrix.tolil()

        all_item_idx = torch.arange(1, n_items + 1, device=device)
        emb_chunks = []
        with torch.no_grad():
            for s in range(0, n_items, 2048):
                emb_chunks.append(model.get_item_vectors(all_item_idx[s:s + 2048]))
        all_item_emb = torch.cat(emb_chunks, dim=0)

        user_index = {uid: i for i, uid in enumerate(user_ids)}
        uid_list = list(sequences.keys())
        with torch.no_grad():
            for batch_start in range(0, len(uid_list), 256):
                batch_uids = uid_list[batch_start:batch_start + 256]
                batch_seqs = []
                for uid in batch_uids:
                    seq = sequences[uid]
                    pad_seq = np.zeros(max_seq, dtype=np.int64)
                    items_only = [it for it, _ in seq][-max_seq:]
                    pad_seq[-len(items_only):] = np.array(items_only, dtype=np.int64) + 1
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

        print(f"[SASRec] 推荐生成完成，共 {len(recommendations):,} 位用户。")
        return recommendations
