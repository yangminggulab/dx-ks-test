"""
文件用途：SASRec + Side Information（序列推荐 + 视频侧特征融合）。

【核心思想 vs SASRec】
  SASRec  item embedding 只依赖 video_id，冷启动视频没有历史交互 → 无法推荐。
  SideInfo-SASRec 把视频 embedding 替换为"ID embedding + 内容特征向量"的融合，
  新视频有内容特征就能参与序列建模，解决冷启动。

【Side Information 融合方式】
  item_repr = LayerNorm( W_id * id_emb  +  W_feat * feature_emb )
                                           ↑
                       Linear(cat_multihot(31) + log_duration(1)) → emb_dim

  两路求和后进 Transformer，让模型自己学"ID 信号"和"内容信号"各贡献多少。
  比拼接（concat）参数量少，与 FDSA / S³-Rec 等论文一致。

【KuaiRec 适配】
  视频特征来源：
    item_categories.csv  → 类别 multi-hot(31 维)
    item_daily_features.csv → video_duration → log1p 标准化(1 维)
  用户序列构建逻辑与 SASRec 相同（big_matrix，无时间戳）。
  训练目标：WBPR（与 SASRec 保持一致，公平对比）。
  返回格式与 run_sasrec_pipeline 完全兼容。
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

# ── 常量 ──────────────────────────────────────────────────────────────
DEFAULT_EMB_DIM   = 64
DEFAULT_N_HEADS   = 2
DEFAULT_N_LAYERS  = 2
DEFAULT_MAX_SEQ   = 50
DEFAULT_DROPOUT   = 0.2
DEFAULT_N_EPOCHS  = 50
DEFAULT_LR        = 1e-3
DEFAULT_BATCH     = 2048
DEFAULT_TOP_K     = 50
DEFAULT_PATIENCE  = 5

N_CATEGORIES  = 31    # KuaiRec item_categories feat 范围 0~30
FEAT_DIM      = N_CATEGORIES + 1   # 31 类别 multi-hot + 1 时长


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_data_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "KuaiRec 2.0" / "data",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


# ══════════════════════════════════════════════════════════════════════
# 视频侧特征加载
# ══════════════════════════════════════════════════════════════════════

def _build_item_features(item_ids: np.ndarray) -> np.ndarray:
    """
    加载视频特征，按 item_ids 顺序对齐。

    Returns:
        feat_arr: (n_items, FEAT_DIM) float32
            列 0-30: 类别 multi-hot
            列   31: log1p(video_duration)
    """
    data_dir = _find_data_dir()
    n_items  = len(item_ids)
    feat_arr = np.zeros((n_items, FEAT_DIM), dtype=np.float32)
    id2idx   = {int(vid): i for i, vid in enumerate(item_ids)}

    ic_path = data_dir / "item_categories.csv"
    if ic_path.exists():
        ic = pd.read_csv(ic_path)
        for _, row in ic.iterrows():
            vid = int(row["video_id"])
            if vid not in id2idx:
                continue
            try:
                cats = ast.literal_eval(str(row["feat"]))
                for c in cats:
                    if 0 <= c < N_CATEGORIES:
                        feat_arr[id2idx[vid], c] = 1.0
            except Exception:
                pass
    else:
        print("[SideInfo] item_categories.csv 不存在，类别特征全零。")

    idf_path = data_dir / "item_daily_features.csv"
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
                    feat_arr[idx, N_CATEGORIES] = float(np.log1p(dur))
    else:
        print("[SideInfo] item_daily_features.csv 不存在，时长特征全零。")

    print(f"[SideInfo] 视频特征加载完成：{n_items:,} 视频，特征维度 {FEAT_DIM}。")
    return feat_arr


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
# Dataset：序列 + WBPR 负采样
# ══════════════════════════════════════════════════════════════════════

class SideInfoDataset(Dataset):
    """与 SASRecDataset 相同的 next-item prediction 范式。"""

    def __init__(
        self,
        sequences: dict[Any, list[tuple[int, float]]],
        n_items: int,
        max_seq: int,
    ):
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

    def __getitem__(self, idx: int):
        seq, pos, w = self.samples[idx]
        neg = random.randint(1, self.n_items)
        return torch.from_numpy(seq), pos, neg, np.float32(w)


# ══════════════════════════════════════════════════════════════════════
# SideInfo-SASRec 模型
# ══════════════════════════════════════════════════════════════════════

class SideInfoSASRec(nn.Module):
    """
    SASRec 的 item embedding 替换为 ID + 内容特征的融合。

    token 空间：0=PAD，1..n_items = item（item_index + 1）
    feat_matrix: (n_items+1, FEAT_DIM)，第 0 行全零对应 PAD。
    """

    def __init__(
        self,
        n_items: int,
        feat_matrix: torch.Tensor,   # (n_items, FEAT_DIM)
        emb_dim: int = DEFAULT_EMB_DIM,
        n_heads: int = DEFAULT_N_HEADS,
        n_layers: int = DEFAULT_N_LAYERS,
        max_seq: int = DEFAULT_MAX_SEQ,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_seq = max_seq

        # ID embedding：0=PAD
        self.id_emb  = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq, emb_dim)

        # 内容特征投影：FEAT_DIM → emb_dim
        self.feat_proj = nn.Linear(feat_matrix.size(1), emb_dim, bias=False)

        # 融合归一化（两路求和后 LayerNorm）
        self.fuse_norm   = nn.LayerNorm(emb_dim)
        self.emb_dropout = nn.Dropout(dropout)

        # 注册 feat_matrix 为 buffer（不参与梯度，随模型保存）
        # 在第 0 行前加一行全零 → 对应 PAD token
        pad_row = torch.zeros(1, feat_matrix.size(1))
        full_feat = torch.cat([pad_row, feat_matrix], dim=0)   # (n_items+1, FEAT_DIM)
        self.register_buffer("feat_matrix", full_feat)

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

        nn.init.normal_(self.id_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _item_repr(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: 任意形状的 LongTensor（包含 0=PAD）
        返回：与 token_ids 同形状的 (..., emb_dim) float 向量。
        ID embedding + 内容特征 embedding，求和后 LayerNorm。
        """
        id_vec   = self.id_emb(token_ids)                                  # (..., D)
        raw_feat = self.feat_matrix[token_ids]                             # (..., FEAT_DIM)
        feat_vec = self.feat_proj(raw_feat)                                # (..., D)
        return self.fuse_norm(id_vec + feat_vec)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (batch, max_seq) → (batch, max_seq, emb_dim)"""
        B, L = seq.shape
        device = seq.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        x = self._item_repr(seq) + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)
        causal = self._causal_mask(L, device)
        x = self.transformer(x, mask=causal)
        return self.out_norm(x)   # (batch, L, D)

    def forward(
        self,
        seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_out  = self.encode(seq)
        last_out = seq_out[:, -1, :]                         # (batch, D)

        pos_emb = self._item_repr(pos_items)                 # (batch, D)
        neg_emb = self._item_repr(neg_items)                 # (batch, D)

        pos_score = (last_out * pos_emb).sum(-1)
        neg_score = (last_out * neg_emb).sum(-1)
        return pos_score, neg_score

    def get_user_vector(self, seq: torch.Tensor) -> torch.Tensor:
        seq_out  = self.encode(seq)
        last_out = seq_out[:, -1, :]
        return F.normalize(last_out, dim=-1)

    def get_all_item_vectors(self) -> torch.Tensor:
        """返回 (n_items, emb_dim) 归一化向量，索引 0 对应 item_index=0。"""
        n_items = self.id_emb.num_embeddings - 1   # 排除 PAD(0)
        device  = self.id_emb.weight.device
        idx     = torch.arange(1, n_items + 1, device=device)
        vecs    = self._item_repr(idx)               # (n_items, D)
        return F.normalize(vecs, dim=-1)


# ══════════════════════════════════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════════════════════════════════

def run_sideinfo_pipeline(
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
    """SideInfo-SASRec 完整流程，返回格式与 run_sasrec_pipeline 兼容。"""
    device = _get_device()
    print(f"[SideInfo] device = {device}")

    df = _test_df if _test_df is not None else load_big_matrix_interactions(eligible_video_ids)
    matrix, user_ids, item_ids = build_sparse_matrix(df)
    n_users, n_items = len(user_ids), len(item_ids)

    user_index = {uid: i for i, uid in enumerate(user_ids)}
    item_index = {iid: i for i, iid in enumerate(item_ids)}

    # 加载视频侧特征
    feat_np = _build_item_features(item_ids)
    feat_t  = torch.from_numpy(feat_np)   # CPU，后面随模型 .to(device)

    print(f"[SideInfo] 构建用户行为序列（max_seq={max_seq}）……")
    sequences = _build_user_sequences(df, item_index, max_seq)
    print(f"[SideInfo] 共 {len(sequences):,} 用户有序列。")

    all_uids = list(sequences.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(all_uids)
    n_val = max(1, int(len(all_uids) * val_frac))
    val_uids   = set(all_uids[:n_val])
    train_uids = set(all_uids[n_val:])

    train_ds = SideInfoDataset({u: sequences[u] for u in train_uids}, n_items, max_seq)
    val_ds   = SideInfoDataset({u: sequences[u] for u in val_uids},   n_items, max_seq)

    if len(train_ds) == 0:
        raise ValueError("[SideInfo] 训练集为空。")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    print(f"[SideInfo] 训练集 {len(train_ds):,} 样本，验证集 {len(val_ds):,} 样本。")

    model = SideInfoSASRec(
        n_items=n_items, feat_matrix=feat_t,
        emb_dim=emb_dim, n_heads=n_heads, n_layers=n_layers,
        max_seq=max_seq, dropout=dropout,
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
        ckpt_path      = checkpoint_dir / "sideinfo_latest.pt"
        best_ckpt_path = checkpoint_dir / "sideinfo_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch      = ckpt["epoch"] + 1
            best_val_loss    = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            if patience_counter >= patience:
                start_epoch = n_epochs
                print(f"[SideInfo] 从 checkpoint 恢复：Early Stopping 已完成，直接推理。")
            else:
                print(f"[SideInfo] 从 checkpoint 恢复：epoch {start_epoch}/{n_epochs}")

    if n_epochs <= 0 and not (
        (best_ckpt_path is not None and best_ckpt_path.exists()) or
        (ckpt_path is not None and ckpt_path.exists())
    ):
        raise ValueError("[SideInfo] 当前是仅推理模式，但找不到可恢复的 checkpoint。")

    print(
        f"[SideInfo] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，"
        f"emb={emb_dim}，heads={n_heads}，layers={n_layers}，feat_dim={FEAT_DIM}"
    )

    avg_loss = 0.0
    avg_val_loss = float("inf")

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        model.train()
        total_loss, n_seen = 0.0, 0
        for seq_b, pos_b, neg_b, w_b in train_loader:
            seq_b = seq_b.to(device)
            pos_b = pos_b.to(device)
            neg_b = neg_b.to(device)
            w_b   = w_b.to(device)
            pos_score, neg_score = model(seq_b, pos_b, neg_b)
            loss = -(w_b.clamp(min=1e-8) * F.logsigmoid(pos_score - neg_score)).mean()
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
            for seq_b, pos_b, neg_b, w_b in val_loader:
                seq_b = seq_b.to(device)
                pos_b = pos_b.to(device)
                neg_b = neg_b.to(device)
                w_b   = w_b.to(device)
                ps, ns = model(seq_b, pos_b, neg_b)
                vl = -(w_b.clamp(min=1e-8) * F.logsigmoid(ps - ns)).mean()
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
                    "val_loss": avg_val_loss,
                }, best_ckpt_path)
                print(f"  └─ 最佳模型已保存（val_loss={avg_val_loss:.6f}）")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[SideInfo] Early Stopping：连续 {patience} 轮未改善，"
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
        print(f"[SideInfo] 已加载最佳权重（val_loss={best_val_loss:.6f}）")
    elif best_ckpt_path is not None and best_ckpt_path.exists():
        best = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        print(f"[SideInfo] 已从磁盘加载最佳权重（val_loss={best['val_loss']:.6f}）")

    print(f"[SideInfo] 训练完成，最终 loss={avg_loss:.6f}，最佳 val_loss={best_val_loss:.6f}")

    # ── 推理 ──────────────────────────────────────────────────────────
    print(f"\n[SideInfo] 为 {n_users:,} 位用户生成个性化 top-{top_k} 推荐……")
    model.eval()
    recommendations: dict = {}
    seen = matrix.tolil()

    with torch.no_grad():
        all_item_emb = model.get_all_item_vectors()   # (n_items, D)

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

    print(f"[SideInfo] 推荐生成完成，共 {len(recommendations):,} 位用户。")
    rec_df = recommendations_to_dataframe(recommendations)

    result: dict[str, Any] = {
        "model":    "SideInfo-SASRec",
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
        rec_path = output_dir / f"sideinfo_top{top_k}_recommendations.csv"
        rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
        print(f"[SideInfo] 推荐列表已保存：{rec_path}")
        result["output_path"] = str(rec_path)

    return result


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SideInfo-SASRec：序列推荐 + 视频内容特征。")
    parser.add_argument("--emb-dim",    type=int,   default=DEFAULT_EMB_DIM)
    parser.add_argument("--n-heads",    type=int,   default=DEFAULT_N_HEADS)
    parser.add_argument("--n-layers",   type=int,   default=DEFAULT_N_LAYERS)
    parser.add_argument("--max-seq",    type=int,   default=DEFAULT_MAX_SEQ)
    parser.add_argument("--dropout",    type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--n-epochs",   type=int,   default=DEFAULT_N_EPOCHS)
    parser.add_argument("--patience",   type=int,   default=DEFAULT_PATIENCE)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--top-k",      type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--output-dir", type=str,   default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )

    run_sideinfo_pipeline(
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
