"""
models/two_tower.py — TwoTower 推荐模型类。

BPR 和 WBPR 两个版本均保留，通过 weighted 参数控制。
checkpoint 文件名与原版一致（two_tower_wbpr_best.pt 等）。
"""
from __future__ import annotations

import ast
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
)

DEFAULT_EMB_DIM  = 32
DEFAULT_HIDDEN   = 128
DEFAULT_OUT_DIM  = 64
DEFAULT_N_EPOCHS = 20
DEFAULT_LR       = 1e-3
DEFAULT_BATCH    = 4096

N_CATEGORIES     = 31
N_ACTIVE_DEGREES = 4
ACTIVE_DEGREE_MAP = {
    "full_active": 0, "high_active": 1, "middle_active": 2, "UNKNOWN": 3,
}


# ══════════════════════════════════════════════════════════════════════
# 特征工程辅助（直接读取原始 CSV）
# ══════════════════════════════════════════════════════════════════════

def _find_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "KuaiRec 2.0" / "data"


def _build_user_features(user_ids: np.ndarray):
    """返回 (active_arr, num_arr)。"""
    n_users = len(user_ids)
    active_arr = np.full(n_users, ACTIVE_DEGREE_MAP["UNKNOWN"], dtype=np.int64)
    num_arr    = np.zeros((n_users, 3), dtype=np.float32)
    uf_path = _find_data_dir() / "user_features.csv"
    if not uf_path.exists():
        print("[TwoTower] user_features.csv 不存在，使用全零用户特征。")
        return active_arr, num_arr
    uf = __import__("pandas").read_csv(uf_path).set_index("user_id")
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


def _build_item_features_tt(item_ids: np.ndarray):
    """返回 (cat_arr, dur_arr)（TwoTower 专用，格式与 video_features DataFrame 不同）。"""
    import pandas as pd
    n_items = len(item_ids)
    cat_arr = np.zeros((n_items, N_CATEGORIES), dtype=np.float32)
    dur_arr = np.zeros((n_items, 1), dtype=np.float32)
    item_idx_map = {int(iid): i for i, iid in enumerate(item_ids)}
    data_dir = _find_data_dir()

    ic_path = data_dir / "item_categories.csv"
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
        print("[TwoTower] item_categories.csv 不存在。")

    idf_path = data_dir / "item_daily_features.csv"
    if idf_path.exists():
        idf = (
            pd.read_csv(idf_path, usecols=["video_id", "video_duration"])
            .drop_duplicates("video_id").set_index("video_id")
        )
        import pandas as _pd
        for idx, vid in enumerate(item_ids):
            vid_int = int(vid)
            if vid_int in idf.index:
                dur = idf.loc[vid_int, "video_duration"]
                if _pd.notna(dur) and float(dur) > 0:
                    dur_arr[idx, 0] = float(np.log1p(float(dur)))
    else:
        print("[TwoTower] item_daily_features.csv 不存在。")

    print(f"[TwoTower] 视频特征加载完成：{n_items:,} 视频。")
    return cat_arr, dur_arr


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════

class _TwoTowerDataset(Dataset):
    def __init__(self, u_arr, i_arr, r_arr, n_items):
        self.u_arr = u_arr; self.i_arr = i_arr; self.r_arr = r_arr; self.n_items = n_items

    def __len__(self): return len(self.u_arr)

    def __getitem__(self, idx):
        return int(self.u_arr[idx]), int(self.i_arr[idx]), random.randint(0, self.n_items - 1), self.r_arr[idx]


# ══════════════════════════════════════════════════════════════════════
# 神经网络（完整保留，一行不改）
# ══════════════════════════════════════════════════════════════════════

class _UserTower(nn.Module):
    def __init__(self, n_users, emb_dim=DEFAULT_EMB_DIM, hidden_dim=DEFAULT_HIDDEN, out_dim=DEFAULT_OUT_DIM):
        super().__init__()
        self.user_emb   = nn.Embedding(n_users, emb_dim)
        self.active_emb = nn.Embedding(N_ACTIVE_DEGREES, 4)
        in_dim = emb_dim + 4 + 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.active_emb.weight, std=0.01)

    def forward(self, user_idx, active_idx, num_feats):
        u = self.user_emb(user_idx); a = self.active_emb(active_idx)
        x = torch.cat([u, a, num_feats], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class _ItemTower(nn.Module):
    def __init__(self, n_items, emb_dim=DEFAULT_EMB_DIM, hidden_dim=DEFAULT_HIDDEN, out_dim=DEFAULT_OUT_DIM):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim)
        in_dim = emb_dim + N_CATEGORIES + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, item_idx, cat_multihot, dur_feat):
        v = self.item_emb(item_idx)
        x = torch.cat([v, cat_multihot, dur_feat], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class _TwoTowerModel(nn.Module):
    def __init__(self, user_tower, item_tower):
        super().__init__()
        self.user_tower = user_tower; self.item_tower = item_tower

    def forward(self, user_idx, active_idx, num_feats, item_idx, cat_multihot, dur_feat):
        u = self.user_tower(user_idx, active_idx, num_feats)
        v = self.item_tower(item_idx, cat_multihot, dur_feat)
        return (u * v).sum(dim=-1)


# ══════════════════════════════════════════════════════════════════════
# BaseRecommender 子类
# ══════════════════════════════════════════════════════════════════════

class TwoTower(BaseRecommender):
    """
    TwoTower 双塔召回模型（BPR / WBPR 两个版本）。

    __init__(data, output_dir, checkpoint_dir=None,
             weighted=True,        # True=WBPR, False=BPR
             emb_dim=32, hidden_dim=128, out_dim=64,
             lr=1e-3, batch_size=4096)
    train(n_epochs=20, patience=5, val_frac=0.1)
    recommend(top_k=50) -> dict[uid, list[vid]]
    """

    def __init__(
        self,
        data: ModelData,
        output_dir: Path,
        checkpoint_dir: Path | None = None,
        weighted: bool = True,
        emb_dim: int = DEFAULT_EMB_DIM,
        hidden_dim: int = DEFAULT_HIDDEN,
        out_dim: int = DEFAULT_OUT_DIM,
        lr: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH,
    ):
        super().__init__(data, output_dir, checkpoint_dir)
        self.weighted   = weighted
        self.emb_dim    = emb_dim
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.lr         = lr
        self.batch_size = batch_size
        self.device = get_device()
        self._model: _TwoTowerModel | None = None
        self._item_cat_t: torch.Tensor | None = None
        self._item_dur_t: torch.Tensor | None = None
        self._user_active_t: torch.Tensor | None = None
        self._user_num_t: torch.Tensor | None = None

    def _variant(self):
        return "wbpr" if self.weighted else "bpr"

    def _load_features(self, device):
        item_ids = self.data.item_ids
        user_ids = self.data.user_ids
        user_active_np, user_num_np = _build_user_features(user_ids)
        item_cat_np,    item_dur_np = _build_item_features_tt(item_ids)
        self._user_active_t = torch.from_numpy(user_active_np).to(device)
        self._user_num_t    = torch.from_numpy(user_num_np).to(device)
        self._item_cat_t    = torch.from_numpy(item_cat_np).to(device)
        self._item_dur_t    = torch.from_numpy(item_dur_np).to(device)

    def train(self, n_epochs: int = 20, patience: int = 5, val_frac: float = 0.1) -> None:
        device = self.device
        print(f"[TwoTower] device = {device}")

        df       = self.data.interaction_df
        user_ids = self.data.user_ids
        item_ids = self.data.item_ids
        n_users, n_items = len(user_ids), len(item_ids)

        u_index = {uid: i for i, uid in enumerate(user_ids)}
        i_index = {iid: i for i, iid in enumerate(item_ids)}
        u_arr = df["user_id"].map(u_index).values.astype(np.int64)
        i_arr = df["video_id"].map(i_index).values.astype(np.int64)
        r_arr = df["watch_ratio"].values.astype(np.float32)

        self._load_features(device)

        n_total = len(u_arr)
        perm  = np.random.default_rng(42).permutation(n_total)
        n_val = max(1, min(500_000, int(n_total * val_frac)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        train_ds = _TwoTowerDataset(u_arr[train_idx], i_arr[train_idx], r_arr[train_idx], n_items)
        val_ds   = _TwoTowerDataset(u_arr[val_idx],   i_arr[val_idx],   r_arr[val_idx],   n_items)
        pin = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

        user_tower = _UserTower(n_users, self.emb_dim, self.hidden_dim, self.out_dim).to(device)
        item_tower = _ItemTower(n_items, self.emb_dim, self.hidden_dim, self.out_dim).to(device)
        model      = _TwoTowerModel(user_tower, item_tower)
        optimizer  = torch.optim.Adam(model.parameters(), lr=self.lr)

        variant = self._variant()
        ckpt_path      = self.checkpoint_dir / f"two_tower_{variant}_latest.pt"
        best_ckpt_path = self.checkpoint_dir / f"two_tower_{variant}_best.pt"
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
                print(f"[TwoTower] Early Stopping 已完成，直接推理。")
            else:
                print(f"[TwoTower] 从 checkpoint 恢复：epoch {start_epoch}/{n_epochs}")

        weighted = self.weighted
        loss_label = "WBPR-loss" if weighted else "BPR-loss"
        print(f"[TwoTower] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，max_epochs={n_epochs}")

        user_active_t = self._user_active_t
        user_num_t    = self._user_num_t
        item_cat_t    = self._item_cat_t
        item_dur_t    = self._item_dur_t

        avg_loss = 0.0
        for epoch in range(start_epoch, n_epochs):
            t0 = time.time()
            model.train()
            total_loss, n_seen = 0.0, 0
            for u_b, pos_b, neg_b, w_b in train_loader:
                u_b = u_b.to(device); pos_b = pos_b.to(device)
                neg_b = neg_b.to(device); w_b = w_b.to(device)
                u_emb   = model.user_tower(u_b, user_active_t[u_b], user_num_t[u_b])
                pos_emb = model.item_tower(pos_b, item_cat_t[pos_b], item_dur_t[pos_b])
                neg_emb = model.item_tower(neg_b, item_cat_t[neg_b], item_dur_t[neg_b])
                pos_s = (u_emb * pos_emb).sum(-1); neg_s = (u_emb * neg_emb).sum(-1)
                if weighted:
                    loss = -(w_b.clamp(min=1e-8) * F.logsigmoid(pos_s - neg_s)).mean()
                else:
                    loss = -F.logsigmoid(pos_s - neg_s).mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item() * len(u_b); n_seen += len(u_b)
            avg_loss = total_loss / n_seen

            model.eval()
            val_total, n_val = 0.0, 0
            with torch.no_grad():
                for u_b, pos_b, neg_b, w_b in val_loader:
                    u_b, pos_b, neg_b, w_b = u_b.to(device), pos_b.to(device), neg_b.to(device), w_b.to(device)
                    u_emb   = model.user_tower(u_b, user_active_t[u_b], user_num_t[u_b])
                    pos_emb = model.item_tower(pos_b, item_cat_t[pos_b], item_dur_t[pos_b])
                    neg_emb = model.item_tower(neg_b, item_cat_t[neg_b], item_dur_t[neg_b])
                    ps = (u_emb * pos_emb).sum(-1); ns = (u_emb * neg_emb).sum(-1)
                    vl = -(w_b.clamp(min=1e-8) * F.logsigmoid(ps - ns)).mean() if weighted else -F.logsigmoid(ps - ns).mean()
                    val_total += vl.item() * len(u_b); n_val += len(u_b)
            avg_val = val_total / n_val
            improved = avg_val < best_val_loss - 1e-6
            star = " ★" if improved else ""
            print(f"  epoch {epoch+1:>2}/{n_epochs}  {loss_label}={avg_loss:.6f}  val={avg_val:.6f}{star}  ({time.time()-t0:.1f}s)")

            if improved:
                best_val_loss = avg_val; patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                save_checkpoint(best_ckpt_path, {"epoch": epoch, "model": best_model_state,
                    "optimizer": optimizer.state_dict(), "val_loss": avg_val})
                print(f"  └─ 最佳模型已保存（val_loss={avg_val:.6f}）")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[TwoTower] Early Stopping：停止在 epoch {epoch+1}")
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

        self._model = model
        print(f"[TwoTower] 训练完成，最终 {loss_label}={avg_loss:.6f}")

    def recommend(self, top_k: int = 50) -> dict:
        device   = self.device
        item_ids = self.data.item_ids
        user_ids = self.data.user_ids
        matrix   = self.data.matrix
        n_users, n_items = len(user_ids), len(item_ids)

        if self._model is None:
            self._load_features(device)
            user_tower = _UserTower(n_users, self.emb_dim, self.hidden_dim, self.out_dim).to(device)
            item_tower = _ItemTower(n_items, self.emb_dim, self.hidden_dim, self.out_dim).to(device)
            model      = _TwoTowerModel(user_tower, item_tower)
            variant    = self._variant()
            best_ckpt  = self.checkpoint_dir / f"two_tower_{variant}_best.pt"
            ckpt_path  = self.checkpoint_dir / f"two_tower_{variant}_latest.pt"
            ckpt = load_checkpoint(best_ckpt, device) or load_checkpoint(ckpt_path, device)
            if ckpt is None:
                raise RuntimeError("[TwoTower] 没有 checkpoint，请先调用 train()。")
            model.load_state_dict(ckpt["model"])
            self._model = model

        model = self._model
        model.eval()
        print(f"\n[TwoTower] 为 {n_users:,} 位用户生成个性化 top-{top_k} 推荐……")

        item_cat_t  = self._item_cat_t
        item_dur_t  = self._item_dur_t
        user_active_t = self._user_active_t
        user_num_t    = self._user_num_t

        recommendations: dict = {}
        seen = matrix.tolil()

        with torch.no_grad():
            all_idx  = torch.arange(n_items, device=device)
            emb_list = []
            for s in range(0, n_items, 2048):
                idx_b = all_idx[s:s + 2048]
                emb_list.append(model.item_tower(idx_b, item_cat_t[idx_b], item_dur_t[idx_b]))
            all_item_emb = torch.cat(emb_list, dim=0)

            for u_start in range(0, n_users, 256):
                u_end   = min(u_start + 256, n_users)
                u_idx_b = torch.arange(u_start, u_end, device=device)
                u_emb_b = model.user_tower(u_idx_b, user_active_t[u_idx_b], user_num_t[u_idx_b])
                scores_np = (u_emb_b @ all_item_emb.T).cpu().numpy()
                for local_i, u_global in enumerate(range(u_start, u_end)):
                    scores    = scores_np[local_i].copy()
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
        return recommendations
