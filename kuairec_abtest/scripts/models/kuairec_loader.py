"""
models/kuairec_loader.py — KuaiRec 数据统一加载，只读一次。

从 CSV 文件加载 big_matrix、构建稀疏矩阵、用户行为序列、
视频侧特征和 small_matrix ground truth，封装进 ModelData。

用法:
    from models.kuairec_loader import load_model_data
    data = load_model_data()
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 确保 scripts/ 目录在 sys.path（从 models/ 子目录调用时需要）
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from svd_recommender import (
    build_sparse_matrix,
    load_big_matrix_interactions,
)
from models.base import ModelData, build_user_sequences_with_ratio

# ── 数据目录 ──────────────────────────────────────────────────────────

def _find_data_dir() -> Path:
    candidate = Path(__file__).resolve().parents[2] / "data" / "KuaiRec 2.0" / "data"
    return candidate


def _find_small_matrix_csv() -> Path:
    return _find_data_dir() / "small_matrix.csv"


# ══════════════════════════════════════════════════════════════════════
# 视频侧特征加载（供 SideInfo 和 TwoTower 使用）
# ══════════════════════════════════════════════════════════════════════

def _load_video_features(item_ids: np.ndarray) -> pd.DataFrame:
    """
    加载并对齐视频侧特征，返回 DataFrame（行对应 item_ids 顺序）。

    列：
        video_id          原始 ID
        category_multihot (31,) 类别 multi-hot，列名 cat_0..cat_30
        duration_log      log1p(video_duration)
    """
    import ast

    data_dir = _find_data_dir()
    n_items  = len(item_ids)
    N_CAT    = 31

    cat_arr = np.zeros((n_items, N_CAT), dtype=np.float32)
    dur_arr = np.zeros(n_items, dtype=np.float32)
    id2idx  = {int(vid): i for i, vid in enumerate(item_ids)}

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
                    if 0 <= c < N_CAT:
                        cat_arr[id2idx[vid], c] = 1.0
            except Exception:
                pass
    else:
        print("[DataLoader] item_categories.csv 不存在，类别特征全零。")

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
                    dur_arr[idx] = float(np.log1p(dur))
    else:
        print("[DataLoader] item_daily_features.csv 不存在，时长特征全零。")

    # 组装成 DataFrame
    cat_df = pd.DataFrame(cat_arr, columns=[f"cat_{i}" for i in range(N_CAT)])
    cat_df.insert(0, "video_id", item_ids)
    cat_df["duration_log"] = dur_arr

    print(f"[DataLoader] 视频特征加载完成：{n_items:,} 视频，{N_CAT + 1} 个特征列。")
    return cat_df


# ══════════════════════════════════════════════════════════════════════
# Ground truth 加载
# ══════════════════════════════════════════════════════════════════════

def _load_ground_truth() -> dict[Any, dict[Any, float]]:
    path = _find_small_matrix_csv()
    if not path.exists():
        raise FileNotFoundError(f"small_matrix.csv 不存在：{path}")

    df = pd.read_csv(path, usecols=["user_id", "video_id", "watch_ratio"])
    df["watch_ratio"] = pd.to_numeric(df["watch_ratio"], errors="coerce").fillna(0.0).clip(0.0)

    gt: dict[Any, dict[Any, float]] = {}
    for uid, grp in df.groupby("user_id"):
        gt[uid] = dict(zip(grp["video_id"], grp["watch_ratio"]))

    print(f"[DataLoader] Ground truth 加载完成：{len(gt):,} 用户，{len(df):,} 条记录。")
    return gt


# ══════════════════════════════════════════════════════════════════════
# 统一入口
# ══════════════════════════════════════════════════════════════════════

def load_model_data(
    eligible_video_ids: set | None = None,
    max_seq: int = 50,
) -> "ModelData":
    """
    加载 KuaiRec 数据并返回 ModelData。

    参数:
        eligible_video_ids: 只保留这些视频（通常从 small_matrix 中提取）
        max_seq: 用户行为序列最大长度（默认 50，与各模型默认值一致）

    返回:
        ModelData dataclass，包含所有模型所需数据。
    """
    print("\n[DataLoader] 开始加载 KuaiRec 数据……")

    # 1. 交互数据
    df = load_big_matrix_interactions(eligible_video_ids)
    matrix, user_ids, item_ids = build_sparse_matrix(df)
    print(f"[DataLoader] 交互矩阵：{len(user_ids):,} 用户 × {len(item_ids):,} 视频。")

    # 2. 用户行为序列（带 watch_ratio）
    item_index = {iid: i for i, iid in enumerate(item_ids)}
    sequences = build_user_sequences_with_ratio(df, item_index, max_seq)
    print(f"[DataLoader] 用户行为序列：{len(sequences):,} 用户有序列（>= 1 条交互）。")

    # 3. 视频侧特征
    video_features = _load_video_features(item_ids)

    # 4. Ground truth
    ground_truth = _load_ground_truth()

    data = ModelData(
        interaction_df=df,
        matrix=matrix,
        user_ids=user_ids,
        item_ids=item_ids,
        sequences=sequences,
        video_features=video_features,
        ground_truth=ground_truth,
    )
    print("[DataLoader] 数据加载完成。\n")
    return data
