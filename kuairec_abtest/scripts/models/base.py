"""
models/base.py — 抽象基类 + 共享工具函数

所有推荐模型继承 BaseRecommender：
  - __init__(data: ModelData, output_dir: Path, checkpoint_dir: Path)
  - train(**kwargs) -> None          (抽象方法)
  - recommend(top_k: int) -> dict    (抽象方法)
  - run(n_epochs, top_k, ...) -> dict (模板方法，调用 train + recommend)

ModelData dataclass 统一装载数据，train-time 和 infer-time 共享同一份数据。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


# ══════════════════════════════════════════════════════════════════════
# 设备检测（所有模型复用）
# ══════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """返回最优可用设备：cuda > mps > cpu。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# 用户行为序列构建（SASRec / SideInfo / CL4SRec 共用）
# ══════════════════════════════════════════════════════════════════════

def build_user_sequences_with_ratio(
    df: pd.DataFrame,
    item_index: dict,
    max_seq: int,
) -> dict[Any, list[tuple[int, float]]]:
    """
    返回 {user_id: [(item_idx, watch_ratio), ...]}，按原始顺序，长度 <= max_seq。
    用于 SASRec / SideInfo / CL4SRec（需要 watch_ratio 权重）。
    """
    sequences: dict[Any, list[tuple[int, float]]] = {}
    for uid, grp in df.groupby("user_id"):
        items = grp["video_id"].map(item_index).values
        ratios = grp["watch_ratio"].values.astype(np.float32)
        valid = [(int(i), float(r)) for i, r in zip(items, ratios) if pd.notna(i)]
        sequences[uid] = valid[-max_seq:]
    return sequences


def build_user_sequences_ids_only(
    df: pd.DataFrame,
    item_index: dict,
    max_seq: int,
    offset: int = 2,
) -> dict[Any, list[int]]:
    """
    返回 {user_id: [item_token, ...]}，item_token = item_idx + offset，长度 <= max_seq。
    用于 BERT4Rec（token_offset=2，0=PAD，1=MASK）。
    """
    sequences: dict[Any, list[int]] = {}
    for uid, grp in df.groupby("user_id"):
        items = grp["video_id"].map(item_index).dropna().astype(int).tolist()
        sequences[uid] = [it + offset for it in items[-max_seq:]]
    return sequences


# ══════════════════════════════════════════════════════════════════════
# Checkpoint 保存 / 加载（通用格式）
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(path: Path, payload: dict) -> None:
    """保存 checkpoint（payload 需含 model、optimizer、epoch 等字段）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict | None:
    """加载 checkpoint；文件不存在则返回 None。"""
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)


# ══════════════════════════════════════════════════════════════════════
# 统一数据容器
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ModelData:
    """统一数据容器，一次加载后供所有模型共享。"""
    interaction_df: pd.DataFrame      # 原始交互数据 (user_id, video_id, watch_ratio)
    matrix: Any                       # scipy sparse matrix (CSR)
    user_ids: np.ndarray              # 去重后的用户 ID 数组
    item_ids: np.ndarray              # 去重后的视频 ID 数组
    sequences: dict                   # {uid: [(item_idx, wr), ...]} 时序排列（带 ratio）
    video_features: pd.DataFrame      # 视频侧特征（category_id, duration 等）
    ground_truth: dict                # {uid: {vid: watch_ratio}} 来自 small_matrix


# ══════════════════════════════════════════════════════════════════════
# 抽象基类
# ══════════════════════════════════════════════════════════════════════

class BaseRecommender(ABC):
    """
    所有模型的抽象基类。

    子类必须实现：
      train(**kwargs)   → None
      recommend(top_k)  → dict[uid, list[vid]]

    run() 是模板方法，按顺序调用 train + recommend。
    """

    def __init__(
        self,
        data: ModelData,
        output_dir: Path,
        checkpoint_dir: Path | None = None,
    ):
        self.data = data
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def train(self, **kwargs) -> None:
        """训练模型（含 Early Stopping + checkpoint 保存）。"""

    @abstractmethod
    def recommend(self, top_k: int = 50) -> dict:
        """
        基于训练好的模型为所有用户生成推荐列表。

        Returns:
            {uid: [vid, ...]}  按相关性降序排列
        """

    def run(
        self,
        n_epochs: int = 50,
        top_k: int = 50,
        patience: int = 5,
        **kwargs,
    ) -> dict:
        """
        模板方法：train → recommend → 返回 pipeline 结果 dict。

        返回 dict 格式与原 run_*_pipeline 函数兼容：
          {
            "model": str,
            "n_users": int,
            "n_items": int,
            "recommendations": {uid: [vid, ...]},
            ...
          }
        """
        self.train(n_epochs=n_epochs, patience=patience, **kwargs)
        recommendations = self.recommend(top_k=top_k)
        return {
            "model":           self.__class__.__name__,
            "n_users":         len(self.data.user_ids),
            "n_items":         len(self.data.item_ids),
            "top_k":           top_k,
            "recommendations": recommendations,
        }
