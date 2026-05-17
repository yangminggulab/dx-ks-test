"""
文件用途：PyTorch 版矩阵分解，与 mf_optimized.py 接口兼容。

本文件是学习对照版：把 mf_optimized.py 里的 numpy/手写 SGD 换成 PyTorch。

  BiasedMF —— nn.Module + Adam（标准深度学习训练范式，走 MPS GPU）
  iALS     —— torch.linalg.solve（线性代数，不走梯度下降，强制 CPU/float64）

【PyTorch 核心概念速查】
  nn.Module     所有模型基类；__init__ 注册参数，forward 定义前向计算
  nn.Embedding  可学习查找表，本质是 (N, k) 权重矩阵，按整数索引取对应行
  DataLoader    把数据集切成 mini-batch，自动 shuffle 和并行预取
  optimizer     封装参数更新；Adam = SGD + 动量估计 + 自适应学习率
  loss.backward 反向传播：自动微分，计算 loss 对所有参数的梯度
  optimizer.step 用梯度更新参数（一步走完"链式法则 → 更新权重"）
  torch.no_grad 推理时关闭自动微分，节省内存和计算
  .to(device)   把 tensor / 模型移到指定设备（cpu / mps / cuda）
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from svd_recommender import (
    build_sparse_matrix,
    load_big_matrix_interactions,
    recommendations_to_dataframe,
)
from mf_optimized import _top_k_from_factors


def _get_device(float64_required: bool = False) -> torch.device:
    """
    MPS（Apple Silicon GPU）不支持 float64，只支持 float32。
    BiasedMF 用 float32 → 走 MPS；iALS 需要 float64 → 强制 CPU。
    """
    if not float64_required and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════
# BiasedMF 模型定义
# ══════════════════════════════════════════════════════════════════════

class BiasedMFModel(nn.Module):
    """
    Biased Matrix Factorization：
      预测 = μ + b_用户 + b_视频 + 用户向量 · 视频向量

    继承 nn.Module 后：
    - __init__ 里定义所有可学习参数（用 nn.Embedding / nn.Parameter 注册）
    - forward 定义"输入索引 → 预测分数"的计算逻辑
    - 框架自动追踪参数，计算梯度，保存/加载模型
    """

    def __init__(self, n_users: int, n_items: int, n_factors: int):
        super().__init__()

        # nn.Embedding(N, k)：N 个实体各用 k 维向量表示
        # 内部是 (N, k) 权重矩阵；输入整数索引，输出对应行向量
        self.user_emb  = nn.Embedding(n_users, n_factors)
        self.item_emb  = nn.Embedding(n_items, n_factors)

        # 偏置用 Embedding(N, 1)，因为需要按用户/视频索引查找各自的偏置值
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        # nn.Parameter：标量全局均值，告诉框架"这是需要训练的参数"
        self.mu = nn.Parameter(torch.zeros(1))

        # 初始化：embedding 用小随机值（避免对称破坏），bias 初始为 0
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """
        u, i: shape (batch,) 的 LongTensor，分别是用户和视频的 0-based 整数索引。
        返回 shape (batch,) 的预测 watch_ratio。
        """
        pu = self.user_emb(u)               # (batch, k) 用户隐向量
        qi = self.item_emb(i)               # (batch, k) 视频隐向量
        bu = self.user_bias(u).squeeze(-1)  # (batch,)   用户偏置
        bi = self.item_bias(i).squeeze(-1)  # (batch,)   视频偏置
        # 点积捕捉用户-视频的个性化匹配，偏置项吸收全局/个体偏差
        return (pu * qi).sum(dim=-1) + bu + bi + self.mu


# ══════════════════════════════════════════════════════════════════════
# BiasedMF 训练流程
# ══════════════════════════════════════════════════════════════════════

def run_biased_mf_torch_pipeline(
    n_factors: int = 50,
    n_epochs: int = 20,
    lr: float = 0.005,
    reg: float = 0.02,
    batch_size: int = 50_000,
    top_k: int = 50,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """PyTorch BiasedMF 完整流程，返回格式与 mf_optimized.run_biased_mf_pipeline 兼容。"""
    device = _get_device(float64_required=False)
    print(f"[BiasedMF-Torch] device = {device}")

    df = _test_df if _test_df is not None else load_big_matrix_interactions(eligible_video_ids)
    matrix, user_ids, item_ids = build_sparse_matrix(df)

    # 把 user_id / video_id 映射到 0-based 整数索引（Embedding 需要整数索引）
    u_index = {uid: idx for idx, uid in enumerate(user_ids)}
    i_index = {iid: idx for idx, iid in enumerate(item_ids)}
    u_arr = df["user_id"].map(u_index).values.astype(np.int64)
    i_arr = df["video_id"].map(i_index).values.astype(np.int64)
    r_arr = df["watch_ratio"].values.astype(np.float32)

    n_users, n_items = len(user_ids), len(item_ids)

    # torch.from_numpy：零拷贝，共享内存；.to(device) 搬到 MPS/CPU
    u_t = torch.from_numpy(u_arr).to(device)
    i_t = torch.from_numpy(i_arr).to(device)
    r_t = torch.from_numpy(r_arr).to(device)

    # TensorDataset 把多个 tensor 打包成 dataset
    # DataLoader 自动切 batch、shuffle（每 epoch 重新打乱顺序）
    dataset = TensorDataset(u_t, i_t, r_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BiasedMFModel(n_users, n_items, n_factors).to(device)

    # Adam optimizer：传入模型全部参数；weight_decay 等价于 L2 正则
    # Adam 比纯 SGD 收敛快：它对每个参数维护独立的自适应学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    loss_fn   = nn.MSELoss()

    print(
        f"[BiasedMF-Torch] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，"
        f"k={n_factors}，epochs={n_epochs}，lr={lr}，reg={reg}"
    )

    for epoch in range(n_epochs):
        t0 = time.time()
        model.train()       # 通知模型进入训练模式（本模型无 dropout，但这是好习惯）
        total_loss = 0.0
        n_seen = 0

        for u_b, i_b, r_b in loader:
            pred = model(u_b, i_b)      # 前向传播：调用 forward()

            loss = loss_fn(pred, r_b)   # MSE = mean((pred - true)²)

            optimizer.zero_grad()       # 清空上一步留下的梯度（不清会累加！）
            loss.backward()             # 反向传播：自动计算 ∂loss/∂每个参数
            optimizer.step()            # 用梯度更新参数：param -= lr * grad（Adam 版）

            total_loss += loss.item() * len(r_b)   # .item() 把单元素 tensor 转成 Python float
            n_seen += len(r_b)

        rmse = math.sqrt(total_loss / n_seen)
        print(f"  epoch {epoch+1:>2}/{n_epochs}  RMSE={rmse:.6f}  ({time.time()-t0:.1f}s)")

    # 提取训练好的权重，转回 numpy 供推荐逻辑复用
    model.eval()
    with torch.no_grad():   # 推理时不需要梯度，关掉节省内存
        U   = model.user_emb.weight.cpu().numpy()           # (n_users, k)
        V   = model.item_emb.weight.cpu().numpy()           # (n_items, k)
        b_u = model.user_bias.weight.cpu().numpy().ravel()  # (n_users,)
        b_i = model.item_bias.weight.cpu().numpy().ravel()  # (n_items,)
        mu  = float(model.mu.cpu().item())

    final_rmse = float(np.sqrt(np.mean(
        (r_arr - (np.sum(U[u_arr] * V[i_arr], axis=1) + b_u[u_arr] + b_i[i_arr] + mu)) ** 2
    )))
    print(f"[BiasedMF-Torch] 训练完成，最终 RMSE={final_rmse:.6f}")

    recs   = _top_k_from_factors(U, V, matrix, user_ids, item_ids, top_k, mu, b_u, b_i)
    rec_df = recommendations_to_dataframe(recs)

    result: dict[str, Any] = {
        "model":     "BiasedMF-Torch",
        "n_users":   n_users,
        "n_items":   n_items,
        "n_factors": n_factors,
        "top_k":     top_k,
        "rmse":      final_rmse,
        "mu":        mu,
        "recommendations":    recs,
        "recommendations_df": rec_df,
        "_U":        U,
        "_Vt":       V.T,
        "_matrix":   matrix,
        "_user_ids": user_ids,
        "_item_ids": item_ids,
        "_interaction_df": df,
        "singular_values": [],
        "b_u": b_u,
        "b_i": b_i,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_df.to_csv(
            output_dir / f"biased_mf_torch_top{top_k}_recommendations.csv",
            index=False, encoding="utf-8-sig",
        )

    return result


# ══════════════════════════════════════════════════════════════════════
# iALS（PyTorch 线性代数版）
#
# iALS 是闭式解（ALS），不走梯度下降，所以不用 nn.Module / optimizer。
# 用 torch.linalg.solve 替代 np.linalg.solve，展示 PyTorch 的线性代数 API。
#
# 注意：MPS 不支持 float64，ALS 求解线性方程组对精度敏感，强制用 CPU。
# ══════════════════════════════════════════════════════════════════════

def run_ials_torch_pipeline(
    n_factors: int = 50,
    n_iters: int = 15,
    reg: float = 0.01,
    alpha: float = 40.0,
    top_k: int = 50,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    iALS PyTorch 版。

    与 mf_optimized.run_ials_pipeline 的唯一区别：
      np.linalg.solve  →  torch.linalg.solve
      numpy 矩阵乘法  →  torch @ 运算符

    强制 CPU：MPS 不支持 float64，而 ALS 的线性方程求解需要 float64 精度。
    """
    device = _get_device(float64_required=True)   # → cpu
    print(f"[iALS-Torch] device = {device}  (float64，MPS 不支持，强制 CPU)")

    df = _test_df if _test_df is not None else load_big_matrix_interactions(eligible_video_ids)
    matrix, user_ids, item_ids = build_sparse_matrix(df)

    n_users, n_items = len(user_ids), len(item_ids)
    k = n_factors
    rng = np.random.default_rng(42)

    u_index = {uid: idx for idx, uid in enumerate(user_ids)}
    i_index = {iid: idx for idx, iid in enumerate(item_ids)}
    u_arr = df["user_id"].map(u_index).values.astype(np.int32)
    i_arr = df["video_id"].map(i_index).values.astype(np.int32)
    r_arr = df["watch_ratio"].values.astype(np.float64)
    c_arr = 1.0 + alpha * r_arr   # 置信度：看完比看一半更"可信"

    # 初始化因子矩阵（torch tensor，dtype=float64）
    U = torch.tensor(rng.normal(0, 0.1, (n_users, k)), dtype=torch.float64, device=device)
    V = torch.tensor(rng.normal(0, 0.1, (n_items, k)), dtype=torch.float64, device=device)

    # 预处理：按用户/视频分组，存成 torch tensor（循环内直接用，省去类型转换）
    user_groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    _tmp_u: dict[int, list] = {i: [] for i in range(n_users)}
    _tmp_cu: dict[int, list] = {i: [] for i in range(n_users)}
    for u, i, c in zip(u_arr, i_arr, c_arr):
        _tmp_u[u].append(i)
        _tmp_cu[u].append(c)
    for u in range(n_users):
        user_groups.append((
            torch.tensor(_tmp_u[u],  dtype=torch.long,    device=device),
            torch.tensor(_tmp_cu[u], dtype=torch.float64, device=device),
        ))

    item_groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    _tmp_i: dict[int, list] = {i: [] for i in range(n_items)}
    _tmp_ci: dict[int, list] = {i: [] for i in range(n_items)}
    for u, i, c in zip(u_arr, i_arr, c_arr):
        _tmp_i[i].append(u)
        _tmp_ci[i].append(c)
    for i in range(n_items):
        item_groups.append((
            torch.tensor(_tmp_i[i],  dtype=torch.long,    device=device),
            torch.tensor(_tmp_ci[i], dtype=torch.float64, device=device),
        ))

    lambda_I = reg * torch.eye(k, dtype=torch.float64, device=device)

    print(
        f"[iALS-Torch] 开始训练：{n_users:,} 用户 × {n_items:,} 视频，"
        f"k={k}，iters={n_iters}，reg={reg}，alpha={alpha}"
    )

    for it in range(n_iters):
        t0 = time.time()

        # 固定 V，更新所有用户向量
        VtV = V.T @ V                          # (k, k)，全局项，只算一次
        for u_idx in range(n_users):
            i_obs, c_obs = user_groups[u_idx]
            if len(i_obs) == 0:
                continue
            V_u = V[i_obs]                     # (|I_u|, k) 该用户看过的视频向量
            # A_u = V^TV + λI + Σ(c-1)·V[i]V[i]^T
            A_u = VtV + lambda_I + (V_u * (c_obs - 1).unsqueeze(1)).T @ V_u
            b_u = V_u.T @ c_obs               # (k,)
            # torch.linalg.solve(A, b)：求解 A·x = b，比 A^{-1}·b 数值更稳定
            U[u_idx] = torch.linalg.solve(A_u, b_u)

        # 固定 U，更新所有视频向量
        UtU = U.T @ U
        for i_idx in range(n_items):
            u_obs, c_obs = item_groups[i_idx]
            if len(u_obs) == 0:
                continue
            U_i = U[u_obs]
            A_i = UtU + lambda_I + (U_i * (c_obs - 1).unsqueeze(1)).T @ U_i
            b_i = U_i.T @ c_obs
            V[i_idx] = torch.linalg.solve(A_i, b_i)

        U_np = U.numpy()
        V_np = V.numpy()
        preds = np.sum(U_np[u_arr] * V_np[i_arr], axis=1)
        rmse = float(np.sqrt(np.mean((r_arr - preds) ** 2)))
        print(f"  iter {it+1:>2}/{n_iters}  RMSE={rmse:.6f}  ({time.time()-t0:.1f}s)")

    U_np = U.numpy()
    V_np = V.numpy()
    final_rmse = float(np.sqrt(np.mean(
        (r_arr - np.sum(U_np[u_arr] * V_np[i_arr], axis=1)) ** 2
    )))
    print(f"[iALS-Torch] 训练完成，最终 RMSE={final_rmse:.6f}")

    recs   = _top_k_from_factors(U_np, V_np, matrix, user_ids, item_ids, top_k)
    rec_df = recommendations_to_dataframe(recs)

    result: dict[str, Any] = {
        "model":     "iALS-Torch",
        "n_users":   n_users,
        "n_items":   n_items,
        "n_factors": k,
        "top_k":     top_k,
        "rmse":      final_rmse,
        "recommendations":    recs,
        "recommendations_df": rec_df,
        "_U":        U_np,
        "_Vt":       V_np.T,
        "_matrix":   matrix,
        "_user_ids": user_ids,
        "_item_ids": item_ids,
        "_interaction_df": df,
        "singular_values": [],
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_df.to_csv(
            output_dir / f"ials_torch_top{top_k}_recommendations.csv",
            index=False, encoding="utf-8-sig",
        )

    return result


# ══════════════════════════════════════════════════════════════════════
# 对比管道：BiasedMF-Torch vs iALS-Torch vs numpy 基线
# ══════════════════════════════════════════════════════════════════════

def run_torch_comparison_pipeline(
    n_factors: int = 50,
    top_k: int = 50,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """依次运行 BiasedMF-Torch 和 iALS-Torch，打印 RMSE 对比。"""
    shared = dict(
        n_factors=n_factors, top_k=top_k,
        eligible_video_ids=eligible_video_ids,
        output_dir=output_dir, _test_df=_test_df,
    )

    print("\n" + "═" * 55)
    print("  PyTorch 模型对比：BiasedMF-Torch vs iALS-Torch")
    print("═" * 55)

    r_bmf  = run_biased_mf_torch_pipeline(**shared)
    r_ials = run_ials_torch_pipeline(**shared)

    rows = [
        {"model": "BiasedMF-Torch", "rmse": r_bmf["rmse"]},
        {"model": "iALS-Torch",     "rmse": r_ials["rmse"]},
    ]
    df_cmp = pd.DataFrame(rows)

    print("\n" + "═" * 55)
    print(df_cmp.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("═" * 55)
    print("注：iALS 优化目标是二值偏好（0/1），RMSE 不与 BiasedMF 直接可比。")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_cmp.to_csv(output_dir / "mf_torch_comparison.csv", index=False, encoding="utf-8-sig")

    return df_cmp


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch 版矩阵分解模型。")
    parser.add_argument("--model", choices=["biased_mf", "ials", "compare"],
                        default="compare", help="运行哪个模型（默认 compare）")
    parser.add_argument("--n-factors", type=int, default=50)
    parser.add_argument("--top-k",     type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )

    if args.model == "biased_mf":
        run_biased_mf_torch_pipeline(n_factors=args.n_factors, top_k=args.top_k, output_dir=out)
    elif args.model == "ials":
        run_ials_torch_pipeline(n_factors=args.n_factors, top_k=args.top_k, output_dir=out)
    else:
        run_torch_comparison_pipeline(n_factors=args.n_factors, top_k=args.top_k, output_dir=out)
