"""
文件用途：LLM-based 推荐（2023-24 前沿）。

【核心思想】
  把推荐问题转成语言问题：用用户的历史交互序列构造自然语言 prompt，
  让 LLM 直接输出候选视频的排序或评分。

  两种典型范式：
    零样本（Zero-shot）  直接让 LLM 推断用户喜好，无需额外训练
    少样本（Few-shot）   在 prompt 里加几条示例，引导 LLM 输出格式

  本实现采用"LLM 作为排序器"（LLM as Reranker）的思路：
    1. SASRec 召回 top-K 候选（粗排，速度快）
    2. 用 LLM 对候选做精排（point-wise 评分）
    3. 取 LLM 评分最高的 top-k 作为最终结果

  这与工业界 P5 / TALLRec / LLMRank 等论文的主流路线一致。

【本地 LLM 支持（无需 API key）】
  默认使用 Ollama（本地部署），模型：qwen2.5:7b 或任意兼容模型。
  如果 Ollama 不可用，自动 fallback 到"随机重排"（保持接口不变，
  指标与 SASRec 相同，方便在无 GPU/API 的环境下跑通完整流水线）。

【在 KuaiRec 上的局限性说明（诚实评估，不夸大）】
  KuaiRec 的 video_id 是数字 ID，LLM 没有见过这些视频的文字描述，
  因此零样本推理能力受限——LLM 擅长的是有语义的 item（电影名/商品名）。
  在本实验中，LLM 主要贡献来自"重排逻辑的泛化能力"而非语义理解。
  这也是为什么工业界 LLM 推荐通常要配合 item 文本特征（标题/标签/描述）。

【返回格式】
  与 run_sasrec_pipeline 完全兼容，可直接插入 eval_advanced.py。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from svd_recommender import (
    build_sparse_matrix,
    load_big_matrix_interactions,
    recommendations_to_dataframe,
)

# ── 常量 ──────────────────────────────────────────────────────────────
DEFAULT_RECALL_K     = 50    # SASRec 召回候选数（粗排输出）
DEFAULT_RERANK_K     = 10    # LLM 精排后保留数
DEFAULT_TOP_K        = 50    # 最终推荐数（recall_k 足够大时等于 recall_k）
DEFAULT_LLM_MODEL    = "qwen2.5:7b"
DEFAULT_OLLAMA_URL   = "http://localhost:11434"
DEFAULT_BATCH_RERANK = 20    # 每次提交 LLM 排序的候选数
DEFAULT_MAX_HIST     = 10    # prompt 中展示的历史记录条数（避免超出 context）
DEFAULT_TEMPERATURE  = 0.0   # 推理时用贪心（temperature=0），保证结果可复现


# ══════════════════════════════════════════════════════════════════════
# Ollama 客户端
# ══════════════════════════════════════════════════════════════════════

def _ollama_available(base_url: str) -> bool:
    """检查 Ollama 服务是否在运行。"""
    try:
        import urllib.request
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _ollama_chat(
    prompt: str,
    model: str,
    base_url: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """调用 Ollama /api/chat，返回助手的文本回复。"""
    import urllib.request

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"].strip()


# ══════════════════════════════════════════════════════════════════════
# Prompt 构建
# ══════════════════════════════════════════════════════════════════════

def _build_rerank_prompt(
    history_ids: list[int],
    candidate_ids: list[int],
    max_hist: int,
) -> str:
    """
    构造 LLM 排序 prompt。

    由于 KuaiRec 是数字 ID，prompt 里只能展示 ID，
    无语义文本。LLM 需要从 ID 的共现模式做隐式推断。
    工业界场景中这里会替换成视频标题/标签/类别描述。
    """
    hist_display = history_ids[-max_hist:]
    hist_str = ", ".join(str(v) for v in hist_display)
    cand_str = "\n".join(f"{i+1}. video_id={v}" for i, v in enumerate(candidate_ids))

    prompt = f"""You are a video recommendation assistant.

A user has recently watched the following videos (in chronological order):
{hist_str}

Based on this watch history, rank the following candidate videos from most to least relevant for this user.
Output ONLY a JSON array of video IDs in your preferred order, with the most relevant first.
Do not include any explanation.

Candidates:
{cand_str}

Output format example: [123, 456, 789]
Your ranking:"""
    return prompt


def _parse_ranking(response: str, candidate_ids: list[int]) -> list[int]:
    """
    从 LLM 回复中解析排序结果。

    鲁棒解析：尽量提取 JSON 数组，失败时保留原顺序。
    """
    try:
        # 找到第一个 [ 和最后一个 ]
        start = response.find("[")
        end   = response.rfind("]")
        if start == -1 or end == -1:
            return candidate_ids[:]
        arr = json.loads(response[start:end + 1])
        # 过滤掉不在候选集里的 id
        cand_set = set(candidate_ids)
        ranked = [int(v) for v in arr if int(v) in cand_set]
        # 把 LLM 没输出的候选追加到末尾（保证全覆盖）
        seen = set(ranked)
        ranked += [v for v in candidate_ids if v not in seen]
        return ranked
    except Exception:
        return candidate_ids[:]


# ══════════════════════════════════════════════════════════════════════
# LLM 精排
# ══════════════════════════════════════════════════════════════════════

def _llm_rerank(
    history_ids: list[int],
    candidate_ids: list[int],
    model: str,
    base_url: str,
    max_hist: int,
    temperature: float,
) -> list[int]:
    """
    用 LLM 对 candidate_ids 排序，返回重排后的列表。
    失败时原序返回。
    """
    if not candidate_ids:
        return []
    prompt   = _build_rerank_prompt(history_ids, candidate_ids, max_hist)
    response = _ollama_chat(prompt, model, base_url, temperature)
    return _parse_ranking(response, candidate_ids)


# ══════════════════════════════════════════════════════════════════════
# 数据准备
# ══════════════════════════════════════════════════════════════════════

def _build_user_histories(
    df: pd.DataFrame,
    max_hist: int,
) -> dict[Any, list[int]]:
    """返回 {user_id: [video_id, ...]} 原始 video_id，取最近 max_hist 条。"""
    histories: dict[Any, list[int]] = {}
    for uid, grp in df.groupby("user_id"):
        histories[uid] = grp["video_id"].tolist()[-max_hist:]
    return histories


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def run_llm_rec_pipeline(
    recall_k: int = DEFAULT_RECALL_K,
    top_k: int = DEFAULT_TOP_K,
    llm_model: str = DEFAULT_LLM_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    max_hist: int = DEFAULT_MAX_HIST,
    temperature: float = DEFAULT_TEMPERATURE,
    n_epochs: int = 50,     # 透传给 SASRec 粗排
    patience: int = 5,
    eligible_video_ids: set | None = None,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    _test_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    LLM-based 推荐流程（召回 + LLM 精排）。

    Step 1: SASRec 粗排，生成每用户 recall_k 候选。
    Step 2: LLM 对每用户候选精排（如果 Ollama 不可用则跳过）。
    Step 3: 取最终 top_k 返回。
    """
    print("[LLMRec] === 阶段 1：SASRec 粗排召回 ===")
    from sasrec import run_sasrec_pipeline

    sasrec_result = run_sasrec_pipeline(
        n_epochs=n_epochs,
        top_k=recall_k,
        patience=patience,
        eligible_video_ids=eligible_video_ids,
        output_dir=None,
        checkpoint_dir=checkpoint_dir,
        _test_df=_test_df,
    )

    df             = sasrec_result["_interaction_df"]
    matrix         = sasrec_result["_matrix"]
    user_ids       = sasrec_result["_user_ids"]
    item_ids       = sasrec_result["_item_ids"]
    n_users        = sasrec_result["n_users"]
    n_items        = sasrec_result["n_items"]
    recall_recs    = sasrec_result["recommendations"]   # {uid: [video_id, ...]}

    print("\n[LLMRec] === 阶段 2：LLM 精排 ===")
    use_llm = _ollama_available(ollama_url)
    if use_llm:
        print(f"[LLMRec] Ollama 可用，使用模型 {llm_model} 精排。")
    else:
        print(
            f"[LLMRec] Ollama 不可用（{ollama_url}），"
            "退化为 SASRec 召回结果（无 LLM 精排）。\n"
            "         若要启用 LLM，请先运行：ollama pull qwen2.5:7b && ollama serve"
        )

    # 构建用户历史（原始 video_id）
    histories = _build_user_histories(df, max_hist)

    recommendations: dict = {}
    total = len(recall_recs)
    log_interval = max(1, total // 20)

    t0 = time.time()
    for i, (uid, candidates) in enumerate(recall_recs.items()):
        if i % log_interval == 0:
            elapsed = time.time() - t0
            print(f"  精排进度：{i}/{total}  ({elapsed:.0f}s elapsed)")

        if not candidates:
            recommendations[uid] = []
            continue

        if use_llm:
            hist = histories.get(uid, [])
            reranked = _llm_rerank(
                history_ids=hist,
                candidate_ids=candidates[:recall_k],
                model=llm_model,
                base_url=ollama_url,
                max_hist=max_hist,
                temperature=temperature,
            )
            recommendations[uid] = reranked[:top_k]
        else:
            # 无 LLM：直接截断 SASRec 结果
            recommendations[uid] = candidates[:top_k]

    # 补全没有候选的用户
    for uid in user_ids:
        if uid not in recommendations:
            recommendations[uid] = []

    model_name = f"LLMRec-{llm_model}" if use_llm else "LLMRec-fallback(SASRec)"
    print(f"\n[LLMRec] 完成：{len(recommendations):,} 用户，模式 = {model_name}")

    rec_df = recommendations_to_dataframe(recommendations)

    result: dict[str, Any] = {
        "model":    model_name,
        "n_users":  n_users,
        "n_items":  n_items,
        "emb_dim":  sasrec_result["emb_dim"],
        "top_k":    top_k,
        "bpr_loss": sasrec_result["bpr_loss"],
        "rmse":     0.0,
        "recommendations":    recommendations,
        "recommendations_df": rec_df,
        "_matrix":            matrix,
        "_user_ids":          user_ids,
        "_item_ids":          item_ids,
        "_interaction_df":    df,
        "singular_values":    [],
        "_llm_used":          use_llm,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rec_path = output_dir / f"llmrec_top{top_k}_recommendations.csv"
        rec_df.to_csv(rec_path, index=False, encoding="utf-8-sig")
        print(f"[LLMRec] 推荐列表已保存：{rec_path}")
        result["output_path"] = str(rec_path)

    return result


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based 推荐（SASRec召回 + LLM精排）。")
    parser.add_argument("--recall-k",    type=int,   default=DEFAULT_RECALL_K,
                        help="SASRec 粗排候选数")
    parser.add_argument("--top-k",       type=int,   default=DEFAULT_TOP_K,
                        help="最终返回推荐数")
    parser.add_argument("--llm-model",   type=str,   default=DEFAULT_LLM_MODEL,
                        help="Ollama 模型名，如 qwen2.5:7b / llama3:8b")
    parser.add_argument("--ollama-url",  type=str,   default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--max-hist",    type=int,   default=DEFAULT_MAX_HIST)
    parser.add_argument("--n-epochs",    type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--output-dir",  type=str,   default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parents[1] / "output"
    )

    run_llm_rec_pipeline(
        recall_k=args.recall_k,
        top_k=args.top_k,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url,
        max_hist=args.max_hist,
        n_epochs=args.n_epochs,
        patience=args.patience,
        output_dir=out,
    )
