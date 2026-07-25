"""Offline lexical reranker for hybrid RAG (Phase 3b).

No cross-encoder dependency: ranks candidates by Chinese-friendly lexical
overlap (CJK n-grams + ZH↔EN financial synonyms) with chunk-type boosts,
then returns top_k. Suitable for demos, CI, and as a default before plugging
in a model-based reranker.
"""

from __future__ import annotations

from typing import Any

from .lexical import lexical_overlap, query_has_any, tokenize_text


def lexical_rerank_score(query: str, hit: dict[str, Any]) -> float:
    text = str(hit.get("text") or "")
    overlap = lexical_overlap(query, text)
    if overlap <= 0.0 and not tokenize_text(query):
        return 0.0
    query_tokens = tokenize_text(query)
    chunk_type = str(hit.get("chunk_type") or "narrative")
    if chunk_type == "financial_metric" and query_has_any(
        query_tokens,
        "revenue",
        "ebitda",
        "margin",
        "operating_margin",
        "r_and_d",
        "研发",
        "收入",
        "gpu",
        "利润率",
    ):
        overlap += 0.12
    if chunk_type == "risk_signal" and query_has_any(
        query_tokens,
        "risk",
        "supply",
        "supply_chain",
        "风险",
        "供应链",
        "capex",
    ):
        overlap += 0.12
    # Prefer hybrid evidence slightly when already fused.
    if str(hit.get("retrieval_method") or "") == "hybrid_rrf":
        overlap += 0.03
    prior = float(hit.get("fusion_score") or hit.get("score") or 0.0)
    # Blend lexical relevance with prior retrieval score (scale prior into ~0-1 band).
    prior_norm = min(1.0, prior if prior <= 1.0 else prior / 10.0)
    return min(1.0, overlap * 0.85 + prior_norm * 0.15)


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if not hits:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for hit in hits:
        score = lexical_rerank_score(query, hit)
        enriched = dict(hit)
        enriched["rerank_score"] = round(score, 6)
        enriched["retrieval_method"] = (
            f"{hit.get('retrieval_method') or 'retrieve'}+rerank"
            if hit.get("retrieval_method")
            else "rerank"
        )
        scored.append((score, enriched))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[: max(1, top_k)]]
