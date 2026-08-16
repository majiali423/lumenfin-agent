"""Per-case retrieval scoring for FinanceBench (page + chunk qrels)."""

from __future__ import annotations

from typing import Any, Sequence

from .constants import CHUNK_K_VALUES, PAGE_K_VALUES
from .metrics import hit_at_k, mean_reciprocal_rank, ndcg_at_k, recall_at_k
from .qrels import retrieved_page_keys
from .schema import CaseQrels, FinanceBenchQuestion
from .taxonomy import classify_case, classify_failure


def first_hit_rank(retrieved: Sequence[object], relevant: set[object]) -> int | None:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return rank
    return None


def _metric_block(
    retrieved: Sequence[object],
    relevant: set[object],
    k_values: Sequence[int],
    *,
    ndcg_ks: Sequence[int],
) -> dict[str, Any]:
    first = first_hit_rank(retrieved, relevant)
    return {
        "hit_at": {str(k): round(hit_at_k(retrieved, relevant, k=k), 4) for k in k_values},
        "recall_at": {str(k): round(recall_at_k(retrieved, relevant, k=k), 4) for k in k_values},
        "mrr": round(mean_reciprocal_rank(retrieved, relevant), 4),
        "ndcg_at": {str(k): round(ndcg_at_k(retrieved, relevant, k=k), 4) for k in ndcg_ks},
        "first_relevant_rank": first or 0,
    }


def public_hit_row(hit: dict[str, Any]) -> dict[str, Any]:
    """Citation metadata only — no passage text."""
    page = hit.get("page")
    try:
        page_i = int(page) if page is not None and page != "" else None
    except (TypeError, ValueError):
        page_i = None
    filename = str(hit.get("filename") or hit.get("document_id") or "")
    citation = str(hit.get("citation") or "")
    if not citation and filename and page_i is not None:
        citation = f"{filename}#p{page_i}"
    return {
        "chunk_id": str(hit.get("chunk_id") or ""),
        "document_id": str(hit.get("document_id") or ""),
        "filename": filename,
        "page": page_i,
        "citation": citation,
        "retrieval_method": str(hit.get("retrieval_method") or ""),
        "score": hit.get("fusion_score", hit.get("rerank_score", hit.get("score"))),
    }


def score_retrieval_case(
    *,
    question: FinanceBenchQuestion,
    qrels: CaseQrels,
    hits: list[dict[str, Any]],
    mode: str,
    retrieval_meta: dict[str, Any] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    meta = dict(retrieval_meta or {})
    retrieved_chunk_ids = [str(hit.get("chunk_id") or "") for hit in hits if hit.get("chunk_id")]
    page_keys = retrieved_page_keys(hits)
    gold_pages = {page.key for page in qrels.gold_pages}
    gold_chunks = set(qrels.gold_chunk_ids)
    page_block = _metric_block(page_keys, gold_pages, PAGE_K_VALUES, ndcg_ks=(5, 10))
    chunk_block = _metric_block(retrieved_chunk_ids, gold_chunks, CHUNK_K_VALUES, ndcg_ks=(10,))
    failure = classify_failure(
        retrieved_pages=page_keys,
        gold_pages=gold_pages,
        top_k=top_k,
        empty=not hits,
        provider_error=str(meta.get("error_type") or ""),
        degraded=bool(meta.get("degraded") or meta.get("rerank_fallback")),
    )
    return {
        "case_id": question.case_id,
        "financebench_id": question.financebench_id,
        "mode": mode,
        "status": "ok" if not meta.get("error_type") else "degraded",
        "company": question.company,
        "doc_name": question.doc_name,
        "single_gold_page": qrels.single_gold_page,
        "page_provenance_ok": qrels.page_provenance_ok,
        "labels": classify_case(question),
        "page": page_block,
        "chunk": chunk_block,
        "failure_class": failure,
        "gold_pages": [
            {
                "doc_name": page.doc_name,
                "page_one": page.page_one,
                "page_zero": page.page_zero,
            }
            for page in qrels.gold_pages
        ],
        "gold_chunk_ids": list(qrels.gold_chunk_ids),
        "retrieved": [public_hit_row(hit) for hit in hits],
        "qrel_notes": list(qrels.notes),
        "retrieval": {
            "degraded": bool(meta.get("degraded")),
            "rerank_fallback": bool(meta.get("rerank_fallback")),
            "rerank_provider": str(meta.get("rerank_provider") or ""),
            "error_type": str(meta.get("error_type") or ""),
            "latency_ms": meta.get("latency_ms"),
            "hit_count": meta.get("hit_count", len(hits)),
            "retrieval_methods": list(meta.get("retrieval_methods") or []),
        },
    }
