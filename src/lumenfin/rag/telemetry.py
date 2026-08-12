"""RAG observability helpers for retrieval/index telemetry."""

from __future__ import annotations

from typing import Any


def summarize_rag_telemetry(
    *,
    rag_index_stats: dict[str, Any] | None,
    company_metas: list[dict[str, Any]],
    sanitized_finding_count: int = 0,
) -> dict[str, Any]:
    """Aggregate per-company retrieve meta + index stats into one telemetry block."""
    stats = dict(rag_index_stats or {})
    vector_hits = sum(int(meta.get("vector_hits") or 0) for meta in company_metas)
    bm25_hits = sum(int(meta.get("bm25_hits") or 0) for meta in company_metas)
    keyword_hits = sum(int(meta.get("keyword_hits") or 0) for meta in company_metas)
    lexical_fallback_hits = sum(
        int(meta.get("lexical_fallback_hits") or 0) for meta in company_metas
    )
    degraded = bool(stats.get("rag_degraded")) or any(bool(meta.get("degraded")) for meta in company_metas)
    modes = sorted({str(meta.get("mode") or "") for meta in company_metas if meta.get("mode")})
    rerank_providers = sorted(
        {
            str(meta.get("rerank_provider") or "")
            for meta in company_metas
            if meta.get("rerank_provider")
        }
    )
    rerank_requested_providers = sorted(
        {
            str(meta.get("rerank_requested_provider") or "")
            for meta in company_metas
            if meta.get("rerank_requested_provider")
        }
    )
    rerank_models = sorted(
        {
            str(meta.get("rerank_model") or "")
            for meta in company_metas
            if meta.get("rerank_model")
        }
    )
    rerank_error_types = sorted(
        {
            str(meta.get("rerank_error_type") or "")
            for meta in company_metas
            if meta.get("rerank_error_type")
        }
    )
    # Primary mode for dashboards: prefer hybrid/rerank over keyword-only when mixed.
    mode = _primary_retrieve_mode(modes)
    return {
        "index_status": _index_status(stats),
        "chunks_indexed": int(stats.get("chunks_indexed") or 0),
        "documents_indexed": int(stats.get("documents_indexed") or 0),
        "search_only": bool(stats.get("search_only")),
        "embed_ms": float(stats.get("embed_ms") or 0.0),
        "embed_chars": int(stats.get("embed_chars") or 0),
        "embed_calls": int(stats.get("embed_calls") or 0),
        "vector_hits": vector_hits,
        "bm25_hits": bm25_hits,
        "keyword_hits": keyword_hits,
        "lexical_fallback_hits": lexical_fallback_hits,
        "degraded": degraded,
        "degraded_companies": list(stats.get("degraded_companies") or []),
        "mode": mode,
        "retrieve_modes": modes,
        "rerank_providers": rerank_providers,
        "rerank_requested_providers": rerank_requested_providers,
        "rerank_models": rerank_models,
        "rerank_latency_ms": round(
            sum(float(meta.get("rerank_latency_ms") or 0.0) for meta in company_metas),
            2,
        ),
        "rerank_attempts": sum(
            int(meta.get("rerank_attempts") or 0) for meta in company_metas
        ),
        "rerank_tokens": sum(
            int(meta.get("rerank_tokens") or 0) for meta in company_metas
        ),
        "rerank_fallbacks": sum(
            1 for meta in company_metas if bool(meta.get("rerank_fallback"))
        ),
        "rerank_error_types": rerank_error_types,
        "sanitized_finding_count": int(sanitized_finding_count),
        "company_count": len(company_metas),
    }


def _primary_retrieve_mode(modes: list[str]) -> str | None:
    if not modes:
        return None
    priority = (
        "hybrid_dense_bm25_rrf+qwen3_rerank",
        "hybrid_dense_bm25_rrf+lexical_rerank_fallback",
        "hybrid_dense_bm25_rrf+lexical_rerank",
        "hybrid_dense_bm25_rrf+rerank",
        "hybrid_dense_bm25_rrf",
        "bm25_only+qwen3_rerank",
        "bm25_only+lexical_rerank_fallback",
        "bm25_only+lexical_rerank",
        "bm25_only+rerank",
        "bm25_only",
        "bm25_only_degraded+qwen3_rerank",
        "bm25_only_degraded+lexical_rerank_fallback",
        "bm25_only_degraded+lexical_rerank",
        "bm25_only_degraded+rerank",
        "bm25_only_degraded",
        "hybrid_dense_lexical_fallback_rrf_degraded+qwen3_rerank",
        "hybrid_dense_lexical_fallback_rrf_degraded+lexical_rerank_fallback",
        "hybrid_dense_lexical_fallback_rrf_degraded+lexical_rerank",
        "hybrid_dense_lexical_fallback_rrf_degraded+rerank",
        "hybrid_dense_lexical_fallback_rrf_degraded",
        "hybrid_rrf+qwen3_rerank",
        "hybrid_rrf+lexical_rerank_fallback",
        "hybrid_rrf+lexical_rerank",
        "hybrid_rrf+rerank",
        "hybrid_rrf",
        "vector_only+qwen3_rerank",
        "vector_only+lexical_rerank_fallback",
        "vector_only+lexical_rerank",
        "vector_only+rerank",
        "vector_only",
        "keyword_only+qwen3_rerank",
        "keyword_only+lexical_rerank_fallback",
        "keyword_only+lexical_rerank",
        "keyword_only+rerank",
        "keyword_only",
        "keyword_only_degraded+qwen3_rerank",
        "keyword_only_degraded+lexical_rerank_fallback",
        "keyword_only_degraded+lexical_rerank",
        "keyword_only_degraded+rerank",
        "keyword_only_degraded",
    )
    for candidate in priority:
        if candidate in modes:
            return candidate
    # Prefer any mode that already includes rerank, else first sorted.
    for mode in modes:
        if "rerank" in mode:
            return mode
    return modes[0]


def _index_status(stats: dict[str, Any]) -> str:
    if stats.get("rag_degraded"):
        return "degraded"
    if stats.get("search_only"):
        return "ready_search_only"
    if int(stats.get("chunks_indexed") or 0) > 0:
        return "indexed"
    if stats.get("documents_failed"):
        return "failed"
    return "skipped"


def evaluate_rag_gates(
    summary: dict[str, Any],
    *,
    min_pass_rate: float = 1.0,
    min_mean_recall_at_3: float = 1.0,
    min_mean_citation_coverage: float = 1.0,
    min_mean_mrr: float = 0.0,
    min_mean_groundedness: float = 0.0,
) -> dict[str, Any]:
    """Compare rag eval summary against release thresholds."""
    checks = {
        "pass_rate": {
            "actual": float(summary.get("pass_rate") or 0.0),
            "minimum": float(min_pass_rate),
        },
        "mean_recall_at_3": {
            "actual": float(summary.get("mean_recall_at_3") or 0.0),
            "minimum": float(min_mean_recall_at_3),
        },
        "mean_citation_coverage": {
            "actual": float(summary.get("mean_citation_coverage") or 0.0),
            "minimum": float(min_mean_citation_coverage),
        },
        "mean_mrr": {
            "actual": float(summary.get("mean_mrr") or 0.0),
            "minimum": float(min_mean_mrr),
        },
        "mean_groundedness": {
            "actual": float(summary.get("mean_groundedness") or 0.0),
            "minimum": float(min_mean_groundedness),
        },
    }
    failures = [
        name
        for name, item in checks.items()
        if item["actual"] + 1e-9 < item["minimum"]
    ]
    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
    }
