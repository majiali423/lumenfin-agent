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
    keyword_hits = sum(int(meta.get("keyword_hits") or 0) for meta in company_metas)
    degraded = bool(stats.get("rag_degraded")) or any(bool(meta.get("degraded")) for meta in company_metas)
    modes = sorted({str(meta.get("mode") or "") for meta in company_metas if meta.get("mode")})
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
        "keyword_hits": keyword_hits,
        "degraded": degraded,
        "degraded_companies": list(stats.get("degraded_companies") or []),
        "mode": mode,
        "retrieve_modes": modes,
        "sanitized_finding_count": int(sanitized_finding_count),
        "company_count": len(company_metas),
    }


def _primary_retrieve_mode(modes: list[str]) -> str | None:
    if not modes:
        return None
    priority = (
        "hybrid_rrf+rerank",
        "hybrid_rrf",
        "vector_only+rerank",
        "vector_only",
        "keyword_only+rerank",
        "keyword_only",
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
