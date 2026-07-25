from __future__ import annotations

import logging
from typing import Any

from .lexical import lexical_overlap, query_has_any, tokenize_text
from .milvus_store import EmbeddingQueryError, MilvusRAGStore
from .rerank import rerank_hits
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


_NUMERIC_QUERY_HINTS = (
    "revenue",
    "net sales",
    "ebitda",
    "margin",
    "income",
    "eps",
    "debt",
    "cash flow",
    "operating",
    "capex",
    "capital expenditure",
    "r&d",
    "研发",
    "收入",
    "营收",
    "利润率",
)

_METRIC_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "net sales", "sales", "营收", "收入", "净销售"),
    "gross_margin": ("gross margin", "gross profit", "毛利率"),
    "operating_margin": ("operating margin", "营业利润率"),
    "operating_income": ("operating income", "income from operations", "营业利润"),
    "net_income": ("net income", "net earnings", "净利润"),
    "eps": ("eps", "earnings per share", "per share"),
    "ebitda": ("ebitda",),
    "r_and_d": ("r&d", "r & d", "research and development", "研发"),
    "debt": ("debt", "长期债务"),
    "operating_cash_flow": ("operating cash", "cash from operations", "cash provided by operating", "经营活动"),
    "capex": ("capex", "capital expenditure", "property, plant", "资本开", "资本支出"),
    "cash": ("cash and cash equivalents", "cash equivalents", "现金及现金"),
}


def _is_numeric_financial_query(query: str) -> bool:
    lowered = (query or "").lower()
    return any(hint in lowered for hint in _NUMERIC_QUERY_HINTS)


def _metric_matches_query(metric: str, query: str) -> bool:
    q = (query or "").lower()
    aliases = _METRIC_QUERY_ALIASES.get(metric) or (metric.replace("_", " "),)
    return any(alias in q for alias in aliases)


def _keyword_score(
    query: str,
    text: str,
    chunk_type: str,
    *,
    financial_fact: dict[str, Any] | None = None,
) -> float:
    overlap = lexical_overlap(query, text)
    if overlap <= 0.0 and not financial_fact:
        return 0.0
    query_tokens = tokenize_text(query)
    if chunk_type == "financial_metric" and query_has_any(
        query_tokens,
        "revenue",
        "ebitda",
        "margin",
        "operating_margin",
        "r_and_d",
        "研发",
        "收入",
        "利润率",
        "net_income",
        "gross_margin",
        "eps",
        "debt",
        "capex",
    ):
        overlap += 0.15
    if financial_fact and _is_numeric_financial_query(query):
        metric = str(financial_fact.get("metric") or "")
        # Only strongly boost facts whose metric matches the query intent
        # (prevents revenue totals winning debt/capex probes).
        if metric and _metric_matches_query(metric, query):
            # Floor so compact fact chunks beat long thematic narrative.
            overlap = max(overlap, 0.55)
            overlap += 0.4
            if str(financial_fact.get("value") or ""):
                overlap += 0.1
            scope = str(financial_fact.get("scope") or "").lower()
            if scope == "consolidated":
                overlap += 0.22
            elif scope == "segment":
                overlap += 0.02
            label = str(financial_fact.get("row_label") or financial_fact.get("alias") or "").lower()
            if "total" in label:
                overlap += 0.08
            if financial_fact.get("source") == "html_table":
                overlap += 0.05
        elif metric:
            # Soft penalty when fact metric conflicts with query
            overlap *= 0.35
    if chunk_type == "risk_signal" and query_has_any(
        query_tokens,
        "risk",
        "supply",
        "supply_chain",
        "风险",
        "供应链",
    ):
        overlap += 0.15
    return min(overlap, 1.0)


def _hits_from_scored_chunks(
    chunks: list[dict[str, Any]],
    *,
    company: str,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        companies = list(chunk.get("companies") or [])
        if companies and company not in companies:
            continue
        fact = chunk.get("financial_fact") if isinstance(chunk.get("financial_fact"), dict) else None
        score = _keyword_score(
            query,
            str(chunk.get("text") or ""),
            str(chunk.get("chunk_type") or "narrative"),
            financial_fact=fact,
        )
        if score <= 0:
            continue
        scored.append(
            (
                score,
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "source_document_id": chunk.get("source_document_id"),
                    "filename": chunk["filename"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "companies": companies,
                    "chunk_type": chunk.get("chunk_type", "narrative"),
                    "financial_fact": fact,
                    "score": score,
                    "retrieval_method": "keyword",
                    "citation": f"{chunk['filename']}#p{chunk['page']}",
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def _keyword_search(
    document_contexts: list[dict[str, Any]],
    *,
    company: str,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    from .chunking import chunk_document

    chunks: list[dict[str, Any]] = []
    for document in document_contexts:
        doc_companies = list(
            document.get("issuer_companies")
            or document.get("detected_companies")
            or []
        )
        if doc_companies and company not in doc_companies:
            continue
        for chunk in chunk_document(document):
            chunks.append(chunk)
    return _hits_from_scored_chunks(chunks, company=company, query=query, top_k=top_k)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int = 60,
) -> list[dict[str, Any]]:
    fused_scores: dict[str, float] = {}
    payload_by_id: dict[str, dict[str, Any]] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            chunk_id = item["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            payload_by_id.setdefault(chunk_id, item)
    ordered = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    merged: list[dict[str, Any]] = []
    for chunk_id, score in ordered:
        hit = dict(payload_by_id[chunk_id])
        hit["fusion_score"] = round(score, 6)
        hit["retrieval_method"] = "hybrid_rrf"
        merged.append(hit)
    return merged


def _apply_min_score(hits: list[dict[str, Any]], min_score: float) -> list[dict[str, Any]]:
    if min_score <= 0:
        return hits
    kept: list[dict[str, Any]] = []
    for hit in hits:
        method = str(hit.get("retrieval_method") or "")
        if method == "hybrid_rrf":
            # RRF scores are small (~0.01–0.03); do not apply keyword-scale thresholds.
            kept.append(hit)
            continue
        score = float(hit.get("score") or 0.0)
        if score >= min_score:
            kept.append(hit)
    return kept


class HybridEvidenceRetriever:
    """Vector + keyword fusion tailored for financial diligence queries."""

    def __init__(
        self,
        rag_store: VectorStore | None,
        *,
        top_k: int = 5,
        chunk_loader: Any | None = None,
        min_score: float = 0.0,
        degrade_on_vector_error: bool = True,
        rerank_enabled: bool = False,
        rerank_candidates: int = 20,
    ) -> None:
        self.rag_store = rag_store
        self.top_k = top_k
        self.chunk_loader = chunk_loader
        self.min_score = float(min_score or 0.0)
        self.degrade_on_vector_error = bool(degrade_on_vector_error)
        self.rerank_enabled = bool(rerank_enabled)
        self.rerank_candidates = max(int(rerank_candidates or top_k), int(top_k))

    def retrieve_for_company(
        self,
        *,
        query: str,
        company: str,
        session_id: str,
        document_contexts: list[dict[str, Any]],
        tenant_id: str | None = None,
        source_document_ids: list[str] | None = None,
        use_stored_chunks: bool = False,
    ) -> list[dict[str, Any]]:
        hits, _meta = self.retrieve_for_company_with_meta(
            query=query,
            company=company,
            session_id=session_id,
            document_contexts=document_contexts,
            tenant_id=tenant_id,
            source_document_ids=source_document_ids,
            use_stored_chunks=use_stored_chunks,
        )
        return hits

    def retrieve_for_company_with_meta(
        self,
        *,
        query: str,
        company: str,
        session_id: str,
        document_contexts: list[dict[str, Any]],
        tenant_id: str | None = None,
        source_document_ids: list[str] | None = None,
        use_stored_chunks: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidate_k = self.rerank_candidates if self.rerank_enabled else self.top_k
        stored_chunks: list[dict[str, Any]] | None = None
        if use_stored_chunks and self.chunk_loader and source_document_ids:
            stored_chunks = list(
                self.chunk_loader(
                    tenant_id=tenant_id or session_id,
                    source_document_ids=source_document_ids,
                )
                or []
            )

        if stored_chunks is not None:
            keyword_hits = _hits_from_scored_chunks(
                stored_chunks,
                company=company,
                query=query,
                top_k=candidate_k,
            )
        else:
            keyword_hits = _keyword_search(
                document_contexts,
                company=company,
                query=query,
                top_k=candidate_k,
            )

        meta: dict[str, Any] = {
            "degraded": False,
            "degrade_reason": "",
            "mode": "keyword_only",
            "vector_hits": 0,
            "keyword_hits": len(keyword_hits),
            "filtered_by_min_score": 0,
            "rerank_enabled": self.rerank_enabled,
            "rerank_candidates": candidate_k if self.rerank_enabled else 0,
        }

        def _finalize(hits: list[dict[str, Any]], *, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            selected = hits
            if self.rerank_enabled and selected:
                selected = rerank_hits(query, selected, top_k=self.top_k)
                meta["mode"] = f"{mode}+rerank"
            else:
                selected = selected[: self.top_k]
                meta["mode"] = mode
            filtered = _apply_min_score(selected, self.min_score)
            meta["filtered_by_min_score"] = max(0, len(selected) - len(filtered))
            return filtered, meta

        if not self.rag_store:
            return _finalize(keyword_hits, mode="keyword_only")
        if not document_contexts and not source_document_ids:
            return _finalize(keyword_hits, mode="keyword_only")

        try:
            vector_hits = self.rag_store.vector_search(
                query,
                session_id=None if (tenant_id and source_document_ids) else session_id,
                tenant_id=tenant_id,
                source_document_ids=source_document_ids,
                companies=[company],
                top_k=candidate_k,
            )
            # Shared path with agent: if company-tagged filter yields empty, retry
            # without company filter then post-filter (issuer tags may be sparse).
            if not vector_hits:
                broadened = self.rag_store.vector_search(
                    query,
                    session_id=None if (tenant_id and source_document_ids) else session_id,
                    tenant_id=tenant_id,
                    source_document_ids=source_document_ids,
                    companies=None,
                    top_k=max(candidate_k * 2, candidate_k),
                )
                vector_hits = [
                    hit
                    for hit in broadened
                    if company in list(hit.get("companies") or [])
                    or company.lower() in str(hit.get("text") or "").lower()
                    or not hit.get("companies")
                ][:candidate_k]
                if vector_hits:
                    meta["company_filter_relaxed"] = True
        except Exception as exc:
            if not self.degrade_on_vector_error:
                raise
            reason = str(exc)
            logger.warning(
                "Vector search failed for company=%s; falling back to keyword-only: %s",
                company,
                reason,
            )
            meta.update(
                {
                    "degraded": True,
                    "degrade_reason": reason[:500],
                    "error_type": type(exc).__name__,
                }
            )
            hits, meta = _finalize(keyword_hits, mode="keyword_only_degraded")
            for hit in hits:
                hit["rag_degraded"] = True
            meta["keyword_hits"] = len(hits)
            return hits, meta

        meta["vector_hits"] = len(vector_hits)
        if not vector_hits:
            if self.rag_store is not None:
                logger.warning(
                    "RAG mode mismatch risk: vector_hits=0 for company=%s tenant=%s; "
                    "falling back to keyword_only (agent/showcase expect hybrid_rrf when indexed).",
                    company,
                    tenant_id or session_id,
                )
            return _finalize(keyword_hits, mode="keyword_only")
        if not keyword_hits:
            return _finalize(vector_hits, mode="vector_only")

        fused = reciprocal_rank_fusion([vector_hits, keyword_hits])[:candidate_k]
        return _finalize(fused, mode="hybrid_rrf")

    def build_source_documents(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for hit in hits:
            documents.append(
                {
                    "document_id": hit.get("document_id"),
                    "filename": hit.get("filename"),
                    "page": hit.get("page"),
                    "excerpt": hit.get("text", "")[:1200],
                    "citation": hit.get("citation"),
                    "chunk_type": hit.get("chunk_type"),
                    "retrieval_method": hit.get("retrieval_method"),
                    "fusion_score": hit.get("fusion_score", hit.get("score")),
                    "rerank_score": hit.get("rerank_score"),
                    "rag_degraded": bool(hit.get("rag_degraded")),
                }
            )
        return documents


# Re-export for callers/tests that catch embed failures via store path.
__all__ = [
    "EmbeddingQueryError",
    "HybridEvidenceRetriever",
    "reciprocal_rank_fusion",
]
