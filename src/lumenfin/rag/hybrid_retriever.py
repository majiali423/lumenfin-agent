from __future__ import annotations

import re
from typing import Any

from .milvus_store import MilvusRAGStore


def _keyword_score(query: str, text: str, chunk_type: str) -> float:
    query_tokens = set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", query.lower()))
    text_tokens = set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower()))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens) / len(query_tokens)
    if chunk_type == "financial_metric" and any(token in query_tokens for token in ("revenue", "ebitda", "margin", "研发", "收入")):
        overlap += 0.15
    if chunk_type == "risk_signal" and any(token in query_tokens for token in ("risk", "supply", "风险", "供应链")):
        overlap += 0.15
    return min(overlap, 1.0)


def _keyword_search(
    document_contexts: list[dict[str, Any]],
    *,
    company: str,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    from .chunking import chunk_document

    scored: list[tuple[float, dict[str, Any]]] = []
    for document in document_contexts:
        if document.get("detected_companies") and company not in document.get("detected_companies", []):
            continue
        for chunk in chunk_document(document):
            score = _keyword_score(query, chunk["text"], chunk["chunk_type"])
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    {
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "filename": chunk["filename"],
                        "page": chunk["page"],
                        "text": chunk["text"],
                        "companies": chunk.get("companies", []),
                        "chunk_type": chunk["chunk_type"],
                        "score": score,
                        "retrieval_method": "keyword",
                        "citation": f"{chunk['filename']}#p{chunk['page']}",
                    },
                )
            )
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


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


class HybridEvidenceRetriever:
    """Vector + keyword fusion tailored for financial diligence queries."""

    def __init__(self, rag_store: MilvusRAGStore | None, *, top_k: int = 5) -> None:
        self.rag_store = rag_store
        self.top_k = top_k

    def retrieve_for_company(
        self,
        *,
        query: str,
        company: str,
        session_id: str,
        document_contexts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        keyword_hits = _keyword_search(
            document_contexts,
            company=company,
            query=query,
            top_k=self.top_k,
        )
        if not self.rag_store or not document_contexts:
            return keyword_hits[: self.top_k]

        vector_hits = self.rag_store.vector_search(
            query,
            session_id=session_id,
            companies=[company],
            top_k=self.top_k,
        )
        if not vector_hits:
            return keyword_hits[: self.top_k]
        if not keyword_hits:
            return vector_hits[: self.top_k]
        return reciprocal_rank_fusion([vector_hits, keyword_hits])[: self.top_k]

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
                }
            )
        return documents
