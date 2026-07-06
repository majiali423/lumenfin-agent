"""Document search for MCP: keyword notes and optional Milvus hybrid RAG."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .doc_context import DOCS_DIR, load_research_document_contexts, resolve_company_hint

_SESSION_ID = "mcp-document-search"


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[\w\u4e00-\u9fff]+", text) if len(token) > 1}


def _keyword_search(query: str, top_k: int, docs_dir: Path) -> dict[str, Any]:
    root = docs_dir
    if not root.exists():
        return {"query": query, "hits": [], "source": "mcp_layer.data.docs", "warning": "docs directory missing"}

    query_tokens = _tokenize(query)
    hits: list[dict[str, Any]] = []

    for path in sorted(root.glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        for paragraph in paragraphs:
            paragraph_tokens = _tokenize(paragraph)
            if not query_tokens:
                continue
            overlap = len(query_tokens & paragraph_tokens)
            substring_hit = bool(query.strip() and query.strip() in paragraph)
            if overlap == 0 and not substring_hit:
                continue
            score = 1.0 if substring_hit else overlap / max(1, len(query_tokens))
            hits.append(
                {
                    "doc": str(path.relative_to(root)),
                    "snippet": paragraph[:400],
                    "score": round(score, 3),
                    "retrieval_method": "keyword",
                }
            )

    hits.sort(key=lambda item: item["score"], reverse=True)
    return {
        "query": query,
        "hits": hits[:top_k],
        "source": "mcp_layer.data.docs",
        "retrieval_mode": "keyword",
        "total_candidates": len(hits),
    }


def _hybrid_search(query: str, top_k: int, company: str | None, docs_dir: Path) -> dict[str, Any]:
    from lumenfin.config import AppConfig
    from lumenfin.rag.factory import build_hybrid_retriever

    config = AppConfig.from_env()
    retriever = build_hybrid_retriever(config)
    if retriever is None or retriever.rag_store is None:
        payload = _keyword_search(query, top_k, docs_dir)
        payload["retrieval_mode"] = "keyword_fallback"
        payload["warning"] = "Milvus RAG unavailable; fell back to keyword search."
        return payload

    document_contexts = load_research_document_contexts(docs_dir)
    if not document_contexts:
        return _keyword_search(query, top_k, docs_dir)

    company_name = company or resolve_company_hint(query)
    if not company_name:
        for context in document_contexts:
            if context.get("detected_companies"):
                company_name = context["detected_companies"][0]
                break
    company_name = company_name or "Apple"
    retriever.rag_store.index_documents(document_contexts, session_id=_SESSION_ID)
    hits = retriever.retrieve_for_company(
        query=query,
        company=company_name,
        session_id=_SESSION_ID,
        document_contexts=document_contexts,
    )
    formatted = [
        {
            "doc": hit.get("citation") or hit.get("filename"),
            "snippet": hit.get("text", "")[:400],
            "score": hit.get("fusion_score", hit.get("score")),
            "retrieval_method": hit.get("retrieval_method", "hybrid_rrf"),
            "page": hit.get("page"),
        }
        for hit in hits[:top_k]
    ]
    return {
        "query": query,
        "company": company_name,
        "hits": formatted,
        "source": "lumenfin.milvus_hybrid",
        "retrieval_mode": "milvus_hybrid",
        "session_id": _SESSION_ID,
        "total_candidates": len(formatted),
    }


def search_research_documents(
    query: str,
    top_k: int = 3,
    *,
    company: str | None = None,
    docs_dir: Path | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    root = docs_dir or DOCS_DIR
    resolved_mode = (mode or os.getenv("MAS_MCP_DOC_SEARCH", "auto")).lower()
    if resolved_mode == "keyword":
        return _keyword_search(query, top_k, root)
    if resolved_mode in {"milvus", "hybrid"}:
        return _hybrid_search(query, top_k, company, root)

    try:
        return _hybrid_search(query, top_k, company, root)
    except Exception:
        payload = _keyword_search(query, top_k, root)
        payload["retrieval_mode"] = "keyword_fallback"
        payload["warning"] = "Hybrid search failed; fell back to keyword search."
        return payload
