from __future__ import annotations

import os
import time
from typing import Any

from ...rag.embeddings import build_embedding_provider
from ...rag.hybrid_retriever import HybridEvidenceRetriever
from ...rag.milvus_store import MilvusRAGStore
from ...rag.rerank import build_reranker
from .constants import (
    DEFAULT_BM25_RRF_WEIGHT,
    DEFAULT_RERANK_CANDIDATES,
    EVAL_ANCHOR_DOCUMENT,
    EVAL_COLLECTION,
    EVAL_COMPANY_TAG,
    EVAL_SESSION_ID,
    REMOTE_EMBEDDING_PROVIDERS,
    REMOTE_MODES,
    RETRIEVAL_MODES,
)


class RemoteEvalBlocked(RuntimeError):
    """Raised when a remote provider would be called without --allow-remote."""


def require_allow_remote(*, mode: str, embedding_provider: str, allow_remote: bool) -> None:
    needs_remote = (
        mode in REMOTE_MODES or embedding_provider.strip().lower() in REMOTE_EMBEDDING_PROVIDERS
    )
    if needs_remote and not allow_remote:
        raise RemoteEvalBlocked(
            f"mode={mode!r} embedding_provider={embedding_provider!r} requires explicit --allow-remote"
        )


def build_eval_store(
    *,
    uri: str,
    embedding_provider: str,
    embedding_dimension: int,
    collection_name: str = EVAL_COLLECTION,
    allow_remote: bool,
    mode: str,
) -> MilvusRAGStore:
    require_allow_remote(mode=mode, embedding_provider=embedding_provider, allow_remote=allow_remote)
    embedder = build_embedding_provider(embedding_provider, embedding_dimension)
    return MilvusRAGStore(uri, embedder, collection_name=collection_name, bm25_enabled=True)


def iter_indexed_chunks(store: Any, *, session_id: str = EVAL_SESSION_ID) -> list[dict[str, Any]]:
    """Return indexed chunk metadata for qrel mapping. Never logs document bodies."""
    custom = getattr(store, "iter_eval_chunks", None)
    if callable(custom):
        return list(custom())
    client = getattr(store, "client", None)
    collection = getattr(store, "collection_name", "")
    if client is None or not collection:
        return []
    fields = [
        "chunk_id",
        "document_id",
        "source_document_id",
        "filename",
        "page",
        "text",
        "companies",
        "primary_company",
    ]
    chunks: list[dict[str, Any]] = []
    page_size = 1000
    offset = 0
    filter_expr = f'session_id == "{session_id}"'
    while True:
        try:
            batch = client.query(
                collection_name=collection,
                filter=filter_expr,
                output_fields=fields,
                limit=page_size,
                offset=offset,
            )
        except Exception:
            if offset == 0:
                batch = client.query(
                    collection_name=collection,
                    filter=filter_expr,
                    output_fields=fields,
                    limit=page_size,
                )
            else:
                break
        if not batch:
            break
        chunks.extend(list(batch))
        if len(batch) < page_size:
            break
        offset += len(batch)
        if offset > 200_000:
            break
    return chunks


def _qwen3_reranker() -> Any:
    return build_reranker(
        "qwen3",
        model=os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank"),
        base_url=os.getenv("DASHSCOPE_RERANK_BASE_URL", ""),
        instruct=os.getenv(
            "MAS_RAG_RERANK_INSTRUCT",
            "Given a financial due diligence query, retrieve passages that directly answer it. "
            "Prefer the correct company, reporting period, metric, scope, and filing context over "
            "merely topical passages.",
        ),
        timeout_seconds=float(os.getenv("MAS_RAG_RERANK_TIMEOUT_SECONDS", "12")),
        max_attempts=int(os.getenv("MAS_RAG_RERANK_MAX_ATTEMPTS", "2")),
        backoff_seconds=float(os.getenv("MAS_RAG_RERANK_BACKOFF_SECONDS", "0.25")),
        max_inflight=int(os.getenv("MAS_RAG_RERANK_MAX_INFLIGHT_PER_PROCESS", "2")),
        max_document_chars=int(os.getenv("MAS_RAG_RERANK_MAX_DOCUMENT_CHARS", "4000")),
    )


def retrieve_for_mode(
    *,
    mode: str,
    store: MilvusRAGStore,
    query: str,
    company: str = EVAL_COMPANY_TAG,
    session_id: str = EVAL_SESSION_ID,
    document_contexts: list[dict[str, Any]] | None = None,
    top_k: int,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    bm25_rrf_weight: float = DEFAULT_BM25_RRF_WEIGHT,
    allow_remote: bool = False,
    embedding_provider: str = "deterministic",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unsupported retrieval mode {mode!r}")
    require_allow_remote(mode=mode, embedding_provider=embedding_provider, allow_remote=allow_remote)
    contexts = document_contexts if document_contexts is not None else [dict(EVAL_ANCHOR_DOCUMENT)]
    meta: dict[str, Any] = {
        "mode": mode,
        "degraded": False,
        "rerank_calls": 0,
        "rerank_fallback": False,
        "rerank_tokens": 0,
        "error_type": "",
        "retrieval_methods": [],
    }
    started = time.perf_counter()
    hits: list[dict[str, Any]] = []
    try:
        if mode == "bm25":
            hits = store.bm25_search(query, session_id=session_id, companies=[company], top_k=top_k)
        elif mode == "dense":
            hits = store.vector_search(query, session_id=session_id, companies=[company], top_k=top_k)
        elif mode == "hybrid":
            retriever = HybridEvidenceRetriever(
                store, top_k=top_k, rerank_enabled=False, bm25_rrf_weight=bm25_rrf_weight
            )
            hits, hybrid_meta = retriever.retrieve_for_company_with_meta(
                query=query,
                company=company,
                session_id=session_id,
                document_contexts=contexts,
            )
            meta.update(hybrid_meta)
        else:
            retriever = HybridEvidenceRetriever(
                store,
                top_k=top_k,
                rerank_enabled=True,
                rerank_candidates=max(rerank_candidates, top_k),
                reranker=_qwen3_reranker(),
                bm25_rrf_weight=bm25_rrf_weight,
            )
            hits, hybrid_meta = retriever.retrieve_for_company_with_meta(
                query=query,
                company=company,
                session_id=session_id,
                document_contexts=contexts,
            )
            meta.update(hybrid_meta)
            meta["rerank_calls"] = 1
    except Exception as exc:  # noqa: BLE001
        meta["degraded"] = True
        meta["error_type"] = type(exc).__name__
        meta["error"] = str(exc)
        hits = []
    meta["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    meta["retrieval_methods"] = sorted(
        {str(hit.get("retrieval_method") or "") for hit in hits if hit.get("retrieval_method")}
    )
    meta["hit_count"] = len(hits)
    return hits, meta
