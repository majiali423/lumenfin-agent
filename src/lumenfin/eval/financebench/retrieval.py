from __future__ import annotations

import os
import time
from typing import Any

from ...rag.embeddings import DashScopeEmbeddingProvider, ResilientEmbeddingProvider, build_embedding_provider
from ...rag.hybrid_retriever import HybridEvidenceRetriever, reciprocal_rank_fusion
from ...rag.milvus_store import MilvusRAGStore
from ...rag.rerank import DEFAULT_RERANK_INSTRUCT, build_reranker
from .constants import (
    DEFAULT_BM25_RRF_WEIGHT,
    DEFAULT_DASHSCOPE_EMBEDDING_DIM,
    DEFAULT_RERANK_CANDIDATES,
    INDEX_SCOPES,
    REMOTE_EMBEDDING_PROVIDERS,
    REMOTE_MODES,
    RETRIEVAL_MODES,
)


class RemoteEvalBlocked(RuntimeError):
    """Raised when a remote provider would be called without --allow-remote."""


def require_allow_remote(
    *,
    mode: str,
    embedding_provider: str,
    allow_remote: bool,
) -> None:
    needs_remote = mode in REMOTE_MODES or embedding_provider.strip().lower() in REMOTE_EMBEDDING_PROVIDERS
    if needs_remote and not allow_remote:
        raise RemoteEvalBlocked(
            f"mode={mode!r} embedding_provider={embedding_provider!r} requires explicit --allow-remote"
        )


def resolve_modes(mode: str) -> tuple[str, ...]:
    if mode == "all":
        return RETRIEVAL_MODES
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unsupported retrieval mode {mode!r}")
    return (mode,)


def resolve_embedding_dimension(provider: str, dimension: int) -> int:
    name = provider.strip().lower()
    if name in REMOTE_EMBEDDING_PROVIDERS:
        if dimension <= 0 or dimension == 384:
            return DEFAULT_DASHSCOPE_EMBEDDING_DIM
        return dimension
    return 384 if dimension <= 0 else dimension


def build_eval_store(
    *,
    uri: str,
    embedding_provider: str,
    embedding_dimension: int,
    collection_name: str,
    allow_remote: bool,
    mode: str,
    embedding_model: str = "",
) -> MilvusRAGStore:
    require_allow_remote(mode=mode, embedding_provider=embedding_provider, allow_remote=allow_remote)
    provider = embedding_provider.strip().lower()
    pinned_model = str(embedding_model or "").strip()
    if pinned_model and provider in REMOTE_EMBEDDING_PROVIDERS:
        inner = DashScopeEmbeddingProvider(model=pinned_model, dimension=embedding_dimension)
        embedder = ResilientEmbeddingProvider(inner)
    else:
        embedder = build_embedding_provider(embedding_provider, embedding_dimension)
    return MilvusRAGStore(
        uri,
        embedder,
        collection_name=collection_name,
        bm25_enabled=True,
    )


def _qwen3_reranker(*, model: str | None = None, instruct: str | None = None) -> Any:
    return build_reranker(
        "qwen3",
        model=(model or os.getenv("DASHSCOPE_RERANK_MODEL") or "qwen3-rerank").strip(),
        base_url=os.getenv("DASHSCOPE_RERANK_BASE_URL", ""),
        instruct=(
            instruct
            if instruct is not None
            else os.getenv("MAS_RAG_RERANK_INSTRUCT", DEFAULT_RERANK_INSTRUCT)
        ),
        timeout_seconds=float(os.getenv("MAS_RAG_RERANK_TIMEOUT_SECONDS", "12")),
        max_attempts=int(os.getenv("MAS_RAG_RERANK_MAX_ATTEMPTS", "2")),
        backoff_seconds=float(os.getenv("MAS_RAG_RERANK_BACKOFF_SECONDS", "0.25")),
        max_inflight=int(os.getenv("MAS_RAG_RERANK_MAX_INFLIGHT_PER_PROCESS", "2")),
        max_document_chars=int(os.getenv("MAS_RAG_RERANK_MAX_DOCUMENT_CHARS", "4000")),
    )


def _company_filter(company: str, index_scope: str) -> list[str] | None:
    if index_scope == "corpus":
        return None
    return [company]


def _hit_companies(hit: dict[str, Any]) -> list[str]:
    raw = hit.get("companies")
    names: list[str] = []
    if isinstance(raw, list):
        names.extend(str(item) for item in raw if item)
    elif isinstance(raw, str) and raw.strip():
        names.extend(part.strip() for part in raw.split(",") if part.strip())
    primary = str(hit.get("primary_company") or "").strip()
    if primary:
        names.append(primary)
    return names


def _restrict_company_hits(
    hits: list[dict[str, Any]],
    *,
    company: str,
    index_scope: str,
) -> list[dict[str, Any]]:
    if index_scope != "company" or not str(company or "").strip():
        return hits
    wanted = company.strip().lower()
    kept: list[dict[str, Any]] = []
    for hit in hits:
        names = [name.lower() for name in _hit_companies(hit)]
        if not names or wanted in names:
            kept.append(hit)
    return kept


def _retrieve_corpus_hybrid(
    *,
    mode: str,
    store: MilvusRAGStore,
    query: str,
    session_id: str,
    top_k: int,
    rerank_candidates: int,
    bm25_rrf_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Eval-only full-corpus hybrid. Does not use production company filters."""
    candidate_k = max(rerank_candidates, top_k) if mode == "hybrid-qwen3" else top_k
    meta: dict[str, Any] = {
        "degraded": False,
        "rerank_calls": 0,
        "rerank_fallback": False,
        "rerank_tokens": 0,
        "index_scope": "corpus",
    }
    bm25_hits = store.bm25_search(
        query,
        session_id=session_id,
        companies=None,
        top_k=candidate_k,
    )
    vector_hits = store.vector_search(
        query,
        session_id=session_id,
        companies=None,
        top_k=candidate_k,
    )
    meta["bm25_hits"] = len(bm25_hits)
    meta["vector_hits"] = len(vector_hits)
    if vector_hits and bm25_hits:
        hits = reciprocal_rank_fusion(
            [vector_hits, bm25_hits],
            retrieval_method="hybrid_dense_bm25_rrf",
            weights=[1.0, bm25_rrf_weight],
        )[:candidate_k]
        meta["mode"] = "hybrid_dense_bm25_rrf"
    elif bm25_hits:
        hits = bm25_hits[:candidate_k]
        meta["mode"] = "bm25_only"
    else:
        hits = vector_hits[:candidate_k]
        meta["mode"] = "vector_only"
    if mode == "hybrid-qwen3":
        selected, rerank_meta = _qwen3_reranker().rerank(query, hits, top_k=top_k)
        meta.update(rerank_meta)
        meta["rerank_calls"] = 1
        suffix = str(rerank_meta.get("rerank_mode_suffix") or "rerank")
        meta["mode"] = f"{meta.get('mode')}+{suffix}"
        hits = selected
    else:
        hits = hits[:top_k]
    return hits, meta


def retrieve_for_mode(
    *,
    mode: str,
    store: MilvusRAGStore,
    query: str,
    company: str,
    session_id: str,
    document_contexts: list[dict[str, Any]],
    top_k: int,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    bm25_rrf_weight: float = DEFAULT_BM25_RRF_WEIGHT,
    allow_remote: bool = False,
    embedding_provider: str = "deterministic",
    index_scope: str = "company",
    rerank_model: str | None = None,
    rerank_instruct: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Eval-only mode isolation. Production HybridEvidenceRetriever is unchanged."""
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"unsupported retrieval mode {mode!r}")
    if index_scope not in INDEX_SCOPES:
        raise ValueError(f"unsupported index scope {index_scope!r}")
    require_allow_remote(mode=mode, embedding_provider=embedding_provider, allow_remote=allow_remote)

    meta: dict[str, Any] = {
        "mode": mode,
        "degraded": False,
        "rerank_calls": 0,
        "rerank_fallback": False,
        "rerank_tokens": 0,
        "error_type": "",
        "retrieval_methods": [],
        "index_scope": index_scope,
    }
    started = time.perf_counter()
    hits: list[dict[str, Any]] = []
    companies = _company_filter(company, index_scope)
    try:
        if mode == "bm25":
            hits = store.bm25_search(
                query,
                session_id=session_id,
                companies=companies,
                top_k=top_k,
            )
        elif mode == "dense":
            hits = store.vector_search(
                query,
                session_id=session_id,
                companies=companies,
                top_k=top_k,
            )
        elif index_scope == "corpus":
            hits, hybrid_meta = _retrieve_corpus_hybrid(
                mode=mode,
                store=store,
                query=query,
                session_id=session_id,
                top_k=top_k,
                rerank_candidates=rerank_candidates,
                bm25_rrf_weight=bm25_rrf_weight,
            )
            meta.update(hybrid_meta)
        elif mode == "hybrid":
            retriever = HybridEvidenceRetriever(
                store,
                top_k=top_k,
                rerank_enabled=False,
                bm25_rrf_weight=bm25_rrf_weight,
            )
            hits, hybrid_meta = retriever.retrieve_for_company_with_meta(
                query=query,
                company=company,
                session_id=session_id,
                document_contexts=document_contexts,
            )
            meta.update(hybrid_meta)
        else:
            retriever = HybridEvidenceRetriever(
                store,
                top_k=top_k,
                rerank_enabled=True,
                rerank_candidates=max(rerank_candidates, top_k),
                reranker=_qwen3_reranker(model=rerank_model, instruct=rerank_instruct),
                bm25_rrf_weight=bm25_rrf_weight,
            )
            hits, hybrid_meta = retriever.retrieve_for_company_with_meta(
                query=query,
                company=company,
                session_id=session_id,
                document_contexts=document_contexts,
            )
            meta.update(hybrid_meta)
            meta["rerank_calls"] = 1
    except Exception as exc:  # noqa: BLE001
        meta["degraded"] = True
        meta["error_type"] = type(exc).__name__
        meta["error"] = str(exc)
        hits = []
    hits = _restrict_company_hits(hits, company=company, index_scope=index_scope)
    meta["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    meta["retrieval_methods"] = sorted(
        {str(hit.get("retrieval_method") or "") for hit in hits if hit.get("retrieval_method")}
    )
    meta["hit_count"] = len(hits)
    meta["index_scope"] = index_scope
    return hits, meta


def mode_is_isolated(mode: str, retrieval_methods: list[str]) -> bool:
    methods = [item for item in retrieval_methods if item]
    if mode == "bm25":
        return bool(methods) and all(item == "bm25" for item in methods)
    if mode == "dense":
        return bool(methods) and all(item == "vector" for item in methods)
    if mode == "hybrid":
        return any("rrf" in item or item in {"bm25_only", "vector_only"} for item in methods) or not methods
    if mode == "hybrid-qwen3":
        return any("rerank" in item or "qwen3" in item for item in methods) or not methods
    return False
