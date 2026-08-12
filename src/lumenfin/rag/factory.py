from __future__ import annotations

from pathlib import Path

from ..config import AppConfig
from ..database import RagDocumentRepository
from .embeddings import build_embedding_provider
from .hybrid_retriever import HybridEvidenceRetriever
from .indexer import DocumentIndexer
from .milvus_client import resolve_milvus_uri
from .milvus_store import MilvusRAGStore
from .rerank import build_reranker


def build_rag_store(config: AppConfig) -> MilvusRAGStore | None:
    if not config.rag_enabled:
        return None
    resolved = resolve_milvus_uri(config.milvus_uri, isolate=config.milvus_isolate)
    if resolved.endswith(".db"):
        Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    embedder = build_embedding_provider(
        config.embedding_provider,
        config.embedding_dimension,
        max_retries=config.embedding_max_retries,
        backoff_seconds=config.embedding_backoff_seconds,
        timeout_seconds=config.embedding_timeout_seconds,
    )
    return MilvusRAGStore(
        uri=resolved,
        embedder=embedder,
        collection_name=config.milvus_collection,
        bm25_enabled=config.rag_bm25_enabled,
    )


def build_document_indexer(
    config: AppConfig,
    *,
    rag_store: MilvusRAGStore | None = None,
    repository: RagDocumentRepository | None = None,
) -> DocumentIndexer:
    store = rag_store if rag_store is not None else build_rag_store(config)
    repo = repository or RagDocumentRepository(config.database_url, db_path=config.db_path)
    return DocumentIndexer(
        rag_store=store,
        repository=repo,
        tenant_id=config.rag_tenant_id,
        lease_seconds=config.rag_index_lease_seconds,
    )


def build_hybrid_retriever(
    config: AppConfig,
    *,
    rag_store: MilvusRAGStore | None = None,
    indexer: DocumentIndexer | None = None,
) -> HybridEvidenceRetriever | None:
    if not config.rag_enabled:
        return None
    store = rag_store if rag_store is not None else build_rag_store(config)
    document_indexer = indexer
    chunk_loader = None
    if document_indexer is not None:

        def chunk_loader(*, tenant_id: str, source_document_ids: list[str]):
            return document_indexer.list_chunks(
                tenant_id=tenant_id,
                source_document_ids=source_document_ids,
            )

    return HybridEvidenceRetriever(
        store,
        top_k=config.rag_top_k,
        chunk_loader=chunk_loader,
        min_score=config.rag_min_score,
        degrade_on_vector_error=config.rag_degrade_on_vector_error,
        rerank_enabled=config.rag_rerank_enabled,
        rerank_candidates=config.rag_rerank_candidates,
        reranker=(
            build_reranker(
                config.rag_rerank_provider,
                model=config.rag_rerank_model,
                base_url=config.rag_rerank_base_url,
                instruct=config.rag_rerank_instruct,
                timeout_seconds=config.rag_rerank_timeout_seconds,
                max_attempts=config.rag_rerank_max_attempts,
                backoff_seconds=config.rag_rerank_backoff_seconds,
                max_inflight=config.rag_rerank_max_inflight_per_process,
                acquire_timeout_seconds=config.provider_acquire_timeout_seconds,
                max_document_chars=config.rag_rerank_max_document_chars,
            )
            if config.rag_rerank_enabled
            else None
        ),
        bm25_rrf_weight=config.rag_bm25_rrf_weight,
    )


# Backward-compatible re-export for older imports.
__all__ = [
    "build_document_indexer",
    "build_hybrid_retriever",
    "build_rag_store",
    "resolve_milvus_uri",
]
