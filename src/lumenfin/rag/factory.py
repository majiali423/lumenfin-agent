from __future__ import annotations

from pathlib import Path

from ..config import AppConfig
from .embeddings import build_embedding_provider
from .hybrid_retriever import HybridEvidenceRetriever
from .milvus_store import MilvusRAGStore


def build_hybrid_retriever(config: AppConfig) -> HybridEvidenceRetriever | None:
    if not config.rag_enabled:
        return None
    milvus_path = Path(config.milvus_uri)
    if milvus_path.suffix == ".db":
        milvus_path.parent.mkdir(parents=True, exist_ok=True)
    embedder = build_embedding_provider(config.embedding_provider, config.embedding_dimension)
    rag_store = MilvusRAGStore(
        uri=str(milvus_path),
        embedder=embedder,
        collection_name=config.milvus_collection,
    )
    return HybridEvidenceRetriever(rag_store, top_k=config.rag_top_k)
