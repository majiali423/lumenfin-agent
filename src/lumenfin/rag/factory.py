from __future__ import annotations

import os
from pathlib import Path

from ..config import AppConfig
from .embeddings import build_embedding_provider
from .hybrid_retriever import HybridEvidenceRetriever
from .milvus_store import MilvusRAGStore


def resolve_milvus_uri(uri: str) -> str:
    """Avoid Milvus Lite multi-process file lock by isolating per PID by default.

    Set MAS_MILVUS_ISOLATE=false to share one .db (single-process API only).
    """
    path = Path(uri)
    if path.suffix != ".db":
        return uri
    isolate = os.getenv("MAS_MILVUS_ISOLATE", "true").strip().lower()
    if isolate in {"0", "false", "no"}:
        return uri
    return str(path.with_name(f"{path.stem}_p{os.getpid()}{path.suffix}"))


def build_hybrid_retriever(config: AppConfig) -> HybridEvidenceRetriever | None:
    if not config.rag_enabled:
        return None
    milvus_path = Path(resolve_milvus_uri(config.milvus_uri))
    if milvus_path.suffix == ".db":
        milvus_path.parent.mkdir(parents=True, exist_ok=True)
    embedder = build_embedding_provider(config.embedding_provider, config.embedding_dimension)
    rag_store = MilvusRAGStore(
        uri=str(milvus_path),
        embedder=embedder,
        collection_name=config.milvus_collection,
    )
    return HybridEvidenceRetriever(rag_store, top_k=config.rag_top_k)
