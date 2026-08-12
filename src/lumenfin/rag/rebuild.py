"""Controlled rebuild of a Milvus RAG collection from durable database chunks."""

from __future__ import annotations

from typing import Any

from ..database import RagDocumentRepository
from .milvus_store import MilvusRAGStore


def build_rebuild_manifest(
    repository: RagDocumentRepository,
    *,
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    """Validate all durable chunks before any destructive collection reset."""
    manifest: list[dict[str, Any]] = []
    for document in repository.list_documents(tenant_id=tenant_id, index_status="ready"):
        chunks = repository.list_chunks(
            tenant_id=str(document["tenant_id"]),
            source_document_ids=[str(document["document_id"])],
        )
        expected = int(document.get("chunk_count") or 0)
        if len(chunks) != expected:
            raise RuntimeError(
                f"RAG rebuild preflight failed for {document['document_id']}: "
                f"metadata chunk_count={expected}, durable chunks={len(chunks)}"
            )
        if not chunks:
            raise RuntimeError(
                f"RAG rebuild preflight failed for {document['document_id']}: ready document has no chunks"
            )
        manifest.append(
            {
                "document_id": str(document["document_id"]),
                "tenant_id": str(document["tenant_id"]),
                "content_hash": str(document.get("content_hash") or ""),
                "chunk_count": len(chunks),
            }
        )
    return manifest


def rebuild_vector_index(
    *,
    repository: RagDocumentRepository,
    store: MilvusRAGStore,
    tenant_id: str | None = None,
    reset_collection: bool = False,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Rebuild and verify dense/BM25 retrieval from durable chunks."""
    if reset_collection and tenant_id is not None:
        raise ValueError(
            "tenant-scoped rebuild cannot reset a shared collection; rebuild all tenants instead"
        )
    manifest = build_rebuild_manifest(repository, tenant_id=tenant_id)
    if not manifest and not allow_empty:
        raise RuntimeError("RAG rebuild refused: no ready documents found")

    if reset_collection:
        store.reset_collection()

    chunks_indexed = 0
    documents_verified = 0
    bm25_documents_verified = 0
    for item in manifest:
        chunks = repository.list_chunks(
            tenant_id=item["tenant_id"],
            source_document_ids=[item["document_id"]],
        )
        stats = store.index_chunks(
            chunks,
            tenant_id=item["tenant_id"],
            source_document_id=item["document_id"],
            content_hash=item["content_hash"],
            replace_existing=True,
        )
        indexed = int(stats.get("chunks_indexed") or 0)
        if indexed != item["chunk_count"]:
            raise RuntimeError(
                f"RAG rebuild write count mismatch for {item['document_id']}: "
                f"expected={item['chunk_count']}, indexed={indexed}"
            )
        hits = store.vector_search(
            str(chunks[0]["text"]),
            tenant_id=item["tenant_id"],
            source_document_ids=[item["document_id"]],
            top_k=1,
        )
        if not hits:
            raise RuntimeError(
                f"RAG rebuild visibility check failed for {item['document_id']}: "
                "first dense search returned no hits"
            )
        if store.bm25_enabled:
            bm25_hits = store.bm25_search(
                str(chunks[0]["text"]),
                tenant_id=item["tenant_id"],
                source_document_ids=[item["document_id"]],
                top_k=1,
            )
            if not bm25_hits:
                raise RuntimeError(
                    f"RAG rebuild visibility check failed for {item['document_id']}: "
                    "first BM25 search returned no hits"
                )
            bm25_documents_verified += 1
        chunks_indexed += indexed
        documents_verified += 1

    return {
        "collection": store.collection_name,
        "documents_indexed": len(manifest),
        "documents_verified": documents_verified,
        "bm25_documents_verified": bm25_documents_verified,
        "bm25_enabled": store.bm25_enabled,
        "chunks_indexed": chunks_indexed,
        "reset_collection": reset_collection,
        "tenant_id": tenant_id,
    }
