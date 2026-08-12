"""Vector store protocol for RAG backends (Lite / Server)."""

from __future__ import annotations

from typing import Any, Protocol


class VectorStore(Protocol):
    """Minimal contract used by DocumentIndexer and HybridEvidenceRetriever."""

    uri: str
    collection_name: str

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        tenant_id: str,
        source_document_id: str,
        content_hash: str = "",
        session_id: str | None = None,
        replace_existing: bool = True,
    ) -> dict[str, int | str]: ...

    def index_documents(self, documents: list[dict[str, Any]], session_id: str) -> dict[str, int | str]: ...

    def vector_search(
        self,
        query: str,
        *,
        session_id: str | None = None,
        tenant_id: str | None = None,
        document_ids: list[str] | None = None,
        source_document_ids: list[str] | None = None,
        companies: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]: ...

    def bm25_search(
        self,
        query: str,
        *,
        session_id: str | None = None,
        tenant_id: str | None = None,
        document_ids: list[str] | None = None,
        source_document_ids: list[str] | None = None,
        companies: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]: ...

    def delete_by_source_document(self, *, tenant_id: str, source_document_id: str) -> None: ...

    def prime_query_embedding(self, query: str) -> list[float]: ...

    def clear_query_cache(self) -> None: ...

    def close(self) -> None: ...

    def health(self) -> dict[str, Any]: ...
