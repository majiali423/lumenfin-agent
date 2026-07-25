from __future__ import annotations

import hashlib
import time
from typing import Any

from .chunking import chunk_document
from .embeddings import EmbeddingProvider
from .milvus_client import (
    build_vector_filter_expr,
    get_shared_milvus_client,
    is_milvus_server_uri,
    milvus_backend_kind,
)

class EmbeddingQueryError(RuntimeError):
    """Raised when query embedding fails (after retries)."""


def _stable_row_id(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


class MilvusRAGStore:
    """Milvus-backed vector store (Lite file or shared Server URI)."""

    def __init__(
        self,
        uri: str,
        embedder: EmbeddingProvider,
        *,
        collection_name: str = "lumenfin_chunks",
        shared_client: bool | None = None,
    ) -> None:
        from pymilvus import MilvusClient

        self.uri = uri
        self.embedder = embedder
        self.collection_name = collection_name
        self.backend = milvus_backend_kind(uri)
        # Server URIs share one process-local client by default; Lite always owns its client.
        if is_milvus_server_uri(uri) and shared_client is not False:
            self.client = get_shared_milvus_client(uri)
            self._owns_client = False
        else:
            self.client = MilvusClient(uri)
            self._owns_client = True
        self._query_vector_cache: dict[str, list[float]] = {}
        self.last_embed_ms = 0.0
        self.last_embed_chars = 0
        self._ensure_collection()

    def clear_query_cache(self) -> None:
        self._query_vector_cache.clear()

    def _record_embed_stats(self, texts: list[str], started: float) -> None:
        self.last_embed_ms = round((time.perf_counter() - started) * 1000, 2)
        self.last_embed_chars = sum(len(text or "") for text in texts)
        embedder = self.embedder
        if hasattr(embedder, "last_embed_ms"):
            # Prefer provider-measured latency when resilient wrapper is present.
            self.last_embed_ms = float(getattr(embedder, "last_embed_ms") or self.last_embed_ms)
        if hasattr(embedder, "last_embed_chars"):
            self.last_embed_chars = int(getattr(embedder, "last_embed_chars") or self.last_embed_chars)

    def prime_query_embedding(self, query: str) -> list[float]:
        """Embed once and cache so parallel per-company searches reuse the vector."""
        return self._embed_query(query)

    def _embed_query(self, query: str) -> list[float]:
        cache_key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        cached = self._query_vector_cache.get(cache_key)
        if cached is not None:
            return cached
        started = time.perf_counter()
        try:
            vector = self.embedder.embed([query])[0]
        except Exception as exc:
            self._record_embed_stats([query], started)
            raise EmbeddingQueryError(f"Query embedding failed: {exc}") from exc
        self._record_embed_stats([query], started)
        self._query_vector_cache[cache_key] = vector
        return vector

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self._validate_collection_dimension()
            self._ensure_loaded()
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedder.dimension,
            auto_id=False,
            enable_dynamic_field=True,
        )
        self._ensure_loaded()

    def _validate_collection_dimension(self) -> None:
        """Fail fast when an existing collection was built with a different embed dim."""
        try:
            info = self.client.describe_collection(self.collection_name)
        except Exception:
            return
        fields = []
        if isinstance(info, dict):
            fields = list(info.get("fields") or info.get("schema", {}).get("fields") or [])
        elif hasattr(info, "fields"):
            fields = list(getattr(info, "fields") or [])
        for field in fields:
            name = field.get("name") if isinstance(field, dict) else getattr(field, "name", None)
            if name not in {"vector", "embedding"}:
                continue
            params = {}
            if isinstance(field, dict):
                params = dict(field.get("params") or field.get("type_params") or {})
                dim = field.get("dim") or params.get("dim") or params.get("dimension")
            else:
                params = dict(getattr(field, "params", None) or {})
                dim = getattr(field, "dim", None) or params.get("dim") or params.get("dimension")
            if dim is None:
                continue
            dim_int = int(dim)
            if dim_int != int(self.embedder.dimension):
                raise RuntimeError(
                    f"Milvus collection '{self.collection_name}' dimension={dim_int} does not match "
                    f"embedder dimension={self.embedder.dimension}. Use a new MAS_MILVUS_URI / "
                    f"MAS_MILVUS_COLLECTION or rebuild the index."
                )
            return

    def _ensure_loaded(self) -> None:
        if not self.client.has_collection(self.collection_name):
            return
        try:
            load_state = self.client.get_load_state(self.collection_name)
            if isinstance(load_state, dict) and load_state.get("state") == "Loaded":
                return
        except Exception:
            pass
        self.client.load_collection(self.collection_name)

    def reset_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self._ensure_collection()

    def delete_by_source_document(self, *, tenant_id: str, source_document_id: str) -> int:
        """Delete all chunks for a source document. Returns best-effort deleted count."""
        expr = build_vector_filter_expr(
            tenant_id=tenant_id,
            source_document_ids=[source_document_id],
        )
        try:
            result = self.client.delete(collection_name=self.collection_name, filter=expr)
        except Exception:
            # Older Lite collections may lack tenant fields; try source-only.
            try:
                result = self.client.delete(
                    collection_name=self.collection_name,
                    filter=f'source_document_id == "{source_document_id}"',
                )
            except Exception:
                return 0
        if isinstance(result, dict):
            for key in ("delete_count", "deleted", "count"):
                if key in result:
                    try:
                        return int(result[key])
                    except (TypeError, ValueError):
                        pass
        return 0

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        tenant_id: str,
        source_document_id: str,
        content_hash: str = "",
        session_id: str | None = None,
        replace_existing: bool = True,
    ) -> dict[str, int | str]:
        deleted = 0
        if replace_existing:
            # Always delete-before-write so updates cannot leave stale pages.
            deleted = self.delete_by_source_document(
                tenant_id=tenant_id,
                source_document_id=source_document_id,
            )

        scope = session_id or tenant_id
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            row_key = f"{tenant_id}:{source_document_id}:{chunk['chunk_id']}"
            companies = list(chunk.get("companies") or [])
            rows.append(
                {
                    "id": _stable_row_id(row_key),
                    "vector": None,
                    "row_key": row_key,
                    "session_id": scope,
                    "tenant_id": tenant_id,
                    "source_document_id": source_document_id,
                    "content_hash": content_hash,
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "companies": ",".join(companies),
                    "primary_company": companies[0] if companies else "",
                    "chunk_type": chunk["chunk_type"],
                    "char_count": chunk["char_count"],
                }
            )

        if not rows:
            return {
                "chunks_indexed": 0,
                "documents_indexed": 1 if source_document_id else 0,
                "chunks_deleted": deleted,
                "embed_calls": 0,
                "backend": self.backend,
                "uri": self.uri,
            }

        texts = [row["text"] for row in rows]
        started = time.perf_counter()
        vectors = self.embedder.embed(texts)
        for row, vector in zip(rows, vectors, strict=True):
            row["vector"] = vector
        self._record_embed_stats(texts, started)

        self.client.upsert(collection_name=self.collection_name, data=rows)
        self._ensure_loaded()
        return {
            "chunks_indexed": len(rows),
            "documents_indexed": 1,
            "chunks_deleted": deleted,
            "embed_calls": 1,
            "embed_ms": self.last_embed_ms,
            "embed_chars": self.last_embed_chars,
            "backend": self.backend,
            "uri": self.uri,
        }

    def index_documents(self, documents: list[dict[str, Any]], session_id: str) -> dict[str, int | str]:
        """Legacy run-scoped indexing (sync_on_run). Prefer index_chunks for upload-time path."""
        rows: list[dict[str, Any]] = []
        for document in documents:
            source_document_id = str(document.get("source_document_id") or document.get("document_id") or "doc")
            for chunk in chunk_document(document):
                row_key = f"{session_id}:{chunk['chunk_id']}"
                companies = list(chunk.get("companies") or [])
                rows.append(
                    {
                        "id": _stable_row_id(row_key),
                        "vector": None,
                        "row_key": row_key,
                        "session_id": session_id,
                        "tenant_id": session_id,
                        "source_document_id": source_document_id,
                        "content_hash": "",
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "filename": chunk["filename"],
                        "page": chunk["page"],
                        "text": chunk["text"],
                        "companies": ",".join(companies),
                        "primary_company": companies[0] if companies else "",
                        "chunk_type": chunk["chunk_type"],
                        "char_count": chunk["char_count"],
                    }
                )

        if not rows:
            return {"chunks_indexed": 0, "documents_indexed": 0, "backend": self.backend}

        texts = [row["text"] for row in rows]
        started = time.perf_counter()
        vectors = self.embedder.embed(texts)
        for row, vector in zip(rows, vectors, strict=True):
            row["vector"] = vector
        self._record_embed_stats(texts, started)

        self.client.upsert(collection_name=self.collection_name, data=rows)
        self._ensure_loaded()
        return {
            "chunks_indexed": len(rows),
            "documents_indexed": len(documents),
            "embed_calls": 1,
            "embed_ms": self.last_embed_ms,
            "embed_chars": self.last_embed_chars,
            "backend": self.backend,
            "uri": self.uri,
        }

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
    ) -> list[dict[str, Any]]:
        query_vector = self._embed_query(query)
        filter_expr = build_vector_filter_expr(
            session_id=session_id,
            tenant_id=tenant_id,
            document_ids=document_ids,
            source_document_ids=source_document_ids,
            companies=companies,
        )
        # Company is pushed into the engine filter; keep a small over-fetch for safety.
        fetch_limit = top_k * 2 if companies else top_k

        self._ensure_loaded()
        search_kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": [query_vector],
            "limit": fetch_limit,
            "output_fields": [
                "chunk_id",
                "document_id",
                "source_document_id",
                "filename",
                "page",
                "text",
                "companies",
                "primary_company",
                "chunk_type",
                "char_count",
                "tenant_id",
            ],
        }
        if filter_expr:
            search_kwargs["filter"] = filter_expr
        try:
            results = self.client.search(**search_kwargs)
        except Exception:
            # Some Lite builds reject `like` / complex dynamic-field filters; retry without
            # company clause and keep Python post-filter as safety net.
            if companies:
                fallback_expr = build_vector_filter_expr(
                    session_id=session_id,
                    tenant_id=tenant_id,
                    document_ids=document_ids,
                    source_document_ids=source_document_ids,
                    companies=None,
                )
                if fallback_expr:
                    search_kwargs["filter"] = fallback_expr
                else:
                    search_kwargs.pop("filter", None)
                search_kwargs["limit"] = top_k * 4
                results = self.client.search(**search_kwargs)
            else:
                raise
        hits: list[dict[str, Any]] = []
        for batch in results:
            for item in batch:
                entity = item.get("entity", {})
                company_tags = [
                    tag.strip()
                    for tag in str(entity.get("companies", "")).split(",")
                    if tag.strip()
                ]
                if companies and company_tags and not any(company in company_tags for company in companies):
                    continue
                hits.append(
                    {
                        "chunk_id": entity.get("chunk_id"),
                        "document_id": entity.get("document_id"),
                        "source_document_id": entity.get("source_document_id"),
                        "filename": entity.get("filename"),
                        "page": entity.get("page"),
                        "text": entity.get("text", ""),
                        "companies": company_tags,
                        "chunk_type": entity.get("chunk_type", "narrative"),
                        "score": float(item.get("distance", 0.0)),
                        "retrieval_method": "vector",
                        "citation": f"{entity.get('filename')}#p{entity.get('page')}",
                    }
                )
                if len(hits) >= top_k:
                    break
        return hits

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "collection": self.collection_name,
            "uri": self.uri,
            "dimension": self.embedder.dimension,
            "ready": self.client.has_collection(self.collection_name),
            "shared_client": not self._owns_client,
        }
