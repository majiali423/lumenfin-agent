from __future__ import annotations

import hashlib
from typing import Any

from pymilvus import MilvusClient

from .chunking import chunk_document
from .embeddings import EmbeddingProvider


def _stable_row_id(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


class MilvusRAGStore:
    """Milvus Lite backed vector store for session-scoped financial document chunks."""

    def __init__(
        self,
        uri: str,
        embedder: EmbeddingProvider,
        *,
        collection_name: str = "lumenfin_chunks",
    ) -> None:
        self.uri = uri
        self.embedder = embedder
        self.collection_name = collection_name
        self.client = MilvusClient(uri)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.embedder.dimension,
            auto_id=False,
            enable_dynamic_field=True,
        )

    def reset_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        self._ensure_collection()

    def index_documents(self, documents: list[dict[str, Any]], session_id: str) -> dict[str, int | str]:
        rows: list[dict[str, Any]] = []
        for document in documents:
            for chunk in chunk_document(document):
                row_key = f"{session_id}:{chunk['chunk_id']}"
                rows.append(
                    {
                        "id": _stable_row_id(row_key),
                        "vector": None,
                        "row_key": row_key,
                        "session_id": session_id,
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "filename": chunk["filename"],
                        "page": chunk["page"],
                        "text": chunk["text"],
                        "companies": ",".join(chunk.get("companies", [])),
                        "chunk_type": chunk["chunk_type"],
                        "char_count": chunk["char_count"],
                    }
                )

        if not rows:
            return {"chunks_indexed": 0, "documents_indexed": 0}

        vectors = self.embedder.embed([row["text"] for row in rows])
        for row, vector in zip(rows, vectors, strict=True):
            row["vector"] = vector

        self.client.upsert(collection_name=self.collection_name, data=rows)
        return {
            "chunks_indexed": len(rows),
            "documents_indexed": len(documents),
            "backend": "milvus-lite",
            "uri": self.uri,
        }

    def vector_search(
        self,
        query: str,
        *,
        session_id: str,
        companies: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedder.embed([query])[0]
        filter_expr = f'session_id == "{session_id}"'
        fetch_limit = top_k * 4 if companies else top_k

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            filter=filter_expr,
            limit=fetch_limit,
            output_fields=[
                "chunk_id",
                "document_id",
                "filename",
                "page",
                "text",
                "companies",
                "chunk_type",
                "char_count",
            ],
        )
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
        self.client.close()

    def health(self) -> dict[str, Any]:
        return {
            "backend": "milvus-lite",
            "collection": self.collection_name,
            "uri": self.uri,
            "dimension": self.embedder.dimension,
            "ready": self.client.has_collection(self.collection_name),
        }
