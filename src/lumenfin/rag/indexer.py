"""Upload-time document indexing: hash dedupe, chunk once, embed once."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, TypedDict

from ..database import RagDocumentRepository
from ..document_ingest import parse_upload_documents
from .chunking import chunk_document
from .vector_store import VectorStore


IndexStatus = Literal["pending", "ready", "failed", "skipped_duplicate"]


class IndexReceipt(TypedDict):
    document_id: str
    tenant_id: str
    filename: str
    content_hash: str
    status: IndexStatus
    chunk_count: int
    error: str | None
    contexts: list[dict[str, Any]]
    embed_calls: int


def content_hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def content_hash_file(path: Path) -> str:
    return content_hash_bytes(path.read_bytes())


def canonical_document_id(tenant_id: str, content_hash: str) -> str:
    identity = hashlib.sha256(f"{tenant_id}\0{content_hash}".encode("utf-8")).hexdigest()
    return f"doc-{identity}"


def summarize_index_receipts(receipts: list[IndexReceipt]) -> dict[str, Any]:
    ready = [item for item in receipts if item["status"] in {"ready", "skipped_duplicate"}]
    failed = [item for item in receipts if item["status"] == "failed"]
    pending = [item for item in receipts if item["status"] == "pending"]
    return {
        "mode": "async_on_upload",
        "chunks_indexed": sum(int(item["chunk_count"]) for item in ready),
        "documents_indexed": len(ready),
        "documents_failed": len(failed),
        "documents_pending": len(pending),
        "embed_calls": sum(int(item.get("embed_calls") or 0) for item in receipts),
        "document_ids": [item["document_id"] for item in receipts],
        "receipts": [
            {
                "document_id": item["document_id"],
                "filename": item["filename"],
                "status": item["status"],
                "chunk_count": item["chunk_count"],
                "content_hash": item["content_hash"],
                "error": item["error"],
            }
            for item in receipts
        ],
        "backend": "milvus",
    }


class DocumentIndexer:
    """Parse → chunk → persist metadata/chunks → upsert vectors (with hash dedupe)."""

    def __init__(
        self,
        *,
        rag_store: VectorStore | None,
        repository: RagDocumentRepository,
        tenant_id: str = "default",
    ) -> None:
        self.rag_store = rag_store
        self.repository = repository
        self.tenant_id = tenant_id or "default"

    def enqueue_file(self, path: Path, *, tenant_id: str | None = None) -> IndexReceipt:
        """Register a pending index job without embedding (async path)."""
        tenant = (tenant_id or self.tenant_id).strip() or "default"
        file_path = Path(path)
        filename = file_path.name
        digest = content_hash_file(file_path)

        existing = self.repository.find_ready_by_hash(tenant_id=tenant, content_hash=digest)
        if existing is not None:
            return {
                "document_id": existing["document_id"],
                "tenant_id": tenant,
                "filename": existing["filename"],
                "content_hash": digest,
                "status": "skipped_duplicate",
                "chunk_count": int(existing.get("chunk_count") or 0),
                "error": None,
                "contexts": list(existing.get("contexts") or []),
                "embed_calls": 0,
            }

        document_id = canonical_document_id(tenant, digest)
        record, owns_processing = self.repository.register_pending_document(
            document_id=document_id,
            tenant_id=tenant,
            filename=filename,
            content_hash=digest,
            source_path=str(file_path),
        )
        if not owns_processing:
            return {
                "document_id": record["document_id"],
                "tenant_id": tenant,
                "filename": record["filename"],
                "content_hash": digest,
                "status": "skipped_duplicate",
                "chunk_count": int(record.get("chunk_count") or 0),
                "error": None,
                "contexts": list(record.get("contexts") or []),
                "embed_calls": 0,
            }
        return {
            "document_id": document_id,
            "tenant_id": tenant,
            "filename": filename,
            "content_hash": digest,
            "status": "pending",
            "chunk_count": 0,
            "error": None,
            "contexts": [],
            "embed_calls": 0,
        }

    def process_pending(self, document_id: str, *, tenant_id: str | None = None) -> IndexReceipt:
        """Complete indexing for a pending document_id."""
        tenant = (tenant_id or self.tenant_id).strip() or "default"
        record = self.repository.get_document(document_id, tenant_id=tenant)
        if record is None:
            return {
                "document_id": document_id,
                "tenant_id": tenant,
                "filename": "",
                "content_hash": "",
                "status": "failed",
                "chunk_count": 0,
                "error": "document not found",
                "contexts": [],
                "embed_calls": 0,
            }
        if record.get("index_status") == "ready":
            return {
                "document_id": record["document_id"],
                "tenant_id": tenant,
                "filename": record["filename"],
                "content_hash": record["content_hash"],
                "status": "skipped_duplicate",
                "chunk_count": int(record.get("chunk_count") or 0),
                "error": None,
                "contexts": list(record.get("contexts") or []),
                "embed_calls": 0,
            }
        record, claimed = self.repository.claim_pending_document(
            document_id=document_id,
            tenant_id=tenant,
        )
        if record is None:
            return {
                "document_id": document_id,
                "tenant_id": tenant,
                "filename": "",
                "content_hash": "",
                "status": "failed",
                "chunk_count": 0,
                "error": "document not found",
                "contexts": [],
                "embed_calls": 0,
            }
        if not claimed:
            return {
                "document_id": record["document_id"],
                "tenant_id": tenant,
                "filename": record["filename"],
                "content_hash": record["content_hash"],
                "status": "skipped_duplicate",
                "chunk_count": int(record.get("chunk_count") or 0),
                "error": None,
                "contexts": list(record.get("contexts") or []),
                "embed_calls": 0,
            }

        source_path = record.get("source_path")
        if not source_path:
            return self._fail(record, "missing source_path for pending document")

        file_path = Path(str(source_path))
        if not file_path.exists():
            return self._fail(record, f"source file missing: {file_path}")

        filename = record["filename"] or file_path.name
        digest = str(record.get("content_hash") or content_hash_file(file_path))
        try:
            contexts = parse_upload_documents(file_path)
            contexts = [_stabilize_context(ctx, source_document_id=document_id) for ctx in contexts]
            chunks: list[dict[str, Any]] = []
            for context in contexts:
                for chunk in chunk_document(context):
                    chunks.append(chunk)

            embed_calls = 0
            if self.rag_store is not None:
                stats = self.rag_store.index_chunks(
                    chunks,
                    tenant_id=tenant,
                    source_document_id=document_id,
                    content_hash=digest,
                    replace_existing=True,
                )
                embed_calls = int(stats.get("embed_calls") or 0)

            self.repository.replace_chunks(
                source_document_id=document_id,
                tenant_id=tenant,
                chunks=chunks,
                content_hash=digest,
            )
            self.repository.upsert_document(
                document_id=document_id,
                tenant_id=tenant,
                filename=filename,
                content_hash=digest,
                index_status="ready",
                contexts=contexts,
                chunk_count=len(chunks),
                source_path=str(file_path),
            )
            return {
                "document_id": document_id,
                "tenant_id": tenant,
                "filename": filename,
                "content_hash": digest,
                "status": "ready",
                "chunk_count": len(chunks),
                "error": None,
                "contexts": contexts,
                "embed_calls": embed_calls,
            }
        except Exception as exc:
            return self._fail(record, str(exc))

    def index_file(self, path: Path, *, tenant_id: str | None = None) -> IndexReceipt:
        """Synchronous path: enqueue then process immediately."""
        receipt = self.enqueue_file(path, tenant_id=tenant_id)
        if receipt["status"] == "skipped_duplicate":
            return receipt
        return self.process_pending(receipt["document_id"], tenant_id=receipt["tenant_id"])

    def index_paths(self, paths: list[str | Path], *, tenant_id: str | None = None) -> list[IndexReceipt]:
        return [self.index_file(Path(path), tenant_id=tenant_id) for path in paths]

    def enqueue_paths(self, paths: list[str | Path], *, tenant_id: str | None = None) -> list[IndexReceipt]:
        return [self.enqueue_file(Path(path), tenant_id=tenant_id) for path in paths]

    def get_status(self, document_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        return self.repository.get_document(document_id, tenant_id=tenant_id or self.tenant_id)

    def load_contexts_for_documents(
        self,
        document_ids: list[str],
        *,
        tenant_id: str | None = None,
        require_ready: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        tenant = tenant_id or self.tenant_id
        contexts: list[dict[str, Any]] = []
        statuses: list[dict[str, Any]] = []
        for document_id in document_ids:
            record = self.repository.get_document(document_id, tenant_id=tenant)
            if record is None:
                statuses.append({"document_id": document_id, "index_status": "missing", "error": "not found"})
                if require_ready:
                    raise ValueError(f"RAG document not found: {document_id}")
                continue
            statuses.append(
                {
                    "document_id": document_id,
                    "index_status": record["index_status"],
                    "error": record.get("error"),
                    "chunk_count": record.get("chunk_count", 0),
                }
            )
            if record["index_status"] != "ready":
                if require_ready:
                    raise ValueError(
                        f"RAG document {document_id} is not ready (status={record['index_status']})."
                    )
                continue
            contexts.extend(list(record.get("contexts") or []))
        return contexts, statuses

    def list_chunks(
        self,
        *,
        tenant_id: str | None = None,
        source_document_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return self.repository.list_chunks(
            tenant_id=tenant_id or self.tenant_id,
            source_document_ids=source_document_ids,
            document_ids=document_ids,
        )

    def _fail(self, record: dict[str, Any], error: str) -> IndexReceipt:
        document_id = str(record["document_id"])
        tenant = str(record["tenant_id"])
        self.repository.upsert_document(
            document_id=document_id,
            tenant_id=tenant,
            filename=str(record.get("filename") or ""),
            content_hash=str(record.get("content_hash") or ""),
            index_status="failed",
            contexts=[],
            chunk_count=0,
            error=error,
            source_path=record.get("source_path"),
        )
        return {
            "document_id": document_id,
            "tenant_id": tenant,
            "filename": str(record.get("filename") or ""),
            "content_hash": str(record.get("content_hash") or ""),
            "status": "failed",
            "chunk_count": 0,
            "error": error,
            "contexts": [],
            "embed_calls": 0,
        }


def _stabilize_context(context: dict[str, Any], *, source_document_id: str) -> dict[str, Any]:
    stabilized = dict(context)
    raw_id = str(stabilized.get("document_id") or stabilized.get("filename") or "doc").strip()
    if not raw_id.startswith(source_document_id):
        stabilized["document_id"] = f"{source_document_id}:{raw_id}"
    stabilized["source_document_id"] = source_document_id
    return stabilized
