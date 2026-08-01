import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import Integer, Text, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class WorkflowCheckpoint(Base):
    __tablename__ = "workflow_checkpoints"

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    workflow_status: Mapped[str] = mapped_column(Text, nullable=False)
    state_json: Mapped[str] = mapped_column(Text, nullable=False)
    clarification_questions_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    last_node: Mapped[str] = mapped_column(Text, nullable=False)
    llm_backend: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    job_id: Mapped[str] = mapped_column(Text, primary_key=True)
    thread_id: Mapped[str] = mapped_column(Text, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    llm_backend: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    artifacts_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


class RagDocument(Base):
    """Persistent RAG document index metadata (upload-time indexing)."""

    __tablename__ = "rag_documents"

    document_id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    index_status: Mapped[str] = mapped_column(Text, nullable=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    indexed_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    contexts_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    source_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)


class RagChunk(Base):
    """Canonical chunk text shared by keyword + vector retrieval."""

    __tablename__ = "rag_chunks"

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    source_document_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    companies_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    chunk_type: Mapped[str] = mapped_column(Text, nullable=False, default="narrative")
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[str] = mapped_column(Text, nullable=False)


class JobRepository:
    def __init__(self, database_url: str, db_path: Optional[Path] = None) -> None:
        if database_url.startswith("sqlite:///") and db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, future=True)
        Base.metadata.create_all(self.engine)

    def create_job(self, job_id: str, thread_id: str, query: str) -> None:
        now = utc_now()
        with Session(self.engine) as session:
            session.add(
                AnalysisJob(
                    job_id=job_id,
                    thread_id=thread_id,
                    query=query,
                    status="pending",
                    llm_backend=None,
                    result_json=None,
                    artifacts_json=None,
                    error_message=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()

    def update_job_status(
        self,
        job_id: str,
        status: str,
        llm_backend: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
        artifacts: Optional[dict[str, str]] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with Session(self.engine) as session:
            job = session.get(AnalysisJob, job_id)
            if job is None:
                return
            job.status = status
            if llm_backend is not None:
                job.llm_backend = llm_backend
            if result is not None:
                job.result_json = json.dumps(result, ensure_ascii=False)
            if artifacts is not None:
                job.artifacts_json = json.dumps(artifacts, ensure_ascii=False)
            job.error_message = error_message
            job.updated_at = utc_now()
            session.commit()

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with Session(self.engine) as session:
            job = session.get(AnalysisJob, job_id)
            return self._row_to_dict(job) if job is not None else None

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            rows = session.scalars(select(AnalysisJob).order_by(AnalysisJob.created_at.desc()).limit(limit)).all()
            return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row: AnalysisJob) -> dict[str, Any]:
        return {
            "job_id": row.job_id,
            "thread_id": row.thread_id,
            "query": row.query,
            "status": row.status,
            "llm_backend": row.llm_backend,
            "result": json.loads(row.result_json) if row.result_json else None,
            "artifacts": json.loads(row.artifacts_json) if row.artifacts_json else {},
            "error_message": row.error_message,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }


class RagDocumentRepository:
    """CRUD for upload-time RAG document + canonical chunk persistence."""

    def __init__(self, database_url: str, db_path: Optional[Path] = None) -> None:
        if database_url.startswith("sqlite:///") and db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(database_url, future=True)
        Base.metadata.create_all(self.engine)
        self._ensure_sqlite_columns()

    def _ensure_sqlite_columns(self) -> None:
        """Best-effort additive migration for existing Lite SQLite files."""
        if not str(self.engine.url).startswith("sqlite"):
            return
        with self.engine.begin() as conn:
            rows = conn.exec_driver_sql("PRAGMA table_info(rag_documents)").fetchall()
            columns = {str(row[1]) for row in rows}
            if rows and "source_path" not in columns:
                conn.exec_driver_sql("ALTER TABLE rag_documents ADD COLUMN source_path TEXT")

    def find_ready_by_hash(self, *, tenant_id: str, content_hash: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            row = session.scalars(
                select(RagDocument).where(
                    RagDocument.tenant_id == tenant_id,
                    RagDocument.content_hash == content_hash,
                    RagDocument.index_status == "ready",
                )
            ).first()
            return self._doc_to_dict(row) if row is not None else None

    def get_document(self, document_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            row = session.get(RagDocument, document_id)
            if row is None:
                return None
            if tenant_id is not None and row.tenant_id != tenant_id:
                return None
            return self._doc_to_dict(row)

    def upsert_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
        filename: str,
        content_hash: str,
        index_status: str,
        contexts: list[dict[str, Any]],
        chunk_count: int = 0,
        error: str | None = None,
        source_path: str | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with Session(self.engine) as session:
            row = session.get(RagDocument, document_id)
            if row is None:
                row = RagDocument(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    filename=filename,
                    content_hash=content_hash,
                    index_status=index_status,
                    error=error,
                    indexed_at=now if index_status == "ready" else None,
                    chunk_count=chunk_count,
                    contexts_json=json.dumps(contexts, ensure_ascii=False),
                    source_path=source_path,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.tenant_id = tenant_id
                row.filename = filename
                row.content_hash = content_hash
                row.index_status = index_status
                row.error = error
                row.chunk_count = chunk_count
                row.contexts_json = json.dumps(contexts, ensure_ascii=False)
                if source_path is not None:
                    row.source_path = source_path
                row.updated_at = now
                if index_status == "ready":
                    row.indexed_at = now
            session.commit()
            return self._doc_to_dict(row)

    def list_pending(self, *, tenant_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            stmt = select(RagDocument).where(RagDocument.index_status == "pending")
            if tenant_id:
                stmt = stmt.where(RagDocument.tenant_id == tenant_id)
            stmt = stmt.order_by(RagDocument.created_at.asc()).limit(limit)
            return [self._doc_to_dict(row) for row in session.scalars(stmt).all()]

    def replace_chunks(
        self,
        *,
        source_document_id: str,
        tenant_id: str,
        chunks: list[dict[str, Any]],
        content_hash: str,
    ) -> None:
        now = utc_now()
        with Session(self.engine) as session:
            session.execute(delete(RagChunk).where(RagChunk.source_document_id == source_document_id))
            for chunk in chunks:
                session.add(
                    RagChunk(
                        chunk_id=str(chunk["chunk_id"]),
                        document_id=str(chunk["document_id"]),
                        source_document_id=source_document_id,
                        tenant_id=tenant_id,
                        filename=str(chunk.get("filename") or ""),
                        page=int(chunk.get("page") or 1),
                        text=str(chunk.get("text") or ""),
                        companies_json=json.dumps(list(chunk.get("companies") or []), ensure_ascii=False),
                        chunk_type=str(chunk.get("chunk_type") or "narrative"),
                        char_count=int(chunk.get("char_count") or len(str(chunk.get("text") or ""))),
                        content_hash=content_hash,
                        created_at=now,
                    )
                )
            session.commit()

    def list_chunks(
        self,
        *,
        tenant_id: str,
        document_ids: list[str] | None = None,
        source_document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            stmt = select(RagChunk).where(RagChunk.tenant_id == tenant_id)
            if source_document_ids:
                stmt = stmt.where(RagChunk.source_document_id.in_(source_document_ids))
            elif document_ids:
                stmt = stmt.where(RagChunk.document_id.in_(document_ids))
            rows = session.scalars(stmt).all()
            return [self._chunk_to_dict(row) for row in rows]

    def _doc_to_dict(self, row: RagDocument) -> dict[str, Any]:
        return {
            "document_id": row.document_id,
            "tenant_id": row.tenant_id,
            "filename": row.filename,
            "content_hash": row.content_hash,
            "index_status": row.index_status,
            "error": row.error,
            "indexed_at": row.indexed_at,
            "chunk_count": row.chunk_count,
            "contexts": json.loads(row.contexts_json or "[]"),
            "source_path": row.source_path,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def _chunk_to_dict(self, row: RagChunk) -> dict[str, Any]:
        return {
            "chunk_id": row.chunk_id,
            "document_id": row.document_id,
            "source_document_id": row.source_document_id,
            "tenant_id": row.tenant_id,
            "filename": row.filename,
            "page": row.page,
            "text": row.text,
            "companies": json.loads(row.companies_json or "[]"),
            "chunk_type": row.chunk_type,
            "char_count": row.char_count,
            "content_hash": row.content_hash,
        }
