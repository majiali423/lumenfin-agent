from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from lumenfin.database import RagChunk, RagDocument, WorkflowCheckpoint


def engine_for(database_url: str):
    return create_engine(database_url, future=True)


def fetch_checkpoint(database_url: str, thread_id: str) -> dict[str, Any] | None:
    engine = engine_for(database_url)
    with Session(engine) as session:
        row = session.get(WorkflowCheckpoint, thread_id)
        if row is None:
            return None
        return {
            "thread_id": row.thread_id,
            "query": row.query,
            "workflow_status": row.workflow_status,
            "revision": row.revision,
            "state": json.loads(row.state_json),
        }


def count_checkpoints(database_url: str, thread_id: str | None = None) -> int:
    engine = engine_for(database_url)
    with Session(engine) as session:
        stmt = select(func.count()).select_from(WorkflowCheckpoint)
        if thread_id:
            stmt = stmt.where(WorkflowCheckpoint.thread_id == thread_id)
        return int(session.scalar(stmt) or 0)


def fetch_document(database_url: str, document_id: str, tenant_id: str) -> dict[str, Any] | None:
    engine = engine_for(database_url)
    with Session(engine) as session:
        row = session.scalars(
            select(RagDocument).where(
                RagDocument.document_id == document_id,
                RagDocument.tenant_id == tenant_id,
            )
        ).first()
        if row is None:
            return None
        return {
            "document_id": row.document_id,
            "tenant_id": row.tenant_id,
            "filename": row.filename,
            "content_hash": row.content_hash,
            "index_status": row.index_status,
            "index_owner": row.index_owner,
            "index_lease_expires": row.index_lease_expires,
            "index_attempt": row.index_attempt,
            "chunk_count": row.chunk_count,
            "error": row.error,
            "source_path": row.source_path,
        }


def count_documents(
    database_url: str,
    *,
    tenant_id: str | None = None,
    content_hash: str | None = None,
) -> int:
    engine = engine_for(database_url)
    with Session(engine) as session:
        stmt = select(func.count()).select_from(RagDocument)
        if tenant_id:
            stmt = stmt.where(RagDocument.tenant_id == tenant_id)
        if content_hash:
            stmt = stmt.where(RagDocument.content_hash == content_hash)
        return int(session.scalar(stmt) or 0)


def count_chunks(database_url: str, *, tenant_id: str | None = None) -> int:
    engine = engine_for(database_url)
    with Session(engine) as session:
        stmt = select(func.count()).select_from(RagChunk)
        if tenant_id:
            stmt = stmt.where(RagChunk.tenant_id == tenant_id)
        return int(session.scalar(stmt) or 0)


def list_chunk_ids(database_url: str, *, tenant_id: str | None = None) -> list[str]:
    engine = engine_for(database_url)
    with Session(engine) as session:
        stmt = select(RagChunk.chunk_id)
        if tenant_id:
            stmt = stmt.where(RagChunk.tenant_id == tenant_id)
        return [str(value) for value in session.scalars(stmt).all()]


def column_names(database_url: str, table: str) -> set[str]:
    engine = engine_for(database_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = :table"
            ),
            {"table": table},
        ).fetchall()
    return {str(row[0]) for row in rows}


def execute_sql(database_url: str, sql: str) -> None:
    engine = engine_for(database_url)
    with engine.begin() as conn:
        conn.execute(text(sql))
