from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import inspect, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .database import Base, WorkflowCheckpoint, utc_now


class CheckpointConflictError(RuntimeError):
    """Raised when a checkpoint write is based on a stale revision."""


def infer_last_node(state: dict[str, Any]) -> str:
    workflow_status = state.get("workflow_status", "running")
    if workflow_status == "needs_clarification":
        return "await_clarification"
    if workflow_status == "blocked_by_guardrail":
        return "input_guardrail"
    audit_log = state.get("audit_log") or []
    if audit_log:
        return str(audit_log[-1].get("step") or "query_planner")
    return "query_planner"


class WorkflowCheckpointRepository:
    """SQLite-backed workflow snapshots for HITL resume across process restarts."""

    def __init__(self, engine) -> None:
        self.engine = engine
        Base.metadata.create_all(self.engine)
        self._ensure_sqlite_revision_column()
        self._validate_external_revision_column()

    def _ensure_sqlite_revision_column(self) -> None:
        if not str(self.engine.url).startswith("sqlite"):
            return
        with self.engine.begin() as conn:
            rows = conn.exec_driver_sql("PRAGMA table_info(workflow_checkpoints)").fetchall()
            columns = {str(row[1]) for row in rows}
            if rows and "revision" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE workflow_checkpoints "
                    "ADD COLUMN revision INTEGER NOT NULL DEFAULT 1"
                )
            if rows and "tenant_id" not in columns:
                conn.exec_driver_sql(
                    "ALTER TABLE workflow_checkpoints "
                    "ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
                )

    def _validate_external_revision_column(self) -> None:
        if str(self.engine.url).startswith("sqlite"):
            return
        schema = inspect(self.engine)
        if not schema.has_table("workflow_checkpoints"):
            return
        columns = {str(column["name"]) for column in schema.get_columns("workflow_checkpoints")}
        if "revision" not in columns:
            raise RuntimeError(
                "Database schema is missing workflow_checkpoints.revision. "
                "Apply migrations/postgresql/001_add_workflow_checkpoint_revision.sql "
                "with psql before starting LumenFin."
            )

    @classmethod
    def from_database_url(cls, database_url: str, db_path=None) -> "WorkflowCheckpointRepository":
        from .database import JobRepository

        repo = JobRepository(database_url, db_path=db_path)
        return cls(repo.engine)

    def upsert(
        self,
        *,
        thread_id: str,
        query: str,
        state: dict[str, Any],
        llm_backend: str | None = None,
        expected_revision: int | None = None,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        if expected_revision is None:
            raise ValueError("expected_revision is required for checkpoint writes")
        if expected_revision < 0:
            raise ValueError("expected_revision must be non-negative")
        now = utc_now()
        tenant = (tenant_id or "default").strip() or "default"
        payload = json.dumps(state, ensure_ascii=False, default=str)
        clarification_questions = json.dumps(
            state.get("clarification_questions") or [],
            ensure_ascii=False,
        )
        last_node = infer_last_node(state)
        workflow_status = str(state.get("workflow_status") or "running")
        with Session(self.engine) as session:
            if expected_revision == 0:
                row = WorkflowCheckpoint(
                    thread_id=thread_id,
                    tenant_id=tenant,
                    query=query,
                    workflow_status=workflow_status,
                    state_json=payload,
                    clarification_questions_json=clarification_questions,
                    last_node=last_node,
                    llm_backend=llm_backend,
                    created_at=now,
                    updated_at=now,
                    revision=1,
                )
                session.add(row)
                try:
                    session.flush()
                    committed = self._row_to_dict(row)
                    session.commit()
                except IntegrityError as exc:
                    session.rollback()
                    raise CheckpointConflictError(
                        f"Checkpoint conflict for thread_id={thread_id}: expected revision 0"
                    ) from exc
            else:
                values: dict[str, Any] = {
                    "query": query,
                    "workflow_status": workflow_status,
                    "state_json": payload,
                    "clarification_questions_json": clarification_questions,
                    "last_node": last_node,
                    "updated_at": now,
                    "revision": expected_revision + 1,
                    "tenant_id": tenant,
                }
                if llm_backend is not None:
                    values["llm_backend"] = llm_backend
                result = session.execute(
                    update(WorkflowCheckpoint)
                    .where(
                        WorkflowCheckpoint.thread_id == thread_id,
                        WorkflowCheckpoint.revision == expected_revision,
                    )
                    .values(**values)
                )
                if result.rowcount != 1:
                    session.rollback()
                    raise CheckpointConflictError(
                        f"Checkpoint conflict for thread_id={thread_id}: "
                        f"expected revision {expected_revision}"
                    )
                row = session.get(WorkflowCheckpoint, thread_id)
                assert row is not None
                committed = self._row_to_dict(row)
                session.commit()
        return committed

    def get(self, thread_id: str, *, tenant_id: str | None = None) -> Optional[dict[str, Any]]:
        with Session(self.engine) as session:
            row = session.get(WorkflowCheckpoint, thread_id)
            if row is None:
                return None
            if tenant_id is not None and str(getattr(row, "tenant_id", "default") or "default") != str(
                tenant_id
            ):
                return None
            return self._row_to_dict(row)

    def delete(self, thread_id: str) -> None:
        with Session(self.engine) as session:
            row = session.get(WorkflowCheckpoint, thread_id)
            if row is not None:
                session.delete(row)
                session.commit()

    def list_threads(self, limit: int = 20) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(WorkflowCheckpoint).order_by(WorkflowCheckpoint.updated_at.desc()).limit(limit)
            ).all()
            return [self._row_to_dict(row) for row in rows]

    def load_state(self, thread_id: str) -> Optional[dict[str, Any]]:
        record = self.get(thread_id)
        if record is None:
            return None
        return dict(record["state"])

    @staticmethod
    def _row_to_dict(row: WorkflowCheckpoint) -> dict[str, Any]:
        return {
            "thread_id": row.thread_id,
            "tenant_id": getattr(row, "tenant_id", None) or "default",
            "query": row.query,
            "workflow_status": row.workflow_status,
            "state": json.loads(row.state_json),
            "clarification_questions": json.loads(row.clarification_questions_json or "[]"),
            "last_node": row.last_node,
            "llm_backend": row.llm_backend,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "revision": row.revision,
        }
