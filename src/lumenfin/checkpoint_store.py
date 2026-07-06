from __future__ import annotations

import json
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import Base, WorkflowCheckpoint, utc_now


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
    ) -> dict[str, Any]:
        now = utc_now()
        payload = json.dumps(state, ensure_ascii=False, default=str)
        clarification_questions = json.dumps(
            state.get("clarification_questions") or [],
            ensure_ascii=False,
        )
        last_node = infer_last_node(state)
        workflow_status = str(state.get("workflow_status") or "running")
        with Session(self.engine) as session:
            row = session.get(WorkflowCheckpoint, thread_id)
            if row is None:
                row = WorkflowCheckpoint(
                    thread_id=thread_id,
                    query=query,
                    workflow_status=workflow_status,
                    state_json=payload,
                    clarification_questions_json=clarification_questions,
                    last_node=last_node,
                    llm_backend=llm_backend,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.query = query or row.query
                row.workflow_status = workflow_status
                row.state_json = payload
                row.clarification_questions_json = clarification_questions
                row.last_node = last_node
                if llm_backend is not None:
                    row.llm_backend = llm_backend
                row.updated_at = now
            session.commit()
        return self.get(thread_id) or {}

    def get(self, thread_id: str) -> Optional[dict[str, Any]]:
        with Session(self.engine) as session:
            row = session.get(WorkflowCheckpoint, thread_id)
            return self._row_to_dict(row) if row is not None else None

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
            "query": row.query,
            "workflow_status": row.workflow_status,
            "state": json.loads(row.state_json),
            "clarification_questions": json.loads(row.clarification_questions_json or "[]"),
            "last_node": row.last_node,
            "llm_backend": row.llm_backend,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
