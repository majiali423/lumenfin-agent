import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import Column, Text, create_engine, select
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
