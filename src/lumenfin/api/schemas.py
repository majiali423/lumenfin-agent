from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalyzeDataRequest(BaseModel):
    query: str = Field(..., description="User query for financial multi-agent analysis.")
    company_metrics: dict[str, dict[str, Any]] = Field(
        ...,
        description='Structured metrics per company, e.g. {"NVIDIA": {"revenue_2025": 130.5, "ebitda_2025": 75.2}}.',
    )
    thread_id: Optional[str] = Field(default=None, description="Optional workflow thread id.")
    export_artifacts: bool = Field(default=True, description="Whether to persist report and state files.")
    include_state: bool = Field(
        default=False,
        description="When true, return the full internal run state. Default is a compact summary only.",
    )


class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="User query for financial multi-agent analysis.")
    thread_id: Optional[str] = Field(default=None, description="Optional workflow thread id.")
    export_artifacts: bool = Field(default=True, description="Whether to persist report and state files.")
    include_state: bool = Field(
        default=False,
        description="When true, return the full internal run state. Default is a compact summary only.",
    )


class AnalyzeResponse(BaseModel):
    thread_id: str
    llm_backend: str
    workflow_status: str = "completed"
    clarification_questions: list[str] = Field(default_factory=list)
    final_report: str
    executive_summary: Optional[str] = None
    compliance_summary: Optional[str] = None
    audit_log: list[dict[str, Any]]
    artifacts: dict[str, str]
    state: dict[str, Any]
    chart_data: Optional[dict[str, Any]] = None
    run_telemetry: Optional[dict[str, Any]] = None
    run_manifest: Optional[dict[str, Any]] = None
    provider_health: Optional[dict[str, Any]] = None
    checkpoint: Optional[dict[str, Any]] = None


class ClarifyRequest(BaseModel):
    thread_id: str = Field(..., description="Existing workflow thread awaiting clarification.")
    clarification: dict[str, Any] = Field(
        ...,
        description="Structured answers, e.g. {\"company\": \"Apple\", \"time_range\": \"FY2025\"}.",
    )
    export_artifacts: bool = Field(default=True, description="Whether to persist report and state files.")
    include_state: bool = Field(
        default=False,
        description="When true, return the full internal run state. Default is a compact summary only.",
    )


class HealthResponse(BaseModel):
    status: str
    llm_backend: str
    llm_configured: bool = False
    market_provider: str = "yahoo"
    market_provider_ok: bool = False
    embedding_provider: str = "deterministic"
    rag_enabled: bool = True


class SubmitJobRequest(BaseModel):
    query: str = Field(..., description="User query for asynchronous financial analysis.")
    thread_id: Optional[str] = Field(default=None, description="Optional workflow thread id.")
    export_artifacts: bool = Field(default=True, description="Whether the background job should export files.")


class SubmitJobResponse(BaseModel):
    job_id: str
    thread_id: str
    status: str
    queue_backend: Optional[str] = None


class JobResponse(BaseModel):
    job_id: str
    thread_id: str
    query: str
    status: str
    llm_backend: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: str
    updated_at: str
