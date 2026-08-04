from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalyzeDataRequest(BaseModel):
    query: str = Field(..., description="User query for financial multi-agent analysis.")
    company_metrics: dict[str, dict[str, Any]] = Field(
        ...,
        description='Structured metrics per company, e.g. {"NVIDIA": {"revenue": 130.5, "ebitda": 75.2}}.',
    )
    thread_id: Optional[str] = Field(default=None, description="Optional workflow thread id.")
    export_artifacts: bool = Field(default=True, description="Whether to persist report and state files.")
    include_state: bool = Field(
        default=False,
        description="When true, return the full internal run state. Default is a compact summary only.",
    )
    output_format: Optional[str] = Field(
        default=None,
        description=(
            "Explicit report length mode: research_report | executive_summary | table_summary. "
            "Omitted/invalid values keep the full research report (keywords never auto-trim)."
        ),
    )


class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="User query for financial multi-agent analysis.")
    thread_id: Optional[str] = Field(default=None, description="Optional workflow thread id.")
    export_artifacts: bool = Field(default=True, description="Whether to persist report and state files.")
    include_state: bool = Field(
        default=False,
        description="When true, return the full internal run state. Default is a compact summary only.",
    )
    output_format: Optional[str] = Field(
        default=None,
        description=(
            "Explicit report length mode: research_report | executive_summary | table_summary. "
            "Omitted/invalid values keep the full research report (keywords never auto-trim)."
        ),
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
    degraded: bool = False
    provider_degraded: Optional[dict[str, Any]] = None
    provider_call_summary: Optional[dict[str, Any]] = None


class ClarifyRequest(BaseModel):
    thread_id: str = Field(..., description="Existing workflow thread awaiting clarification.")
    clarification: dict[str, Any] = Field(
        ...,
        description=(
            "Structured answers, e.g. "
            '{"company": "Apple", "time_range": "FY2025"} or '
            '{"company_scope": "uploaded|query|both"} when query issuers disagree with uploads.'
        ),
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
    pid: Optional[int] = None
    worker_id: Optional[str] = None


class ProviderProbeRequest(BaseModel):
    scenario: str = Field(default="success", description="Provider stub scenario name (integration/test only).")
    prompt: str = Field(default="ping", description="User prompt for a single logical chat call.")
    max_attempts: Optional[int] = Field(default=None, description="Optional override for LLM max attempts.")


class ProviderProbeResponse(BaseModel):
    ok: bool
    degraded: bool = False
    fallback: bool = False
    attempts: int = 0
    error_class: Optional[str] = None
    text_preview: str = ""
    request_id: str
    thread_id: Optional[str] = None
    worker_id: Optional[str] = None
    pid: int
    hostname: Optional[str] = None
    container_id_hint: Optional[str] = None
    client_instance_id: Optional[str] = None
    client_id: Optional[str] = None
    llm_inflight_current: int = 0
    llm_max_inflight_seen: int = 0
    llm_max_inflight_configured: int = 0
    trace: list[dict[str, Any]] = Field(default_factory=list)
    provider_call_summary: dict[str, Any] = Field(default_factory=dict)


class ProviderIdentityResponse(BaseModel):
    worker_id: Optional[str] = None
    pid: int
    hostname: str
    container_id_hint: Optional[str] = None
    http_client_instance_id: Optional[str] = None
    llm_inflight_current: int = 0
    llm_max_inflight_seen: int = 0
    llm_max_inflight_configured: int = 0
    note: str = "per-process bulkhead ≠ cross-process global rate limit"


class SubmitJobRequest(BaseModel):
    query: str = Field(..., description="User query for asynchronous financial analysis.")
    thread_id: Optional[str] = Field(default=None, description="Optional workflow thread id.")
    export_artifacts: bool = Field(default=True, description="Whether the background job should export files.")
    output_format: Optional[str] = Field(
        default=None,
        description="Explicit report length mode; same semantics as AnalyzeRequest.output_format.",
    )


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


class DocumentReceipt(BaseModel):
    document_id: str
    filename: str
    content_hash: str
    status: str
    chunk_count: int = 0
    error: Optional[str] = None
    embed_calls: int = 0


class DocumentIndexResponse(BaseModel):
    tenant_id: str
    documents: list[DocumentReceipt]


class DocumentStatusResponse(BaseModel):
    document_id: str
    tenant_id: str
    filename: str
    content_hash: str
    index_status: str
    chunk_count: int = 0
    error: Optional[str] = None
    indexed_at: Optional[str] = None
