from typing import Any, Optional

from typing_extensions import TypedDict


class AuditEvent(TypedDict, total=False):
    step: str
    status: str
    detail: str
    started_at: str
    ended_at: str
    latency_ms: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    tool_calls: int
    retry_count: int


class FinanceState(TypedDict, total=False):
    query: str
    companies: list[str]
    target_symbols: dict[str, str]
    query_plan: dict[str, Any]
    required_skills: list[str]
    skill_specs: list[dict[str, Any]]
    missing_fields: list[str]
    clarification_questions: list[str]
    plan: list[str]
    task_brief: str
    document_contexts: list[dict[str, Any]]
    retrieved_docs: dict[str, dict[str, Any]]
    market_snapshots: dict[str, dict[str, Any]]
    market_data_status: dict[str, Any]
    appendix_search_done: bool
    financial_metrics: dict[str, dict[str, float]]
    metric_confidence: dict[str, dict[str, dict[str, Any]]]
    sentiment_analysis: dict[str, dict[str, Any]]
    compliance_findings: list[str]
    compliance_violations: list[dict[str, Any]]
    compliance_summary: str
    report_sections: list[str]
    executive_summary: str
    final_report: str
    llm_backend: str
    audit_log: list[AuditEvent]
    knowledge_snapshot: dict[str, Any]
    reasoning_memory: list[str]
    replan_reason: Optional[str]
    retries: int
    degraded_mode: bool
    fatal_data_gap: bool
    data_gap_detail: str
    company_profiles: dict[str, str]
    swot_analysis: dict[str, dict[str, str]]
    risk_scores: dict[str, dict[str, float]]
    investment_thesis: dict[str, dict[str, str]]
    chart_data: dict[str, Any]
    peer_comparison: dict[str, Any]
    thread_id: str
    rag_evidence: dict[str, list[dict[str, Any]]]
    rag_index_stats: dict[str, Any]
    critic_iterations: int
    critic_max_iterations: int
    critic_repair_target: str
    workflow_status: str
    user_clarification: dict[str, Any]
    run_telemetry: dict[str, Any]
    run_started_at: str
    run_ended_at: str
    input_guardrail_findings: list[dict[str, Any]]
    input_guardrail_summary: dict[str, Any]
    retrieval_provenance: dict[str, dict[str, Any]]
    data_mode: str
