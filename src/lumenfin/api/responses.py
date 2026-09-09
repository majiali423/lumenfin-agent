"""Map service payloads to public API models (no route handlers)."""

from __future__ import annotations

from typing import Any

from ..config import AppConfig
from ..provider_resilience import redact_provider_message
from ..reporting import build_run_manifest, load_run_manifest
from ..structured_answer import public_structured_answer_fields
from .schemas import AnalyzeResponse


def compact_state(result: dict[str, Any], *, data_mode: str) -> dict[str, Any]:
    compact = {
        "run_id": result.get("run_id"),
        "thread_id": result.get("thread_id"),
        "companies": result.get("companies"),
        "workflow_status": result.get("workflow_status"),
        "degraded_mode": result.get("degraded_mode"),
        "fatal_data_gap": result.get("fatal_data_gap"),
        "data_mode": result.get("data_mode") or data_mode,
        "llm_backend": result.get("llm_backend"),
        "clarification_questions": result.get("clarification_questions", []),
        "audit_log": result.get("audit_log") or [],
        "financial_metrics": result.get("financial_metrics") or {},
        "risk_scores": result.get("risk_scores") or {},
        "sentiment_analysis": result.get("sentiment_analysis") or {},
    }
    structured = public_structured_answer_fields(result)
    if structured is not None:
        compact["answer"] = structured["answer"]
        compact["citations"] = structured["citations"]
        compact["structured_answer_schema_version"] = structured[
            "structured_answer_schema_version"
        ]
    return compact


def public_job(job: dict[str, Any], *, data_mode: str) -> dict[str, Any]:
    public = dict(job)
    for internal in (
        "execution_token",
        "execution_owner",
        "execution_attempt",
        "delivery_state",
        "delivery_payload",
    ):
        public.pop(internal, None)
    result = job.get("result")
    if isinstance(result, dict):
        public_result = compact_state(result, data_mode=data_mode)
        for key in (
            "final_report",
            "executive_summary",
            "compliance_summary",
            "chart_data",
            "run_manifest",
            "run_telemetry",
            "answer",
            "citations",
        ):
            if key in result:
                public_result[key] = result.get(key)
        public["result"] = public_result
        public["workflow_status"] = result.get("workflow_status") or job.get("status")
        public["audit_log"] = list(result.get("audit_log") or [])
    if public.get("error_message"):
        public["error_message"] = redact_provider_message(str(public["error_message"]))
    return public


def analyze_response(
    payload: dict[str, Any],
    *,
    app_config: AppConfig,
    include_state: bool = False,
) -> AnalyzeResponse:
    result = payload["result"]
    artifacts = payload.get("artifacts", {})
    run_manifest = load_run_manifest(artifacts) or build_run_manifest(
        result,
        thread_id=payload["thread_id"],
        llm_backend=payload.get("llm_backend"),
        artifact_paths=artifacts,
        embedding_provider=app_config.embedding_provider,
        rag_enabled=app_config.rag_enabled,
        market_provider=app_config.market_data_provider,
    )
    checkpoint = payload.get("checkpoint")
    if checkpoint and "state" in checkpoint:
        checkpoint = {
            "thread_id": checkpoint.get("thread_id"),
            "workflow_status": checkpoint.get("workflow_status"),
            "last_node": checkpoint.get("last_node"),
            "clarification_questions": checkpoint.get("clarification_questions"),
            "revision": checkpoint.get("revision"),
            "created_at": checkpoint.get("created_at"),
            "updated_at": checkpoint.get("updated_at"),
        }
    structured = public_structured_answer_fields(result)
    return AnalyzeResponse(
        thread_id=payload["thread_id"],
        llm_backend=payload["llm_backend"],
        workflow_status=payload.get("workflow_status", result.get("workflow_status", "completed")),
        clarification_questions=result.get("clarification_questions", []),
        final_report=result.get("final_report", ""),
        executive_summary=result.get("executive_summary"),
        compliance_summary=result.get("compliance_summary"),
        audit_log=result.get("audit_log", []),
        artifacts=artifacts,
        state=result if include_state else compact_state(result, data_mode=app_config.data_mode),
        chart_data=result.get("chart_data"),
        run_telemetry=result.get("run_telemetry"),
        run_manifest=run_manifest,
        provider_health=payload.get("provider_health"),
        checkpoint=checkpoint,
        degraded=bool(payload.get("degraded")),
        provider_degraded=payload.get("provider_degraded"),
        provider_call_summary=payload.get("provider_call_summary"),
        answer=None if structured is None else structured["answer"],
        citations=[] if structured is None else structured["citations"],
        structured_answer_schema_version=(
            None if structured is None else structured["structured_answer_schema_version"]
        ),
    )
