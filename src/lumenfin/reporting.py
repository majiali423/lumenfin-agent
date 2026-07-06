from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .evaluation import evaluate_run_state


def build_data_sources(
    result: dict[str, Any],
    *,
    llm_backend: str | None = None,
    embedding_provider: str = "deterministic",
    rag_enabled: bool = True,
    market_provider: str = "yahoo",
) -> dict[str, Any]:
    companies = list(result.get("companies") or [])
    document_contexts = list(result.get("document_contexts") or [])
    rag_evidence = result.get("rag_evidence") or {}
    rag_index_stats = result.get("rag_index_stats") or {}
    market_snapshots = result.get("market_snapshots") or {}

    structured_source = "none"
    has_structured_upload = any(
        doc.get("source_type") == "structured_json"
        or str(doc.get("filename", "")).endswith("_metrics.json")
        for doc in document_contexts
    )
    has_pdf = any(
        doc.get("source_type") != "structured_json"
        and not str(doc.get("filename", "")).endswith("_metrics.json")
        for doc in document_contexts
    )
    if has_structured_upload:
        structured_source = "uploaded_json"
    elif any(company in SAMPLE_FINANCIAL_DATA for company in companies):
        structured_source = "sample_db"
    elif document_contexts and not has_structured_upload:
        structured_source = "pdf_extracted"

    rag_used = any(bool(hits) for hits in rag_evidence.values())
    chunks_indexed = int(rag_index_stats.get("chunks_indexed") or 0)
    if not rag_enabled:
        rag_status = "disabled"
    elif rag_used:
        rag_status = "milvus_hybrid"
    elif chunks_indexed > 0:
        rag_status = "indexed_no_hits"
    elif has_pdf:
        rag_status = "pdf_no_index"
    else:
        rag_status = "skipped"

    market_ok = False
    resolved_market_provider = market_provider
    for snapshot in market_snapshots.values():
        if snapshot.get("provider"):
            resolved_market_provider = str(snapshot.get("provider"))
        if snapshot.get("current_price") is not None:
            market_ok = True
            break

    return {
        "structured": structured_source,
        "market": resolved_market_provider,
        "market_ok": market_ok,
        "rag": rag_status,
        "llm": llm_backend or result.get("llm_backend") or "unknown",
        "embedding": embedding_provider,
        "pdf_uploaded": has_pdf,
        "structured_uploaded": has_structured_upload,
    }


def build_run_manifest(
    result: dict[str, Any],
    *,
    thread_id: str,
    llm_backend: str | None = None,
    artifact_paths: dict[str, str] | None = None,
    embedding_provider: str = "deterministic",
    rag_enabled: bool = True,
    market_provider: str = "yahoo",
) -> dict[str, Any]:
    telemetry = result.get("run_telemetry") or {}
    evaluation = evaluate_run_state(result)
    guardrail_findings = result.get("input_guardrail_findings") or []
    spans = telemetry.get("node_spans") or []
    started_at = result.get("run_started_at") or (spans[0].get("started_at") if spans else None)
    ended_at = result.get("run_ended_at") or (spans[-1].get("ended_at") if spans else None)
    artifacts = artifact_paths or {}
    resolved_backend = llm_backend or result.get("llm_backend")
    return {
        "thread_id": thread_id,
        "workflow_status": result.get("workflow_status"),
        "llm_backend": resolved_backend,
        "started_at": started_at,
        "ended_at": ended_at,
        "total_latency_ms": telemetry.get("total_latency_ms", 0),
        "total_prompt_tokens": telemetry.get("total_prompt_tokens", 0),
        "total_completion_tokens": telemetry.get("total_completion_tokens", 0),
        "degraded_mode": bool(result.get("degraded_mode")),
        "guardrail_findings": len(guardrail_findings),
        "evaluator_score": evaluation.score,
        "evaluator_grade": evaluation.grade,
        "data_sources": build_data_sources(
            result,
            llm_backend=resolved_backend,
            embedding_provider=embedding_provider,
            rag_enabled=rag_enabled,
            market_provider=market_provider,
        ),
        "artifacts": {
            "report": artifacts.get("report_path"),
            "state": artifacts.get("state_path"),
            "audit": artifacts.get("audit_path"),
            "manifest": artifacts.get("manifest_path"),
        },
    }


def load_run_manifest(artifact_paths: dict[str, str]) -> dict[str, Any] | None:
    manifest_path = artifact_paths.get("manifest_path")
    if not manifest_path:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def export_run_artifacts(
    result: dict[str, Any],
    output_dir: Path,
    thread_id: str,
    *,
    llm_backend: str | None = None,
    embedding_provider: str = "deterministic",
    rag_enabled: bool = True,
    market_provider: str = "yahoo",
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{thread_id}_{timestamp}"

    manifest_path = output_dir / f"{base_name}_manifest.json"
    artifacts: dict[str, str] = {}

    workflow_status = result.get("workflow_status")
    if workflow_status == "completed":
        report_path = output_dir / f"{base_name}_report.md"
        audit_path = output_dir / f"{base_name}_audit.json"
        state_path = output_dir / f"{base_name}_state.json"

        report_path.write_text(result.get("final_report", ""), encoding="utf-8")
        audit_path.write_text(
            json.dumps(result.get("audit_log", []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        artifacts.update(
            {
                "report_path": str(report_path),
                "audit_path": str(audit_path),
                "state_path": str(state_path),
            }
        )

    manifest = build_run_manifest(
        result,
        thread_id=thread_id,
        llm_backend=llm_backend,
        artifact_paths={**artifacts, "manifest_path": str(manifest_path)},
        embedding_provider=embedding_provider,
        rag_enabled=rag_enabled,
        market_provider=market_provider,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["manifest_path"] = str(manifest_path)
    return artifacts
