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
    tool_backend: str | None = None,
) -> dict[str, Any]:
    companies = list(result.get("companies") or [])
    document_contexts = list(result.get("document_contexts") or [])
    retrieved_docs = result.get("retrieved_docs") or {}
    rag_evidence = result.get("rag_evidence") or {}
    rag_index_stats = result.get("rag_index_stats") or {}
    market_snapshots = result.get("market_snapshots") or {}
    market_data_status = result.get("market_data_status") or {}
    data_mode = str(result.get("data_mode") or "demo").lower()
    allow_sample = data_mode == "demo"

    structured_source = "none"
    source_types = {
        str(doc.get("source_type") or "").strip().lower()
        for doc in document_contexts
        if doc.get("source_type")
    }
    has_structured_upload = any(
        doc.get("source_type") in {"structured_json", "csv", "excel"}
        or str(doc.get("filename", "")).endswith("_metrics.json")
        for doc in document_contexts
    )
    has_narrative_upload = any(
        doc.get("source_type") in {"pdf", "markdown"}
        for doc in document_contexts
    )
    has_pdf = "pdf" in source_types or any(
        str(doc.get("filename", "")).lower().endswith(".pdf") for doc in document_contexts
    )
    if has_structured_upload:
        if "csv" in source_types:
            structured_source = "uploaded_csv"
        elif "excel" in source_types:
            structured_source = "uploaded_excel"
        else:
            structured_source = "uploaded_json"
    elif allow_sample and any(company in SAMPLE_FINANCIAL_DATA for company in companies):
        structured_source = "sample_db"
    elif any(
        str((retrieved_docs.get(c) or {}).get("structured_source") or "") == "sec_companyfacts"
        for c in companies
    ):
        structured_source = "sec_companyfacts"
    elif any(
        str((retrieved_docs.get(c) or {}).get("structured_source") or "") == "yahoo_fundamentals"
        for c in companies
    ):
        structured_source = "yahoo_fundamentals"
    elif has_narrative_upload:
        structured_source = "document_extracted"

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
    elif has_narrative_upload:
        rag_status = "document_no_index"
    else:
        rag_status = "skipped"

    market_ok = False
    resolved_market_provider = market_provider
    market_ok_count = int(market_data_status.get("ok_count") or 0)
    market_total_count = int(market_data_status.get("total_count") or 0)
    for snapshot in market_snapshots.values():
        if snapshot.get("provider"):
            resolved_market_provider = str(snapshot.get("provider"))
        if snapshot.get("current_price") is not None:
            market_ok = True
    if market_total_count and market_ok_count == 0:
        market_ok = False

    per_company_market = market_data_status.get("companies") or {}
    if not per_company_market:
        per_company_market = {
            company: {
                "status": snap.get("status") or ("ok" if snap.get("current_price") is not None else "failed"),
                "provider": snap.get("provider"),
                "fetched_at": snap.get("fetched_at"),
                "has_price": snap.get("current_price") is not None,
            }
            for company, snap in market_snapshots.items()
        }

    return {
        "data_mode": data_mode,
        "structured": structured_source,
        "market": resolved_market_provider,
        "market_ok": market_ok,
        "market_ok_count": market_ok_count or sum(1 for s in market_snapshots.values() if s.get("current_price") is not None),
        "market_total_count": market_total_count or len(market_snapshots),
        "market_by_company": per_company_market,
        "rag": rag_status,
        "llm": llm_backend or result.get("llm_backend") or "unknown",
        "embedding": embedding_provider,
        "tool_transport": tool_backend or result.get("tool_backend") or "local",
        "pdf_uploaded": has_pdf,
        "structured_uploaded": has_structured_upload,
        "upload_formats": sorted(source_types) if source_types else [],
        "markdown_uploaded": "markdown" in source_types,
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
            tool_backend=result.get("tool_backend"),
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
    exportable = workflow_status in {
        "completed",
        "incomplete_data",
        "needs_clarification",
        "blocked_by_guardrail",
    }
    if exportable:
        report_path = output_dir / f"{base_name}_report.md"
        audit_path = output_dir / f"{base_name}_audit.json"
        state_path = output_dir / f"{base_name}_state.json"

        report_path.write_text(result.get("final_report", "") or "", encoding="utf-8")
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
