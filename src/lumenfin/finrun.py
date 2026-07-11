from __future__ import annotations

from typing import Any


FORMULA_BY_METRIC = {
    "ebitda_margin": ("ebitda / revenue", {"ebitda": "ebitda_2025", "revenue": "revenue_2025"}),
    "r_and_d_intensity": ("r_and_d / revenue", {"r_and_d": "r_and_d_2025", "revenue": "revenue_2025"}),
    "operating_margin": ("operating_income / revenue", {"operating_income": "operating_income_2025", "revenue": "revenue_2025"}),
}


def export_finrun_state(state: dict[str, Any]) -> dict[str, Any]:
    """Map a LumenFin exported state into the FinRun evaluation schema."""

    return {
        "run_id": str(state.get("run_id") or state.get("thread_id") or "lumenfin-run"),
        "query": str(state.get("query") or ""),
        "metadata": {
            "adapter": "lumenfin",
            "source_project": "lumenfin-agent",
            "thread_id": state.get("thread_id"),
            "workflow_status": state.get("workflow_status"),
            "llm_backend": state.get("llm_backend"),
        },
        "entities": [{"name": company} for company in _companies(state)],
        "steps": _steps(state),
        "metrics": _metrics(state),
        "evidence": _evidence(state),
        "market_data": _market_data(state),
        "final_output": str(state.get("final_report") or ""),
    }


def _companies(state: dict[str, Any]) -> list[str]:
    companies = state.get("companies") or []
    return [str(company) for company in companies]


def _steps(state: dict[str, Any]) -> list[dict[str, str]]:
    steps = []
    for event in state.get("audit_log") or []:
        name = event.get("step")
        if name:
            steps.append({"name": str(name), "status": str(event.get("status") or "ok")})
    return steps


def _metrics(state: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    financial_metrics = state.get("financial_metrics") or {}
    retrieved_docs = state.get("retrieved_docs") or {}
    metric_confidence = state.get("metric_confidence") or {}
    for company, metrics in financial_metrics.items():
        source_values = (retrieved_docs.get(company) or {}).get("market_data") or {}
        for name, value in metrics.items():
            formula, input_map = FORMULA_BY_METRIC.get(name, ("", {}))
            item = {
                "entity": str(company),
                "name": str(name),
                "period": "FY2025",
                "value": value,
                "formula": formula,
                "inputs": _metric_inputs(input_map, source_values),
                "confidence": (metric_confidence.get(company) or {}).get(name, {}),
            }
            output.append(item)
    return output


def _metric_inputs(input_map: dict[str, str], source_values: dict[str, Any]) -> dict[str, Any]:
    inputs = {}
    for input_name, source_key in input_map.items():
        if source_key not in source_values:
            continue
        inputs[input_name] = {
            "value": source_values[source_key],
            "unit": "billion",
            "currency": "USD",
            "period": "FY2025",
        }
    return inputs


def _evidence(state: dict[str, Any]) -> list[dict[str, str]]:
    evidence = []
    seen = set()
    retrieved_docs = state.get("retrieved_docs") or {}
    rag_evidence = state.get("rag_evidence") or {}

    for company, hits in rag_evidence.items():
        for index, hit in enumerate(hits):
            citation = str(hit.get("citation") or hit.get("source") or hit.get("filename") or f"rag:{company}:{index}")
            _append_evidence(
                evidence,
                seen,
                company=str(company),
                citation=citation,
                source_type=str(hit.get("source_type") or "rag"),
                text=str(hit.get("text") or hit.get("snippet") or hit.get("excerpt") or ""),
            )

    for company, payload in retrieved_docs.items():
        for index, doc in enumerate(payload.get("source_documents") or []):
            citation = str(doc.get("citation") or doc.get("filename") or doc.get("source") or f"source:{company}:{index}")
            _append_evidence(
                evidence,
                seen,
                company=str(company),
                citation=citation,
                source_type=str(doc.get("source_type") or "document"),
                text=str(doc.get("excerpt") or doc.get("text") or ""),
            )
        market_data = payload.get("market_data") or {}
        if market_data:
            text = (
                f"{company} FY2025 revenue was {market_data.get('revenue_2025')} billion USD, "
                f"EBITDA was {market_data.get('ebitda_2025')} billion USD, "
                f"R&D was {market_data.get('r_and_d_2025')} billion USD, and "
                f"operating income was {market_data.get('operating_income_2025')} billion USD."
            )
            _append_evidence(
                evidence,
                seen,
                company=str(company),
                citation=f"lumenfin:sample_financial_data:{company}:FY2025",
                source_type="sample_db",
                text=text,
            )
    return evidence


def _append_evidence(
    evidence: list[dict[str, str]],
    seen: set[tuple[str, str]],
    *,
    company: str,
    citation: str,
    source_type: str,
    text: str,
) -> None:
    key = (company, citation)
    if key in seen:
        return
    seen.add(key)
    evidence.append(
        {
            "entity": company,
            "citation": citation,
            "period": "FY2025",
            "source_type": source_type,
            "provider": "lumenfin",
            "text": text,
        }
    )


def _market_data(state: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for company, snapshot in (state.get("market_snapshots") or {}).items():
        output.append(
            {
                "entity": str(company),
                "status": str(snapshot.get("status") or ("ok" if snapshot.get("current_price") is not None else "failed")),
                "provider": snapshot.get("provider") or "",
                "as_of": snapshot.get("fetched_at") or snapshot.get("as_of") or "",
                "error": snapshot.get("error") or "",
            }
        )
    return output
