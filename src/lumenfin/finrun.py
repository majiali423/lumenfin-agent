from __future__ import annotations

from typing import Any

from .metrics_schema import get_fundamental, period_label_from_meta

FINRUN_SCHEMA_VERSION = "1.0"

FORMULA_BY_METRIC = {
    "ebitda_margin": ("ebitda / revenue", {"ebitda": "ebitda", "revenue": "revenue"}),
    "r_and_d_intensity": ("r_and_d / revenue", {"r_and_d": "r_and_d", "revenue": "revenue"}),
    "operating_margin": (
        "operating_income / revenue",
        {"operating_income": "operating_income", "revenue": "revenue"},
    ),
}


def export_finrun_state(state: dict[str, Any]) -> dict[str, Any]:
    """Map a LumenFin exported state into the FinRun evaluation schema."""

    return {
        "schema_version": FINRUN_SCHEMA_VERSION,
        "run_id": str(state.get("run_id") or state.get("thread_id") or "lumenfin-run"),
        "query": str(state.get("query") or ""),
        "metadata": {
            "adapter": "lumenfin",
            "source_project": "lumenfin-agent",
            "thread_id": state.get("thread_id"),
            "workflow_status": state.get("workflow_status"),
            "llm_backend": state.get("llm_backend"),
            "data_mode": state.get("data_mode"),
            "input_guardrail_summary": state.get("input_guardrail_summary") or {},
            "input_guardrail_findings": state.get("input_guardrail_findings") or [],
            "compliance_violations": state.get("compliance_violations") or [],
            "retrieval_provenance": _retrieval_provenance(state),
            "claim_binding": state.get("claim_binding") or {},
            "verified_claim_count": len(state.get("verified_claims") or []),
            "claim_count": len(state.get("claims") or []),
        },
        "entities": [{"name": company} for company in _companies(state)],
        "steps": _steps(state),
        "metrics": _metrics(state),
        "evidence": _evidence(state),
        "market_data": _market_data(state),
        "final_output": str(state.get("final_report") or ""),
        "claims": list(state.get("verified_claims") or []),
    }


def _retrieval_provenance(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    explicit = state.get("retrieval_provenance") or {}
    if explicit:
        merged = {str(company): dict(value) for company, value in explicit.items()}
        for company, bundle in (state.get("retrieved_docs") or {}).items():
            merged.setdefault(str(company), {})["fundamental_provenance"] = dict(
                bundle.get("fundamental_provenance") or {}
            )
        return merged

    derived: dict[str, dict[str, Any]] = {}
    for company, bundle in (state.get("retrieved_docs") or {}).items():
        provenance = bundle.get("provenance")
        if isinstance(provenance, dict):
            derived[str(company)] = dict(provenance)
            continue
        structured_source = str(bundle.get("structured_source") or "none")
        derived[str(company)] = {"structured_source": structured_source}
    return derived


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
        bundle = retrieved_docs.get(company) or {}
        source_values = bundle.get("market_data") or {}
        period = period_label_from_meta(bundle.get("fundamentals_meta"))
        for name, value in metrics.items():
            formula, input_map = FORMULA_BY_METRIC.get(name, ("", {}))
            item = {
                "entity": str(company),
                "name": str(name),
                "period": period,
                "value": value,
                "formula": formula,
                "inputs": _metric_inputs(
                    input_map,
                    source_values,
                    period=period,
                    provenance=bundle.get("fundamental_provenance"),
                ),
                "confidence": _metric_confidence(
                    metric_confidence.get(company) or {},
                    name,
                    bundle,
                ),
            }
            output.append(item)
    return output


def _metric_inputs(
    input_map: dict[str, str],
    source_values: dict[str, Any],
    *,
    period: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = {}
    for input_name, source_key in input_map.items():
        value = get_fundamental(source_values, source_key)
        if value is None:
            continue
        field_provenance = (provenance or {}).get(source_key) or {}
        input_period = field_provenance.get("period") if isinstance(provenance, dict) else period
        inputs[input_name] = {
            "value": value,
            "unit": "billion",
            "currency": "USD",
            "period": input_period,
            "source": field_provenance.get("source") or "market_data",
            "period_source": field_provenance.get("period_source"),
            "period_alignment": field_provenance.get("period_alignment"),
            "citation": field_provenance.get("citation"),
            "source_record_id": (
                field_provenance.get("source_record_id")
                or field_provenance.get("provider_record_id")
            ),
        }
    return inputs


def _metric_confidence(
    company_confidence: dict[str, Any],
    metric_name: str,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    confidence = dict(company_confidence.get(metric_name) or {})
    provenance = bundle.get("provenance") or {}
    if provenance:
        confidence.setdefault("structured_source", provenance.get("structured_source"))
        confidence.setdefault("data_mode", provenance.get("data_mode"))
        confidence.setdefault("market_status", provenance.get("market_status"))
    retrieval_confidence = bundle.get("confidence") or {}
    if retrieval_confidence:
        confidence.setdefault("retrieval_overall", retrieval_confidence.get("overall"))
    return confidence


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
        supply_chain = payload.get("supply_chain") or {}
        if supply_chain:
            signals = [str(signal) for signal in supply_chain.get("signals") or []]
            text = (
                f"{company} supply chain risk level is {supply_chain.get('risk_level', 'unknown')}. "
                f"Signals: {'; '.join(signals)}"
            )
            _append_evidence(
                evidence,
                seen,
                company=str(company),
                citation=f"lumenfin:supply_chain:{company}:{period_label_from_meta(payload.get('fundamentals_meta'))}",
                source_type="sample_db",
                text=text,
            )
        quotes = [str(quote) for quote in payload.get("earnings_call_quotes") or []]
        if quotes:
            _append_evidence(
                evidence,
                seen,
                company=str(company),
                citation=f"lumenfin:earnings_call_quotes:{company}:{period_label_from_meta(payload.get('fundamentals_meta'))}",
                source_type="sample_db",
                text=f"{company} management commentary: {'; '.join(quotes)}",
            )
        market_data = payload.get("market_data") or {}
        if market_data:
            structured = str(payload.get("structured_source") or "sample_financial_data")
            text = (
                f"{company} {period_label_from_meta(payload.get('fundamentals_meta'))} revenue was {get_fundamental(market_data, 'revenue')} billion USD, "
                f"EBITDA was {get_fundamental(market_data, 'ebitda')} billion USD, "
                f"R&D was {get_fundamental(market_data, 'r_and_d')} billion USD, and "
                f"operating income was {get_fundamental(market_data, 'operating_income')} billion USD."
            )
            _append_evidence(
                evidence,
                seen,
                company=str(company),
                citation=f"lumenfin:{structured}:{company}:{period_label_from_meta(payload.get('fundamentals_meta'))}",
                source_type=structured if structured != "none" else "fundamentals",
                text=text,
            )
    # Verified claims contribute their bound evidence (ensures claim text ↔ citation in FinRun).
    for claim in state.get("verified_claims") or []:
        if not isinstance(claim, dict):
            continue
        entity = str(claim.get("entity") or "")
        statement = str(claim.get("statement") or "")
        for ref in claim.get("evidence_refs") or []:
            if not isinstance(ref, dict):
                continue
            citation = str(ref.get("citation") or "")
            if not entity or not citation:
                continue
            text = str(ref.get("text") or "")
            if statement and statement not in text:
                text = f"{statement} Evidence: {text}".strip()
            _append_evidence(
                evidence,
                seen,
                company=entity,
                citation=citation,
                source_type=str(ref.get("source_type") or "claim"),
                text=text,
            )
    for company, scores in (state.get("risk_scores") or {}).items():
        if not isinstance(scores, dict):
            continue
        parts = [
            f"{name}={value}"
            for name, value in scores.items()
            if isinstance(value, (int, float))
        ]
        if parts:
            _append_evidence(
                evidence,
                seen,
                company=str(company),
                citation=f"lumenfin:risk_model:{company}:model",
                source_type="risk_model",
                text=(
                    f"{company} model-derived risk scores are screening indicators, not standalone cited facts: "
                    + ", ".join(parts)
                    + "."
                ),
            )
    for company, snapshot in (state.get("market_snapshots") or {}).items():
        if snapshot.get("current_price") is None:
            continue
        details = []
        for key in ("current_price", "trailing_pe", "monthly_return", "fifty_two_week_high", "fifty_two_week_low"):
            if snapshot.get(key) is not None:
                details.append(f"{key}={snapshot.get(key)}")
        if details:
            _append_evidence(
                evidence,
                seen,
                company=str(company),
                citation=f"lumenfin:market_snapshot:{company}:{snapshot.get('fetched_at') or 'latest'}",
                source_type="market_data",
                text=(
                    f"{company} live market snapshot from {snapshot.get('provider') or 'unknown'} "
                    f"as_of={snapshot.get('fetched_at') or 'n/a'}: "
                    + ", ".join(details)
                    + "."
                ),
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
    period: str | None = None,
) -> None:
    key = (company, citation)
    if key in seen:
        return
    seen.add(key)
    evidence.append(
        {
            "entity": company,
            "citation": citation,
            "period": period or "latest",
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
                "current_price": snapshot.get("current_price"),
                "trailing_pe": snapshot.get("trailing_pe"),
                "monthly_return": snapshot.get("monthly_return"),
                "fifty_two_week_high": snapshot.get("fifty_two_week_high"),
                "fifty_two_week_low": snapshot.get("fifty_two_week_low"),
            }
        )
    return output
