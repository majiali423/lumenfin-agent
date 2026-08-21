"""Evidence pool collection and numeric claim verification."""

from __future__ import annotations

import re
from typing import Any

from ..metrics_schema import get_fundamental, period_label_from_meta
from .models import Claim, EvidenceRef
from .numeric import (
    _find_number_span,
    _number_token_end,
    _text_contains_number,
    match_numeric_evidence,
)
from .period import (
    PeriodIdentity,
    _PERIOD_TYPE_TOKENS,
    _nearest_period_identity,
    _period_identities_compatible,
    is_factual_period_provenance,
    parse_period_identity,
)

def _page_from_citation(citation: str) -> int | None:
    match = re.search(r"#p(\d+)\b", citation or "", flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _collect_evidence_pool(state: dict[str, Any], company: str) -> list[EvidenceRef]:
    pool: list[EvidenceRef] = []
    seen: set[str] = set()

    def add(
        *,
        evidence_id: str,
        citation: str,
        source_type: str,
        text: str,
        period: str | None = None,
        period_type: str | None = None,
        metric_name: str | None = None,
        value: float | None = None,
        unit: str | None = None,
        confidence: str | None = None,
        period_source: str | None = None,
        period_alignment: str | None = None,
        source_record_id: str | None = None,
        citation_trusted: bool = False,
        chunk_id: str | None = None,
        tenant_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if not citation or evidence_id in seen:
            return
        seen.add(evidence_id)
        specific = period
        ptype = period_type
        if specific and str(specific).lower() in _PERIOD_TYPE_TOKENS:
            ptype = ptype or str(specific).lower()
            specific = None
        if specific and not is_factual_period_provenance(
            period=specific,
            period_source=period_source,
            period_alignment=period_alignment,
        ):
            specific = None
        pool.append(
            EvidenceRef(
                evidence_id=evidence_id,
                entity=company,
                citation=citation,
                source_type=source_type,
                text=text or "",
                page=_page_from_citation(citation),
                period=specific,
                period_type=ptype,
                metric_name=metric_name,
                value=value,
                unit=unit,
                confidence=confidence,
                period_source=period_source,
                period_alignment=period_alignment,
                source_record_id=source_record_id,
                citation_trusted=citation_trusted,
                chunk_id=chunk_id or None,
                tenant_id=tenant_id or None,
                session_id=session_id or None,
            )
        )

    for index, hit in enumerate((state.get("rag_evidence") or {}).get(company) or []):
        supplied_citation = hit.get("citation") or hit.get("source")
        citation = str(supplied_citation or f"rag:{company}:{index}")
        hit_period = str(hit.get("period") or "") or None
        hit_period_source = str(hit.get("period_source") or "") or None
        hit_period_alignment = str(hit.get("period_alignment") or "") or None
        hit_record = str(hit.get("source_record_id") or hit.get("provider_record_id") or "") or None
        trusted_citation = bool(supplied_citation)
        chunk_id = str(hit.get("chunk_id") or "").strip() or None
        tenant_id = str(hit.get("tenant_id") or state.get("rag_tenant_id") or "") or None
        session_id = str(hit.get("session_id") or state.get("thread_id") or "") or None
        if hit_period and not is_factual_period_provenance(
            period=hit_period,
            period_source=hit_period_source,
            period_alignment=hit_period_alignment,
        ):
            hit_period = None
        add(
            evidence_id=f"ev_rag_{company}_{index}",
            citation=citation,
            source_type=str(hit.get("source_type") or "rag"),
            text=str(hit.get("text") or hit.get("snippet") or ""),
            period=hit_period,
            period_source=hit_period_source,
            period_alignment=hit_period_alignment,
            source_record_id=hit_record,
            citation_trusted=trusted_citation,
            chunk_id=chunk_id,
            tenant_id=tenant_id,
            session_id=session_id,
        )

    payload = (state.get("retrieved_docs") or {}).get(company) or {}
    period = period_label_from_meta(payload.get("fundamentals_meta"))
    structured = str(payload.get("structured_source") or "none")
    market_data = payload.get("market_data") or {}
    provenance = payload.get("fundamental_provenance") or {}
    if market_data:
        labels = {
            "revenue": "revenue",
            "ebitda": "EBITDA",
            "r_and_d": "R&D",
            "operating_income": "operating income",
        }
        for key, label in labels.items():
            raw = get_fundamental(market_data, key)
            if raw is None:
                continue
            field_prov = provenance.get(key) if isinstance(provenance, dict) else None
            if not isinstance(field_prov, dict):
                field_prov = {}
            field_period = str(field_prov.get("period") or "") or None
            field_period_type = str(field_prov.get("period_type") or "") or None
            field_period_source = str(field_prov.get("period_source") or "") or None
            field_period_alignment = str(field_prov.get("period_alignment") or "") or None
            source_record_id = str(
                field_prov.get("source_record_id") or field_prov.get("provider_record_id") or ""
            ) or None
            supplied_citation = str(field_prov.get("citation") or "") or None
            if field_period and field_period.lower() in _PERIOD_TYPE_TOKENS:
                field_period_type = field_period_type or field_period.lower()
                field_period = None
            record_proof = bool(supplied_citation or source_record_id)
            if not record_proof or not is_factual_period_provenance(
                period=field_period,
                period_source=field_period_source,
                period_alignment=field_period_alignment,
            ):
                field_period = None
            field_source = str(field_prov.get("source") or "fundamentals")
            field_conf = (
                str(field_prov["confidence"])
                if field_prov.get("confidence") is not None
                else None
            )
            source_type = field_source if field_source != "none" else "fundamentals"
            display = f"{company} {label} was {float(raw)} billion USD."
            display_citation = (
                supplied_citation
                or f"lumenfin:{source_type}:{company}:{field_period or 'unknown'}:{key}"
            )
            add(
                evidence_id=f"ev_fund_{company}_{key}_{field_period or 'unknown'}",
                citation=display_citation,
                source_type=source_type,
                text=display,
                period=field_period,
                period_type=field_period_type,
                metric_name=key,
                value=float(raw),
                unit="billion_usd",
                confidence=field_conf,
                period_source=field_period_source,
                period_alignment=field_period_alignment,
                source_record_id=source_record_id,
                citation_trusted=bool(supplied_citation),
            )

    supply = payload.get("supply_chain") or {}
    if supply:
        signals = [str(s) for s in (supply.get("signals") or [])]
        text = (
            f"{company} supply chain risk level is {supply.get('risk_level', 'unknown')}. "
            f"Signals: {'; '.join(signals)}"
        )
        add(
            evidence_id=f"ev_supply_{company}",
            citation=f"lumenfin:supply_chain:{company}:{period}",
            source_type="supply_chain",
            text=text,
            period=period,
        )

    for index, doc in enumerate(payload.get("source_documents") or []):
        citation = str(doc.get("citation") or doc.get("filename") or f"source:{company}:{index}")
        add(
            evidence_id=f"ev_doc_{company}_{index}",
            citation=citation,
            source_type=str(doc.get("source_type") or "document"),
            text=str(doc.get("excerpt") or doc.get("text") or ""),
            period=str(doc.get("period") or "") or None,
            period_source=str(doc.get("period_source") or "") or None,
            period_alignment=str(doc.get("period_alignment") or "") or None,
            source_record_id=str(doc.get("source_record_id") or "") or None,
            citation_trusted=bool(doc.get("citation") or doc.get("filename")),
        )

    scores = (state.get("risk_scores") or {}).get(company) or {}
    if isinstance(scores, dict) and scores:
        parts = [f"{k}={v}" for k, v in scores.items() if isinstance(v, (int, float))]
        if parts:
            add(
                evidence_id=f"ev_risk_{company}",
                citation=f"lumenfin:risk_model:{company}:model",
                source_type="risk_model",
                text=(
                    f"{company} model-derived risk scores are screening indicators: "
                    + ", ".join(parts)
                    + "."
                ),
                period="model",
            )

    snapshot = (state.get("market_snapshots") or {}).get(company) or {}
    if snapshot.get("current_price") is not None:
        details = []
        for key in ("current_price", "trailing_pe", "monthly_return", "fifty_two_week_high", "fifty_two_week_low"):
            if snapshot.get(key) is not None:
                details.append(f"{key}={snapshot.get(key)}")
        add(
            evidence_id=f"ev_mkt_{company}",
            citation=f"lumenfin:market_snapshot:{company}:{snapshot.get('fetched_at') or 'latest'}",
            source_type="market_data",
            text=(
                f"{company} live market snapshot from {snapshot.get('provider') or 'unknown'}: "
                + ", ".join(details)
                + "."
            ),
            period="latest",
        )
    return pool


def _fund_refs_for_values(
    pool: list[EvidenceRef],
    values: list[float | None],
    *,
    entity: str | None = None,
    metric_name: str | None = None,
    unit: str | None = None,
    period: str | None = None,
    formula_inputs: dict[str, float] | None = None,
) -> list[EvidenceRef]:
    """Return fundamentals evidence that structurally matches the claim."""
    needed = [v for v in values if v is not None]
    fund_refs = [r for r in pool if r.evidence_id.startswith("ev_fund_")]

    if formula_inputs and entity:
        bound: list[EvidenceRef] = []
        for input_name, input_value in formula_inputs.items():
            hit = None
            for ref in fund_refs:
                result = match_numeric_evidence(
                    ref,
                    entity=entity,
                    metric_name=str(input_name),
                    value=float(input_value),
                    unit="billion_usd",
                    period=period,
                )
                if result.matched and result.confidence == "high":
                    hit = ref
                    break
            if hit is None:
                for ref in pool:
                    if ref in bound:
                        continue
                    result = match_numeric_evidence(
                        ref,
                        entity=entity,
                        metric_name=str(input_name),
                        value=float(input_value),
                        unit="billion_usd",
                        period=period,
                    )
                    if result.matched and result.confidence in {"high", "medium"}:
                        hit = ref
                        break
            if hit is None:
                return []
            bound.append(hit)
        return bound

    if fund_refs and metric_name and entity and needed:
        matched = [
            r
            for r in fund_refs
            if match_numeric_evidence(
                r,
                entity=entity,
                metric_name=metric_name,
                value=float(needed[0]),
                unit=unit,
                period=period,
            ).matched
        ]
        if matched:
            return matched[:1]
        return []
    if metric_name:
        # Structured metric context present: do not fall back to whole-text number match.
        return []
    if fund_refs and (
        not needed or any(_text_contains_number(r.text, v) for r in fund_refs for v in needed)
    ):
        usable = [
            r
            for r in fund_refs
            if not needed or any(_text_contains_number(r.text, v) for v in needed)
        ]
        if usable:
            return usable[:1]
    for r in pool:
        cite = r.citation or ""
        if not any(
            key in cite
            for key in (
                "sample_financial_data",
                "sec_companyfacts",
                "yahoo_fundamentals",
                "document_extracted",
            )
        ) and r.source_type not in {
            "sec_companyfacts",
            "yahoo_fundamentals",
            "document_extracted",
            "fundamentals",
            "sample_db",
        }:
            continue
        if metric_name and entity and needed:
            if match_numeric_evidence(
                r,
                entity=entity,
                metric_name=metric_name,
                value=float(needed[0]),
                unit=unit,
                period=period,
            ).matched:
                return [r]
            continue
        if needed and not any(_text_contains_number(r.text, v) for v in needed):
            continue
        return [r]
    return []


def _prefer_refs_for_values(
    pool: list[EvidenceRef],
    values: list[float | None],
    *,
    require_values: list[float | None] | None = None,
    entity: str | None = None,
    metric_name: str | None = None,
    unit: str | None = None,
    period: str | None = None,
    formula_inputs: dict[str, float] | None = None,
) -> list[EvidenceRef]:
    """Prefer page-anchored RAG that structurally matches; else fundamentals text."""
    gate = [v for v in (require_values if require_values is not None else values) if v is not None]
    scan = [v for v in values if v is not None]
    matched: list[EvidenceRef] = []
    for ref in pool:
        if ref.source_type not in {"rag", "document"} and "#p" not in (ref.citation or "").lower():
            continue
        if metric_name and entity:
            probe_value = float((gate or scan)[0]) if (gate or scan) else float(values[0] or 0)
            result = match_numeric_evidence(
                ref,
                entity=entity,
                metric_name=metric_name,
                value=probe_value if not formula_inputs else float(values[-1] or probe_value),
                unit=unit,
                period=period,
                formula_inputs=formula_inputs,
            )
            if result.matched:
                matched.append(ref)
            continue
        probe = gate or scan
        if probe and any(_text_contains_number(ref.text, v) for v in probe):
            matched.append(ref)
    if matched:
        matched.sort(key=lambda r: (0 if r.page is not None else 1, r.evidence_id))
        return matched[:2]

    return _fund_refs_for_values(
        pool,
        gate or scan,
        entity=entity,
        metric_name=metric_name,
        unit=unit,
        period=period,
        formula_inputs=formula_inputs,
    )


def _verify_numeric(
    claim: Claim,
    refs: list[EvidenceRef],
    input_values: list[float | None],
    *,
    pool: list[EvidenceRef] | None = None,
    formula_inputs: dict[str, float] | None = None,
) -> Claim:
    needed = [v for v in input_values if v is not None]

    def _bind_formula(candidates: list[EvidenceRef]) -> tuple[list[EvidenceRef], str]:
        assert formula_inputs is not None
        bound: list[EvidenceRef] = []
        last_reason = "formula_input_incomplete"
        input_ids: dict[str, str] = {}
        input_periods: list[PeriodIdentity] = []
        search_space = list(candidates)
        if pool:
            for ref in pool:
                if ref not in search_space:
                    search_space.append(ref)
        for input_name, input_value in formula_inputs.items():
            hit = None
            for ref in search_space:
                result = match_numeric_evidence(
                    ref,
                    entity=claim.entity,
                    metric_name=str(input_name),
                    value=float(input_value),
                    unit="billion_usd",
                    period=claim.period,
                )
                if result.matched and result.matched_period and (
                    result.confidence == "high"
                    or (
                        not ref.has_structured_fields
                        and result.confidence == "medium"
                    )
                ):
                    hit = ref
                    last_reason = result.reason
                    break
                last_reason = result.reason
            if hit is None:
                return [], last_reason
            bound.append(hit)
            input_ids[str(input_name)] = hit.evidence_id
            if hit.period:
                input_periods.append(parse_period_identity(hit.period))
            else:
                input_span = _find_number_span(
                    hit.text or "",
                    float(input_value),
                    unit="billion_usd",
                    metric_name=str(input_name),
                )
                if input_span is None:
                    input_periods.append(PeriodIdentity("unknown"))
                else:
                    input_start = input_span[0]
                    input_end = _number_token_end(hit.text or "", input_start)
                    input_periods.append(
                        _nearest_period_identity(hit.text or "", input_start, input_end)
                        or PeriodIdentity("unknown")
                    )
        claim_id = parse_period_identity(claim.period) if claim.period else PeriodIdentity("unknown")
        if claim_id.kind != "unknown":
            for item in input_periods:
                if item.kind == "unknown" or not _period_identities_compatible(claim_id, item):
                    return [], "formula_input_period_mismatch"
            for left, right in zip(input_periods, input_periods[1:]):
                if not _period_identities_compatible(left, right):
                    return [], "formula_input_period_mismatch"
        # Keep formula input map in verify_reason for auditability.
        claim.verify_reason = f"formula_inputs={input_ids}"
        return bound, "formula_inputs_bound"

    def _usable_from(candidates: list[EvidenceRef]) -> tuple[list[EvidenceRef], str]:
        if formula_inputs and claim.metric_name:
            return _bind_formula(candidates)
        usable: list[EvidenceRef] = []
        last_reason = "number_not_found"
        for ref in candidates:
            if claim.metric_name and isinstance(claim.value, (int, float)):
                result = match_numeric_evidence(
                    ref,
                    entity=claim.entity,
                    metric_name=str(claim.metric_name),
                    value=float(claim.value),
                    unit=claim.unit,
                    period=claim.period,
                )
                if (
                    result.matched
                    and result.matched_period
                    and result.confidence in {"high", "medium"}
                ):
                    # Verified numeric facts require high confidence when structured fields exist.
                    if ref.has_structured_fields and result.confidence != "high":
                        last_reason = result.reason
                        continue
                    usable.append(ref)
                    last_reason = result.reason
                else:
                    last_reason = result.reason
            elif not claim.metric_name and needed and any(
                _text_contains_number(ref.text, v) for v in needed
            ):
                usable.append(ref)
                last_reason = "legacy_number_match"
        return usable, last_reason

    usable, reason = _usable_from(refs)
    if not usable and pool is not None:
        usable, reason = _usable_from(
            _fund_refs_for_values(
                pool,
                input_values,
                entity=claim.entity,
                metric_name=claim.metric_name,
                unit=claim.unit,
                period=claim.period,
                formula_inputs=formula_inputs,
            )
        )
    if not refs and not usable:
        if pool and claim.period:
            relevant = [
                ref
                for ref in pool
                if ref.evidence_id.startswith("ev_fund_")
                and (
                    (formula_inputs and ref.metric_name in formula_inputs)
                    or (not formula_inputs and ref.metric_name == claim.metric_name)
                )
            ]
            if relevant and any(ref.period is None for ref in relevant):
                reason = "formula_input_period_unknown" if formula_inputs else "period_unknown"
        claim.verification = "rejected"
        claim.verify_reason = f"No evidence text contains the metric inputs ({reason})."
        claim.evidence_refs = []
        return claim
    if not usable:
        claim.verification = "rejected"
        claim.verify_reason = f"Candidate evidence rejected: {reason}."
        claim.evidence_refs = []
        return claim
    claim.evidence_refs = usable
    claim.verification = "verified"
    detail = claim.verify_reason if claim.verify_reason.startswith("formula_inputs=") else ""
    claim.verify_reason = (
        f"Metric/period/unit-bound evidence ({reason})"
        + (f"; {detail}" if detail else "")
        + (
            f"; formula_inputs={sorted(formula_inputs)}"
            if formula_inputs and not detail
            else ""
        )
    )
    return claim

