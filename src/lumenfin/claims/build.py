"""Build and summarize verified claims from structured state."""

from __future__ import annotations

import re
from typing import Any

from ..metrics_schema import get_fundamental, period_label_from_meta
from ..tools import AST_RATIO_KEYS, has_computable_fundamentals
from .binding import _collect_evidence_pool, _prefer_refs_for_values, _verify_numeric
from .models import (
    FORMULA_INPUTS,
    METRIC_LABELS,
    Claim,
    EvidenceRef,
    filter_verified,
    verified_by_entity,
)
from .numeric import _fmt_num, _fmt_pct, _text_contains_number

def build_claims(state: dict[str, Any]) -> list[Claim]:
    """Build and verify claims from structured state (no LLM)."""
    claims: list[Claim] = []
    companies = list(state.get("companies") or [])
    metrics_by_co = state.get("financial_metrics") or {}
    retrieved = state.get("retrieved_docs") or {}
    fatal_gap = bool(state.get("fatal_data_gap"))

    for company in companies:
        pool = _collect_evidence_pool(state, company)
        payload = retrieved.get(company) or {}
        market_data = payload.get("market_data") or {}
        period = period_label_from_meta(payload.get("fundamentals_meta"))
        metrics = metrics_by_co.get(company) or {}
        structured = str(payload.get("structured_source") or "none")
        # Fail-closed: do not mint verified numeric "facts" when fundamentals are absent.
        block_numeric = fatal_gap or structured == "none" or not has_computable_fundamentals(payload)

        # --- Numeric ratio claims (AST) ---
        if not block_numeric:
            for metric_name in AST_RATIO_KEYS:
                value = metrics.get(metric_name)
                if not isinstance(value, (int, float)):
                    continue
                input_map = FORMULA_INPUTS.get(metric_name) or {}
                input_vals = [get_fundamental(market_data, src) for src in input_map.values()]
                formula_inputs = {
                    src: float(get_fundamental(market_data, src))
                    for src in input_map.values()
                    if get_fundamental(market_data, src) is not None
                }
                label = METRIC_LABELS.get(metric_name, metric_name)
                statement = f"{company} {label} is {_fmt_pct(float(value))} for {period}."
                claim = Claim(
                    claim_id=f"cl_num_{company}_{metric_name}",
                    entity=company,
                    claim_type="numeric",
                    statement=statement,
                    value=float(value),
                    unit="ratio",
                    period=period,
                    metric_name=metric_name,
                )
                refs = _prefer_refs_for_values(
                    pool,
                    input_vals + [float(value)],
                    require_values=input_vals,
                    entity=company,
                    metric_name=metric_name,
                    unit="ratio",
                    period=period,
                    formula_inputs=formula_inputs or None,
                )
                claims.append(
                    _verify_numeric(
                        claim,
                        refs,
                        input_vals,
                        pool=pool,
                        formula_inputs=formula_inputs or None,
                    )
                )

            # Absolute fundamentals as numeric claims (for ledger + consistency)
            for key, label in (
                ("revenue", "Revenue"),
                ("ebitda", "EBITDA"),
                ("operating_income", "Operating income"),
                ("r_and_d", "R&D expense"),
            ):
                raw = get_fundamental(market_data, key)
                if raw is None:
                    continue
                statement = f"{company} {label} is {_fmt_num(float(raw))} billion USD for {period}."
                claim = Claim(
                    claim_id=f"cl_abs_{company}_{key}",
                    entity=company,
                    claim_type="numeric",
                    statement=statement,
                    value=float(raw),
                    unit="billion_usd",
                    period=period,
                    metric_name=key,
                )
                refs = _prefer_refs_for_values(
                    pool,
                    [float(raw)],
                    require_values=[float(raw)],
                    entity=company,
                    metric_name=key,
                    unit="billion_usd",
                    period=period,
                )
                claims.append(_verify_numeric(claim, refs, [float(raw)], pool=pool))

            # Market snapshot numerics only when structured fundamentals exist.
            snapshot = (state.get("market_snapshots") or {}).get(company) or {}
            pe = snapshot.get("trailing_pe")
            if isinstance(pe, (int, float)):
                claim = Claim(
                    claim_id=f"cl_num_{company}_pe_ratio",
                    entity=company,
                    claim_type="numeric",
                    statement=f"{company} trailing P/E is {float(pe):.2f}x (live market snapshot).",
                    value=float(pe),
                    unit="multiple",
                    period="latest",
                    metric_name="pe_ratio",
                )
                mkt_refs = [r for r in pool if r.evidence_id.startswith("ev_mkt_")]
                if mkt_refs and _text_contains_number(mkt_refs[0].text, float(pe)):
                    claim.evidence_refs = mkt_refs[:1]
                    claim.verification = "verified"
                    claim.verify_reason = "Bound to live market snapshot evidence."
                else:
                    claim.verification = "rejected"
                    claim.verify_reason = "No market snapshot evidence for P/E."
                claims.append(claim)
        else:
            claim = Claim(
                claim_id=f"cl_num_{company}_blocked",
                entity=company,
                claim_type="numeric",
                statement=(
                    f"{company}: numeric claims withheld — structured fundamentals unavailable "
                    f"(fatal_data_gap={fatal_gap}, structured_source={structured})."
                ),
                metric_name="numeric_blocked",
                verification="rejected",
                verify_reason="Fail-closed: refusing verified numeric claims without computable fundamentals.",
            )
            claims.append(claim)

        # --- Growth claims: only with multi-period fundamentals ---
        if not block_numeric:
            growth_claim = _build_growth_claim(company, market_data, period, pool)
            if growth_claim is not None:
                claims.append(growth_claim)
        else:
            claims.append(
                Claim(
                    claim_id=f"cl_growth_{company}_revenue",
                    entity=company,
                    claim_type="growth",
                    statement=f"{company}: revenue growth claim withheld under fail-closed data gap.",
                    metric_name="revenue_growth",
                    period=period,
                    verification="rejected",
                    verify_reason="Fail-closed: no computable fundamentals for growth.",
                )
            )

        # --- Risk conclusions ---
        supply = payload.get("supply_chain") or {}
        risk_level = str(supply.get("risk_level") or "unknown")
        risk_scores = (state.get("risk_scores") or {}).get(company) or {}
        if supply or risk_scores or block_numeric:
            if block_numeric:
                statement = (
                    f"{company} data-limitation risk is elevated: no AST-computable fundamentals "
                    f"(structured_source={structured})."
                )
                metric_name = "data_limitation_risk"
            else:
                statement = (
                    f"{company} supply-chain risk signal is '{risk_level}'"
                    + (
                        f" with model supply_chain_risk={risk_scores.get('supply_chain_risk')}"
                        if isinstance(risk_scores.get("supply_chain_risk"), (int, float))
                        else ""
                    )
                    + "."
                )
                metric_name = "supply_chain_risk"
            claim = Claim(
                claim_id=f"cl_risk_{company}_supply",
                entity=company,
                claim_type="risk_conclusion",
                statement=statement,
                value=risk_level if not block_numeric else "elevated",
                period=period,
                metric_name=metric_name,
            )
            refs = [r for r in pool if r.evidence_id.startswith("ev_supply_")]
            rag_risk = [
                r
                for r in pool
                if (r.page is not None or "#p" in r.citation.lower())
                and re.search(r"risk|supply|supplier|concentration", r.text, re.I)
            ]
            if rag_risk:
                refs = rag_risk[:1] + refs[:1]
            if block_numeric:
                # Bind to explicit fail-closed provenance rather than market snapshots.
                claim.evidence_refs = [
                    EvidenceRef(
                        evidence_id=f"ev_gap_{company}",
                        entity=company,
                        citation=f"lumenfin:data_gap:{company}:{structured}",
                        source_type="data_gap",
                        text=statement,
                        period=period,
                    )
                ]
                claim.verification = "verified"
                claim.verify_reason = "Fail-closed data-limitation risk bound to structured_source=none provenance."
            elif risk_level.lower() in {"unknown", "n/a", "none"} and not block_numeric:
                # Do not promote an "unknown" supply-chain placeholder into material risk/thesis.
                claim.verification = "rejected"
                claim.verify_reason = (
                    "Supply-chain signal is unknown; withheld from material risk conclusions."
                )
                claim.evidence_refs = []
            elif refs:
                claim.evidence_refs = refs[:2]
                claim.verification = "verified"
                claim.verify_reason = "Bound to supply-chain / risk evidence."
            else:
                risk_model = [r for r in pool if r.evidence_id.startswith("ev_risk_")]
                if risk_model:
                    claim.evidence_refs = risk_model[:1]
                    claim.verification = "verified"
                    claim.verify_reason = "Bound to risk-model screening evidence (not a filing fact)."
                else:
                    claim.verification = "rejected"
                    claim.verify_reason = "No risk evidence available."
            claims.append(claim)

        if not block_numeric:
            for dim in ("financial_risk", "market_risk", "operational_risk"):
                score = risk_scores.get(dim)
                if not isinstance(score, (int, float)):
                    continue
                level = "Low" if score < 3.5 else ("Moderate" if score < 6.5 else "Elevated")
                claim = Claim(
                    claim_id=f"cl_risk_{company}_{dim}",
                    entity=company,
                    claim_type="risk_conclusion",
                    statement=f"{company} {dim.replace('_', ' ')} screening score is {score:.1f}/10 ({level}).",
                    value=float(score),
                    unit="score_1_10",
                    period="model",
                    metric_name=dim,
                )
                risk_refs = [r for r in pool if r.evidence_id.startswith("ev_risk_")]
                mkt_refs = [r for r in pool if r.evidence_id.startswith("ev_mkt_")] if dim == "market_risk" else []
                refs = (mkt_refs or risk_refs)[:1]
                if refs:
                    claim.evidence_refs = refs
                    claim.verification = "verified"
                    claim.verify_reason = "Bound to risk-model / market evidence as screening conclusion."
                else:
                    claim.verification = "rejected"
                    claim.verify_reason = "Missing risk-model evidence."
                claims.append(claim)

    # --- Investment conclusions: only compose from verified numeric+risk ---
    verified_so_far = filter_verified(claims)
    for company in companies:
        payload = retrieved.get(company) or {}
        structured = str(payload.get("structured_source") or "none")
        block_numeric = fatal_gap or structured == "none" or not has_computable_fundamentals(payload)
        num = verified_by_entity(verified_so_far, company, claim_type="numeric")
        risk = verified_by_entity(verified_so_far, company, claim_type="risk_conclusion")
        # Prefer EBITDA margin; fall back to operating margin for PDF extracts lacking EBITDA.
        profit_claims = [c for c in num if c.metric_name == "ebitda_margin"] or [
            c for c in num if c.metric_name == "operating_margin"
        ]
        claim = Claim(
            claim_id=f"cl_inv_{company}_thesis",
            entity=company,
            claim_type="investment_conclusion",
            statement="",
            metric_name="research_thesis",
        )
        if block_numeric or not profit_claims or not risk:
            claim.verification = "rejected"
            claim.verify_reason = (
                "Investment conclusion omitted: requires verified numeric profitability "
                "and verified risk conclusion."
            )
            claim.statement = (
                f"{company}: no evidence-backed investment conclusion — missing verified "
                "profitability and/or risk claims."
            )
            claims.append(claim)
            continue
        profit = profit_claims[0]
        # Prefer scored screening dimensions; never anchor thesis on unknown supply-chain.
        usable_risk = [
            c
            for c in risk
            if not (
                c.metric_name == "supply_chain_risk"
                and str(c.value).lower() in {"unknown", "n/a", "none"}
            )
        ]
        scored_risk = [
            c
            for c in usable_risk
            if c.metric_name in {"financial_risk", "operational_risk", "market_risk", "data_limitation_risk"}
        ]
        risk_c = scored_risk[0] if scored_risk else (usable_risk[0] if usable_risk else None)
        if risk_c is None:
            claim.verification = "rejected"
            claim.verify_reason = (
                "Investment conclusion omitted: no material verified risk conclusion "
                "(unknown supply-chain alone is insufficient)."
            )
            claim.statement = (
                f"{company}: no evidence-backed investment conclusion — missing material "
                "verified risk claims."
            )
            claims.append(claim)
            continue
        profit_v = float(profit.value) if isinstance(profit.value, (int, float)) else 0.0
        # Operating-margin thresholds are slightly lower than EBITDA-margin screens.
        strong, adequate = (0.25, 0.15) if profit.metric_name == "ebitda_margin" else (0.20, 0.12)
        if profit_v >= strong:
            stance = "quality-screening research thesis (not a recommendation)"
        elif profit_v >= adequate:
            stance = "neutral quality-compounder research screen (not a recommendation)"
        else:
            stance = "defensive research posture pending operational evidence (not a recommendation)"
        risk_label = (
            f"{risk_c.metric_name}={risk_c.value}"
            if isinstance(risk_c.value, (int, float))
            else str(risk_c.value)
        )
        claim.statement = (
            f"{company} supports a {stance} based on verified {profit.metric_name} "
            f"({_fmt_pct(profit_v)}) and verified risk screen ({risk_label})."
        )
        claim.evidence_refs = list(profit.evidence_refs[:1]) + list(risk_c.evidence_refs[:1])
        claim.verification = "verified"
        claim.verify_reason = "Composed only from verified numeric + risk claims."
        claim.value = stance
        claims.append(claim)

    return claims


def _build_growth_claim(
    company: str,
    market_data: dict[str, Any],
    period: str,
    pool: list[EvidenceRef],
) -> Claim | None:
    """Emit growth claim only when two periods of revenue exist structurally."""
    current = get_fundamental(market_data, "revenue")
    prior = None
    prior_key = None
    for key, value in (market_data or {}).items():
        match = re.match(r"revenue_(20\d{2})$", str(key))
        if not match:
            continue
        year = int(match.group(1))
        # Prefer an older year than current period tag if present.
        try:
            prior = float(value)
            prior_key = str(key)
        except (TypeError, ValueError):
            continue
        _ = year
    # Also accept explicit prior_revenue
    if prior is None and market_data.get("prior_revenue") is not None:
        try:
            prior = float(market_data["prior_revenue"])
            prior_key = "prior_revenue"
        except (TypeError, ValueError):
            prior = None

    claim = Claim(
        claim_id=f"cl_growth_{company}_revenue",
        entity=company,
        claim_type="growth",
        statement="",
        metric_name="revenue_growth",
        period=period,
    )
    if current is None or prior is None or prior == 0:
        # Explicitly reject heuristic/scenario growth — do not invent.
        claim.verification = "rejected"
        claim.verify_reason = (
            "No multi-period revenue pair in structured fundamentals; "
            "refusing heuristic growth claims."
        )
        claim.statement = (
            f"{company}: revenue growth claim withheld — multi-period fundamentals unavailable."
        )
        return claim

    growth = (float(current) - float(prior)) / abs(float(prior))
    claim.value = growth
    claim.unit = "ratio"
    claim.statement = (
        f"{company} revenue growth is {_fmt_pct(growth)} "
        f"(from {prior_key}={_fmt_num(float(prior))} to revenue={_fmt_num(float(current))} billion USD)."
    )
    refs = _prefer_refs_for_values(pool, [float(current), float(prior), growth])
    return _verify_numeric(claim, refs, [float(current), float(prior)])


def binding_summary(claims: list[Claim]) -> dict[str, Any]:
    verified = filter_verified(claims)
    by_type: dict[str, dict[str, int]] = {}
    for claim in claims:
        bucket = by_type.setdefault(claim.claim_type, {"total": 0, "verified": 0, "rejected": 0})
        bucket["total"] += 1
        if claim.verification == "verified":
            bucket["verified"] += 1
        elif claim.verification == "rejected":
            bucket["rejected"] += 1
    page_bound = sum(1 for c in verified if any(r.page is not None or "#p" in r.citation.lower() for r in c.evidence_refs))
    return {
        "total_claims": len(claims),
        "verified_claims": len(verified),
        "rejected_claims": sum(1 for c in claims if c.verification == "rejected"),
        "page_anchored_verified": page_bound,
        "by_type": by_type,
        "bind_rate": round(len(verified) / len(claims), 4) if claims else 0.0,
    }


def format_verified_claims_ledger(claims: list[Claim]) -> list[str]:
    from ..reporting import humanize_citation

    verified = filter_verified(claims)
    lines = [
        "## Appendix A. Verified Claims Ledger",
        "",
        "*Audit appendix: only structurally verified claims may back material assertions. "
        "Internal claim IDs are retained here for traceability.*",
        "",
    ]
    if not verified:
        lines.append("- (none — report will withhold evidence-backed financial assertions)")
        lines.append("")
        return lines
    lines.append("| Entity | Type | Statement | Source |")
    lines.append("|--------|------|-----------|--------|")
    for claim in verified:
        cite = humanize_citation(claim.primary_citation).replace("|", "/")
        stmt = claim.statement.replace("|", "/")
        lines.append(
            f"| {claim.entity} | {claim.claim_type} | {stmt} | {cite} |"
        )
    lines.append("")
    return lines

