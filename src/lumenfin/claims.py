"""Claim → Evidence Binding layer.

Builds internal claim objects from structured state, verifies them against
evidence (RAG page anchors and/or fundamentals text), and exposes only
verified claims for the report synthesizer.

No prompt-forced citations: citations come from evidence already present
on the claim after structural verification.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .metrics_schema import get_fundamental, period_label_from_meta
from .tools import AST_RATIO_KEYS, has_computable_fundamentals

ClaimType = Literal["numeric", "growth", "risk_conclusion", "investment_conclusion"]
Verification = Literal["verified", "unverified", "rejected"]

FORMULA_INPUTS: dict[str, dict[str, str]] = {
    "ebitda_margin": {"ebitda": "ebitda", "revenue": "revenue"},
    "operating_margin": {"operating_income": "operating_income", "revenue": "revenue"},
    "r_and_d_intensity": {"r_and_d": "r_and_d", "revenue": "revenue"},
}

METRIC_LABELS = {
    "ebitda_margin": "EBITDA margin",
    "operating_margin": "Operating margin",
    "r_and_d_intensity": "R&D intensity",
    "estimated_net_margin": "Estimated net margin",
    "estimated_fcf_margin": "Estimated FCF margin",
    "pe_ratio": "P/E (TTM)",
    "monthly_return": "Monthly return",
    "range_position": "52-week range position",
}


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    entity: str
    citation: str
    source_type: str
    text: str
    page: int | None = None
    period: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    claim_id: str
    entity: str
    claim_type: ClaimType
    statement: str
    value: float | str | None = None
    unit: str | None = None
    period: str | None = None
    metric_name: str | None = None
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    verification: Verification = "unverified"
    verify_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @property
    def primary_citation(self) -> str:
        if not self.evidence_refs:
            return ""
        return self.evidence_refs[0].citation

    def render_with_citation(self) -> str:
        cite = self.primary_citation
        if cite:
            return f"{self.statement} [{cite}]"
        return self.statement


def claim_to_dict(claim: Claim) -> dict[str, Any]:
    return claim.to_dict()


def claims_from_state(state: dict[str, Any]) -> list[Claim]:
    raw = state.get("claims") or []
    out: list[Claim] = []
    for item in raw:
        if isinstance(item, Claim):
            out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        refs = [
            EvidenceRef(**ref) if isinstance(ref, dict) else ref
            for ref in (item.get("evidence_refs") or [])
        ]
        out.append(
            Claim(
                claim_id=str(item.get("claim_id") or ""),
                entity=str(item.get("entity") or ""),
                claim_type=item.get("claim_type") or "numeric",  # type: ignore[arg-type]
                statement=str(item.get("statement") or ""),
                value=item.get("value"),
                unit=item.get("unit"),
                period=item.get("period"),
                metric_name=item.get("metric_name"),
                evidence_refs=refs,
                verification=item.get("verification") or "unverified",  # type: ignore[arg-type]
                verify_reason=str(item.get("verify_reason") or ""),
            )
        )
    return out


def filter_verified(claims: list[Claim]) -> list[Claim]:
    return [c for c in claims if c.verification == "verified" and c.evidence_refs]


def verified_by_entity(
    claims: list[Claim],
    entity: str,
    *,
    claim_type: ClaimType | None = None,
    metric_name: str | None = None,
) -> list[Claim]:
    out = []
    for claim in filter_verified(claims):
        if claim.entity != entity:
            continue
        if claim_type and claim.claim_type != claim_type:
            continue
        if metric_name and claim.metric_name != metric_name:
            continue
        out.append(claim)
    return out


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def _fmt_num(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4g}"


def _number_variants(value: float) -> list[str]:
    variants = {
        _fmt_num(value),
        f"{value:.1f}",
        f"{value:.2f}",
        f"{value:.0f}",
        f"{value:,.0f}",
        f"{value:,.1f}",
        f"{value * 1000:.0f}",  # billion → million
        f"{value * 1_000_000_000:.0f}",  # billion → raw USD
    }
    # Compact without commas for PDF text matches.
    variants |= {v.replace(",", "") for v in list(variants)}
    return [v for v in variants if v and v not in {".", "-"}]


def _text_contains_number(text: str, value: float | None, *, tol: float = 0.02) -> bool:
    if value is None or not text:
        return False
    lowered = text.lower().replace(",", "")
    for token in _number_variants(float(value)):
        if token.replace(",", "") in lowered:
            return True
    # Relative match for percentages already formatted in text like 34.8%
    pct = float(value) * 100.0 if abs(float(value)) <= 1.5 else float(value)
    for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", text):
        try:
            found = float(match.group(1))
        except ValueError:
            continue
        if abs(found - pct) <= max(0.15, abs(pct) * tol):
            return True
    return False


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
    ) -> None:
        if not citation or citation in seen:
            return
        seen.add(citation)
        pool.append(
            EvidenceRef(
                evidence_id=evidence_id,
                entity=company,
                citation=citation,
                source_type=source_type,
                text=text or "",
                page=_page_from_citation(citation),
                period=period,
            )
        )

    for index, hit in enumerate((state.get("rag_evidence") or {}).get(company) or []):
        citation = str(hit.get("citation") or hit.get("source") or f"rag:{company}:{index}")
        add(
            evidence_id=f"ev_rag_{company}_{index}",
            citation=citation,
            source_type=str(hit.get("source_type") or "rag"),
            text=str(hit.get("text") or hit.get("snippet") or ""),
            period=str(hit.get("period") or "") or None,
        )

    payload = (state.get("retrieved_docs") or {}).get(company) or {}
    period = period_label_from_meta(payload.get("fundamentals_meta"))
    structured = str(payload.get("structured_source") or "none")
    market_data = payload.get("market_data") or {}
    if market_data:
        text = (
            f"{company} {period} revenue was {get_fundamental(market_data, 'revenue')} billion USD, "
            f"EBITDA was {get_fundamental(market_data, 'ebitda')} billion USD, "
            f"R&D was {get_fundamental(market_data, 'r_and_d')} billion USD, and "
            f"operating income was {get_fundamental(market_data, 'operating_income')} billion USD."
        )
        add(
            evidence_id=f"ev_fund_{company}_{period}",
            citation=f"lumenfin:{structured}:{company}:{period}",
            source_type=structured if structured != "none" else "fundamentals",
            text=text,
            period=period,
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
            period=period,
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


def _prefer_refs_for_values(
    pool: list[EvidenceRef],
    values: list[float | None],
) -> list[EvidenceRef]:
    """Prefer page-anchored RAG hits that contain the values; else fundamentals text."""
    matched: list[EvidenceRef] = []
    for ref in pool:
        if ref.source_type not in {"rag", "document"} and "#p" not in (ref.citation or "").lower():
            continue
        if any(_text_contains_number(ref.text, v) for v in values if v is not None):
            matched.append(ref)
    if matched:
        # Prefer real page anchors.
        matched.sort(key=lambda r: (0 if r.page is not None else 1, r.evidence_id))
        return matched[:2]

    fund = [r for r in pool if r.source_type in {"sec_companyfacts", "yahoo_fundamentals", "document_extracted", "sample_db", "fundamentals"} or r.citation.startswith("lumenfin:")]
    fund = [
        r
        for r in fund
        if "sample_financial_data" in r.citation
        or "sec_companyfacts" in r.citation
        or "yahoo_fundamentals" in r.citation
        or "document_extracted" in r.citation
        or r.source_type in {"sec_companyfacts", "yahoo_fundamentals", "document_extracted", "fundamentals", "sample_db"}
    ]
    # Narrow to fundamentals citation specifically.
    fund_refs = [
        r
        for r in pool
        if r.evidence_id.startswith("ev_fund_")
        and any(_text_contains_number(r.text, v) for v in values if v is not None)
    ]
    if fund_refs:
        return fund_refs[:1]
    if fund:
        return fund[:1]
    return []


def _verify_numeric(claim: Claim, refs: list[EvidenceRef], input_values: list[float | None]) -> Claim:
    if not refs:
        claim.verification = "rejected"
        claim.verify_reason = "No evidence text contains the metric inputs."
        claim.evidence_refs = []
        return claim
    # Require at least one ref whose text contains an input OR is page-anchored RAG with overlap.
    usable = []
    for ref in refs:
        if any(_text_contains_number(ref.text, v) for v in input_values if v is not None):
            usable.append(ref)
        elif ref.page is not None and any(_text_contains_number(ref.text, v) for v in input_values if v is not None):
            usable.append(ref)
    if not usable:
        claim.verification = "rejected"
        claim.verify_reason = "Candidate evidence did not contain metric input numbers."
        claim.evidence_refs = []
        return claim
    claim.evidence_refs = usable
    claim.verification = "verified"
    claim.verify_reason = "Metric value and inputs bound to evidence containing those numbers."
    return claim


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
                refs = _prefer_refs_for_values(pool, input_vals + [float(value)])
                claims.append(_verify_numeric(claim, refs, input_vals))

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
                refs = _prefer_refs_for_values(pool, [float(raw)])
                claims.append(_verify_numeric(claim, refs, [float(raw)]))

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
        ebitda_claims = [c for c in num if c.metric_name == "ebitda_margin"]
        claim = Claim(
            claim_id=f"cl_inv_{company}_thesis",
            entity=company,
            claim_type="investment_conclusion",
            statement="",
            metric_name="research_thesis",
        )
        if block_numeric or not ebitda_claims or not risk:
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
        ebitda = ebitda_claims[0]
        risk_c = risk[0]
        ebitda_v = float(ebitda.value) if isinstance(ebitda.value, (int, float)) else 0.0
        if ebitda_v >= 0.25:
            stance = "quality-screening research thesis (not a recommendation)"
        elif ebitda_v >= 0.15:
            stance = "neutral quality-compounder research screen (not a recommendation)"
        else:
            stance = "defensive research posture pending operational evidence (not a recommendation)"
        claim.statement = (
            f"{company} supports a {stance} based on verified {ebitda.metric_name} "
            f"({_fmt_pct(ebitda_v)}) and verified risk signal ({risk_c.value})."
        )
        claim.evidence_refs = list(ebitda.evidence_refs[:1]) + list(risk_c.evidence_refs[:1])
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
    verified = filter_verified(claims)
    lines = [
        "## 0. Verified Claims Ledger",
        "",
        "*Only structurally verified claims are eligible for material assertions in this report. "
        "Unverified or rejected claims are omitted from the ledger.*",
        "",
    ]
    if not verified:
        lines.append("- (none — report will withhold evidence-backed financial assertions)")
        lines.append("")
        return lines
    lines.append("| ID | Entity | Type | Statement | Citation |")
    lines.append("|----|--------|------|-----------|----------|")
    for claim in verified:
        cite = claim.primary_citation.replace("|", "/")
        stmt = claim.statement.replace("|", "/")
        lines.append(
            f"| `{claim.claim_id}` | {claim.entity} | {claim.claim_type} | {stmt} | `{cite}` |"
        )
    lines.append("")
    return lines
