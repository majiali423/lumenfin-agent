from __future__ import annotations

import json
import re
from typing import Any

from ..claims import (
    binding_summary,
    build_claims,
    claim_to_dict,
    filter_verified,
    format_verified_claims_ledger,
    verified_by_entity,
)
from ..metrics_schema import get_fundamental
from ..observability import StepTimer
from ..quant_contract import non_comparable_companies
from ..reporting import (
    build_analyst_executive_summary,
    effective_report_output_format,
    filter_claims_for_brief,
    format_next_actions,
    format_peer_metric_matrix,
    format_period_alignment_notice,
    format_rag_citation_section,
    humanize_citation,
    is_low_signal_claim,
)
from ..state import FinanceState
from ..tools import build_chart_data
from .report_gap import render_fatal_data_gap_sections


def metric_verification_status_label(basis: str) -> str:
    """Render metric status without treating ``unverified`` as ``verified``."""
    normalized = (basis or "").strip().lower()
    if "unverified" in normalized or "computed" in normalized:
        return "Computed (unverified)"
    if normalized in {"verified", "ast_verified"}:
        return "Verified"
    if "live" in normalized:
        return "Live market"
    return basis or "—"


class SynthesisMixin:
    def claim_binder(self, state: FinanceState) -> FinanceState:
        with self._track_step("claim_binder") as timer:
            claims = build_claims(state)
            verified = filter_verified(claims)
            summary = binding_summary(claims)
            detail = (
                f"Built {summary['total_claims']} claims; verified={summary['verified_claims']}; "
                f"rejected={summary['rejected_claims']}; page_anchored={summary['page_anchored_verified']}; "
                f"bind_rate={summary['bind_rate']}."
            )
            update: FinanceState = {
                "claims": [claim_to_dict(c) for c in claims],
                "verified_claims": [claim_to_dict(c) for c in verified],
                "claim_binding": summary,
            }
            update.update(self._record("claim_binder", "ok", detail, state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update

    def _attach_structured_answer(
        self,
        state: FinanceState,
        update: FinanceState,
        *,
        window: list[dict[str, Any]] | None = None,
    ) -> None:
        from ..structured_answer import (
            StructuredAnswerError,
            build_structured_answer_from_state,
            degraded_structured_answer,
        )

        merged = {**state, **update}
        if window:
            from ..citation_alias import final_allowed_hits_sha256

            merged["citation_final_window"] = {
                "hits": window,
                "sha256": final_allowed_hits_sha256(window),
            }
        try:
            update["structured_answer"] = build_structured_answer_from_state(merged).to_dict()
        except StructuredAnswerError as exc:
            update["structured_answer"] = degraded_structured_answer(
                answer=str(merged.get("final_report") or merged.get("executive_summary") or ""),
                workflow_status=str(merged.get("workflow_status") or "completed"),
                error=exc,
            )

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIZER — Investment-Grade Report Assembly
    # ═══════════════════════════════════════════════════════════════
    def synthesizer(self, state: FinanceState) -> FinanceState:
        with self._track_step("synthesizer") as timer:
            return self._synthesize_report(state, timer)

    def _synthesize_report(self, state: FinanceState, timer: StepTimer) -> FinanceState:
        def ensure_sentence_complete(text: str) -> str:
            cleaned = (text or "").strip()
            if not cleaned:
                return cleaned
            if cleaned[-1] not in ".!?。！？)]】":
                return cleaned + "。"
            return cleaned

        sections: list[str] = []
        def S(line: str = "") -> None:
            sections.append(line)

        fatal_data_gap = bool(state.get("fatal_data_gap"))
        if fatal_data_gap:
            sections, detail = render_fatal_data_gap_sections(state, data_mode=self.data_mode)
            final_report = "\n".join(sections)
            update: FinanceState = {
                "report_sections": sections,
                "executive_summary": detail,
                "final_report": final_report,
                "llm_backend": self.llm_client.backend_name,
                "swot_analysis": {},
                "investment_thesis": {},
                "chart_data": {},
                "workflow_status": "incomplete_data",
                "degraded_mode": True,
            }
            self._attach_structured_answer(state, update)
            update.update(
                self._record(
                    "synthesizer",
                    "incomplete_data",
                    "Fail-loud incomplete report: no computable fundamentals; skipped inventing metrics.",
                    state,
                    timer.metrics(),
                )
            )
            self.session_memory.save({**state, **update})
            return update

        from ..structured_answer import bind_production_citation_window

        citation_window = bind_production_citation_window(state)
        doc_context = ""
        if not citation_window and state.get("document_contexts"):
            excerpts = [d["excerpt"][:600] for d in state["document_contexts"] if d.get("excerpt")]
            if excerpts:
                doc_context = "\nUploaded PDF excerpts:\n" + "\n---\n".join(excerpts)

        has_metrics = any(state.get("financial_metrics", {}).values())
        knowledge_hint = ""
        if not has_metrics and not doc_context:
            knowledge_hint = (
                "\nNote: Limited structured data available. Leverage your public knowledge of these companies "
                "to provide insightful analysis. Do not simply state 'insufficient data'."
            )

        profile_lines = [f"{c}: {state.get('company_profiles', {}).get(c, '')}" for c in state["companies"]]
        profile_context = "\n".join(profile_lines)
        metrics_context = json.dumps(state.get("financial_metrics", {}), ensure_ascii=False)
        sentiment_context = json.dumps(state.get("sentiment_analysis", {}), ensure_ascii=False)
        risk_context = json.dumps(state.get("risk_scores", {}), ensure_ascii=False)
        peer_context = state.get("peer_comparison", {}).get("summary", "")
        has_uploaded_docs = bool(state.get("document_contexts"))
        market_snapshots = state.get("market_snapshots", {})
        market_ok = any(snap.get("current_price") is not None for snap in market_snapshots.values())
        unverified_note = "_Source: LLM knowledge (unverified in this run)._"

        def fmt_pct(value: Any) -> str:
            return f"{value:.1%}" if isinstance(value, (int, float)) else "n/a"

        def fmt_x(value: Any) -> str:
            return f"{value:.2f}x" if isinstance(value, (int, float)) else "n/a"

        # Claim → Evidence: synthesizer may only assert from verified claims.
        from ..claims import claims_from_state

        if not state.get("verified_claims") and not state.get("claims"):
            built = build_claims(state)
            state = {
                **state,
                "claims": [claim_to_dict(c) for c in built],
                "verified_claims": [claim_to_dict(c) for c in filter_verified(built)],
                "claim_binding": binding_summary(built),
            }
        all_claims = claims_from_state(state)
        verified_claims = filter_verified(all_claims)

        def cite_for(company: str, *, claim_type: str | None = None, metric_name: str | None = None) -> str:
            hits = verified_by_entity(
                verified_claims,
                company,
                claim_type=claim_type,  # type: ignore[arg-type]
                metric_name=metric_name,
            )
            if not hits:
                return ""
            return hits[0].primary_citation

        def build_grounded_summary() -> str:
            return build_analyst_executive_summary(
                state,
                verified_claims,
                brief=effective_report_output_format(state) != "research_report",
            )

        llm_summary = build_grounded_summary()

        output_format = effective_report_output_format(state)
        is_full = output_format == "research_report"
        is_table = output_format == "table_summary"
        # Brief/table: keep source, summary/ledger (except pure table), metrics, gaps, compliance, disclaimer.
        include_narrative_sections = is_full
        include_summary_and_ledger = not is_table
        ledger_claims = filter_claims_for_brief(verified_claims) if not is_full else verified_claims

        # ── Report Construction (analyst-first; audit details in appendices) ──
        S("# LumenFin Diligence Report")
        S("")
        if is_full:
            S(
                "**Report Type:** Diligence Screening Report (AI-assisted) | "
                "**Classification:** For internal research reference only"
            )
        elif is_table:
            S("**Report Type:** Table Summary | **Classification:** AI-Generated, For Reference Only")
            S("")
            S(f"**Report Mode:** `{output_format}` (explicit UI/API selection; keywords do not auto-trim).")
        else:
            S("**Report Type:** Brief Diligence | **Classification:** AI-Generated, For Reference Only")
            S("")
            S(f"**Report Mode:** `{output_format}` (explicit UI/API selection; keywords do not auto-trim).")
        S("")
        if include_summary_and_ledger:
            S("## 1. Executive Summary")
            S("")
            S(llm_summary)
            S("")
            S(
                "**Evidence Boundary:** Material numeric and investment assertions are limited to "
                "verified claims with bound sources. Incomplete or unbound inputs are treated as "
                "data limitations (including Computed/unverified ratios). Risk-model scores remain "
                "screening indicators even when bound to risk-model evidence."
            )
            S("")
        if state.get("partial_data_gap"):
            coverage = state.get("coverage_matrix") or {}
            comparable = [
                company for company in state.get("companies") or [] if (coverage.get(company) or {}).get("comparable")
            ]
            skipped = state.get("non_comparable_companies") or non_comparable_companies(
                list(state.get("companies") or []),
                coverage,
            )
            S("**Partial Peer Coverage Notice:**")
            S(
                f"- Comparable ratio set: {', '.join(comparable) if comparable else '(none)'}"
            )
            S(
                f"- Non-comparable peers: {', '.join(skipped) if skipped else '(none)'} "
                "(missing extractable revenue/EBITDA/R&D inputs)."
            )
            S("")
        for line in format_next_actions(state):
            S(line)

        # Period + source alignment (merged)
        S("## 2. Period & Source Alignment")
        S("")
        for line in format_period_alignment_notice(state):
            # Downgrade nested "## Period Alignment" to a plain label inside §2.
            if line.startswith("## Period Alignment"):
                S("### Period Alignment")
                continue
            S(line)
        source_resolution = state.get("source_resolution") or {}
        company_resolutions = source_resolution.get("companies") or {}
        fallback_rows = [
            (company, info)
            for company, info in company_resolutions.items()
            if info.get("live_fallback_used")
        ]
        if source_resolution.get("prefer_uploaded_only") or fallback_rows or state.get("document_contexts"):
            mode = str(source_resolution.get("mode") or "hybrid")
            S("### Source Resolution")
            S("")
            if source_resolution.get("prefer_uploaded_only"):
                S(
                    "**Mode: uploaded materials only.** Structured fundamentals were not backfilled "
                    "from SEC/Yahoo/sample even if the upload lacked computable metrics."
                )
            elif fallback_rows:
                S(
                    "**Mode: hybrid.** Uploads are preferred; when they lack extractable "
                    "revenue/EBITDA/R&D, live providers may fill the gap — listed below."
                )
            else:
                S(
                    f"**Mode: {mode}.** Per-company structured source is listed so document narrative "
                    "is not confused with SEC/Yahoo numbers."
                )
            S("")
            S("| Company | Fundamentals source | Upload had metrics? | Notes |")
            S("|---------|---------------------|---------------------|-------|")
            for company in state.get("companies") or []:
                info = company_resolutions.get(company) or {}
                source = str(
                    info.get("structured_source")
                    or (state.get("retrieved_docs") or {}).get(company, {}).get("structured_source")
                    or "none"
                )
                had_metrics = (
                    "yes"
                    if info.get("upload_had_computable_metrics")
                    else ("n/a" if not state.get("document_contexts") else "no")
                )
                note = str(info.get("fallback_reason") or "")
                if not note and source == "document_extracted":
                    note = "Numbers taken from uploaded materials."
                elif not note and info.get("live_fallback_used"):
                    note = f"Upload lacked metrics; used {source}."
                S(f"| {company} | {source} | {had_metrics} | {note or '—'} |")
            S("")

        # Profiles are LLM narrative noise in production; keep only in demo for orientation.
        if include_narrative_sections and has_uploaded_docs and self.data_mode == "demo":
            shown_profile = False
            for company in state["companies"]:
                profile = ensure_sentence_complete(
                    state.get("company_profiles", {}).get(company, "")
                )
                if not profile or "not available" in profile.lower() or "pending" in profile.lower():
                    continue
                if len(profile) > 420:
                    profile = profile[:419].rstrip() + "…"
                if not shown_profile:
                    S("## Company Profiles *(LLM profile — unverified)*")
                    S("")
                    shown_profile = True
                S(f"### {company}")
                S(profile)
                S("")

        S("## 3. Financial Performance Analysis")
        S("")
        S("*Ratios below are computed from structured fundamentals. Prefer verified rows for decision use.*")
        S("")
        for line in format_peer_metric_matrix(state):
            S(line)
        for company in state["companies"]:
            metrics = state.get("financial_metrics", {}).get(company, {})

            S(f"### {company}")
            S("")

            if metrics:
                S("**Key Financial Indicators**")
                S("")
                S(
                    "*Internal screen thresholds are LumenFin heuristics (not industry peer medians). "
                    "P/E is TTM live and may not match the FY window of the statement ratios.*"
                )
                S("")
                S("| Metric | Value | Internal screen | Vs screen | Status | Source |")
                S("|--------|-------|-----------------|-----------|--------|--------|")

                metric_conf = state.get("metric_confidence", {}).get(company, {})

                def assess_metric(metric_key: str, value: float) -> tuple[str, str]:
                    if metric_key == "ebitda_margin":
                        if value >= 0.25:
                            return "Above", "internal screen >25%"
                        if value >= 0.15:
                            return "Near", "internal screen >25%"
                        return "Below", "internal screen >25%"
                    if metric_key == "operating_margin":
                        if value >= 0.20:
                            return "Above", "internal screen >20%"
                        if value >= 0.12:
                            return "Near", "internal screen >20%"
                        return "Below", "internal screen >20%"
                    if metric_key == "estimated_net_margin":
                        if value >= 0.15:
                            return "Above", "internal screen >15%"
                        if value >= 0.08:
                            return "Near", "internal screen >15%"
                        return "Below", "internal screen >15%"
                    if metric_key == "estimated_fcf_margin":
                        if value >= 0.10:
                            return "Above", "internal screen >10%"
                        if value >= 0.05:
                            return "Near", "internal screen >10%"
                        return "Below", "internal screen >10%"
                    if metric_key == "r_and_d_intensity":
                        if 0.05 <= value <= 0.15:
                            return "In range", "internal screen 5-15%"
                        if 0.03 <= value < 0.05 or 0.15 < value <= 0.20:
                            return "Near", "internal screen 5-15%"
                        return "Outside", "internal screen 5-15%"
                    return "—", "—"

                def add_row(metric_key, label, screen, value=None):
                    v = value if value is not None else metrics.get(metric_key)
                    if v is None:
                        return
                    verified_hit = verified_by_entity(
                        verified_claims, company, metric_name=metric_key
                    )
                    allow_computed = metric_key in (
                        "ebitda_margin",
                        "operating_margin",
                        "r_and_d_intensity",
                    )
                    if metric_key in ("ebitda_margin", "operating_margin", "r_and_d_intensity", "pe_ratio"):
                        if not verified_hit and not (allow_computed and isinstance(v, (int, float))):
                            return
                    conf = metric_conf.get(metric_key, {})
                    if verified_hit:
                        raw_basis = str(conf.get("basis", "Verified"))
                        raw_cite = verified_hit[0].primary_citation
                        citation = (
                            f"{humanize_citation(raw_cite)} [{raw_cite}]" if raw_cite else humanize_citation(raw_cite)
                        )
                    else:
                        raw_basis = "Computed (unverified)"
                        citation = "structured fundamentals (claim not bound)"
                    status = metric_verification_status_label(raw_basis)
                    if metric_key in (
                        "ebitda_margin",
                        "r_and_d_intensity",
                        "operating_margin",
                        "estimated_net_margin",
                        "estimated_fcf_margin",
                    ):
                        if metric_key.startswith("estimated_") and not verified_hit:
                            return
                        vs_screen, _ = assess_metric(metric_key, float(v))
                        S(
                            f"| {label} | {v:.2%} | {screen} | {vs_screen} | {status} | {citation} |"
                        )
                    elif metric_key == "pe_ratio":
                        if not verified_hit:
                            return
                        S(
                            f"| {label} | {v:.2f}x | {screen} | — | {status} | {citation} |"
                        )

                asked = [
                    str(item)
                    for item in ((state.get("query_plan") or {}).get("requested_metrics") or [])
                    if str(item).strip()
                ]
                if not asked or "ebitda_margin" in asked:
                    add_row("ebitda_margin", "EBITDA Margin", ">25%")
                if not asked or "operating_margin" in asked:
                    add_row("operating_margin", "Operating Margin", ">20%")
                if not asked or "r_and_d_intensity" in asked:
                    add_row("r_and_d_intensity", "R&D Intensity", "5-15%")
                if not asked:
                    add_row("pe_ratio", "P/E (TTM, live)", "—")
                # Absolute fundamentals (once) for analyst context
                market = ((state.get("retrieved_docs") or {}).get(company) or {}).get("market_data") or {}
                from ..query_focus import format_billion_amount

                abs_keys = (
                    ("revenue", "Revenue"),
                    ("operating_income", "Operating income"),
                    ("r_and_d", "R&D"),
                )
                if asked:
                    abs_keys = tuple((key, label) for key, label in abs_keys if key in asked)
                abs_bits = []
                for key, label in abs_keys:
                    hits = verified_by_entity(verified_claims, company, metric_name=key)
                    if hits:
                        abs_bits.append(hits[0].render_with_citation(humanize=True))
                        continue
                    raw_abs = get_fundamental(market, key)
                    if isinstance(raw_abs, (int, float)):
                        payload = (state.get("retrieved_docs") or {}).get(company) or {}
                        field_prov = (payload.get("fundamental_provenance") or {}).get(key) or {}
                        cite = str(field_prov.get("citation") or "").strip()
                        period = str(field_prov.get("period") or "").strip()
                        if period:
                            period_bit = f" for {period}"
                        else:
                            period_bit = " (period not stated in the source)"
                        if cite:
                            abs_bits.append(
                                f"{company} {label} is {format_billion_amount(float(raw_abs))} billion USD{period_bit} [{cite}]."
                            )
                        else:
                            abs_bits.append(
                                f"{company} {label} is {format_billion_amount(float(raw_abs))} billion USD "
                                f"(structured fundamentals; claim not bound)."
                            )
                if abs_bits:
                    S("")
                    S("**Key absolute figures**")
                    for bit in abs_bits[:3]:
                        S(f"- {bit}")
                S("")
            else:
                S(
                    "*[Partial Coverage] Insufficient structured data for ratio comparison. "
                    "Market-only or risk-screening context may still appear below.*"
                )
                if company in (state.get("non_comparable_companies") or []):
                    source = state.get("retrieved_docs", {}).get(company, {}).get("structured_source", "none")
                    S(f"*Structured source for {company}: {source}. Peer margin comparison skipped.*")
                S("")

        if has_uploaded_docs and state.get("rag_evidence"):
            S("### Uploaded Filing Evidence for the Requested Topics")
            S("")
            S(
                "*These are page-anchored excerpts selected by hybrid retrieval and reranking. "
                "They are evidence excerpts, not additional model-verified conclusions; use them "
                "to inspect segment, guidance, liquidity and GAAP/non-GAAP details requested above.*"
            )
            S("")
            for line in format_rag_citation_section(
                state.get("rag_evidence"),
                max_excerpt=360,
                heading="#### Evidence Coverage",
                max_rows_per_company=10,
            ):
                S(line)

        # ── Risk (dedicated section; screening scores labeled honestly) ──
        swot: dict[str, dict[str, str]] = {}
        investment_thesis: dict[str, dict[str, str]] = {}
        any_risk = False
        for company in state["companies"]:
            if verified_by_entity(verified_claims, company, claim_type="risk_conclusion") or state.get(
                "risk_scores", {}
            ).get(company):
                any_risk = True
                break
        if any_risk:
            S("## 4. Risk")
            S("")
            S(
                "*Screening scores (model-derived; not a 10-K Item 1A extract). "
                "Use as diligence flags, not as independently audited risk conclusions.*"
            )
            S("")
            for company in state["companies"]:
                risk_claims = verified_by_entity(verified_claims, company, claim_type="risk_conclusion")
                risk_data = state.get("risk_scores", {}).get(company, {}) or {}
                if not risk_claims and not risk_data:
                    continue
                S(f"### {company} — Risk Screening Matrix")
                S("")
                S("| Dimension | Screening score (1-10) | Level | Source |")
                S("|-----------|------------------------|-------|--------|")
                dim_labels = {
                    "financial_risk": "Financial",
                    "operational_risk": "Operational",
                    "market_risk": "Market",
                    "regulatory_risk": "Regulatory",
                    "supply_chain_risk": "Supply Chain",
                }
                unknown_supply = False
                for dim, label in dim_labels.items():
                    hits = verified_by_entity(verified_claims, company, metric_name=dim)
                    if dim == "supply_chain_risk" and not hits:
                        hits = [c for c in risk_claims if c.metric_name == "supply_chain_risk"]
                    if not hits:
                        continue
                    claim = hits[0]
                    if dim == "supply_chain_risk" and str(claim.value).lower() in {
                        "unknown",
                        "n/a",
                        "none",
                    }:
                        unknown_supply = True
                        continue
                    score = risk_data.get(dim, claim.value if isinstance(claim.value, (int, float)) else 5.0)
                    if not isinstance(score, (int, float)):
                        score = 5.0
                    level = "Low" if score < 3.5 else ("Moderate" if score < 6.5 else "Elevated")
                    S(
                        f"| {label} | {score:.1f} | {level} | "
                        f"{humanize_citation(claim.primary_citation)} [{claim.primary_citation}] |"
                    )
                S("")
                if unknown_supply:
                    S(
                        "*Supply-chain screen: no clear filing signal in this run "
                        "(not shown as a Moderate/Elevated score).*"
                    )
                    S("")
                material_risk = [
                    c
                    for c in risk_claims
                    if not (
                        c.metric_name == "supply_chain_risk"
                        and str(c.value).lower() in {"unknown", "n/a", "none"}
                    )
                ]
                if material_risk:
                    S("**Screening conclusions**")
                    S("")
                    for claim in material_risk:
                        if not is_full and is_low_signal_claim(claim):
                            continue
                        S(f"- {claim.render_with_citation(humanize=True)}")
                    S("")

        # ── Research thesis (verified investment claims only) ──
        if include_narrative_sections:
            S("## 5. Research Thesis & Positioning")
            S("")
            S(
                "*Not a buy/sell recommendation. Emitted only from verified investment conclusions "
                "backed by verified numeric + risk evidence.*"
            )
            S("")
            for company in state["companies"]:
                inv = verified_by_entity(verified_claims, company, claim_type="investment_conclusion")
                S(f"### {company}")
                if inv:
                    bull = inv[0].render_with_citation(humanize=True)
                    risk_lines = [
                        c
                        for c in verified_by_entity(
                            verified_claims, company, claim_type="risk_conclusion"
                        )
                        if not (
                            c.metric_name == "supply_chain_risk"
                            and str(c.value).lower() in {"unknown", "n/a", "none"}
                        )
                    ]
                    # Prefer scored dimensions for the bear line over supply-chain noise.
                    scored = [
                        c
                        for c in risk_lines
                        if c.metric_name in {"financial_risk", "operational_risk", "market_risk"}
                    ]
                    bear_claim = scored[0] if scored else (risk_lines[0] if risk_lines else None)
                    bear = (
                        bear_claim.render_with_citation(humanize=True)
                        if bear_claim
                        else "See Risk screening section; no separate unverified bear narrative is invented."
                    )
                    investment_thesis[company] = {"bull_case": bull, "bear_case": bear}
                    S(f"- **Bull case (screening):** {bull}")
                    S(f"- **Bear / risk case (screening):** {bear}")
                else:
                    rejected = [
                        c
                        for c in all_claims
                        if c.entity == company
                        and c.claim_type == "investment_conclusion"
                        and c.verification == "rejected"
                    ]
                    msg = rejected[0].statement if rejected else (
                        f"{company}: investment conclusion withheld — missing verified claims."
                    )
                    investment_thesis[company] = {"bull_case": msg, "bear_case": msg}
                    S(f"- {msg}")
                S("")

        # ── Compliance ──
        S("## 6. Compliance Review & Data Integrity")
        S("")
        if state.get("compliance_summary") and state.get("compliance_findings"):
            compliance_summary = str(state["compliance_summary"]).strip()
            compliance_summary = re.sub(r"^\**\s*Audit Opinion:\s*\**\s*", "", compliance_summary, flags=re.IGNORECASE)
            S(f"**Audit Opinion:** {compliance_summary}")
            S("")
        if state.get("compliance_findings"):
            S("**Identified Issues:**")
            for item in state["compliance_findings"]:
                S(f"- {item}")
            if state.get("critic_iterations", 0) >= state.get("critic_max_iterations", 2):
                S("")
                S(
                    f"*Evaluator-optimizer loop exhausted after {state['critic_iterations']} iteration(s); "
                    "report generated with acknowledged compliance gaps.*"
                )
        else:
            S(
                "Core compliance checks passed. Material assertions are limited to verified claims "
                "with bound evidence."
            )
        S("")

        # ── Appendices ──
        if include_summary_and_ledger:
            for line in format_verified_claims_ledger(ledger_claims):
                S(line)
            binding = state.get("claim_binding") or binding_summary(all_claims)
            S(
                f"*Binding stats: verified={binding.get('verified_claims', 0)}/"
                f"{binding.get('total_claims', 0)} "
                f"(bind_rate={binding.get('bind_rate', 0)}, "
                f"page_anchored={binding.get('page_anchored_verified', 0)}).*"
            )
            S("")

        S("## Appendix B. Methodology, Data Sources & Disclaimer")
        S("")
        S(
            "**Methods:** Deterministic ratio engine on structured fundamentals; claim→evidence binder; "
            "optional LLM screening for sentiment/profile; multi-factor risk screening scores."
        )
        S("")
        document_contexts = state.get("document_contexts", [])
        market_snapshots = state.get("market_snapshots", {})
        rag_evidence = state.get("rag_evidence", {})
        companies = state.get("companies", [])
        sample_companies = [
            c
            for c in companies
            if str((state.get("retrieved_docs") or {}).get(c, {}).get("structured_source") or "")
            == "sample_db"
        ]
        market_ok = sum(1 for snap in market_snapshots.values() if snap.get("current_price") is not None)
        market_total = len(market_snapshots)
        rag_chunks = sum(len(hits) for hits in rag_evidence.values())

        source_parts: list[str] = []
        if document_contexts:
            source_types = sorted(
                {
                    str(doc.get("source_type") or "unknown")
                    for doc in document_contexts
                }
            )
            source_parts.append(
                f"Uploaded documents: {len(document_contexts)} file(s), types={', '.join(source_types)}."
            )
        else:
            source_parts.append("Uploaded documents: none (no user files were provided for this run).")

        if rag_chunks > 0:
            source_parts.append(f"RAG evidence: Milvus hybrid retrieval returned {rag_chunks} cited chunk(s).")
        elif document_contexts:
            source_parts.append("RAG evidence: enabled but no cited chunk was retrieved in this run.")
        else:
            source_parts.append("RAG evidence: not applicable because no documents were uploaded.")

        if market_total:
            source_parts.append(
                f"Market data API: {market_ok}/{market_total} company snapshots succeeded; "
                "per-company failures degrade only that entity's live-market metrics."
            )
        else:
            source_parts.append("Market data API: no market snapshots requested.")

        yahoo_companies = [
            c
            for c in companies
            if str((state.get("retrieved_docs") or {}).get(c, {}).get("structured_source") or "")
            == "yahoo_fundamentals"
        ]
        sec_companies = [
            c
            for c in companies
            if str((state.get("retrieved_docs") or {}).get(c, {}).get("structured_source") or "")
            == "sec_companyfacts"
        ]
        if sample_companies:
            source_parts.append(
                f"Structured fundamentals: DEMO sample financial database used for {', '.join(sample_companies)} "
                f"(data_mode={self.data_mode})."
            )
        elif sec_companies:
            source_parts.append(
                f"Structured fundamentals: SEC EDGAR companyfacts for {', '.join(sec_companies)} "
                f"(structured_source=sec_companyfacts, data_mode={self.data_mode})."
            )
        elif yahoo_companies:
            source_parts.append(
                f"Structured fundamentals: Yahoo Finance annual income statement for "
                f"{', '.join(yahoo_companies)} (structured_source=yahoo_fundamentals, data_mode={self.data_mode})."
            )
        else:
            source_parts.append(
                f"Structured fundamentals: derived from uploaded structured documents when available "
                f"(data_mode={self.data_mode})."
            )

        resolution = state.get("source_resolution") or {}
        if resolution.get("prefer_uploaded_only"):
            source_parts.append(
                "Source policy: prefer_uploaded_only=true (SEC/Yahoo/sample backfill disabled for this run)."
            )
        else:
            fallback_companies = [
                name
                for name, info in (resolution.get("companies") or {}).items()
                if info.get("live_fallback_used")
            ]
            if fallback_companies:
                source_parts.append(
                    "Live/sample backfill after sparse upload for: "
                    + ", ".join(fallback_companies)
                    + " (see Period & Source Alignment)."
                )

        source_parts.append("Narrative analysis: generated by the configured LLM using retrieved evidence and computed metrics.")
        S(f"**Data Sources:** {' '.join(source_parts)}")
        S("")
        for line in format_rag_citation_section(rag_evidence):
            S(line)
        if market_total:
            S("")
            S("**Market Data by Company:**")
            for company in companies:
                snap = market_snapshots.get(company, {})
                symbol = snap.get("symbol") or state.get("target_symbols", {}).get(company, company)
                status = snap.get("status") or ("ok" if snap.get("current_price") is not None else "failed")
                provider = snap.get("provider") or "unknown"
                as_of = snap.get("fetched_at") or "n/a"
                if snap.get("current_price") is not None:
                    S(
                        f"- {company} ({symbol}): status={status}, provider={provider}, "
                        f"as_of={as_of}, price={snap.get('current_price')}."
                    )
                else:
                    from ..provider_resilience import redact_provider_message

                    err = redact_provider_message(
                        str(snap.get("error") or "no live price returned"),
                        limit=240,
                    )
                    S(f"- {company} ({symbol}): status=failed, error={err}.")
            S("")
        S(
            "**Source Attribution:** Quant tables use deterministic calculations on structured inputs. "
            "Market rows use live snapshots when available. Company profiles and thesis language are "
            "LLM-assisted unless a row cites bound evidence. Risk matrix values are model-derived "
            "screening indicators and should not be treated as independently audited facts."
        )
        S("")
        if self.data_mode == "demo" or sample_companies:
            S(
                "**Disclaimer:** DEMO MODE -- some or all structured fundamentals may come from the built-in sample database, "
                "not audited filings. This report is for research and demonstration only. It does not constitute investment advice."
            )
        else:
            S(
                "**Disclaimer:** This report is generated by an AI-powered multi-agent system for research purposes only. "
                "It does not constitute investment advice, a solicitation, or a recommendation to buy or sell any security."
            )

        final_report = "\n".join(sections)
        if citation_window:
            from ..citation_alias import (
                build_citation_alias_map,
                replace_ephemeral_aliases_in_user_text,
            )

            alias_map = build_citation_alias_map(
                citation_window,
                case_id=str(state.get("thread_id") or state.get("run_id") or ""),
                attempt_id=str(state.get("repair_attempt_id") or state.get("thread_id") or "attempt-1"),
                tenant_id=str(state.get("rag_tenant_id") or state.get("tenant_id") or ""),
                session_id=str(state.get("thread_id") or state.get("run_id") or ""),
            )
            final_report = replace_ephemeral_aliases_in_user_text(
                final_report,
                citation_window,
                alias_map,
            )

        # ── Chart Data ──
        chart_data = build_chart_data(
            companies=state["companies"],
            financial_metrics=state.get("financial_metrics", {}),
            sentiment_analysis=state.get("sentiment_analysis", {}),
            risk_scores=state.get("risk_scores", {}),
            audit_log=state.get("audit_log", []),
        )

        update: FinanceState = {
            "report_sections": sections,
            "executive_summary": llm_summary,
            "final_report": final_report,
            "llm_backend": self.llm_client.backend_name,
            "swot_analysis": swot,
            "investment_thesis": investment_thesis,
            "chart_data": chart_data,
            "workflow_status": "completed",
        }
        self._attach_structured_answer(state, update, window=citation_window or None)
        synth_detail = (
            f"Report assembled from verified claims only "
            f"(mode={output_format}; verified={len(verified_claims)}/{len(all_claims)}; "
            f"bind_rate={(state.get('claim_binding') or {}).get('bind_rate', 0)})."
        )
        update.update(self._record("synthesizer", "ok", synth_detail, state, timer.metrics()))
        self.session_memory.save({**state, **update})
        return update
