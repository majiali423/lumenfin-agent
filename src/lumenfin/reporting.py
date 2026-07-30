from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .evaluation import evaluate_run_state

_PAGE_CITATION_RE = re.compile(r"#p\d+", re.IGNORECASE)

REPORT_OUTPUT_FORMATS = frozenset({"research_report", "executive_summary", "table_summary"})
DEFAULT_REPORT_OUTPUT_FORMAT = "research_report"


def normalize_requested_output_format(value: Any) -> str | None:
    """Return a valid explicit report mode, or None when absent/invalid (treat as full)."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    aliases = {
        "full": "research_report",
        "complete": "research_report",
        "brief": "executive_summary",
        "summary": "executive_summary",
        "executive": "executive_summary",
        "table": "table_summary",
    }
    normalized = aliases.get(text, text)
    if normalized in REPORT_OUTPUT_FORMATS:
        return normalized
    return None


def effective_report_output_format(state: dict[str, Any] | None) -> str:
    """Report length mode: only explicit requested_output_format may shorten the report.

    Keyword-detected query_plan.output_format is intentionally ignored so phrases like
    "摘要" inside a full-report request cannot silently trim diligence sections.
    """
    requested = normalize_requested_output_format((state or {}).get("requested_output_format"))
    return requested or DEFAULT_REPORT_OUTPUT_FORMAT


def format_next_actions(state: dict[str, Any] | None) -> list[str]:
    """Rule-based补件清单 from existing gap fields (no LLM)."""
    state = state or {}
    companies = [str(c) for c in (state.get("companies") or [])]
    coverage = state.get("coverage_matrix") or {}
    comparable = [
        company for company in companies if (coverage.get(company) or {}).get("comparable")
    ]
    non_comparable = list(state.get("non_comparable_companies") or [])
    if not non_comparable and coverage:
        non_comparable = [
            company for company in companies if not (coverage.get(company) or {}).get("comparable")
        ]
    detail = str(state.get("data_gap_detail") or "").strip()
    fatal = bool(state.get("fatal_data_gap"))
    partial = bool(state.get("partial_data_gap"))
    status = str(state.get("workflow_status") or "")

    if not fatal and not partial and status != "incomplete_data":
        return []

    lines = [
        "## Next Actions（补件清单）",
        "",
        f"- Status: `{status or ('incomplete_data' if fatal else 'partial')}`"
        f"{'; fatal_data_gap' if fatal else ''}{' / partial_data_gap' if partial else ''}",
        f"- Comparable: {', '.join(comparable) if comparable else '(none)'}",
        f"- Non-comparable: {', '.join(non_comparable) if non_comparable else '(none)'}",
    ]
    if detail:
        lines.append(f"- Gap detail: {detail}")
    lines.extend(
        [
            "- Suggested next steps:",
            "  1. Upload 10-K/annual filings (PDF) for non-comparable issuers with extractable FY metrics, or",
            "  2. Narrow the compare set to comparable companies only, or",
            "  3. Clarify fiscal year / company_scope via HITL and resume the same thread.",
            "",
        ]
    )
    return lines


_FY_RE = re.compile(r"(?:FY\s*)?(20\d{2})", re.IGNORECASE)

# Peer metrics shown in comparison capsule / matrix (pct vs multiple).
COMPARE_METRIC_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("ebitda_margin", "EBITDA Margin", True),
    ("operating_margin", "Operating Margin", True),
    ("r_and_d_intensity", "R&D Intensity", True),
    ("pe_ratio", "P/E (TTM)", False),
)


def parse_requested_fiscal_year(*texts: Any) -> int | None:
    """Extract a single FY year from planner/query strings (e.g. FY2024, 2024)."""
    for raw in texts:
        text = str(raw or "").strip()
        if not text:
            continue
        # Prefer explicit FY#### tokens.
        fy_hits = re.findall(r"FY\s*(20\d{2})", text, flags=re.IGNORECASE)
        if fy_hits:
            return int(fy_hits[-1])
        hits = _FY_RE.findall(text)
        if hits:
            return int(hits[-1])
    return None


def requested_fiscal_year_from_state(state: dict[str, Any] | None) -> int | None:
    state = state or {}
    plan = state.get("query_plan") or {}
    return parse_requested_fiscal_year(
        plan.get("time_range"),
        state.get("query"),
        (state.get("user_clarification") or {}).get("time_range"),
        (state.get("user_clarification") or {}).get("fiscal_year"),
    )


def used_fiscal_year_for_company(state: dict[str, Any] | None, company: str) -> int | None:
    payload = ((state or {}).get("retrieved_docs") or {}).get(company) or {}
    meta = payload.get("fundamentals_meta") or {}
    for key in ("fiscal_year", "fy"):
        value = meta.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            text = str(value)
            if re.fullmatch(r"20\d{2}", text):
                return int(text)
    period = str(meta.get("period") or meta.get("period_end") or "")
    match = re.search(r"(20\d{2})", period)
    if match:
        return int(match.group(1))
    return None


def format_period_alignment_notice(state: dict[str, Any] | None) -> list[str]:
    """Disclose requested vs used fiscal periods (never silent FY swap)."""
    state = state or {}
    requested = requested_fiscal_year_from_state(state)
    companies = [str(c) for c in (state.get("companies") or [])]
    if not companies:
        return []

    rows: list[tuple[str, str, str]] = []
    any_mismatch = False
    for company in companies:
        used = used_fiscal_year_for_company(state, company)
        payload = (state.get("retrieved_docs") or {}).get(company) or {}
        meta = payload.get("fundamentals_meta") or {}
        alignment = str(meta.get("period_alignment") or "")
        used_label = f"FY{used}" if used is not None else "n/a"
        if requested is None:
            status = alignment or "no FY requested"
        elif used is None:
            status = "requested FY not found in structured fundamentals"
            any_mismatch = True
        elif int(used) == int(requested):
            status = "exact match"
        else:
            status = f"FALLBACK — requested FY{requested}, using {used_label}"
            any_mismatch = True
        req_label = f"FY{requested}" if requested is not None else "—"
        rows.append((company, req_label, f"{used_label} ({status})"))

    if requested is None and not any(
        ((state.get("retrieved_docs") or {}).get(c) or {}).get("fundamentals_meta")
        for c in companies
    ):
        return []

    lines = [
        "## Period Alignment",
        "",
        "| Company | Requested | Used (status) |",
        "|---------|-----------|---------------|",
    ]
    for company, req, used in rows:
        lines.append(f"| {company} | {req} | {used} |")
    lines.append("")
    if any_mismatch:
        lines.append(
            "**Period notice:** Structured fundamentals are not aligned to the requested fiscal year "
            "for at least one issuer. Treat YoY / same-year peer statements as non-comparable until "
            "the requested FY is available or the query is restated."
        )
        lines.append("")
    elif requested is not None:
        lines.append(f"**Period notice:** Structured fundamentals match requested FY{requested}.")
        lines.append("")
    return lines


def _fmt_metric_value(value: Any, *, is_pct: bool) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    if is_pct:
        return f"{float(value):.1%}"
    return f"{float(value):.2f}x"


def format_peer_metric_matrix(state: dict[str, Any] | None) -> list[str]:
    """Wide compare table with explicit n/a cells when a peer lacks a metric."""
    state = state or {}
    companies = [str(c) for c in (state.get("companies") or [])]
    if len(companies) < 2:
        return []
    metrics_by_company = state.get("financial_metrics") or {}
    lines = [
        "### Peer Metric Matrix (comparable columns; gaps are explicit)",
        "",
        "| Metric | " + " | ".join(companies) + " | Notes |",
        "|--------|" + "|".join(["------"] * len(companies)) + "|-------|",
    ]
    for key, label, is_pct in COMPARE_METRIC_SPECS:
        cells: list[str] = []
        present: list[str] = []
        missing: list[str] = []
        for company in companies:
            metrics = metrics_by_company.get(company) or {}
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                cells.append(_fmt_metric_value(value, is_pct=is_pct))
                present.append(company)
            else:
                cells.append("n/a")
                missing.append(company)
        if missing and present:
            note = f"asymmetric: missing for {', '.join(missing)}"
        elif missing and not present:
            note = "unavailable for all peers"
        else:
            note = "comparable"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {note} |")
    lines.append("")
    return lines


def format_comparison_capsule(state: dict[str, Any] | None) -> list[str]:
    """Rule-based compare conclusions from financial_metrics only (no LLM)."""
    state = state or {}
    companies = [str(c) for c in (state.get("companies") or [])]
    if len(companies) < 2:
        return []
    metrics_by_company = state.get("financial_metrics") or {}
    bullets: list[str] = []
    for key, label, is_pct in COMPARE_METRIC_SPECS:
        scored: list[tuple[str, float]] = []
        missing: list[str] = []
        for company in companies:
            value = (metrics_by_company.get(company) or {}).get(key)
            if isinstance(value, (int, float)):
                scored.append((company, float(value)))
            else:
                missing.append(company)
        if len(scored) >= 2:
            ranked = sorted(scored, key=lambda item: item[1], reverse=True)
            leader, lead_v = ranked[0]
            trailer, trail_v = ranked[-1]
            if lead_v == trail_v:
                bullets.append(
                    f"- {label}: tied at {_fmt_metric_value(lead_v, is_pct=is_pct)} "
                    f"({', '.join(c for c, _ in ranked)})."
                )
            else:
                delta = lead_v - trail_v
                delta_txt = _fmt_metric_value(delta, is_pct=is_pct) if is_pct else f"{delta:.2f}"
                bullets.append(
                    f"- {label}: **{leader}** {_fmt_metric_value(lead_v, is_pct=is_pct)} > "
                    f"{trailer} {_fmt_metric_value(trail_v, is_pct=is_pct)} "
                    f"(delta {delta_txt})."
                )
            if missing:
                bullets.append(
                    f"  - Not comparable for {', '.join(missing)} (metric n/a)."
                )
        elif len(scored) == 1 and missing:
            only_co, only_v = scored[0]
            bullets.append(
                f"- {label}: only {only_co} has "
                f"{_fmt_metric_value(only_v, is_pct=is_pct)}; "
                f"missing for {', '.join(missing)} — peer comparison withheld."
            )
        elif missing:
            bullets.append(f"- {label}: n/a for all peers in this run.")
    if not bullets:
        return []
    return [
        "**Comparison capsule (rule-based from AST metrics):**",
        *bullets,
        "",
    ]


def is_low_signal_claim(claim: Any) -> bool:
    """True for brief-unfriendly template noise (unknown supply-chain / empty thesis)."""
    claim_type = getattr(claim, "claim_type", None) or (claim.get("claim_type") if isinstance(claim, dict) else None)
    metric = getattr(claim, "metric_name", None) or (claim.get("metric_name") if isinstance(claim, dict) else None)
    statement = str(
        getattr(claim, "statement", None)
        or (claim.get("statement") if isinstance(claim, dict) else "")
        or ""
    ).lower()
    if claim_type == "investment_conclusion":
        return True
    if claim_type == "risk_conclusion":
        if metric == "supply_chain_risk" or "supply-chain risk signal is 'unknown'" in statement:
            return True
        if "unknown" in statement and "supply" in statement:
            return True
    return False


def filter_claims_for_brief(claims: list[Any]) -> list[Any]:
    """Brief ledger/summary: numeric (+ growth) only; drop thesis/unknown-risk filler."""
    kept: list[Any] = []
    for claim in claims:
        claim_type = getattr(claim, "claim_type", None) or (
            claim.get("claim_type") if isinstance(claim, dict) else None
        )
        if claim_type not in {"numeric", "growth"}:
            continue
        if is_low_signal_claim(claim):
            continue
        kept.append(claim)
    return kept


def build_clerk_executive_summary(
    state: dict[str, Any] | None,
    verified_claims: list[Any],
    *,
    brief: bool = False,
) -> str:
    """Clerk-oriented summary: compare capsule first; single-issuer falls back to numeric claims."""
    state = state or {}
    companies = [str(c) for c in (state.get("companies") or [])]
    parts: list[str] = []
    capsule = format_comparison_capsule(state)
    # Multi-company: capsule alone is the clerk conclusion (no claim dump).
    if capsule:
        parts.extend(line for line in capsule if line.strip())
        parts.append(
            "Assertions above are limited to verified claim objects / AST metrics with bound evidence."
        )
        return "\n".join(parts)

    claims = filter_claims_for_brief(verified_claims) if brief else list(verified_claims)
    if brief:
        claims = [
            c
            for c in claims
            if (getattr(c, "claim_type", None) or (c.get("claim_type") if isinstance(c, dict) else None))
            == "numeric"
        ]

    if not companies:
        return (
            "This run produced a research report, but no target company was available "
            "for a grounded executive summary."
        )
    if not claims:
        return (
            "No structurally verified financial claims were available. "
            "This executive summary withholds numeric and investment assertions rather than inventing citations."
        )

    per_company: list[str] = []
    for company in companies:
        company_claims = [
            c
            for c in claims
            if (getattr(c, "entity", None) or (c.get("entity") if isinstance(c, dict) else None)) == company
        ]
        # Prefer margin/intensity metrics for compact summary.
        preferred_order = (
            "operating_margin",
            "ebitda_margin",
            "r_and_d_intensity",
            "pe_ratio",
            "revenue",
        )
        picked: list[Any] = []
        remaining = list(company_claims)
        for metric in preferred_order:
            for claim in list(remaining):
                metric_name = getattr(claim, "metric_name", None) or (
                    claim.get("metric_name") if isinstance(claim, dict) else None
                )
                if metric_name == metric:
                    picked.append(claim)
                    remaining.remove(claim)
                    break
            if len(picked) >= (2 if brief else 3):
                break
        if len(picked) < (2 if brief else 3):
            for claim in remaining:
                if claim in picked:
                    continue
                picked.append(claim)
                if len(picked) >= (2 if brief else 3):
                    break
        if not picked:
            continue
        rendered = []
        for claim in picked:
            if hasattr(claim, "render_with_citation"):
                rendered.append(claim.render_with_citation())
            else:
                rendered.append(str(claim.get("statement") or ""))
        per_company.append(f"{company}: " + " ".join(rendered))

    if per_company:
        parts.append(" ".join(per_company))
    parts.append(
        "Assertions above are limited to verified claim objects / AST metrics with bound evidence."
    )
    return "\n".join(p for p in parts if p)


# Used by AgentRuntime.retrieval for SEC prefer_fiscal_year.
def _requested_fiscal_year_from_state(state: dict[str, Any] | None) -> int | None:
    return requested_fiscal_year_from_state(state)


def build_metrics_csv_rows(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten financial_metrics for CSV export (no recomputation)."""
    result = result or {}
    coverage = result.get("coverage_matrix") or {}
    retrieved = result.get("retrieved_docs") or {}
    workflow_status = str(result.get("workflow_status") or "")
    rows: list[dict[str, Any]] = []
    metrics_by_company = result.get("financial_metrics") or {}
    for company in result.get("companies") or list(metrics_by_company.keys()):
        company_key = str(company)
        metrics = metrics_by_company.get(company) or metrics_by_company.get(company_key) or {}
        if not isinstance(metrics, dict):
            continue
        cov = coverage.get(company) or coverage.get(company_key) or {}
        comparable = bool(cov.get("comparable")) if cov else bool(metrics)
        structured = str(
            cov.get("structured_source")
            or (retrieved.get(company) or retrieved.get(company_key) or {}).get("structured_source")
            or ""
        )
        for metric_name, value in metrics.items():
            rows.append(
                {
                    "company": company_key,
                    "metric": str(metric_name),
                    "value": value,
                    "comparable": "yes" if comparable else "no",
                    "structured_source": structured or "unknown",
                    "workflow_status": workflow_status,
                }
            )
    return rows


def write_metrics_csv(path: Path, result: dict[str, Any] | None) -> Path:
    import csv

    rows = build_metrics_csv_rows(result)
    fieldnames = [
        "company",
        "metric",
        "value",
        "comparable",
        "structured_source",
        "workflow_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def format_rag_citation_section(
    rag_evidence: dict[str, Any] | None,
    *,
    max_excerpt: int = 180,
    heading: str = "### Retrieved Document Citations (page-level)",
) -> list[str]:
    """Build a deterministic markdown block of RAG page citations for final_report.

    Citations keep ``filename#pN`` anchors from retrieval so the report is auditable
    against uploaded PDFs without relying on the LLM to copy them.
    """
    evidence = rag_evidence or {}
    rows: list[tuple[str, str, str, str]] = []
    for company_key in sorted(evidence.keys(), key=lambda c: str(c)):
        company = str(company_key)
        hits = evidence.get(company_key) or []
        if not isinstance(hits, list):
            continue
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            citation = str(hit.get("citation") or "").strip()
            if not citation:
                filename = (
                    str(hit.get("filename") or hit.get("source") or "document").strip()
                    or "document"
                )
                page = hit.get("page")
                citation = f"{filename}#p{page}" if page is not None else filename
            method = (
                str(hit.get("retrieval_method") or hit.get("method") or "hybrid").strip()
                or "hybrid"
            )
            text = str(hit.get("text") or hit.get("excerpt") or "").replace("\n", " ").strip()
            if len(text) > max_excerpt:
                text = text[: max_excerpt - 1].rstrip() + "…"
            rows.append((company, citation, method, text or "—"))
    if not rows:
        return []
    lines = [
        heading,
        "",
        "These anchors come from hybrid RAG hits and are written deterministically "
        "(not paraphrased by the LLM). Use `filename#pN` to locate the source page.",
        "",
        "| Company | Citation | Method | Excerpt |",
        "|---------|----------|--------|---------|",
    ]
    for company, citation, method, text in rows:
        safe_text = text.replace("|", "\\|")
        lines.append(f"| {company} | `{citation}` | {method} | {safe_text} |")
    lines.append("")
    return lines

def report_contains_page_citations(report: str | None) -> bool:
    """True when the report text includes at least one ``#pN`` page anchor."""
    return bool(_PAGE_CITATION_RE.search(report or ""))


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
    search_only = bool(rag_index_stats.get("search_only"))
    if not rag_enabled:
        rag_status = "disabled"
    elif rag_used:
        rag_status = "milvus_hybrid"
    elif chunks_indexed > 0 or search_only:
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
        metrics_csv_path = output_dir / f"{base_name}_metrics.csv"

        report_path.write_text(result.get("final_report", "") or "", encoding="utf-8")
        audit_path.write_text(
            json.dumps(result.get("audit_log", []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        try:
            write_metrics_csv(metrics_csv_path, result)
            metrics_csv_key = str(metrics_csv_path)
        except Exception:
            metrics_csv_key = ""
        artifacts.update(
            {
                "report_path": str(report_path),
                "audit_path": str(audit_path),
                "state_path": str(state_path),
            }
        )
        if metrics_csv_key:
            artifacts["metrics_csv_path"] = metrics_csv_key

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
