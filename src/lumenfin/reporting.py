from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .evaluation import evaluate_run_state

_PAGE_CITATION_RE = re.compile(r"(?:#p\d+|\bp\.\d+\b)", re.IGNORECASE)

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


def infer_fiscal_year_from_documents(
    document_contexts: list[dict[str, Any]] | None,
    *,
    company: str | None = None,
) -> tuple[int | None, str | None]:
    """Infer FY from upload filenames / filing text (not from query alone).

    Returns ``(year, source_tag)`` where source_tag is ``upload_filename`` or
    ``upload_text`` when a year is found.
    """
    docs = document_contexts or []
    # Filename is the strongest clerk-facing signal for derived fixtures
    # (e.g. msft_fy2024_10k_long_excerpt.pdf).
    for doc in docs:
        for key in ("filename", "source", "path", "citation"):
            raw = str(doc.get(key) or "")
            if not raw:
                continue
            name = Path(raw).name if ("/" in raw or "\\" in raw) else raw
            year = parse_requested_fiscal_year(name)
            if year is not None:
                return year, "upload_filename"
    # Filing body / excerpt (FY2024, fiscal year ended ..., etc.)
    for doc in docs:
        detected = doc.get("detected_companies") or []
        if company and detected and company not in detected:
            continue
        blob = " ".join(
            str(doc.get(k) or "") for k in ("filename", "excerpt", "text")
        )[:8000]
        year = parse_requested_fiscal_year(blob)
        if year is not None:
            return year, "upload_text"
    return None, None


# Typical fiscal year-end month/day for common issuers when filing extract has no period_end.
# These are convention hints for clerk disclosure — not a substitute for filing metadata.
_ISSUER_FY_END_HINTS: dict[str, tuple[int, int]] = {
    "Apple": (9, 30),
    "Microsoft": (6, 30),
    "NVIDIA": (1, 26),  # late January fiscal year end (approx)
    "AMD": (12, 28),
    "Tesla": (12, 31),
    "Amazon": (12, 31),
    "Alphabet": (12, 31),
    "Meta": (12, 31),
}


def suggest_period_end_hint(company: str | None, fiscal_year: int | None) -> tuple[str | None, str | None]:
    """Return (YYYY-MM-DD, source_tag) from issuer convention when FY is known."""
    if not company or fiscal_year is None:
        return None, None
    tip = _ISSUER_FY_END_HINTS.get(str(company))
    if not tip:
        return None, None
    month, day = tip
    try:
        return f"{int(fiscal_year):04d}-{month:02d}-{day:02d}", "issuer_convention_hint"
    except (TypeError, ValueError):
        return None, None


def annotate_upload_period_meta(
    meta: dict[str, Any] | None,
    *,
    document_contexts: list[dict[str, Any]] | None = None,
    company: str | None = None,
    prefer_fiscal_year: int | None = None,
) -> dict[str, Any]:
    """Fill fiscal_year / period_alignment for document_extracted payloads.

    Priority: existing meta fiscal_year → filename/text inference → query FY
    (tagged ``assumed_from_query`` so clerks know it was not filing-labeled).
    When period_end is missing, may attach an issuer-convention hint date.
    """
    out = dict(meta or {})
    if prefer_fiscal_year is not None:
        out.setdefault("requested_fiscal_year", int(prefer_fiscal_year))

    def _finalize(used: int | None) -> dict[str, Any]:
        if used is not None and not str(out.get("period_end") or "").strip():
            hint, src = suggest_period_end_hint(company, used)
            if hint:
                out["period_end"] = hint
                out["period_end_source"] = src
        return out

    existing = out.get("fiscal_year")
    if existing not in (None, ""):
        try:
            used = int(existing)
        except (TypeError, ValueError):
            used = None
        if used is not None:
            out["fiscal_year"] = used
            if prefer_fiscal_year is not None:
                out.setdefault(
                    "period_alignment",
                    "exact" if used == int(prefer_fiscal_year) else "fallback_latest",
                )
            else:
                out.setdefault("period_alignment", "upload_labeled")
            return _finalize(used)

    inferred, source = infer_fiscal_year_from_documents(
        document_contexts, company=company
    )
    if inferred is not None:
        out["fiscal_year"] = int(inferred)
        out["fiscal_year_source"] = source
        if prefer_fiscal_year is not None and int(inferred) == int(prefer_fiscal_year):
            out["period_alignment"] = "exact"
        elif prefer_fiscal_year is not None:
            out["period_alignment"] = "fallback_latest"
        else:
            out["period_alignment"] = source or "upload_labeled"
        return _finalize(int(inferred))

    if prefer_fiscal_year is not None:
        # Upload has computable metrics but no FY tag — use query FY with an honest label.
        out["fiscal_year"] = int(prefer_fiscal_year)
        out["fiscal_year_source"] = "query"
        out["period_alignment"] = "assumed_from_query"
        return _finalize(int(prefer_fiscal_year))

    out.setdefault("period_alignment", "unspecified")
    return _finalize(None)


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


def period_end_for_company(state: dict[str, Any] | None, company: str) -> str | None:
    """Return a clerk-facing period-end date string (YYYY-MM-DD) when metadata has one."""
    payload = ((state or {}).get("retrieved_docs") or {}).get(company) or {}
    meta = payload.get("fundamentals_meta") or {}
    for key in ("period_end", "period"):
        raw = str(meta.get(key) or "").strip()
        if not raw:
            continue
        match = re.search(r"(20\d{2}-\d{2}-\d{2})", raw)
        if match:
            return match.group(1)
        # Yahoo sometimes stores column labels like 2024-06-30 00:00:00
        match = re.search(r"(20\d{2}-\d{2}-\d{2})", raw.replace("/", "-"))
        if match:
            return match.group(1)
    return None


def _parse_iso_date(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.match(r"(20\d{2})-(\d{2})-(\d{2})", text)
    if not match:
        return None
    try:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


# Peer FY labels can match while fiscal calendars differ (e.g. AAPL Sep vs MSFT Jun).
_PEER_PERIOD_END_MISMATCH_DAYS = 90


def peer_period_end_span_days(state: dict[str, Any] | None) -> int | None:
    """Max absolute day gap among known peer period_end dates, or None if <2 dates."""
    companies = [str(c) for c in ((state or {}).get("companies") or [])]
    dates = []
    for company in companies:
        parsed = _parse_iso_date(period_end_for_company(state, company))
        if parsed is not None:
            dates.append(parsed)
    if len(dates) < 2:
        return None
    return max(abs((a - b).days) for a in dates for b in dates)


def format_period_alignment_notice(state: dict[str, Any] | None) -> list[str]:
    """Disclose requested vs used fiscal periods (never silent FY swap)."""
    state = state or {}
    requested = requested_fiscal_year_from_state(state)
    companies = [str(c) for c in (state.get("companies") or [])]
    if not companies:
        return []

    rows: list[tuple[str, str, str, str]] = []
    any_mismatch = False
    any_assumed = False
    used_years: list[int] = []
    for company in companies:
        used = used_fiscal_year_for_company(state, company)
        if used is not None:
            used_years.append(int(used))
        payload = (state.get("retrieved_docs") or {}).get(company) or {}
        meta = payload.get("fundamentals_meta") or {}
        alignment = str(meta.get("period_alignment") or "")
        fy_source = str(meta.get("fiscal_year_source") or "")
        period_end = period_end_for_company(state, company) or "n/a"
        pe_source = str(meta.get("period_end_source") or "")
        if period_end != "n/a" and pe_source == "issuer_convention_hint":
            period_end = f"{period_end} (issuer convention hint)"
        used_label = f"FY{used}" if used is not None else "n/a"
        if requested is None:
            status = alignment or "no FY requested"
        elif used is None:
            status = "requested FY not found in structured fundamentals"
            any_mismatch = True
        elif alignment == "assumed_from_query":
            status = "upload extract; FY assumed from query (filing not year-tagged)"
            any_assumed = True
        elif int(used) == int(requested):
            if pe_source == "issuer_convention_hint" and fy_source in {
                "upload_filename",
                "upload_text",
                "query",
            }:
                status = f"FY label match ({fy_source or alignment}); period-end is convention hint"
            elif fy_source in {"upload_filename", "upload_text"} or alignment in {
                "upload_filename",
                "upload_text",
                "upload_labeled",
            }:
                status = f"exact match ({fy_source or alignment})"
            else:
                status = "exact match"
        else:
            status = f"FALLBACK — requested FY{requested}, using {used_label}"
            any_mismatch = True
        req_label = f"FY{requested}" if requested is not None else "—"
        rows.append((company, req_label, f"{used_label} ({status})", period_end))

    if requested is None and not any(
        ((state.get("retrieved_docs") or {}).get(c) or {}).get("fundamentals_meta")
        for c in companies
    ):
        return []

    lines = [
        "## Period Alignment",
        "",
        "| Company | Requested | Used (status) | Period end |",
        "|---------|-----------|---------------|------------|",
    ]
    for company, req, used, period_end in rows:
        lines.append(f"| {company} | {req} | {used} | {period_end} |")
    lines.append("")
    if any_mismatch:
        lines.append(
            "**Period notice:** Structured fundamentals are not aligned to the requested fiscal year "
            "for at least one issuer. Treat YoY / same-year peer statements as non-comparable until "
            "the requested FY is available or the query is restated."
        )
        lines.append("")
    elif any_assumed:
        lines.append(
            "**Period notice:** Upload extract supplied the numbers; fiscal year was taken from the "
            "query because the filing extract did not carry an explicit FY tag in structured metadata."
        )
        lines.append("")
    elif requested is not None:
        lines.append(f"**Period notice:** Structured fundamentals match requested FY{requested}.")
        lines.append("")

    # Multi-issuer: same FY label ≠ same fiscal calendar.
    if len(companies) >= 2:
        span = peer_period_end_span_days(state)
        label_aligned = (
            len(used_years) >= 2
            and len(set(used_years)) == 1
            and (requested is None or all(y == int(requested) for y in used_years))
        )
        if span is not None and span >= _PEER_PERIOD_END_MISMATCH_DAYS:
            lines.append(
                "**Calendar note:** Peer `period_end` dates differ by "
                f"{_PEER_PERIOD_END_MISMATCH_DAYS}+ days (span={span}d). Side-by-side ratios are "
                "**FY-label-aligned research comps**, not a claim that fiscal calendars match "
                "the same natural-year window."
            )
            lines.append("")
        elif label_aligned or (requested is not None and not any_mismatch):
            lines.append(
                "**Calendar note:** Matching FY labels (e.g. Apple FY2024 vs Microsoft FY2024) do not "
                "imply identical fiscal year-end dates. Treat the peer matrix as same-label research "
                "comparison unless period-end dates above are close."
            )
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
        "*FY-label research comps — not a claim that peer fiscal calendars share the same natural-year window. "
        "See Period Alignment for period-end dates.*",
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
        return "\n".join(parts)

    # Clerk summary: numeric highlights only (risk/thesis live in dedicated sections).
    claims = [
        c
        for c in (filter_claims_for_brief(verified_claims) if brief else list(verified_claims))
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
            "revenue",
            "pe_ratio",
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
            if len(picked) >= 3:
                break
        if not picked:
            continue
        bullets = []
        for claim in picked:
            if hasattr(claim, "render_with_citation"):
                bullets.append(f"- {claim.render_with_citation(humanize=True)}")
            else:
                bullets.append(f"- {claim.get('statement') or ''}")
        per_company.append(f"**{company}**\n" + "\n".join(bullets))

    if per_company:
        parts.append("\n\n".join(per_company))
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


def humanize_citation(citation: str) -> str:
    """Turn internal citation URIs into clerk-readable source labels."""
    cite = (citation or "").strip()
    if not cite:
        return ""
    m = re.match(r"^(?P<file>.+\.pdf)#p(?P<page>\d+)$", cite, flags=re.I)
    if m:
        return f"{m.group('file')} p.{m.group('page')}"
    patterns = (
        (r"^lumenfin:sec_companyfacts:([^:]+):(.+)$", "SEC companyfacts ({0}, {1})"),
        (r"^lumenfin:yahoo_fundamentals:([^:]+):(.+)$", "Yahoo fundamentals ({0}, {1})"),
        (r"^lumenfin:document_extracted:([^:]+):(.+)$", "Uploaded filing extract ({0}, {1})"),
        (r"^lumenfin:market_snapshot:([^:]+):", "Live market snapshot ({0})"),
        (r"^lumenfin:risk_model:([^:]+)", "Risk screening model ({0})"),
        (r"^lumenfin:supply_chain:([^:]+):(.+)$", "Supply-chain screen ({0}, {1})"),
        (r"^lumenfin:sample_db:([^:]+):(.+)$", "Demo sample fundamentals ({0}, {1})"),
    )
    for pattern, template in patterns:
        match = re.match(pattern, cite)
        if match:
            return template.format(*match.groups())
    return cite


def format_rag_citation_section(
    rag_evidence: dict[str, Any] | None,
    *,
    max_excerpt: int = 180,
    heading: str = "### Retrieved Document Citations (page-level)",
    max_rows_per_company: int = 3,
) -> list[str]:
    """Build a deterministic markdown block of RAG page citations for final_report.

    Citations keep ``filename#pN`` anchors from retrieval so the report is auditable
    against uploaded PDFs without relying on the LLM to copy them.
    """
    evidence = rag_evidence or {}
    rows: list[tuple[str, str, str, str]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for company_key in sorted(evidence.keys(), key=lambda c: str(c)):
        company = str(company_key)
        hits = evidence.get(company_key) or []
        if not isinstance(hits, list):
            continue
        scored_hits = [h for h in hits if isinstance(h, dict)]
        scored_hits.sort(
            key=lambda h: float(h.get("score") or h.get("rerank_score") or 0.0),
            reverse=True,
        )
        kept_for_company = 0
        for hit in scored_hits:
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
            # Dedupe identical excerpt pairs that repeat across near-duplicate pages.
            file_key = re.sub(r"#p\d+$", "", citation, flags=re.I).lower()
            dedupe_key = (company, file_key, text[:120].lower())
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            if len(text) > max_excerpt:
                text = text[: max_excerpt - 1].rstrip() + "…"
            rows.append((company, citation, method, text or "—"))
            kept_for_company += 1
            if kept_for_company >= max_rows_per_company:
                break
    if not rows:
        return []
    lines = [
        heading,
        "",
        "Page anchors from retrieval (deduplicated; top hits per company). Open the cited PDF page to verify.",
        "",
        "| Company | Citation | Method | Excerpt |",
        "|---------|----------|--------|---------|",
    ]
    for company, citation, method, text in rows:
        safe_text = text.replace("|", "\\|")
        label = humanize_citation(citation)
        lines.append(f"| {company} | {label} | {method} | {safe_text} |")
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
