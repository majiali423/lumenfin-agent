"""Query-asked metrics and per-company periods. No gold, no eval case IDs."""

from __future__ import annotations

import re
from typing import Any

RATIO_METRICS = ("operating_margin", "ebitda_margin", "r_and_d_intensity")
ABSOLUTE_METRICS = ("operating_income", "revenue", "r_and_d", "ebitda")
_FY_NEAR = re.compile(r"FY\s*(20\d{2})", re.IGNORECASE)
_RELATIVE_UNANCHORED = re.compile(
    r"(?:去年|今年|前年|\b(?:this|last|prior)\s+year\b)",
    re.IGNORECASE,
)

_METRIC_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("operating_margin", ("operating margin", "营业利润率", "经营利润率")),
    ("ebitda_margin", ("ebitda margin", "ebitda 利润率")),
    ("r_and_d_intensity", ("r&d intensity", "rd intensity", "r and d intensity", "研发强度")),
    ("operating_income", ("operating income", "operating profit", "income from operations", "营业利润", "经营利润")),
    ("r_and_d", ("r&d expense", "research and development", "r&d", "r and d", "研发支出", "研发费用", "研发投入")),
    ("ebitda", ("ebitda",)),
    ("revenue", ("total net sales", "net sales", "revenue", "revenues", "营收", "收入")),
)


def format_billion_amount(value: float) -> str:
    """Keep thousandths of a billion (source millions) without trailing zeros."""
    text = f"{float(value):.3f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def detect_requested_metrics(query: str) -> list[str]:
    blob = str(query or "").lower()
    found: list[str] = []
    for metric, phrases in _METRIC_PATTERNS:
        if metric in found:
            continue
        if any(phrase in blob for phrase in phrases):
            found.append(metric)
    # More specific phrases already listed first, but drop superseded pairs.
    if "operating_margin" in found:
        found = [item for item in found if item != "operating_income"]
    if "r_and_d_intensity" in found:
        found = [item for item in found if item != "r_and_d"]
    if "ebitda_margin" in found:
        found = [item for item in found if item != "ebitda"]
    return found


def detect_company_periods(query: str, companies: list[str], *, aliases: dict[str, str] | None = None) -> dict[str, str]:
    text = str(query or "")
    if not text or not companies:
        return {}
    alias_map = aliases or {}
    names_for: dict[str, list[str]] = {company: [company] for company in companies}
    for alias, canonical in alias_map.items():
        if canonical in names_for and alias.lower() != canonical.lower():
            names_for[canonical].append(alias)
    out: dict[str, str] = {}
    for company in companies:
        best: tuple[int, int, str] | None = None
        for name in names_for[company]:
            for match in re.finditer(re.escape(name), text, flags=re.IGNORECASE):
                start = max(0, match.start() - 36)
                end = min(len(text), match.end() + 36)
                window = text[start:end]
                name_rel_start = match.start() - start
                name_rel_end = match.end() - start
                for fy in _FY_NEAR.finditer(window):
                    dist = min(
                        abs(fy.start() - name_rel_start),
                        abs(fy.end() - name_rel_end),
                        abs(fy.start() - name_rel_end),
                        abs(fy.end() - name_rel_start),
                    )
                    after_name = 0 if fy.start() >= name_rel_end else 1
                    label = f"FY{fy.group(1)}"
                    candidate = (dist, after_name, label)
                    if best is None or candidate[:2] < best[:2]:
                        best = candidate
        if best:
            out[company] = best[2]
    return out


def query_has_explicit_fiscal_year(*texts: Any) -> bool:
    """True when the user named a calendar/FY year, not merely 'last year'."""
    return bool(distinct_fiscal_years(*texts))


def query_has_relative_unanchored_period(query: str) -> bool:
    """Relative time with no FY/calendar-year anchor must not inherit filename years."""
    text = str(query or "")
    if not _RELATIVE_UNANCHORED.search(text):
        return False
    return not query_has_explicit_fiscal_year(text)


def distinct_fiscal_years(*texts: Any) -> list[int]:
    years: list[int] = []
    seen: set[int] = set()
    for raw in texts:
        for hit in re.findall(r"FY\s*(20\d{2})", str(raw or ""), flags=re.IGNORECASE):
            year = int(hit)
            if year not in seen:
                seen.add(year)
                years.append(year)
    return years


def skip_enrichment_for_plan(plan: dict[str, Any] | None) -> bool:
    plan = plan or {}
    intent = str(plan.get("intent") or "")
    if intent == "risk_compliance_review":
        return False
    metrics = [str(item) for item in (plan.get("requested_metrics") or []) if str(item).strip()]
    if not metrics:
        return False
    query = str(plan.get("normalized_query") or plan.get("query") or "")
    lowered = query.lower()
    if any(token in lowered for token in ("sentiment", "tone", "管理层语气", "supply chain", "供应链", "compliance risk")):
        return False
    return True
