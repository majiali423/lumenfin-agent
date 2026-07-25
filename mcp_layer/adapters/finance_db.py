"""Reads lumenfin SAMPLE_FINANCIAL_DATA (demo only — not live fundamentals)."""
from __future__ import annotations

from typing import Any

from lumenfin.data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from lumenfin.metrics_schema import canonical_fundamental_name, get_fundamental

from .scope import SCOPE_FINANCE_DB, stamp_scope

_METRIC_UNITS = {
    "revenue": "USD billions",
    "ebitda": "USD billions",
    "r_and_d": "USD billions",
    "operating_income": "USD billions",
    # Legacy aliases kept for MCP callers that still request *_2025 keys.
    "revenue_2025": "USD billions",
    "ebitda_2025": "USD billions",
    "r_and_d_2025": "USD billions",
    "operating_income_2025": "USD billions",
}


def _normalize_company(name: str) -> str | None:
    candidate = name.strip()
    for company in SAMPLE_FINANCIAL_DATA:
        if company.lower() == candidate.lower():
            return company
    return None


def query_company_metrics(company: str, metrics: list[str] | None = None) -> dict[str, Any]:
    company_norm = _normalize_company(company)
    if company_norm is None:
        return stamp_scope(
            {
                "company": company.strip(),
                "found": False,
                "metrics": {},
                "source": "lumenfin.sample_financial_data",
                "hint": f"Supported companies: {', '.join(sorted(SAMPLE_FINANCIAL_DATA))}",
            },
            SCOPE_FINANCE_DB,
        )

    market_data = SAMPLE_FINANCIAL_DATA[company_norm].get("market_data", {})
    selected = metrics or list(market_data.keys())
    payload: dict[str, dict[str, Any]] = {}
    for key in selected:
        value = get_fundamental(market_data, key)
        if value is None and key in market_data:
            try:
                value = float(market_data[key])
            except (TypeError, ValueError):
                value = None
        if value is None:
            continue
        payload[key] = {
            "value": value,
            "unit": _METRIC_UNITS.get(key) or _METRIC_UNITS.get(canonical_fundamental_name(key) or "", "USD billions"),
        }

    return stamp_scope(
        {
            "company": company_norm,
            "found": bool(payload),
            "metrics": payload,
            "source": "lumenfin.sample_financial_data",
        },
        SCOPE_FINANCE_DB,
    )
