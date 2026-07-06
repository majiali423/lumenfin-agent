"""Reads lumenfin SAMPLE_FINANCIAL_DATA (same numbers as retrieval/quant nodes)."""
from __future__ import annotations

from typing import Any

from lumenfin.data.sample_financial_data import SAMPLE_FINANCIAL_DATA

_METRIC_UNITS = {
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
        return {
            "company": company.strip(),
            "found": False,
            "metrics": {},
            "source": "lumenfin.sample_financial_data",
            "hint": f"Supported companies: {', '.join(sorted(SAMPLE_FINANCIAL_DATA))}",
        }

    market_data = SAMPLE_FINANCIAL_DATA[company_norm].get("market_data", {})
    selected = metrics or list(market_data.keys())
    payload: dict[str, dict[str, Any]] = {}
    for key in selected:
        if key in market_data:
            payload[key] = {
                "value": market_data[key],
                "unit": _METRIC_UNITS.get(key, "USD billions"),
            }

    return {
        "company": company_norm,
        "found": bool(payload),
        "metrics": payload,
        "source": "lumenfin.sample_financial_data",
    }
