"""AST quant coverage helpers without PDF / tool-adapter imports.

Claim binding and FinRun export can import this module without pulling
``documents`` (PyMuPDF) or the retrieval tool surface.
"""

from __future__ import annotations

from typing import Any

from .metrics_schema import get_fundamental

AST_RATIO_KEYS = ("ebitda_margin", "r_and_d_intensity", "operating_margin")
_AST_NUMERATOR_KEYS = ("ebitda", "operating_income", "r_and_d")


def has_computable_fundamentals(payload: dict[str, Any] | None) -> bool:
    """True when AST quant can compute at least one core ratio from structured inputs."""
    market = (payload or {}).get("market_data") or {}
    revenue = get_fundamental(market, "revenue")
    if revenue in (None, 0):
        return False
    return any(get_fundamental(market, key) is not None for key in _AST_NUMERATOR_KEYS)


def classify_quant_status(metrics: dict[str, float] | None) -> str:
    """Classify per-company quant output for peer-comparison coverage."""
    values = metrics or {}
    if any(key in values for key in AST_RATIO_KEYS):
        return "ast_ok"
    if values:
        return "market_only"
    return "uncomputable"


def build_coverage_matrix(
    companies: list[str],
    retrieved_docs: dict[str, dict[str, Any]],
    financial_metrics: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for company in companies:
        payload = retrieved_docs.get(company) or {}
        metrics = (financial_metrics or {}).get(company) or {}
        has_structured = has_computable_fundamentals(payload)
        if metrics:
            quant_status = classify_quant_status(metrics)
            comparable = quant_status == "ast_ok"
        elif has_structured:
            quant_status = "pending"
            comparable = True
        elif payload.get("market_data") or (payload.get("live_market") or {}).get("current_price"):
            quant_status = "pending_market"
            comparable = False
        else:
            quant_status = "uncomputable"
            comparable = False
        matrix[company] = {
            "structured_source": str(payload.get("structured_source") or "none"),
            "has_computable_fundamentals": has_structured,
            "quant_status": quant_status,
            "ast_ratios": quant_status == "ast_ok",
            "comparable": comparable,
        }
    return matrix


def is_partial_compare_gap(companies: list[str], coverage_matrix: dict[str, dict[str, Any]]) -> bool:
    """True when a multi-company run has both comparable and non-comparable peers."""
    if len(companies) <= 1:
        return False
    comparable = [company for company in companies if (coverage_matrix.get(company) or {}).get("comparable")]
    return bool(comparable) and len(comparable) < len(companies)


def non_comparable_companies(companies: list[str], coverage_matrix: dict[str, dict[str, Any]]) -> list[str]:
    return [company for company in companies if not (coverage_matrix.get(company) or {}).get("comparable")]
