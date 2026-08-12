from __future__ import annotations

from typing import Any

from ..reporting import format_comparison_capsule


_PEER_COMPARISON_LEAK_MARKERS = (
    "we need to",
    "let's draft",
    "the instruction says",
    "the user asked",
    "we must ",
    "i need to",
)


def _single_company_peer_summary(company: str) -> str:
    return (
        f"Peer comparison is unavailable because only {company} has "
        "comparable structured ratio metrics in this run."
    )


def _peer_comparison_is_safe(text: str) -> bool:
    cleaned = (text or "").strip()
    lowered = cleaned.casefold()
    if not cleaned or len(cleaned) > 1200:
        return False
    if any(marker in lowered for marker in _PEER_COMPARISON_LEAK_MARKERS):
        return False
    return cleaned.endswith((".", "!", "?"))


def _peer_comparison_fallback(companies: list[str]) -> str:
    names = ", ".join(companies)
    return (
        f"Structured peer metrics are available for {names}. "
        "See the Executive Summary comparison capsule and Peer Metric Matrix for verified ratios; "
        "no free-form peer narrative is invented beyond those figures."
    )


def _deterministic_peer_comparison(comparable_metrics: dict[str, dict[str, Any]]) -> str:
    """Rule-based peer blurb from AST metrics only (no LLM; no invented margins/returns)."""
    companies = list(comparable_metrics.keys())
    if len(companies) < 2:
        return _single_company_peer_summary(companies[0]) if companies else (
            "No quantitative metrics were available for peer comparison."
        )
    capsule = format_comparison_capsule(
        {"companies": companies, "financial_metrics": comparable_metrics}
    )
    bullets = [line for line in capsule if line.startswith("- ")]
    if not bullets:
        return _peer_comparison_fallback(companies)
    return (
        "Peer comparison is limited to verified structured ratios "
        "(see Executive Summary capsule and Peer Metric Matrix):\n"
        + "\n".join(bullets)
    )
