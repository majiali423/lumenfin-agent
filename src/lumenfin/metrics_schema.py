"""Period-agnostic fundamental metric keys with legacy ``*_2025`` compatibility.

Historical demo code welded the fiscal year into field names (``revenue_2025``).
Canonical keys are period-free (``revenue``); period belongs in ``fundamentals_meta``
or appendix metadata. Readers accept both shapes via ``get_fundamental``.
"""

from __future__ import annotations

import re
from typing import Any

CANONICAL_FUNDAMENTALS = (
    "revenue",
    "ebitda",
    "r_and_d",
    "operating_income",
    "subsidiary_revenue",
)

# Legacy demo keys → canonical names.
LEGACY_FUNDAMENTAL_KEYS: dict[str, str] = {
    "revenue_2025": "revenue",
    "ebitda_2025": "ebitda",
    "r_and_d_2025": "r_and_d",
    "operating_income_2025": "operating_income",
    "subsidiary_revenue_2025": "subsidiary_revenue",
}

_PERIOD_SUFFIX_RE = re.compile(r"^(?P<base>revenue|ebitda|r_and_d|operating_income|subsidiary_revenue)_(?P<year>20\d{2})$")


def canonical_fundamental_name(key: str) -> str | None:
    """Map a market_data / appendix key onto a canonical fundamental name, if any."""
    if key in CANONICAL_FUNDAMENTALS:
        return key
    if key in LEGACY_FUNDAMENTAL_KEYS:
        return LEGACY_FUNDAMENTAL_KEYS[key]
    match = _PERIOD_SUFFIX_RE.match(key)
    if match:
        return match.group("base")
    return None


def get_fundamental(market_data: dict[str, Any] | None, name: str) -> float | None:
    """Read a fundamental, accepting canonical or legacy/period-suffixed keys."""
    data = market_data or {}
    canonical = canonical_fundamental_name(name) or name
    candidates = [canonical, f"{canonical}_2025"]
    # Also accept any revenue_20xx style key present in the payload.
    for key, value in data.items():
        mapped = canonical_fundamental_name(str(key))
        if mapped == canonical:
            candidates.append(str(key))
    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        if key not in data:
            continue
        value = data.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def set_fundamental(market_data: dict[str, Any], name: str, value: float | int) -> None:
    """Write a canonical fundamental key (does not emit legacy ``*_2025`` aliases)."""
    canonical = canonical_fundamental_name(name) or name
    market_data[canonical] = float(value)


def normalize_market_data(market_data: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy with fundamentals collapsed onto canonical keys.

    Non-fundamental keys are preserved. When both legacy and canonical exist,
    canonical wins.
    """
    raw = dict(market_data or {})
    normalized: dict[str, Any] = {}
    pending: dict[str, float] = {}
    for key, value in raw.items():
        mapped = canonical_fundamental_name(str(key))
        if mapped is None:
            normalized[key] = value
            continue
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        # Prefer an already-canonical key over a later legacy duplicate.
        if mapped in pending and key != mapped:
            continue
        pending[mapped] = number
    normalized.update(pending)
    return normalized


def fundamentals_present(market_data: dict[str, Any] | None) -> dict[str, float]:
    """Return only the canonical fundamentals that are present and numeric."""
    out: dict[str, float] = {}
    for name in CANONICAL_FUNDAMENTALS:
        value = get_fundamental(market_data, name)
        if value is not None:
            out[name] = value
    return out


def period_label_from_meta(meta: dict[str, Any] | None, *, default: str = "latest") -> str:
    """Human-readable period for citations / FinRun export."""
    meta = meta or {}
    for key in ("fiscal_year", "period", "fy", "period_end"):
        value = meta.get(key)
        if value in (None, ""):
            continue
        text = str(value)
        if key == "fiscal_year" or (key == "fy" and text.isdigit()):
            return f"FY{text}"
        if re.fullmatch(r"20\d{2}", text):
            return f"FY{text}"
        return text
    return default
