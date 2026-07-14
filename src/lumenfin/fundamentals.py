"""Live fundamentals fetch (Yahoo via yfinance) for AST-computable inputs.

Numbers are converted to billions USD to match the existing diligence metric scale.
Provenance label: structured_source=yahoo_fundamentals.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_REVENUE_KEYS = ("Total Revenue", "Operating Revenue", "Revenue")
_EBITDA_KEYS = ("EBITDA", "Normalized EBITDA")
_OP_INCOME_KEYS = ("Operating Income", "EBIT")
_RD_KEYS = ("Research And Development", "Research & Development")


def _row_value(frame: Any, keys: tuple[str, ...], col_idx: int = 0) -> float | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    for key in keys:
        if key not in frame.index:
            continue
        try:
            raw = frame.loc[key].iloc[col_idx]
        except Exception:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value != value:  # NaN
            continue
        return value
    return None


def _to_billions(value: float | None) -> float | None:
    if value is None:
        return None
    # Yahoo annual statements are in absolute currency units.
    scaled = value / 1_000_000_000.0
    return round(scaled, 4)


def fetch_yahoo_fundamentals(ticker: str) -> dict[str, Any] | None:
    """Return market_data dict + metadata, or None if insufficient rows."""
    symbol = (ticker or "").strip().upper()
    if not symbol or symbol in {"?", "N/A"}:
        return None
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; cannot fetch live fundamentals")
        return None

    try:
        ticker_obj = yf.Ticker(symbol)
        income = ticker_obj.income_stmt
        if income is None or getattr(income, "empty", True):
            income = ticker_obj.financials
        if income is None or getattr(income, "empty", True):
            return None

        col = income.columns[0]
        fiscal_year = None
        try:
            fiscal_year = int(getattr(col, "year", None) or str(col)[:4])
        except Exception:
            fiscal_year = None

        revenue = _to_billions(_row_value(income, _REVENUE_KEYS))
        ebitda = _to_billions(_row_value(income, _EBITDA_KEYS))
        operating_income = _to_billions(_row_value(income, _OP_INCOME_KEYS))
        r_and_d = _to_billions(_row_value(income, _RD_KEYS))

        if revenue in (None, 0):
            return None
        if ebitda is None and operating_income is None and r_and_d is None:
            return None

        market_data: dict[str, float] = {"revenue_2025": float(revenue)}
        if ebitda is not None:
            market_data["ebitda_2025"] = float(ebitda)
        if operating_income is not None:
            market_data["operating_income_2025"] = float(operating_income)
        if r_and_d is not None:
            market_data["r_and_d_2025"] = float(r_and_d)

        return {
            "market_data": market_data,
            "structured_source": "yahoo_fundamentals",
            "fundamentals_meta": {
                "provider": "yahoo",
                "symbol": symbol,
                "fiscal_year": fiscal_year,
                "period_end": str(col),
                "unit": "billion_usd_equiv",
                "scale_note": "Absolute Yahoo statement values divided by 1e9 for LumenFin metric scale.",
            },
            "supply_chain": {
                "risk_level": "unknown",
                "signals": [
                    f"Fundamentals loaded from Yahoo Finance annual income statement for {symbol}."
                ],
            },
            "earnings_call_quotes": [
                f"Yahoo fundamentals snapshot for {symbol} (period ending {col})."
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Yahoo fundamentals fetch failed for %s: %s", symbol, exc)
        return None
