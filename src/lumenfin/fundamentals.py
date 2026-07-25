"""Live fundamentals fetch (Yahoo via yfinance) for AST-computable inputs.

Numbers are converted to billions USD to match the existing diligence metric scale.
Provenance label: structured_source=yahoo_fundamentals.

Yahoo income statements for non-US issuers are often denominated in local currency
(e.g. TSMC financialCurrency=TWD). Blindly dividing by 1e9 and labeling the result
as USD will inflate TWD revenue (NT$3,809B) into a fake ~US$3,809B figure.
"""
from __future__ import annotations

import logging
from typing import Any

from .provider_retry import (
    append_provider_error,
    call_with_transient_retry,
    classify_exception,
)

logger = logging.getLogger(__name__)

_REVENUE_KEYS = ("Total Revenue", "Operating Revenue", "Revenue")
_EBITDA_KEYS = ("EBITDA", "Normalized EBITDA")
_OP_INCOME_KEYS = ("Operating Income", "EBIT")
_RD_KEYS = ("Research And Development", "Research & Development")
_YAHOO_RETRIES = 3
_YAHOO_BACKOFF_SEC = 0.5

# Hard ceiling for annual revenue on the LumenFin billion-USD scale.
# Even mega-cap retailers/oil majors are typically well below this; values above
# almost always indicate a local-currency / unit-scaling bug.
MAX_PLAUSIBLE_REVENUE_BILLION_USD = 800.0

# Approximate USD per 1 unit of local currency (fallback when FX quote unavailable).
# Used only for statement scaling; not for trading decisions.
_APPROX_USD_PER_UNIT: dict[str, float] = {
    "USD": 1.0,
    "TWD": 0.031,  # ~NT$32 / USD
    "JPY": 0.0067,  # ~¥149 / USD
    "KRW": 0.00072,  # ~₩1,390 / USD
    "CNY": 0.14,
    "HKD": 0.128,
    "EUR": 1.08,
    "GBP": 1.27,
}


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
    scaled = value / 1_000_000_000.0
    return round(scaled, 4)


def is_plausible_revenue_billion_usd(revenue_billion: float | None) -> bool:
    """Reject absurd annual-revenue magnitudes on the billion-USD scale."""
    if revenue_billion is None:
        return False
    if revenue_billion <= 0:
        return False
    return revenue_billion <= MAX_PLAUSIBLE_REVENUE_BILLION_USD


def _load_yahoo_income(symbol: str) -> Any:
    import yfinance as yf

    ticker_obj = yf.Ticker(symbol)
    income = ticker_obj.income_stmt
    if income is None or getattr(income, "empty", True):
        income = ticker_obj.financials
    return income


def _load_yahoo_financial_currency(symbol: str) -> str | None:
    """Return Yahoo financialCurrency (statement currency), not listing currency."""
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
        currency = str(info.get("financialCurrency") or info.get("currency") or "").strip().upper()
        return currency or None
    except Exception:  # noqa: BLE001
        return None


def _usd_per_unit(currency: str | None) -> tuple[float, str]:
    """Return (fx_rate, source_note) converting 1 local unit into USD."""
    code = (currency or "USD").strip().upper() or "USD"
    if code == "USD":
        return 1.0, "identity"
    approx = _APPROX_USD_PER_UNIT.get(code)
    if approx is not None:
        return approx, f"approx_table:{code}"
    # Unknown currency: refuse conversion rather than invent a rate.
    return 0.0, f"unsupported_currency:{code}"


def _scale_statement_value_to_billion_usd(
    raw_value: float | None,
    *,
    currency: str | None,
) -> tuple[float | None, dict[str, Any]]:
    """Convert a Yahoo statement absolute amount into billion USD."""
    meta: dict[str, Any] = {
        "statement_currency": (currency or "USD").upper(),
        "fx_usd_per_unit": 1.0,
        "fx_source": "identity",
    }
    if raw_value is None:
        return None, meta
    fx, fx_source = _usd_per_unit(currency)
    meta["fx_usd_per_unit"] = fx
    meta["fx_source"] = fx_source
    if fx <= 0:
        return None, meta
    usd_absolute = float(raw_value) * fx
    return _to_billions(usd_absolute), meta


def fetch_yahoo_fundamentals(
    ticker: str,
    *,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return market_data dict + metadata, or None if insufficient rows."""
    symbol = (ticker or "").strip().upper()
    if not symbol or symbol in {"?", "N/A"}:
        return None
    try:
        import yfinance as yf  # noqa: F401
    except ImportError:
        logger.warning("yfinance not installed; cannot fetch live fundamentals")
        append_provider_error(
            errors,
            provider="yahoo",
            symbol=symbol,
            error_class="unavailable",
            message="yfinance is not installed.",
        )
        return None

    attempts_used = 0
    try:
        def _fetch_income() -> Any:
            nonlocal attempts_used
            attempts_used += 1
            return _load_yahoo_income(symbol)

        income = call_with_transient_retry(
            _fetch_income,
            max_retries=_YAHOO_RETRIES,
            backoff_seconds=_YAHOO_BACKOFF_SEC,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Yahoo fundamentals fetch failed for %s: %s", symbol, exc)
        append_provider_error(
            errors,
            provider="yahoo",
            symbol=symbol,
            error_class=classify_exception(exc),
            message=str(exc),
            attempts=max(attempts_used, 1),
        )
        return None

    try:
        if income is None or getattr(income, "empty", True):
            append_provider_error(
                errors,
                provider="yahoo",
                symbol=symbol,
                error_class="truly_missing",
                message="Yahoo income statement / financials is empty.",
                attempts=max(attempts_used, 1),
            )
            return None

        col = income.columns[0]
        fiscal_year = None
        try:
            fiscal_year = int(getattr(col, "year", None) or str(col)[:4])
        except Exception:
            fiscal_year = None

        statement_currency = _load_yahoo_financial_currency(symbol) or "USD"
        revenue_raw = _row_value(income, _REVENUE_KEYS)
        ebitda_raw = _row_value(income, _EBITDA_KEYS)
        operating_income_raw = _row_value(income, _OP_INCOME_KEYS)
        r_and_d_raw = _row_value(income, _RD_KEYS)

        revenue, scale_meta = _scale_statement_value_to_billion_usd(
            revenue_raw,
            currency=statement_currency,
        )
        ebitda, _ = _scale_statement_value_to_billion_usd(ebitda_raw, currency=statement_currency)
        operating_income, _ = _scale_statement_value_to_billion_usd(
            operating_income_raw,
            currency=statement_currency,
        )
        r_and_d, _ = _scale_statement_value_to_billion_usd(r_and_d_raw, currency=statement_currency)

        if scale_meta.get("fx_source", "").startswith("unsupported_currency"):
            append_provider_error(
                errors,
                provider="yahoo",
                symbol=symbol,
                error_class="unavailable",
                message=(
                    f"Yahoo statement currency {statement_currency} has no USD conversion table entry; "
                    "refusing to invent a FX rate."
                ),
                attempts=max(attempts_used, 1),
            )
            return None

        if revenue in (None, 0):
            append_provider_error(
                errors,
                provider="yahoo",
                symbol=symbol,
                error_class="truly_missing",
                message="Yahoo statement has no usable revenue row.",
                attempts=max(attempts_used, 1),
            )
            return None

        if not is_plausible_revenue_billion_usd(revenue):
            append_provider_error(
                errors,
                provider="yahoo",
                symbol=symbol,
                error_class="implausible_scale",
                message=(
                    f"Yahoo revenue {revenue} billion USD-equiv exceeds plausibility ceiling "
                    f"{MAX_PLAUSIBLE_REVENUE_BILLION_USD} (statement_currency={statement_currency}, "
                    f"raw={revenue_raw}). Likely local-currency / unit mismatch; refusing to publish."
                ),
                attempts=max(attempts_used, 1),
            )
            return None

        if ebitda is None and operating_income is None and r_and_d is None:
            append_provider_error(
                errors,
                provider="yahoo",
                symbol=symbol,
                error_class="truly_missing",
                message="Yahoo statement lacks EBITDA/operating income/R&D peer inputs.",
                attempts=max(attempts_used, 1),
            )
            return None

        market_data: dict[str, float] = {"revenue": float(revenue)}
        if ebitda is not None:
            market_data["ebitda"] = float(ebitda)
        if operating_income is not None:
            market_data["operating_income"] = float(operating_income)
        if r_and_d is not None:
            market_data["r_and_d"] = float(r_and_d)

        return {
            "market_data": market_data,
            "structured_source": "yahoo_fundamentals",
            "fundamentals_meta": {
                "provider": "yahoo",
                "symbol": symbol,
                "fiscal_year": fiscal_year,
                "period_end": str(col),
                "unit": "billion_usd",
                "statement_currency": statement_currency,
                "fx_usd_per_unit": scale_meta.get("fx_usd_per_unit"),
                "fx_source": scale_meta.get("fx_source"),
                "scale_note": (
                    "Yahoo absolute statement values converted to USD (when needed) then divided by 1e9 "
                    "for LumenFin billion-USD metric scale."
                ),
                "fetch_attempts": max(attempts_used, 1),
                "plausibility_ceiling_billion_usd": MAX_PLAUSIBLE_REVENUE_BILLION_USD,
            },
            "supply_chain": {
                "risk_level": "unknown",
                "signals": [
                    f"Fundamentals loaded from Yahoo Finance annual income statement for {symbol} "
                    f"(statement_currency={statement_currency})."
                ],
            },
            "earnings_call_quotes": [
                f"Yahoo fundamentals snapshot for {symbol} (period ending {col}, currency={statement_currency})."
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Yahoo fundamentals parse failed for %s: %s", symbol, exc)
        append_provider_error(
            errors,
            provider="yahoo",
            symbol=symbol,
            error_class=classify_exception(exc),
            message=str(exc),
            attempts=max(attempts_used, 1),
        )
        return None
