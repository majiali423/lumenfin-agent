"""SEC EDGAR companyfacts fundamentals (US filers).

Uses the public XBRL companyfacts JSON:
  https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

Requires a descriptive User-Agent (SEC fair-access policy).
Provenance label: structured_source=sec_companyfacts.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from .provider_retry import (
    append_provider_error,
    classify_exception,
    classify_http_status,
    is_transient_error_class,
)

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Prefer more specific revenue tags before generic Revenues.
_REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
)
_OP_INCOME_TAGS = ("OperatingIncomeLoss",)
_RD_TAGS = ("ResearchAndDevelopmentExpense",)
# EBITDA is rarely a primary US-GAAP tag; we approximate when possible.
_DEPR_TAGS = (
    "DepreciationDepletionAndAmortization",
    "DepreciationAndAmortization",
)

_HTTP_RETRIES = 3
_HTTP_BACKOFF_SEC = 0.5


def _user_agent() -> str:
    import os

    contact = os.getenv("SEC_USER_AGENT", "").strip()
    if contact:
        return contact
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    if app_env not in {"dev", "test"}:
        raise RuntimeError(
            "SEC_USER_AGENT is required outside dev/test so SEC requests include "
            "an operator contact per fair-access guidance."
        )
    # Local/offline development fallback. Controlled deployments must configure
    # an operator-owned contact through SEC_USER_AGENT.
    return "LumenFinAgent/0.1 (financial diligence research; contact=lumenfin-local@example.com)"


def _headers() -> dict[str, str]:
    return {
        "User-Agent": _user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def _get_json_with_retries(
    http: httpx.Client,
    url: str,
    *,
    allow_404: bool = False,
    errors: list[dict[str, Any]] | None = None,
    provider: str = "sec_edgar",
    symbol: str = "",
) -> dict[str, Any] | None:
    """Fetch SEC JSON with bounded retries for transient provider failures only."""
    last_error: Exception | None = None
    last_class = "error"
    attempts_used = 0
    for attempt in range(_HTTP_RETRIES):
        attempts_used = attempt + 1
        try:
            resp = http.get(url)
            if allow_404 and resp.status_code == 404:
                append_provider_error(
                    errors,
                    provider=provider,
                    symbol=symbol,
                    error_class="not_found",
                    message=f"HTTP 404 for {url}",
                    attempts=attempts_used,
                )
                return None
            if resp.status_code in {408, 425, 429, 500, 502, 503, 504}:
                error_class = classify_http_status(resp.status_code)
                last_class = error_class
                last_error = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
                if attempt < _HTTP_RETRIES - 1 and is_transient_error_class(error_class):
                    time.sleep(_HTTP_BACKOFF_SEC * (2**attempt))
                    continue
                append_provider_error(
                    errors,
                    provider=provider,
                    symbol=symbol,
                    error_class=error_class,
                    message=str(last_error),
                    attempts=attempts_used,
                )
                return None
            resp.raise_for_status()
            payload = resp.json()
            return payload if isinstance(payload, dict) else None
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            last_class = classify_exception(exc)
            if attempt < _HTTP_RETRIES - 1 and is_transient_error_class(last_class):
                time.sleep(_HTTP_BACKOFF_SEC * (2**attempt))
                continue
            append_provider_error(
                errors,
                provider=provider,
                symbol=symbol,
                error_class=last_class,
                message=str(exc),
                attempts=attempts_used,
            )
            logger.warning(
                "SEC JSON fetch failed after %s attempts for %s (%s): %s",
                attempts_used,
                url,
                last_class,
                last_error,
            )
            return None
    append_provider_error(
        errors,
        provider=provider,
        symbol=symbol,
        error_class=last_class,
        message=str(last_error) if last_error else "unknown SEC fetch failure",
        attempts=attempts_used or _HTTP_RETRIES,
    )
    logger.warning("SEC JSON fetch failed after %s attempts for %s: %s", _HTTP_RETRIES, url, last_error)
    return None


def resolve_cik(
    ticker: str,
    *,
    client: httpx.Client | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> str | None:
    """Map ticker → zero-padded 10-digit CIK (shared SEC ticker directory cache)."""
    from .ticker_resolve import get_cik_for_ticker

    return get_cik_for_ticker(ticker, client=client, errors=errors)


def _latest_annual_value(concept: dict[str, Any]) -> tuple[float, dict[str, Any]] | None:
    """Pick the latest USD annual (10-K / FY) fact for a US-GAAP concept."""
    units = (concept or {}).get("units") or {}
    series = units.get("USD") or []
    if not series:
        return None

    def is_annual(item: dict[str, Any]) -> bool:
        form = str(item.get("form") or "").upper()
        fp = str(item.get("fp") or "").upper()
        frame = str(item.get("frame") or "").upper()
        if item.get("val") is None:
            return False
        if form in {"10-K", "10-K/A"} and fp in {"FY", ""}:
            return True
        # Some issuers mark annual points with CY#### frames and fp=FY.
        if fp == "FY" and (frame.endswith("Q1") or frame.endswith("Q2") or frame.endswith("Q3") or frame.endswith("Q4")):
            return False
        if fp == "FY":
            return True
        return False

    annual = [item for item in series if isinstance(item, dict) and is_annual(item)]
    if not annual:
        return None

    def sort_key(item: dict[str, Any]) -> tuple:
        return (str(item.get("end") or ""), str(item.get("filed") or ""))

    best = sorted(annual, key=sort_key)[-1]
    try:
        value = float(best["val"])
    except (TypeError, ValueError, KeyError):
        return None
    if value != value:
        return None
    return value, best


def _fact_from_tags(gaap: dict[str, Any], tags: tuple[str, ...]) -> tuple[float, str, dict[str, Any]] | None:
    """Across candidate tags, prefer the fact with the latest period end."""
    best: tuple[float, str, dict[str, Any]] | None = None
    best_end = ""
    for tag in tags:
        concept = gaap.get(tag)
        if not isinstance(concept, dict):
            continue
        hit = _latest_annual_value(concept)
        if hit is None:
            continue
        value, meta = hit
        end = str(meta.get("end") or "")
        if best is None or end > best_end:
            best = (value, tag, meta)
            best_end = end
    return best


def _fact_for_period(
    gaap: dict[str, Any],
    tags: tuple[str, ...],
    *,
    prefer_end: str | None,
) -> tuple[float, str, dict[str, Any]] | None:
    """Prefer a fact ending on prefer_end; else latest annual across tags."""
    if prefer_end:
        for tag in tags:
            concept = gaap.get(tag)
            if not isinstance(concept, dict):
                continue
            units = (concept.get("units") or {}).get("USD") or []
            matches = [
                item
                for item in units
                if isinstance(item, dict)
                and str(item.get("end") or "") == prefer_end
                and str(item.get("form") or "").upper() in {"10-K", "10-K/A", "8-K"}
                and item.get("val") is not None
            ]
            if not matches:
                continue
            best = sorted(matches, key=lambda item: str(item.get("filed") or ""))[-1]
            try:
                value = float(best["val"])
            except (TypeError, ValueError, KeyError):
                continue
            if value == value:
                return value, tag, best
    return _fact_from_tags(gaap, tags)


def _to_billions(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value / 1_000_000_000.0, 4)


def fetch_sec_companyfacts_fundamentals(
    ticker: str,
    *,
    client: httpx.Client | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return LumenFin market_data payload from SEC companyfacts, or None."""
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return None
    if any(symbol.endswith(suffix) for suffix in (".HK", ".KS", ".T", ".L", ".SS", ".SZ")):
        append_provider_error(
            errors,
            provider="sec_edgar",
            symbol=symbol,
            error_class="unavailable",
            message="Non-US exchange suffix; SEC companyfacts not applicable.",
        )
        return None
    owns_client = client is None
    http = client or httpx.Client(timeout=45.0, headers=_headers())
    try:
        prior_errors = len(errors or [])
        cik = resolve_cik(symbol, client=http, errors=errors)
        if not cik:
            if len(errors or []) == prior_errors:
                append_provider_error(
                    errors,
                    provider="sec_edgar",
                    symbol=symbol,
                    error_class="truly_missing",
                    message="Ticker not found in SEC company_tickers map.",
                )
            return None
        facts = _get_json_with_retries(
            http,
            _FACTS_URL.format(cik=cik),
            allow_404=True,
            errors=errors,
            provider="sec_edgar",
            symbol=symbol,
        )
        if facts is None:
            return None
        gaap = ((facts.get("facts") or {}).get("us-gaap") or {})
        if not isinstance(gaap, dict) or not gaap:
            append_provider_error(
                errors,
                provider="sec_edgar",
                symbol=symbol,
                error_class="truly_missing",
                message="SEC companyfacts payload has no us-gaap concepts.",
            )
            return None

        revenue_hit = _fact_from_tags(gaap, _REVENUE_TAGS)
        if revenue_hit is None:
            append_provider_error(
                errors,
                provider="sec_edgar",
                symbol=symbol,
                error_class="truly_missing",
                message="No annual revenue fact found in SEC companyfacts.",
            )
            return None
        revenue_raw, revenue_tag, revenue_meta = revenue_hit
        prefer_end = str(revenue_meta.get("end") or "") or None

        op_hit = _fact_for_period(gaap, _OP_INCOME_TAGS, prefer_end=prefer_end)
        rd_hit = _fact_for_period(gaap, _RD_TAGS, prefer_end=prefer_end)
        depr_hit = _fact_for_period(gaap, _DEPR_TAGS, prefer_end=prefer_end)

        revenue = _to_billions(revenue_raw)
        market_data: dict[str, float] = {"revenue": float(revenue)} if revenue is not None else {}
        if not market_data:
            append_provider_error(
                errors,
                provider="sec_edgar",
                symbol=symbol,
                error_class="truly_missing",
                message="Revenue fact could not be scaled to billions.",
            )
            return None

        operating_income = None
        op_tag = None
        op_meta: dict[str, Any] = {}
        if op_hit:
            operating_income = _to_billions(op_hit[0])
            op_tag, op_meta = op_hit[1], op_hit[2]
            if operating_income is not None:
                market_data["operating_income"] = float(operating_income)

        r_and_d = None
        rd_tag = None
        rd_meta: dict[str, Any] = {}
        if rd_hit:
            r_and_d = _to_billions(rd_hit[0])
            rd_tag, rd_meta = rd_hit[1], rd_hit[2]
            if r_and_d is not None:
                market_data["r_and_d"] = float(r_and_d)

        ebitda = None
        ebitda_note = None
        if op_hit is not None and depr_hit is not None:
            ebitda = _to_billions(op_hit[0] + depr_hit[0])
            ebitda_note = f"approx OperatingIncomeLoss + {depr_hit[1]}"
            if ebitda is not None:
                market_data["ebitda"] = float(ebitda)

        if (
            market_data.get("ebitda") is None
            and market_data.get("operating_income") is None
            and market_data.get("r_and_d") is None
        ):
            append_provider_error(
                errors,
                provider="sec_edgar",
                symbol=symbol,
                error_class="truly_missing",
                message="Revenue present but no EBITDA/operating income/R&D peer inputs.",
            )
            return None

        entity = str((facts.get("entityName") or symbol))
        period_end = str(revenue_meta.get("end") or "")
        fiscal_year = None
        try:
            fiscal_year = int(str(revenue_meta.get("fy") or period_end[:4]))
        except Exception:
            fiscal_year = None

        return {
            "market_data": market_data,
            "structured_source": "sec_companyfacts",
            "fundamentals_meta": {
                "provider": "sec_edgar",
                "symbol": symbol,
                "cik": cik,
                "entity_name": entity,
                "fiscal_year": fiscal_year,
                "period_end": period_end,
                "form": revenue_meta.get("form"),
                "filed": revenue_meta.get("filed"),
                "unit": "billion_usd",
                "tags": {
                    "revenue": revenue_tag,
                    "operating_income": op_tag,
                    "r_and_d": rd_tag,
                    "ebitda_approx": ebitda_note,
                },
                "source_url": _FACTS_URL.format(cik=cik),
            },
            "supply_chain": {
                "risk_level": "unknown",
                "signals": [
                    f"Fundamentals loaded from SEC EDGAR companyfacts for {symbol} (CIK {cik})."
                ],
            },
            "earnings_call_quotes": [
                f"SEC companyfacts annual snapshot for {entity} ({symbol}), period ending {period_end}."
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("SEC companyfacts fetch failed for %s: %s", symbol, exc)
        append_provider_error(
            errors,
            provider="sec_edgar",
            symbol=symbol,
            error_class=classify_exception(exc),
            message=str(exc),
        )
        return None
    finally:
        if owns_client:
            http.close()
