"""SEC EDGAR companyfacts fundamentals (US filers).

Uses the public XBRL companyfacts JSON:
  https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

Requires a descriptive User-Agent (SEC fair-access policy).
Provenance label: structured_source=sec_companyfacts.

Retry owner: ``call_with_policy`` only (no nested SEC retry loop).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .provider_resilience import (
    InvalidProviderResponseError,
    ProviderCallContext,
    ProviderCallPolicy,
    acquire_provider_slot,
    call_with_policy,
    classify_provider_exception,
    get_shared_http_client,
)
from .provider_retry import (
    TRANSIENT_HTTP_STATUS,
    append_provider_error,
    classify_exception,
    classify_http_status,
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

_HTTP_MAX_ATTEMPTS = max(1, int(os.getenv("MAS_MARKET_DATA_MAX_ATTEMPTS", "3")))
_HTTP_BACKOFF_SEC = float(os.getenv("MAS_MARKET_DATA_BACKOFF_SECONDS", "0.5"))
_HTTP_TIMEOUT_SEC = float(os.getenv("MAS_SEC_TIMEOUT_SECONDS", "45"))
_MARKET_MAX_INFLIGHT = max(1, int(os.getenv("MAS_MARKET_DATA_MAX_INFLIGHT_PER_PROCESS", "8")))
_ACQUIRE_TIMEOUT = float(os.getenv("MAS_PROVIDER_ACQUIRE_TIMEOUT_SECONDS", "5"))


def _user_agent() -> str:
    contact = os.getenv("SEC_USER_AGENT", "").strip()
    if contact:
        return contact
    app_env = os.getenv("APP_ENV", "dev").strip().lower()
    if app_env not in {"dev", "test", "integration"}:
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


def _resolve_sec_client(client: httpx.Client | None) -> tuple[httpx.Client, bool]:
    """Return (client, owns_client). Shared process client is never closed here."""
    if client is not None:
        return client, False
    return (
        get_shared_http_client(
            "sec-edgar",
            timeout=_HTTP_TIMEOUT_SEC,
            headers=_headers(),
        ),
        False,
    )


def _get_json_with_retries(
    http: httpx.Client,
    url: str,
    *,
    allow_404: bool = False,
    errors: list[dict[str, Any]] | None = None,
    provider: str = "sec_edgar",
    symbol: str = "",
    call_context: ProviderCallContext | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any] | None:
    """Fetch SEC JSON. Retry owner is ``call_with_policy`` only."""
    attempts = max(1, int(max_attempts if max_attempts is not None else _HTTP_MAX_ATTEMPTS))
    policy = ProviderCallPolicy(
        provider=provider,
        operation="http_get",
        max_attempts=attempts,
        connect_timeout_seconds=min(5.0, _HTTP_TIMEOUT_SEC),
        read_timeout_seconds=_HTTP_TIMEOUT_SEC,
        write_timeout_seconds=_HTTP_TIMEOUT_SEC,
        pool_timeout_seconds=min(5.0, _HTTP_TIMEOUT_SEC),
        base_backoff_seconds=_HTTP_BACKOFF_SEC,
        max_backoff_seconds=max(_HTTP_BACKOFF_SEC * 8, 8.0),
        jitter_ratio=0.2,
    )
    context = call_context or ProviderCallContext.create()
    if context.trace_sink is None:
        context.trace_sink = []
    before = len(context.trace_sink)

    def _once() -> dict[str, Any]:
        remaining = context.remaining_seconds()
        timeout = policy.httpx_timeout(remaining_seconds=remaining)
        resp = http.get(url, timeout=timeout)
        if allow_404 and resp.status_code == 404:
            raise httpx.HTTPStatusError(
                f"HTTP 404 for {url}",
                request=resp.request,
                response=resp,
            )
        if resp.status_code in TRANSIENT_HTTP_STATUS:
            raise httpx.HTTPStatusError(
                f"HTTP {resp.status_code}",
                request=resp.request,
                response=resp,
            )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            raise InvalidProviderResponseError("SEC JSON payload is not an object")
        return payload

    release = None
    try:
        release = acquire_provider_slot(
            "market-data",
            max_inflight=_MARKET_MAX_INFLIGHT,
            context=context,
            acquire_timeout_seconds=_ACQUIRE_TIMEOUT,
        )
        return call_with_policy(_once, policy=policy, context=context)
    except Exception as exc:  # noqa: BLE001
        attempts_used = max(1, len(context.trace_sink) - before)
        if isinstance(exc, httpx.HTTPStatusError):
            error_class = classify_http_status(exc.response.status_code)
        else:
            error_class = classify_provider_exception(exc)
        append_provider_error(
            errors,
            provider=provider,
            symbol=symbol,
            error_class=error_class,
            message=str(exc),
            attempts=attempts_used,
        )
        logger.warning(
            "SEC JSON fetch failed after %s attempts for %s (%s): %s",
            attempts_used,
            url,
            error_class,
            exc,
        )
        return None
    finally:
        if release is not None:
            release()


def resolve_cik(
    ticker: str,
    *,
    client: httpx.Client | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> str | None:
    """Map ticker → zero-padded 10-digit CIK (shared SEC ticker directory cache)."""
    from .ticker_resolve import get_cik_for_ticker

    return get_cik_for_ticker(ticker, client=client, errors=errors)


def _item_fiscal_year(item: dict[str, Any]) -> int | None:
    try:
        if item.get("fy") is not None:
            return int(item["fy"])
    except (TypeError, ValueError):
        pass
    end = str(item.get("end") or "")
    if len(end) >= 4 and end[:4].isdigit():
        try:
            return int(end[:4])
        except ValueError:
            return None
    return None


def _latest_annual_value(
    concept: dict[str, Any],
    *,
    prefer_fiscal_year: int | None = None,
) -> tuple[float, dict[str, Any]] | None:
    """Pick a USD annual (10-K / FY) fact; prefer requested FY when present."""
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

    pool = annual
    if prefer_fiscal_year is not None:
        matched = [item for item in annual if _item_fiscal_year(item) == prefer_fiscal_year]
        if matched:
            pool = matched

    best = sorted(pool, key=sort_key)[-1]
    try:
        value = float(best["val"])
    except (TypeError, ValueError, KeyError):
        return None
    if value != value:
        return None
    return value, best


def _fact_from_tags(
    gaap: dict[str, Any],
    tags: tuple[str, ...],
    *,
    prefer_fiscal_year: int | None = None,
) -> tuple[float, str, dict[str, Any]] | None:
    """Across candidate tags, prefer requested FY when possible, else latest period end."""
    best: tuple[float, str, dict[str, Any]] | None = None
    best_end = ""
    best_exact = False
    for tag in tags:
        concept = gaap.get(tag)
        if not isinstance(concept, dict):
            continue
        hit = _latest_annual_value(concept, prefer_fiscal_year=prefer_fiscal_year)
        if hit is None:
            continue
        value, meta = hit
        end = str(meta.get("end") or "")
        exact = prefer_fiscal_year is not None and _item_fiscal_year(meta) == prefer_fiscal_year
        if best is None:
            best = (value, tag, meta)
            best_end = end
            best_exact = exact
            continue
        if exact and not best_exact:
            best = (value, tag, meta)
            best_end = end
            best_exact = True
            continue
        if exact == best_exact and end > best_end:
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
    prefer_fiscal_year: int | None = None,
    call_context: ProviderCallContext | None = None,
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
    http, owns_client = _resolve_sec_client(client)
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
            call_context=call_context,
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

        revenue_hit = _fact_from_tags(
            gaap, _REVENUE_TAGS, prefer_fiscal_year=prefer_fiscal_year
        )
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

        period_alignment = "unspecified"
        if prefer_fiscal_year is not None and fiscal_year is not None:
            period_alignment = (
                "exact" if int(fiscal_year) == int(prefer_fiscal_year) else "fallback_latest"
            )
        elif prefer_fiscal_year is not None:
            period_alignment = "fallback_latest"

        source_url = _FACTS_URL.format(cik=cik)

        def _record_provenance(tag: str | None, meta: dict[str, Any]) -> dict[str, Any]:
            record_year = _item_fiscal_year(meta)
            record_period = f"FY{record_year}" if record_year is not None else None
            record_bits = [
                "sec",
                cik,
                str(tag or "unknown-tag"),
                str(meta.get("accn") or "unknown-accession"),
                str(meta.get("end") or "unknown-end"),
            ]
            return {
                "source": "sec_companyfacts",
                "provider": "sec",
                "confidence": "high",
                "period": record_period,
                "period_type": "annual",
                "period_source": "provider_record",
                "period_alignment": "exact",
                "citation": source_url,
                "source_record_id": ":".join(record_bits),
            }

        fundamental_provenance = {
            "revenue": _record_provenance(revenue_tag, revenue_meta),
        }
        if operating_income is not None:
            fundamental_provenance["operating_income"] = _record_provenance(op_tag, op_meta)
        if r_and_d is not None:
            fundamental_provenance["r_and_d"] = _record_provenance(rd_tag, rd_meta)
        if ebitda is not None and op_hit is not None and depr_hit is not None:
            ebitda_prov = _record_provenance(op_tag, op_meta)
            ebitda_prov["source_record_id"] = (
                f"{ebitda_prov['source_record_id']}+"
                f"sec:{cik}:{depr_hit[1]}:{depr_hit[2].get('accn') or 'unknown-accession'}:"
                f"{depr_hit[2].get('end') or 'unknown-end'}"
            )
            fundamental_provenance["ebitda"] = ebitda_prov

        return {
            "market_data": market_data,
            "structured_source": "sec_companyfacts",
            "fundamental_provenance": fundamental_provenance,
            "fundamentals_meta": {
                "provider": "sec_edgar",
                "symbol": symbol,
                "cik": cik,
                "entity_name": entity,
                "fiscal_year": fiscal_year,
                "requested_fiscal_year": prefer_fiscal_year,
                "period_alignment": period_alignment,
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
                "source_url": source_url,
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
