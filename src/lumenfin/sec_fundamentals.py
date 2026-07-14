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

_ticker_cik_cache: dict[str, str] | None = None
_ticker_cik_fetched_at = 0.0
_TICKER_CACHE_TTL_SEC = 24 * 3600


def _user_agent() -> str:
    import os

    contact = os.getenv("SEC_USER_AGENT", "").strip()
    if contact:
        return contact
    # SEC asks for an identifying UA string with contact; override via SEC_USER_AGENT.
    return "LumenFinAgent/0.1 (financial diligence research; contact=lumenfin-local@example.com)"


def _headers() -> dict[str, str]:
    return {
        "User-Agent": _user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def resolve_cik(ticker: str, *, client: httpx.Client | None = None) -> str | None:
    """Map ticker → zero-padded 10-digit CIK."""
    global _ticker_cik_cache, _ticker_cik_fetched_at
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return None

    now = time.monotonic()
    if _ticker_cik_cache is None or (now - _ticker_cik_fetched_at) > _TICKER_CACHE_TTL_SEC:
        owns_client = client is None
        http = client or httpx.Client(timeout=30.0, headers=_headers())
        try:
            resp = http.get(_TICKERS_URL)
            resp.raise_for_status()
            payload = resp.json()
            mapping: dict[str, str] = {}
            for row in payload.values() if isinstance(payload, dict) else []:
                if not isinstance(row, dict):
                    continue
                t = str(row.get("ticker") or "").strip().upper()
                cik_raw = row.get("cik_str")
                if not t or cik_raw is None:
                    continue
                mapping[t] = f"{int(cik_raw):010d}"
            _ticker_cik_cache = mapping
            _ticker_cik_fetched_at = now
        except Exception as exc:  # noqa: BLE001
            logger.warning("SEC ticker map fetch failed: %s", exc)
            if _ticker_cik_cache is None:
                return None
        finally:
            if owns_client:
                http.close()

    return (_ticker_cik_cache or {}).get(symbol)


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
) -> dict[str, Any] | None:
    """Return LumenFin market_data payload from SEC companyfacts, or None."""
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return None
    if any(symbol.endswith(suffix) for suffix in (".HK", ".KS", ".T", ".L", ".SS", ".SZ")):
        return None
    owns_client = client is None
    http = client or httpx.Client(timeout=45.0, headers=_headers())
    try:
        cik = resolve_cik(symbol, client=http)
        if not cik:
            return None
        resp = http.get(_FACTS_URL.format(cik=cik))
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        facts = resp.json()
        gaap = ((facts.get("facts") or {}).get("us-gaap") or {})
        if not isinstance(gaap, dict) or not gaap:
            return None

        revenue_hit = _fact_from_tags(gaap, _REVENUE_TAGS)
        if revenue_hit is None:
            return None
        revenue_raw, revenue_tag, revenue_meta = revenue_hit
        prefer_end = str(revenue_meta.get("end") or "") or None

        op_hit = _fact_for_period(gaap, _OP_INCOME_TAGS, prefer_end=prefer_end)
        rd_hit = _fact_for_period(gaap, _RD_TAGS, prefer_end=prefer_end)
        depr_hit = _fact_for_period(gaap, _DEPR_TAGS, prefer_end=prefer_end)

        revenue = _to_billions(revenue_raw)
        market_data: dict[str, float] = {"revenue_2025": float(revenue)} if revenue is not None else {}
        if not market_data:
            return None

        operating_income = None
        op_tag = None
        op_meta: dict[str, Any] = {}
        if op_hit:
            operating_income = _to_billions(op_hit[0])
            op_tag, op_meta = op_hit[1], op_hit[2]
            if operating_income is not None:
                market_data["operating_income_2025"] = float(operating_income)

        r_and_d = None
        rd_tag = None
        rd_meta: dict[str, Any] = {}
        if rd_hit:
            r_and_d = _to_billions(rd_hit[0])
            rd_tag, rd_meta = rd_hit[1], rd_hit[2]
            if r_and_d is not None:
                market_data["r_and_d_2025"] = float(r_and_d)

        ebitda = None
        ebitda_note = None
        if op_hit is not None and depr_hit is not None:
            ebitda = _to_billions(op_hit[0] + depr_hit[0])
            ebitda_note = f"approx OperatingIncomeLoss + {depr_hit[1]}"
            if ebitda is not None:
                market_data["ebitda_2025"] = float(ebitda)

        if (
            market_data.get("ebitda_2025") is None
            and market_data.get("operating_income_2025") is None
            and market_data.get("r_and_d_2025") is None
        ):
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
        return None
    finally:
        if owns_client:
            http.close()
