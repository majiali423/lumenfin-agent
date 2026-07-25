"""Resolve arbitrary issuers → exchange tickers via SEC company_tickers + local map.

Design:
- DEFAULT_TICKER_MAP remains the fast, curated path (demo brands, HK aliases, etc.).
- SEC ``company_tickers.json`` is the broad US-listed fallback (no API key).
- Name matching is conservative (exact normalized title, then unique substring).
- Network failures fail soft: callers keep prior behavior (company name as symbol).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .market_data import DEFAULT_TICKER_MAP
from .provider_retry import append_provider_error, classify_exception

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CACHE_TTL_SEC = 24 * 3600

_TITLE_NOISE = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|LLC|PLC|SA|NV|"
    r"CLASS\s+[A-Z]|NEW|THE)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SecIssuer:
    ticker: str
    title: str
    cik: str


@dataclass(frozen=True)
class TickerResolveResult:
    ticker: str
    company: str
    source: str  # default_map | sec_ticker | sec_title | query_ticker
    title: str = ""


@dataclass
class _Directory:
    by_ticker: dict[str, SecIssuer]
    by_norm_title: dict[str, str]  # normalized title -> ticker
    fetched_at: float


_directory: _Directory | None = None
_fetch_failed_at: float = 0.0
_FETCH_FAIL_COOLDOWN_SEC = 60.0


def _sec_headers() -> dict[str, str]:
    import os

    contact = os.getenv("SEC_USER_AGENT", "").strip()
    ua = contact or (
        "LumenFinAgent/0.1 (financial diligence research; contact=lumenfin-local@example.com)"
    )
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def normalize_issuer_text(text: str) -> str:
    """Normalize issuer / query labels for title matching."""
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", (text or "").upper())
    cleaned = _TITLE_NOISE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def _display_name_from_title(title: str, ticker: str) -> str:
    for company, mapped in DEFAULT_TICKER_MAP.items():
        if str(mapped).upper() == ticker.upper():
            return company
    # Prefer a short readable label: first 1–3 tokens of the SEC title.
    tokens = [t for t in re.sub(r"[^A-Za-z0-9\s]", " ", title).split() if t]
    drop = {"INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "THE", "NEW"}
    kept: list[str] = []
    for token in tokens:
        if token.upper() in drop:
            if kept:
                break
            continue
        kept.append(token.title() if token.isupper() or token.islower() else token)
        if len(kept) >= 2:
            break
    if kept:
        return " ".join(kept)
    return title.strip() or ticker.upper()


def set_ticker_directory_for_tests(rows: list[dict[str, Any]] | None) -> None:
    """Inject a tiny SEC ticker directory (tests). Pass None to clear."""
    global _directory, _fetch_failed_at
    _fetch_failed_at = 0.0
    if rows is None:
        _directory = None
        return
    by_ticker: dict[str, SecIssuer] = {}
    by_norm_title: dict[str, str] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        title = str(row.get("title") or "").strip()
        cik_raw = row.get("cik_str", 0)
        if not ticker or not title:
            continue
        cik = f"{int(cik_raw):010d}"
        by_ticker[ticker] = SecIssuer(ticker=ticker, title=title, cik=cik)
        norm = normalize_issuer_text(title)
        if norm and norm not in by_norm_title:
            by_norm_title[norm] = ticker
    _directory = _Directory(by_ticker=by_ticker, by_norm_title=by_norm_title, fetched_at=time.monotonic())


def _build_directory_from_payload(payload: Any) -> _Directory:
    by_ticker: dict[str, SecIssuer] = {}
    by_norm_title: dict[str, str] = {}
    rows = payload.values() if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        title = str(row.get("title") or "").strip()
        cik_raw = row.get("cik_str")
        if not ticker or not title or cik_raw is None:
            continue
        try:
            cik = f"{int(cik_raw):010d}"
        except (TypeError, ValueError):
            continue
        by_ticker[ticker] = SecIssuer(ticker=ticker, title=title, cik=cik)
        norm = normalize_issuer_text(title)
        if norm and norm not in by_norm_title:
            by_norm_title[norm] = ticker
    return _Directory(by_ticker=by_ticker, by_norm_title=by_norm_title, fetched_at=time.monotonic())


def ensure_sec_ticker_directory(
    *,
    client: httpx.Client | None = None,
    errors: list[dict[str, Any]] | None = None,
    force: bool = False,
) -> _Directory | None:
    """Load/cache SEC company_tickers.json. Returns None when unavailable."""
    global _directory, _fetch_failed_at
    now = time.monotonic()
    if (
        not force
        and _directory is not None
        and (now - _directory.fetched_at) <= _CACHE_TTL_SEC
    ):
        return _directory
    if not force and _directory is None and (now - _fetch_failed_at) < _FETCH_FAIL_COOLDOWN_SEC:
        return None

    owns_client = client is None
    http = client or httpx.Client(timeout=30.0, headers=_sec_headers())
    try:
        response = http.get(_TICKERS_URL)
        response.raise_for_status()
        payload = response.json()
        _directory = _build_directory_from_payload(payload)
        return _directory
    except Exception as exc:  # noqa: BLE001
        logger.warning("SEC ticker directory fetch failed: %s", exc)
        append_provider_error(
            errors,
            provider="sec_edgar",
            symbol="TICKERS",
            error_class=classify_exception(exc),
            message=str(exc),
        )
        _fetch_failed_at = now
        if _directory is not None:
            return _directory
        return None
    finally:
        if owns_client:
            http.close()


def get_cik_for_ticker(
    ticker: str,
    *,
    client: httpx.Client | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> str | None:
    """CIK lookup used by sec_fundamentals (shared directory cache)."""
    symbol = (ticker or "").strip().upper()
    if not symbol:
        return None
    directory = ensure_sec_ticker_directory(client=client, errors=errors)
    if directory is None:
        return None
    issuer = directory.by_ticker.get(symbol)
    return issuer.cik if issuer else None


def _unique_title_substring_match(directory: _Directory, needle: str) -> str | None:
    if len(needle) < 4:
        return None
    hits: list[str] = []
    for norm_title, ticker in directory.by_norm_title.items():
        if needle == norm_title or norm_title.startswith(needle + " ") or f" {needle} " in f" {norm_title} ":
            hits.append(ticker)
            if len(hits) > 1:
                return None
    return hits[0] if len(hits) == 1 else None


def resolve_ticker_for_company(
    company: str,
    *,
    query: str = "",
    client: httpx.Client | None = None,
    errors: list[dict[str, Any]] | None = None,
    allow_network: bool = True,
) -> TickerResolveResult | None:
    """Map a company label or ticker token to an exchange symbol when possible."""
    raw = (company or "").strip()
    if not raw:
        return None

    # Curated map first (includes non-US aliases the SEC file lacks).
    if raw in DEFAULT_TICKER_MAP:
        ticker = str(DEFAULT_TICKER_MAP[raw]).upper()
        return TickerResolveResult(ticker=ticker, company=raw, source="default_map")
    lowered_map = {name.lower(): (name, str(sym).upper()) for name, sym in DEFAULT_TICKER_MAP.items()}
    if raw.lower() in lowered_map:
        name, ticker = lowered_map[raw.lower()]
        return TickerResolveResult(ticker=ticker, company=name, source="default_map")

    upper = raw.upper()
    # Direct ticker token (COST, NVDA).
    if re.fullmatch(r"[A-Z]{1,5}", upper):
        for name, sym in DEFAULT_TICKER_MAP.items():
            if str(sym).upper() == upper:
                return TickerResolveResult(ticker=upper, company=name, source="default_map")
        if allow_network:
            directory = ensure_sec_ticker_directory(client=client, errors=errors)
            if directory and upper in directory.by_ticker:
                issuer = directory.by_ticker[upper]
                return TickerResolveResult(
                    ticker=upper,
                    company=_display_name_from_title(issuer.title, upper),
                    source="sec_ticker",
                    title=issuer.title,
                )

    directory = ensure_sec_ticker_directory(client=client, errors=errors) if allow_network else _directory
    if directory is None:
        return None

    norm = normalize_issuer_text(raw)
    if norm in directory.by_norm_title:
        ticker = directory.by_norm_title[norm]
        issuer = directory.by_ticker[ticker]
        return TickerResolveResult(
            ticker=ticker,
            company=_display_name_from_title(issuer.title, ticker),
            source="sec_title",
            title=issuer.title,
        )

    ticker = _unique_title_substring_match(directory, norm)
    if ticker:
        issuer = directory.by_ticker[ticker]
        return TickerResolveResult(
            ticker=ticker,
            company=_display_name_from_title(issuer.title, ticker),
            source="sec_title",
            title=issuer.title,
        )

    # Optional: explicit ticker mentioned next to the company in the query.
    if query:
        for match in re.finditer(r"\(([A-Z]{1,5})\)|\bticker\s*[:=]\s*([A-Z]{1,5})\b", query, flags=re.I):
            token = (match.group(1) or match.group(2) or "").upper()
            if token in directory.by_ticker:
                issuer = directory.by_ticker[token]
                return TickerResolveResult(
                    ticker=token,
                    company=raw if raw.lower() not in {token.lower()} else _display_name_from_title(issuer.title, token),
                    source="query_ticker",
                    title=issuer.title,
                )
    return None


def enrich_company_universe(
    companies: list[str],
    *,
    query: str = "",
    client: httpx.Client | None = None,
    errors: list[dict[str, Any]] | None = None,
    allow_network: bool = True,
) -> tuple[list[str], dict[str, str], list[str]]:
    """Canonicalize labels and attach tickers; pull bare SEC tickers from the query.

    Returns (companies, symbol_by_company, notes).
    """
    notes: list[str] = []
    symbols: dict[str, str] = {}
    ordered: list[str] = []

    def _append(company: str, ticker: str | None, note: str | None = None) -> None:
        if company and company not in ordered:
            ordered.append(company)
        if company and ticker:
            symbols[company] = ticker
        if note:
            notes.append(note)

    directory = ensure_sec_ticker_directory(client=client, errors=errors) if allow_network else _directory

    for name in companies:
        resolved = resolve_ticker_for_company(
            name,
            query=query,
            client=client,
            errors=errors,
            allow_network=allow_network,
        )
        if resolved is None:
            _append(name, None)
            if directory is not None:
                notes.append(f"No ticker resolved for '{name}' (live SEC/Yahoo may fail).")
            continue
        note = None
        if resolved.source != "default_map":
            note = f"Resolved '{name}' → {resolved.company} ({resolved.ticker}) via {resolved.source}."
        # Keep the caller's label when curated map already knows it; otherwise prefer SEC display name.
        label = name if resolved.source == "default_map" else resolved.company
        if name not in ordered and label != name:
            # Preserve original mention order using the resolved display label.
            pass
        _append(label, resolved.ticker, note)
        if label != name and name not in symbols:
            # Also allow lookups keyed by the original extracted string.
            symbols[name] = resolved.ticker

    # Bare tickers in the query: require original ALL-CAPS (so "live fundamentals"
    # does not become ticker LIVE / Live Ventures). Length >= 3 avoids IT/AI noise.
    if query and directory is not None:
        existing_tickers = {sym.upper() for sym in symbols.values()}
        stopwords = {
            "FY",
            "USD",
            "GAAP",
            "HTTP",
            "HTTPS",
            "PDF",
            "JSON",
            "HTML",
            "YEAR",
            "LIVE",
            "ONLY",
            "FROM",
            "WITH",
            "THAT",
            "THIS",
            "RISK",
            "DATA",
            "COST",
            "CASH",
            "DEBT",
            "RATE",
            "FUND",
            "GAIN",
            "LOSS",
            "OPEN",
            "HIGH",
            "AND",
            "FOR",
            "THE",
            "SEC",
            "API",
            "CEO",
            "CFO",
            "IPO",
            "EPS",
            "TTM",
            "YOY",
            "QOQ",
        }
        for token in re.findall(r"\b([A-Z]{3,5})\b", query):
            if token in existing_tickers or token not in directory.by_ticker:
                continue
            if token in stopwords:
                continue
            issuer = directory.by_ticker[token]
            company = _display_name_from_title(issuer.title, token)
            _append(
                company,
                token,
                f"Picked up query ticker {token} → {company} from SEC directory.",
            )
            existing_tickers.add(token)

    return ordered, symbols, notes
