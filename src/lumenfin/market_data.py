from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import httpx

DEFAULT_TICKER_MAP = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "NVIDIA": "NVDA",
    "AMD": "AMD",
    "BYD": "BYDDF",
    "比亚迪": "1211.HK",
    "Toyota": "TM",
    "Samsung": "005930.KS",
    "TSMC": "TSM",
    "Tencent": "TCEHY",
    "腾讯": "0700.HK",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MarketDataCache:
    """In-process TTL cache keyed by ticker (production would use Redis)."""

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self.ttl_seconds = max(1.0, ttl_seconds)
        self._entries: dict[str, tuple[float, dict[str, Any]]] = {}

    def get(self, ticker: str) -> dict[str, Any] | None:
        entry = self._entries.get(ticker.upper())
        if not entry:
            return None
        expires_at, payload = entry
        if time.monotonic() > expires_at:
            return None
        return dict(payload)

    def get_stale(self, ticker: str) -> dict[str, Any] | None:
        entry = self._entries.get(ticker.upper())
        if not entry:
            return None
        return dict(entry[1])

    def set(self, ticker: str, payload: dict[str, Any]) -> None:
        self._entries[ticker.upper()] = (time.monotonic() + self.ttl_seconds, dict(payload))


class MarketDataClient:
    def __init__(
        self,
        provider: str = "yahoo",
        alphavantage_api_key: str | None = None,
        *,
        fallback_provider: str | None = "yahoo",
        cache_ttl_seconds: int = 60,
        cache: MarketDataCache | None = None,
    ) -> None:
        self.provider = provider
        self.fallback_provider = fallback_provider or "yahoo"
        self.alphavantage_api_key = alphavantage_api_key
        self.cache = cache or MarketDataCache(ttl_seconds=cache_ttl_seconds)

    def _provider_chain(self) -> list[str]:
        chain: list[str] = []
        for name in (self.provider, self.fallback_provider):
            normalized = (name or "").strip().lower()
            if not normalized or normalized in chain:
                continue
            if normalized == "alphavantage" and not self.alphavantage_api_key:
                continue
            chain.append(normalized)
        if not chain:
            chain.append("yahoo")
        return chain

    def _fetch_snapshot(self, provider_name: str, company: str, ticker: str) -> dict[str, Any]:
        if provider_name == "alphavantage":
            return self._fetch_from_alphavantage(company, ticker)
        return self._fetch_from_yahoo(company, ticker)

    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict[str, Any]:
        ticker = symbol or DEFAULT_TICKER_MAP.get(company, company)
        cached = self.cache.get(ticker)
        if cached and cached.get("current_price") is not None:
            return self._finalize_snapshot(
                cached,
                status="cached",
                from_cache=True,
                provider_chain=["cache"],
            )

        errors: list[str] = []
        for provider_name in self._provider_chain():
            try:
                snapshot = self._fetch_snapshot(provider_name, company, ticker)
                if snapshot.get("current_price") is not None:
                    self.cache.set(ticker, snapshot)
                    return self._finalize_snapshot(
                        snapshot,
                        status="ok",
                        from_cache=False,
                        provider_chain=[provider_name],
                    )
                errors.append(f"{provider_name}: empty price")
            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")

        stale = self.cache.get_stale(ticker)
        if stale and stale.get("current_price") is not None:
            return self._finalize_snapshot(
                stale,
                status="stale",
                from_cache=True,
                provider_chain=self._provider_chain(),
                error="; ".join(errors),
            )

        return self._empty_snapshot(
            company=company,
            ticker=ticker,
            status="failed",
            provider_chain=self._provider_chain(),
            error="; ".join(errors) if errors else "all providers failed",
        )

    def _finalize_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        status: str,
        from_cache: bool,
        provider_chain: list[str],
        error: str | None = None,
    ) -> dict[str, Any]:
        result = dict(snapshot)
        result["status"] = status
        result["from_cache"] = from_cache
        result["fetched_at"] = result.get("fetched_at") or _utc_now_iso()
        result["provider_chain"] = provider_chain
        if error:
            result["error"] = error
        return result

    def _empty_snapshot(
        self,
        *,
        company: str,
        ticker: str,
        status: str,
        provider_chain: list[str],
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "provider": provider_chain[0] if provider_chain else self.provider,
            "symbol": ticker,
            "company": company,
            "current_price": None,
            "monthly_return": None,
            "market_cap": None,
            "trailing_pe": None,
            "currency": None,
            "sector": None,
            "industry": None,
            "fifty_two_week_high": None,
            "fifty_two_week_low": None,
            "status": status,
            "from_cache": False,
            "fetched_at": _utc_now_iso(),
            "provider_chain": provider_chain,
            "error": error,
        }

    def _fetch_from_yahoo(self, company: str, ticker: str) -> dict[str, Any]:
        """Yahoo Finance via yfinance (handles cookies/crumbs; raw v7 API often returns 401)."""
        import yfinance as yf

        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info or {}
        history = ticker_obj.history(period="1mo")
        closes = [float(value) for value in history["Close"].tolist() if value is not None] if not history.empty else []

        current_price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if current_price is None and closes:
            current_price = round(closes[-1], 4)

        monthly_change = round((closes[-1] / closes[0]) - 1, 4) if len(closes) >= 2 else None
        return {
            "provider": "yahoo",
            "symbol": ticker,
            "company": company,
            "current_price": current_price,
            "monthly_return": monthly_change,
            "market_cap": _safe_float(info.get("marketCap")),
            "trailing_pe": _safe_float(info.get("trailingPE")),
            "currency": info.get("currency"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "fifty_two_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
            "fetched_at": _utc_now_iso(),
        }

    def _fetch_from_alphavantage(self, company: str, ticker: str) -> dict[str, Any]:
        if not self.alphavantage_api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY is not configured")
        overview_url = "https://www.alphavantage.co/query"
        with httpx.Client(timeout=30) as client:
            quote = client.get(
                overview_url,
                params={"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": self.alphavantage_api_key},
            ).json()
        if "Note" in quote or "Information" in quote:
            raise RuntimeError(quote.get("Note") or quote.get("Information") or "Alpha Vantage rate limit")
        global_quote = quote.get("Global Quote") or {}
        current_price = _safe_float(global_quote.get("05. price"))
        if current_price is None:
            raise RuntimeError("Alpha Vantage returned empty Global Quote")

        overview: dict[str, Any] = {}
        try:
            with httpx.Client(timeout=30) as client:
                overview_resp = client.get(
                    overview_url,
                    params={"function": "OVERVIEW", "symbol": ticker, "apikey": self.alphavantage_api_key},
                ).json()
            if "Note" not in overview_resp and "Information" not in overview_resp:
                overview = overview_resp
        except Exception:
            overview = {}

        return {
            "provider": "alphavantage",
            "symbol": ticker,
            "company": company,
            "current_price": current_price,
            "monthly_return": None,
            "market_cap": _safe_float(overview.get("MarketCapitalization")),
            "trailing_pe": _safe_float(overview.get("PERatio")),
            "currency": overview.get("Currency"),
            "sector": overview.get("Sector"),
            "industry": overview.get("Industry"),
            "fifty_two_week_high": _safe_float(overview.get("52WeekHigh")),
            "fifty_two_week_low": _safe_float(overview.get("52WeekLow")),
            "fetched_at": _utc_now_iso(),
        }


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_market_snapshots(market_snapshots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_company: dict[str, dict[str, Any]] = {}
    ok_count = 0
    for company, snap in market_snapshots.items():
        status = str(snap.get("status") or ("ok" if snap.get("current_price") is not None else "failed"))
        if status in {"ok", "cached", "stale"} and snap.get("current_price") is not None:
            ok_count += 1
        per_company[company] = {
            "status": status,
            "provider": snap.get("provider"),
            "symbol": snap.get("symbol"),
            "fetched_at": snap.get("fetched_at"),
            "from_cache": bool(snap.get("from_cache")),
            "has_price": snap.get("current_price") is not None,
            "error": snap.get("error"),
        }
    total = len(market_snapshots)
    return {
        "companies": per_company,
        "ok_count": ok_count,
        "total_count": total,
        "all_ok": total > 0 and ok_count == total,
    }


def probe_market_provider(client: MarketDataClient, *, probe_symbol: str = "NVDA") -> dict[str, Any]:
    """Lightweight connectivity probe for /health."""
    snapshot = client.fetch_company_snapshot("Probe", symbol=probe_symbol)
    return {
        "provider": client.provider,
        "fallback_provider": client.fallback_provider,
        "ok": snapshot.get("current_price") is not None,
        "symbol": probe_symbol,
        "status": snapshot.get("status"),
        "provider_chain": snapshot.get("provider_chain"),
        "error": snapshot.get("error"),
    }
