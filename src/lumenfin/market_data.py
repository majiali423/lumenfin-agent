from __future__ import annotations

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
    "BYD": "BYDDF",
    "比亚迪": "1211.HK",
    "Toyota": "TM",
    "Samsung": "005930.KS",
    "TSMC": "TSM",
    "Tencent": "TCEHY",
    "腾讯": "0700.HK",
}


class MarketDataClient:
    def __init__(self, provider: str = "yahoo", alphavantage_api_key: str | None = None) -> None:
        self.provider = provider
        self.alphavantage_api_key = alphavantage_api_key

    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict[str, Any]:
        ticker = symbol or DEFAULT_TICKER_MAP.get(company, company)
        if self.provider == "alphavantage" and self.alphavantage_api_key:
            try:
                return self._fetch_from_alphavantage(company, ticker)
            except Exception:
                pass
        try:
            return self._fetch_from_yahoo(company, ticker)
        except Exception:
            return {
                "provider": self.provider,
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
            }

    def _fetch_from_yahoo(self, company: str, ticker: str) -> dict[str, Any]:
        with httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0"}) as client:
            quote_resp = client.get("https://query1.finance.yahoo.com/v7/finance/quote", params={"symbols": ticker})
            quote_resp.raise_for_status()
            quote_data = quote_resp.json()["quoteResponse"]["result"]
            chart_resp = client.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"range": "1mo", "interval": "1d"},
            )
            chart_resp.raise_for_status()
            chart_data = chart_resp.json()["chart"]["result"][0]

        quote = quote_data[0] if quote_data else {}
        closes = chart_data.get("indicators", {}).get("quote", [{}])[0].get("close", [])
        closes = [float(value) for value in closes if value is not None]
        latest_close = round(closes[-1], 4) if closes else None
        monthly_change = round((closes[-1] / closes[0]) - 1, 4) if len(closes) >= 2 else None
        return {
            "provider": "yahoo",
            "symbol": ticker,
            "company": company,
            "current_price": latest_close,
            "monthly_return": monthly_change,
            "market_cap": quote.get("marketCap"),
            "trailing_pe": quote.get("trailingPE"),
            "currency": quote.get("currency"),
            "sector": quote.get("sector"),
            "industry": quote.get("industry"),
            "fifty_two_week_high": quote.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": quote.get("fiftyTwoWeekLow"),
        }

    def _fetch_from_alphavantage(self, company: str, ticker: str) -> dict[str, Any]:
        overview_url = "https://www.alphavantage.co/query"
        with httpx.Client(timeout=30) as client:
            overview = client.get(
                overview_url,
                params={"function": "OVERVIEW", "symbol": ticker, "apikey": self.alphavantage_api_key},
            ).json()
            quote = client.get(
                overview_url,
                params={"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": self.alphavantage_api_key},
            ).json()
        global_quote = quote.get("Global Quote", {})
        return {
            "provider": "alphavantage",
            "symbol": ticker,
            "company": company,
            "current_price": _safe_float(global_quote.get("05. price")),
            "monthly_return": None,
            "market_cap": _safe_float(overview.get("MarketCapitalization")),
            "trailing_pe": _safe_float(overview.get("PERatio")),
            "currency": overview.get("Currency"),
            "sector": overview.get("Sector"),
            "industry": overview.get("Industry"),
            "fifty_two_week_high": _safe_float(overview.get("52WeekHigh")),
            "fifty_two_week_low": _safe_float(overview.get("52WeekLow")),
        }


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_market_provider(client: MarketDataClient, *, probe_symbol: str = "NVDA") -> dict[str, Any]:
    """Lightweight connectivity probe for /health (short timeout)."""
    try:
        with httpx.Client(timeout=5, headers={"User-Agent": "Mozilla/5.0"}) as http_client:
            if client.provider == "alphavantage" and client.alphavantage_api_key:
                response = http_client.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "GLOBAL_QUOTE",
                        "symbol": probe_symbol,
                        "apikey": client.alphavantage_api_key,
                    },
                )
                payload = response.json()
                price = _safe_float(payload.get("Global Quote", {}).get("05. price"))
                return {
                    "provider": "alphavantage",
                    "ok": price is not None,
                    "symbol": probe_symbol,
                }

            response = http_client.get(
                "https://query1.finance.yahoo.com/v7/finance/quote",
                params={"symbols": probe_symbol},
            )
            response.raise_for_status()
            rows = response.json().get("quoteResponse", {}).get("result", [])
            price = rows[0].get("regularMarketPrice") if rows else None
            return {
                "provider": "yahoo",
                "ok": price is not None,
                "symbol": probe_symbol,
            }
    except Exception as exc:
        return {
            "provider": client.provider,
            "ok": False,
            "symbol": probe_symbol,
            "error": str(exc),
        }
