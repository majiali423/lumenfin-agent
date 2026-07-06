from __future__ import annotations

from typing import Any

from lumenfin.market_data import DEFAULT_TICKER_MAP


class FakeMarketDataClient:
    """Deterministic offline market snapshots for unit tests."""

    backend_name = "fake"
    provider = "fake"

    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict[str, Any]:
        ticker = symbol or DEFAULT_TICKER_MAP.get(company, company)
        base_price = 180.0 if company == "Apple" else 420.0
        return {
            "provider": "fake",
            "symbol": ticker,
            "company": company,
            "current_price": base_price,
            "monthly_return": 0.024,
            "market_cap": 2_800_000_000_000 if company == "Apple" else 3_100_000_000_000,
            "trailing_pe": 28.5,
            "currency": "USD",
            "sector": "Technology",
            "industry": "Consumer Electronics" if company == "Apple" else "Software",
            "fifty_two_week_high": base_price * 1.15,
            "fifty_two_week_low": base_price * 0.82,
        }


class UnauthorizedMarketDataClient(FakeMarketDataClient):
    provider = "unauthorized"

    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict[str, Any]:
        raise PermissionError("401 Unauthorized from market provider")


class TimeoutLLMClient:
    backend_name = "timeout-llm"
    model_name = "timeout-llm"

    def __init__(self) -> None:
        self._usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
        self._usage_mark = {"prompt_tokens": 0, "completion_tokens": 0}

    def mark_usage_start(self) -> None:
        self._usage_mark = dict(self._usage_totals)

    def usage_since_mark(self) -> dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        raise TimeoutError("LLM request timed out")


class EmptyHybridRetriever:
    def search(
        self,
        *,
        company: str,
        query: str,
        document_contexts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return []
