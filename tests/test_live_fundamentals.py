from __future__ import annotations

import unittest
from unittest.mock import patch

from lumenfin.fundamentals import fetch_yahoo_fundamentals
from lumenfin.tools import retrieve_company_payload


class _FakeFrame:
    def __init__(self, data: dict[str, float], year: int = 2025) -> None:
        import pandas as pd

        self._df = pd.DataFrame(
            {pd.Timestamp(f"{year}-01-31"): data},
        )

    @property
    def empty(self) -> bool:
        return False

    @property
    def columns(self):
        return self._df.columns

    @property
    def index(self):
        return self._df.index

    @property
    def loc(self):
        return self._df.loc


class LiveFundamentalsTests(unittest.TestCase):
    def test_fetch_yahoo_fundamentals_scales_to_billions(self) -> None:
        frame = _FakeFrame(
            {
                "Total Revenue": 215_938_000_000.0,
                "EBITDA": 144_552_000_000.0,
                "Operating Income": 130_387_000_000.0,
                "Research And Development": 18_497_000_000.0,
            }
        )

        class FakeTicker:
            income_stmt = frame
            financials = frame

        with patch("yfinance.Ticker", return_value=FakeTicker()):
            payload = fetch_yahoo_fundamentals("NVDA")

        assert payload is not None
        self.assertEqual(payload["structured_source"], "yahoo_fundamentals")
        self.assertAlmostEqual(payload["market_data"]["revenue_2025"], 215.938, places=3)
        self.assertAlmostEqual(payload["market_data"]["ebitda_2025"], 144.552, places=3)

    def test_retrieve_prefers_live_over_empty_when_enabled(self) -> None:
        live = {
            "market_data": {"revenue_2025": 10.0, "ebitda_2025": 4.0, "r_and_d_2025": 1.0},
            "structured_source": "yahoo_fundamentals",
            "supply_chain": {"risk_level": "unknown", "signals": []},
            "earnings_call_quotes": [],
            "fundamentals_meta": {"provider": "yahoo", "symbol": "ORCL"},
        }
        with patch("lumenfin.fundamentals.fetch_yahoo_fundamentals", return_value=live):
            payload = retrieve_company_payload(
                "Oracle",
                allow_sample_data=False,
                ticker="ORCL",
                fetch_live_fundamentals=True,
            )
        self.assertEqual(payload["structured_source"], "yahoo_fundamentals")
        self.assertEqual(payload["market_data"]["revenue_2025"], 10.0)


if __name__ == "__main__":
    unittest.main()
