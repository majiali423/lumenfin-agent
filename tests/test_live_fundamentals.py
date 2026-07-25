from __future__ import annotations

import unittest
from unittest.mock import patch

from lumenfin.fundamentals import (
    MAX_PLAUSIBLE_REVENUE_BILLION_USD,
    fetch_yahoo_fundamentals,
    is_plausible_revenue_billion_usd,
)
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

        with (
            patch("lumenfin.fundamentals._load_yahoo_income", return_value=frame),
            patch("lumenfin.fundamentals._load_yahoo_financial_currency", return_value="USD"),
        ):
            payload = fetch_yahoo_fundamentals("NVDA")

        assert payload is not None
        self.assertEqual(payload["structured_source"], "yahoo_fundamentals")
        self.assertAlmostEqual(payload["market_data"]["revenue"], 215.938, places=3)
        self.assertAlmostEqual(payload["market_data"]["ebitda"], 144.552, places=3)
        self.assertEqual(payload["fundamentals_meta"]["statement_currency"], "USD")

    def test_fetch_yahoo_converts_twd_statement_to_usd_billions(self) -> None:
        # Real TSMC-scale TWD absolute amount (~NT$3,809B).
        frame = _FakeFrame(
            {
                "Total Revenue": 3_809_054_300_000.0,
                "EBITDA": 2_742_121_500_000.0,
                "Operating Income": 1_936_095_600_000.0,
                "Research And Development": 246_427_200_000.0,
            }
        )
        errors: list[dict] = []
        with (
            patch("lumenfin.fundamentals._load_yahoo_income", return_value=frame),
            patch("lumenfin.fundamentals._load_yahoo_financial_currency", return_value="TWD"),
        ):
            payload = fetch_yahoo_fundamentals("TSM", errors=errors)

        assert payload is not None
        # 3,809,054,300,000 * 0.031 / 1e9 ≈ 118.08
        self.assertAlmostEqual(payload["market_data"]["revenue"], 118.0807, places=2)
        self.assertLess(payload["market_data"]["revenue"], MAX_PLAUSIBLE_REVENUE_BILLION_USD)
        self.assertEqual(payload["fundamentals_meta"]["statement_currency"], "TWD")
        self.assertEqual(errors, [])

    def test_fetch_yahoo_rejects_implausible_usd_scale(self) -> None:
        # Same TWD raw magnitude, but mislabeled as USD -> would become ~3809B.
        frame = _FakeFrame(
            {
                "Total Revenue": 3_809_054_300_000.0,
                "EBITDA": 2_742_121_500_000.0,
                "Operating Income": 1_936_095_600_000.0,
            }
        )
        errors: list[dict] = []
        with (
            patch("lumenfin.fundamentals._load_yahoo_income", return_value=frame),
            patch("lumenfin.fundamentals._load_yahoo_financial_currency", return_value="USD"),
        ):
            payload = fetch_yahoo_fundamentals("TSM", errors=errors)

        self.assertIsNone(payload)
        self.assertEqual(errors[0]["error_class"], "implausible_scale")
        self.assertFalse(errors[0]["transient"])

    def test_plausibility_helper(self) -> None:
        self.assertTrue(is_plausible_revenue_billion_usd(122.4))
        self.assertTrue(is_plausible_revenue_billion_usd(416.0))
        self.assertFalse(is_plausible_revenue_billion_usd(3809.0))
        self.assertFalse(is_plausible_revenue_billion_usd(0))

    def test_retrieve_prefers_live_over_empty_when_enabled(self) -> None:
        live = {
            "market_data": {"revenue": 10.0, "ebitda": 4.0, "r_and_d": 1.0},
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
        self.assertEqual(payload["market_data"]["revenue"], 10.0)


if __name__ == "__main__":
    unittest.main()
