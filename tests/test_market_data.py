from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.market_data import (  # noqa: E402
    MarketDataCache,
    MarketDataClient,
    probe_market_provider,
    summarize_market_snapshots,
)


def _yahoo_snapshot(company: str = "NVIDIA", ticker: str = "NVDA") -> dict:
    return {
        "provider": "yahoo",
        "symbol": ticker,
        "company": company,
        "current_price": 120.5,
        "monthly_return": 0.03,
        "market_cap": 3_000_000_000_000,
        "trailing_pe": 45.0,
        "currency": "USD",
        "sector": "Technology",
        "industry": "Semiconductors",
        "fifty_two_week_high": 140.0,
        "fifty_two_week_low": 80.0,
        "fetched_at": "2026-07-07T10:00:00+00:00",
    }


def _av_snapshot(company: str = "NVIDIA", ticker: str = "NVDA") -> dict:
    snap = _yahoo_snapshot(company, ticker)
    snap["provider"] = "alphavantage"
    snap["current_price"] = 121.0
    return snap


class MarketDataCacheTestCase(unittest.TestCase):
    def test_ttl_expiry(self) -> None:
        cache = MarketDataCache(ttl_seconds=1)
        cache.set("NVDA", {"current_price": 100.0})
        self.assertIsNotNone(cache.get("NVDA"))
        time.sleep(1.05)
        self.assertIsNone(cache.get("NVDA"))
        self.assertIsNotNone(cache.get_stale("NVDA"))


class MarketDataClientTestCase(unittest.TestCase):
    def test_primary_success_caches_result(self) -> None:
        client = MarketDataClient(provider="yahoo", cache_ttl_seconds=60)
        with patch.object(client, "_fetch_from_yahoo", return_value=_yahoo_snapshot()) as yahoo:
            first = client.fetch_company_snapshot("NVIDIA")
            second = client.fetch_company_snapshot("NVIDIA")
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "cached")
        self.assertTrue(second["from_cache"])
        yahoo.assert_called_once()

    def test_fallback_when_primary_fails(self) -> None:
        client = MarketDataClient(
            provider="alphavantage",
            alphavantage_api_key="test-key",
            fallback_provider="yahoo",
        )
        with patch.object(client, "_fetch_from_alphavantage", side_effect=RuntimeError("rate limit")):
            with patch.object(client, "_fetch_from_yahoo", return_value=_yahoo_snapshot()) as yahoo:
                snap = client.fetch_company_snapshot("NVIDIA")
        self.assertEqual(snap["status"], "ok")
        self.assertEqual(snap["provider"], "yahoo")
        yahoo.assert_called_once()

    def test_yfinance_yahoo_snapshot_has_price(self) -> None:
        client = MarketDataClient(provider="yahoo")
        with patch("yfinance.Ticker") as ticker_cls:
            ticker = ticker_cls.return_value
            ticker.info = {
                "currentPrice": 123.45,
                "marketCap": 1_000_000_000,
                "trailingPE": 30.0,
                "currency": "USD",
                "sector": "Technology",
                "industry": "Semiconductors",
                "fiftyTwoWeekHigh": 140.0,
                "fiftyTwoWeekLow": 90.0,
            }
            ticker.history.return_value.empty = False
            ticker.history.return_value = type(
                "Frame",
                (),
                {
                    "empty": False,
                    "__getitem__": lambda self, key: type(
                        "Series",
                        (),
                        {"tolist": lambda self: [100.0, 123.45]},
                    )(),
                },
            )()
            snap = client.fetch_company_snapshot("NVIDIA", "NVDA")
        self.assertEqual(snap["current_price"], 123.45)
        self.assertEqual(snap["provider"], "yahoo")

    def test_returns_failed_when_all_providers_fail(self) -> None:
        client = MarketDataClient(provider="yahoo", fallback_provider="yahoo")
        with patch.object(client, "_fetch_from_yahoo", side_effect=RuntimeError("401 Unauthorized")):
            snap = client.fetch_company_snapshot("NVIDIA")
        self.assertEqual(snap["status"], "failed")
        self.assertIsNone(snap["current_price"])
        self.assertIn("401", snap.get("error", ""))

    def test_stale_cache_when_providers_fail(self) -> None:
        client = MarketDataClient(provider="yahoo", cache_ttl_seconds=1)
        with patch.object(client, "_fetch_from_yahoo", return_value=_yahoo_snapshot()):
            fresh = client.fetch_company_snapshot("NVIDIA")
        self.assertEqual(fresh["status"], "ok")
        time.sleep(1.05)
        with patch.object(client, "_fetch_from_yahoo", side_effect=RuntimeError("network down")):
            stale = client.fetch_company_snapshot("NVIDIA")
        self.assertEqual(stale["status"], "stale")
        self.assertEqual(stale["current_price"], 120.5)

    def test_alphavantage_primary_without_key_skips_to_fallback(self) -> None:
        client = MarketDataClient(provider="alphavantage", alphavantage_api_key=None, fallback_provider="yahoo")
        self.assertEqual(client._provider_chain(), ["yahoo"])
        with patch.object(client, "_fetch_from_yahoo", return_value=_yahoo_snapshot()):
            snap = client.fetch_company_snapshot("NVIDIA")
        self.assertEqual(snap["provider"], "yahoo")


class MarketDataSummaryTestCase(unittest.TestCase):
    def test_summarize_market_snapshots(self) -> None:
        summary = summarize_market_snapshots(
            {
                "NVIDIA": {"status": "ok", "provider": "yahoo", "current_price": 120.0, "fetched_at": "t1"},
                "AMD": {"status": "failed", "provider": "yahoo", "current_price": None, "error": "401"},
            }
        )
        self.assertEqual(summary["ok_count"], 1)
        self.assertEqual(summary["total_count"], 2)
        self.assertFalse(summary["all_ok"])
        self.assertEqual(summary["companies"]["AMD"]["status"], "failed")

    def test_probe_market_provider(self) -> None:
        client = MarketDataClient(provider="yahoo")
        with patch.object(client, "_fetch_from_yahoo", return_value=_av_snapshot()):
            probe = probe_market_provider(client)
        self.assertTrue(probe["ok"])
        self.assertEqual(probe["status"], "ok")


if __name__ == "__main__":
    unittest.main()
