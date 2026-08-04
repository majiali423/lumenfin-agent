from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.fundamentals import fetch_yahoo_fundamentals
from lumenfin.provider_retry import (
    call_with_transient_retry,
    classify_exception,
    is_transient_exception,
    summarize_provider_errors,
)
from lumenfin.sec_fundamentals import fetch_sec_companyfacts_fundamentals
from lumenfin.tools import extract_companies_from_query, retrieve_company_payload


class ProviderRetryHelpersTestCase(unittest.TestCase):
    def test_classify_timeout_and_rate_limit(self) -> None:
        self.assertEqual(classify_exception(TimeoutError("timed out")), "timeout")
        self.assertTrue(is_transient_exception(TimeoutError("timed out")))
        self.assertEqual(classify_exception(RuntimeError("HTTP 429 Too Many Requests")), "rate_limited")

    def test_call_with_transient_retry_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("temporary")
            return "ok"

        sleeps: list[float] = []
        result = call_with_transient_retry(
            flaky,
            max_retries=3,
            backoff_seconds=0.5,
            sleep=sleeps.append,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(sleeps, [0.5, 1.0])

    def test_call_with_transient_retry_does_not_retry_client_errors(self) -> None:
        calls = {"n": 0}

        def permanent() -> str:
            calls["n"] += 1
            raise ValueError("bad request")

        with self.assertRaises(ValueError):
            call_with_transient_retry(permanent, max_retries=3, sleep=lambda _: None)
        self.assertEqual(calls["n"], 1)

    def test_summarize_provider_errors_splits_transient_and_missing(self) -> None:
        summary = summarize_provider_errors(
            [
                {"provider": "sec_edgar", "error_class": "timeout", "transient": True},
                {"provider": "yahoo", "error_class": "truly_missing", "transient": False},
                {"provider": "yahoo", "error_class": "rate_limited", "transient": True},
            ]
        )
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["transient_count"], 2)
        self.assertEqual(summary["missing_count"], 1)
        self.assertTrue(summary["has_transient"])
        self.assertFalse(summary["has_truly_missing"])


class YahooFundamentalsRetryTestCase(unittest.TestCase):
    def test_fetch_yahoo_retries_transient_failure(self) -> None:
        frame = pd.DataFrame(
            {
                pd.Timestamp("2025-01-31"): {
                    "Total Revenue": 10_000_000_000.0,
                    "EBITDA": 4_000_000_000.0,
                    "Operating Income": 3_000_000_000.0,
                    "Research And Development": 1_000_000_000.0,
                }
            }
        )
        errors: list[dict] = []
        calls = {"n": 0}

        def fake_load(symbol: str):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("yahoo timeout")
            return frame

        with (
            patch("lumenfin.fundamentals._load_yahoo_income", side_effect=fake_load),
            patch("lumenfin.provider_resilience.time.sleep", return_value=None),
        ):
            payload = fetch_yahoo_fundamentals("ORCL", errors=errors)

        assert payload is not None
        self.assertEqual(payload["structured_source"], "yahoo_fundamentals")
        self.assertEqual(calls["n"], 2)
        self.assertEqual(errors, [])

    def test_fetch_yahoo_records_truly_missing(self) -> None:
        empty = pd.DataFrame()
        errors: list[dict] = []
        with (
            patch("lumenfin.fundamentals._load_yahoo_income", return_value=empty),
            patch("lumenfin.provider_resilience.time.sleep", return_value=None),
        ):
            payload = fetch_yahoo_fundamentals("OPENAI", errors=errors)
        self.assertIsNone(payload)
        self.assertEqual(errors[0]["error_class"], "truly_missing")
        self.assertFalse(errors[0]["transient"])


class SecTransientOnlyRetryTestCase(unittest.TestCase):
    def test_sec_does_not_retry_http_400(self) -> None:
        bad = MagicMock()
        bad.status_code = 400
        bad.raise_for_status.side_effect = Exception("HTTP 400")
        mock_client = MagicMock()
        mock_client.get.return_value = bad
        errors: list[dict] = []
        with (
            patch("lumenfin.sec_fundamentals.resolve_cik", return_value="0000320193"),
            patch("lumenfin.provider_resilience.time.sleep", return_value=None),
        ):
            payload = fetch_sec_companyfacts_fundamentals("AAPL", client=mock_client, errors=errors)
        self.assertIsNone(payload)
        self.assertEqual(mock_client.get.call_count, 1)


class RetrieveProviderErrorsTestCase(unittest.TestCase):
    def test_retrieve_surfaces_provider_errors_when_live_sources_fail(self) -> None:
        def fake_sec(symbol, errors=None, **kwargs):
            if errors is not None:
                errors.append(
                    {
                        "provider": "sec_edgar",
                        "symbol": symbol,
                        "error_class": "truly_missing",
                        "message": "not found",
                        "attempts": 1,
                        "transient": False,
                    }
                )
            return None

        def fake_yahoo(symbol, errors=None, **kwargs):
            if errors is not None:
                errors.append(
                    {
                        "provider": "yahoo",
                        "symbol": symbol,
                        "error_class": "timeout",
                        "message": "timed out",
                        "attempts": 3,
                        "transient": True,
                    }
                )
            return None

        with (
            patch("lumenfin.sec_fundamentals.fetch_sec_companyfacts_fundamentals", side_effect=fake_sec),
            patch("lumenfin.fundamentals.fetch_yahoo_fundamentals", side_effect=fake_yahoo),
        ):
            payload = retrieve_company_payload(
                "OpenAI",
                allow_sample_data=False,
                ticker="OPEN",
                fetch_live_fundamentals=True,
                fetch_sec_fundamentals=True,
            )
        self.assertEqual(payload.get("structured_source"), "none")
        classes = {item["error_class"] for item in payload.get("provider_errors") or []}
        self.assertIn("truly_missing", classes)
        self.assertIn("timeout", classes)


class CompanyAliasFalsePositiveTestCase(unittest.TestCase):
    def test_blockchain_does_not_resolve_to_block_company(self) -> None:
        companies = extract_companies_from_query("Discuss blockchain settlement risk without naming issuers.")
        self.assertNotIn("Block", companies)

    def test_explicit_block_still_resolves(self) -> None:
        companies = extract_companies_from_query("Analyze Block FY2025 operating margin.")
        self.assertIn("Block", companies)


if __name__ == "__main__":
    unittest.main()
