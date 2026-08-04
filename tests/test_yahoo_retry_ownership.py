"""Yahoo fundamentals single retry-owner tests (no live Yahoo)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.fundamentals import fetch_yahoo_fundamentals
from lumenfin.provider_resilience import ProviderCallContext


class YahooRetryOwnershipTestCase(unittest.TestCase):
    def test_timeout_timeout_success_calls_three(self) -> None:
        calls = {"n": 0}
        frame = pd.DataFrame(
            {"2024-12-31": [100_000_000_000.0, 40_000_000_000.0]},
            index=["Total Revenue", "Operating Income"],
        )

        def fake_load(_symbol: str):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("yahoo timeout")
            return frame

        errors: list[dict] = []
        ctx = ProviderCallContext.create(request_id="yahoo-ok")
        ctx.sleep = lambda _: None
        with (
            patch("lumenfin.fundamentals._load_yahoo_income", side_effect=fake_load),
            patch("lumenfin.fundamentals._load_yahoo_financial_currency", return_value="USD"),
        ):
            payload = fetch_yahoo_fundamentals(
                "ORCL",
                errors=errors,
                call_context=ctx,
                max_attempts=3,
            )
        self.assertIsNotNone(payload)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(errors, [])
        self.assertEqual(payload["fundamentals_meta"]["fetch_attempts"], 3)

    def test_permanent_value_error_calls_once(self) -> None:
        calls = {"n": 0}

        def fake_load(_symbol: str):
            calls["n"] += 1
            raise ValueError("permanent yahoo parse failure")

        errors: list[dict] = []
        ctx = ProviderCallContext.create()
        ctx.sleep = lambda _: None
        with patch("lumenfin.fundamentals._load_yahoo_income", side_effect=fake_load):
            payload = fetch_yahoo_fundamentals(
                "BAD",
                errors=errors,
                call_context=ctx,
                max_attempts=3,
            )
        self.assertIsNone(payload)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(errors[-1]["attempts"], 1)

    def test_empty_frame_truly_missing_no_retry(self) -> None:
        calls = {"n": 0}

        def fake_load(_symbol: str):
            calls["n"] += 1
            return pd.DataFrame()

        errors: list[dict] = []
        ctx = ProviderCallContext.create()
        ctx.sleep = lambda _: None
        with patch("lumenfin.fundamentals._load_yahoo_income", side_effect=fake_load):
            payload = fetch_yahoo_fundamentals(
                "OPENAI",
                errors=errors,
                call_context=ctx,
                max_attempts=3,
            )
        self.assertIsNone(payload)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(errors[-1]["error_class"], "truly_missing")

    def test_deadline_exhausted_stops_loader(self) -> None:
        calls = {"n": 0}

        def fake_load(_symbol: str):
            calls["n"] += 1
            raise TimeoutError("slow yahoo")

        errors: list[dict] = []
        ctx = ProviderCallContext.create(deadline_seconds=0.01)
        ctx.sleep = lambda _: None
        # Force remaining budget exhausted after first failure classification path.
        original_ensure = ctx.ensure_budget
        ticks = {"n": 0}

        def gated_ensure(*, minimum_seconds: float = 0.05):
            ticks["n"] += 1
            if ticks["n"] > 1:
                from lumenfin.provider_resilience import DeadlineExceededError

                raise DeadlineExceededError("budget gone")
            return original_ensure(minimum_seconds=minimum_seconds)

        ctx.ensure_budget = gated_ensure  # type: ignore[method-assign]
        with patch("lumenfin.fundamentals._load_yahoo_income", side_effect=fake_load):
            payload = fetch_yahoo_fundamentals(
                "NVDA",
                errors=errors,
                call_context=ctx,
                max_attempts=5,
            )
        self.assertIsNone(payload)
        self.assertLessEqual(calls["n"], 2)
        self.assertIn(errors[-1]["error_class"], {"timeout", "deadline_exceeded"})


if __name__ == "__main__":
    unittest.main()
