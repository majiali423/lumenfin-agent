"""SEC physical-attempt and policy ownership tests (no live SEC)."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.provider_resilience import ProviderCallContext
from lumenfin.sec_fundamentals import _get_json_with_retries


def _request(url: str = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json") -> httpx.Request:
    return httpx.Request("GET", url)


class SecUnifiedPolicyTestCase(unittest.TestCase):
    def test_503_then_success_physical_attempts_three(self) -> None:
        request = _request()
        responses = [
            httpx.Response(503, request=request),
            httpx.Response(503, request=request),
            httpx.Response(200, json={"ok": True}, request=request),
        ]
        mock = MagicMock()
        mock.get.side_effect = responses
        ctx = ProviderCallContext.create(request_id="sec-503-ok")
        ctx.sleep = lambda _: None
        payload = _get_json_with_retries(
            mock,
            str(request.url),
            call_context=ctx,
            max_attempts=3,
        )
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(mock.get.call_count, 3)

    def test_always_503_uses_max_attempts(self) -> None:
        request = _request()
        mock = MagicMock()
        mock.get.side_effect = [httpx.Response(503, request=request) for _ in range(5)]
        errors: list[dict] = []
        ctx = ProviderCallContext.create()
        ctx.sleep = lambda _: None
        payload = _get_json_with_retries(
            mock,
            str(request.url),
            errors=errors,
            call_context=ctx,
            max_attempts=3,
        )
        self.assertIsNone(payload)
        self.assertEqual(mock.get.call_count, 3)
        self.assertEqual(errors[-1]["attempts"], 3)
        self.assertEqual(errors[-1]["error_class"], "transient_http")

    def test_http_400_no_retry(self) -> None:
        request = _request()
        mock = MagicMock()
        mock.get.return_value = httpx.Response(400, request=request, text="bad")
        errors: list[dict] = []
        ctx = ProviderCallContext.create()
        ctx.sleep = lambda _: None
        payload = _get_json_with_retries(
            mock,
            str(request.url),
            errors=errors,
            call_context=ctx,
            max_attempts=3,
        )
        self.assertIsNone(payload)
        self.assertEqual(mock.get.call_count, 1)
        self.assertEqual(errors[-1]["error_class"], "client_error")

    def test_http_404_not_found_no_retry(self) -> None:
        request = _request()
        mock = MagicMock()
        mock.get.return_value = httpx.Response(404, request=request, text="missing")
        errors: list[dict] = []
        ctx = ProviderCallContext.create()
        ctx.sleep = lambda _: None
        payload = _get_json_with_retries(
            mock,
            str(request.url),
            allow_404=True,
            errors=errors,
            call_context=ctx,
            max_attempts=3,
        )
        self.assertIsNone(payload)
        self.assertEqual(mock.get.call_count, 1)
        self.assertEqual(errors[-1]["error_class"], "not_found")
        self.assertEqual(errors[-1]["attempts"], 1)

    def test_429_retry_after_respects_deadline(self) -> None:
        request = _request()
        response = httpx.Response(
            429,
            request=request,
            headers={"Retry-After": "30"},
            text="slow down",
        )
        mock = MagicMock()
        mock.get.return_value = response
        errors: list[dict] = []
        sleeps: list[float] = []
        ctx = ProviderCallContext.create(deadline_seconds=0.2)
        ctx.sleep = sleeps.append
        started = time.monotonic()
        payload = _get_json_with_retries(
            mock,
            str(request.url),
            errors=errors,
            call_context=ctx,
            max_attempts=5,
        )
        elapsed = time.monotonic() - started
        self.assertIsNone(payload)
        self.assertLess(elapsed, 2.0)
        self.assertTrue(errors)
        self.assertIn(errors[-1]["error_class"], {"rate_limited", "deadline_exceeded"})
        if sleeps:
            self.assertTrue(all(s <= 0.2 for s in sleeps))

    def test_slow_response_deadline_exceeded(self) -> None:
        request = _request()
        mock = MagicMock()
        errors: list[dict] = []
        sleeps: list[float] = []
        clock = {"t": 1000.0}
        ctx = ProviderCallContext.create(deadline_seconds=0.2)
        ctx.now = lambda: clock["t"]
        ctx.deadline_monotonic = clock["t"] + 0.2

        def sleep(delay: float) -> None:
            sleeps.append(delay)
            clock["t"] += float(delay)

        def slow_get(*_args, **_kwargs):
            clock["t"] += 0.12
            raise httpx.TimeoutException("SEC read timed out", request=request)

        ctx.sleep = sleep
        mock.get.side_effect = slow_get
        payload = _get_json_with_retries(
            mock,
            str(request.url),
            errors=errors,
            call_context=ctx,
            max_attempts=5,
        )
        self.assertIsNone(payload)
        self.assertTrue(errors)
        self.assertIn(errors[-1]["error_class"], {"timeout", "deadline_exceeded"})
        self.assertGreaterEqual(mock.get.call_count, 1)
        self.assertLess(mock.get.call_count, 5)
        if sleeps:
            self.assertTrue(all(s <= 0.25 for s in sleeps))


if __name__ == "__main__":
    unittest.main()
