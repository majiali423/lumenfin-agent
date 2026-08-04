"""Expose SEC nested/custom retry ownership before unified policy routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.sec_fundamentals import _get_json_with_retries


class SecRetryOwnershipGapTestCase(unittest.TestCase):
    def test_sec_transient_failures_use_exactly_max_attempts(self) -> None:
        request = httpx.Request("GET", "https://data.sec.gov/api/xbrl/companyfacts/CIK0001045810.json")
        responses = [
            httpx.Response(503, request=request),
            httpx.Response(503, request=request),
            httpx.Response(503, request=request),
        ]
        mock = MagicMock()
        mock.get.side_effect = responses
        errors: list[dict] = []
        with patch("lumenfin.provider_resilience.time.sleep", return_value=None):
            payload = _get_json_with_retries(
                mock,
                str(request.url),
                errors=errors,
                provider="sec_edgar",
                symbol="NVDA",
            )
        self.assertIsNone(payload)
        self.assertEqual(mock.get.call_count, 3)
        self.assertEqual(errors[-1]["attempts"], 3)


if __name__ == "__main__":
    unittest.main()
