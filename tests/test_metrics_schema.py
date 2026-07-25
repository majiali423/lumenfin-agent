from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.metrics_schema import get_fundamental, normalize_market_data
from lumenfin.planning import build_query_plan


class MetricsSchemaTestCase(unittest.TestCase):
    def test_get_fundamental_accepts_legacy_and_canonical(self) -> None:
        self.assertEqual(get_fundamental({"revenue_2025": 10.0}, "revenue"), 10.0)
        self.assertEqual(get_fundamental({"revenue": 11.0}, "revenue_2025"), 11.0)

    def test_normalize_collapses_legacy_keys(self) -> None:
        normalized = normalize_market_data({"revenue_2025": 10.0, "ebitda_2025": 4.0, "note": "x"})
        self.assertEqual(normalized["revenue"], 10.0)
        self.assertEqual(normalized["ebitda"], 4.0)
        self.assertNotIn("revenue_2025", normalized)
        self.assertEqual(normalized["note"], "x")

    def test_time_range_relative_phrases(self) -> None:
        self.assertNotIn("time_range", build_query_plan("分析苹果这两年盈利").missing_fields)
        self.assertNotIn("time_range", build_query_plan("Compare Apple for 2015-2018").missing_fields)
        self.assertIn("time_range", build_query_plan("Clarify Apple profitability").missing_fields)


if __name__ == "__main__":
    unittest.main()
