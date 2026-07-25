from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.documents import extract_columnar_peer_metrics, merge_per_company_metric_hints


class ColumnarTableMetricsTestCase(unittest.TestCase):
    def test_vertical_peer_table_assigns_distinct_columns(self) -> None:
        text = """
Consolidated Peer Fundamentals Table
Metric
Apple
Microsoft
Revenue
383.3
245.1
EBITDA
130.1
128.4
R&D Expense
31.4
29.5
"""
        out = extract_columnar_peer_metrics(text, ["Apple", "Microsoft"])
        self.assertEqual(out["Apple"]["revenue"], 383.3)
        self.assertEqual(out["Microsoft"]["revenue"], 245.1)
        self.assertEqual(out["Apple"]["ebitda"], 130.1)
        self.assertEqual(out["Microsoft"]["ebitda"], 128.4)
        self.assertEqual(out["Apple"]["r_and_d"], 31.4)
        self.assertEqual(out["Microsoft"]["r_and_d"], 29.5)

    def test_merge_prefers_columnar_over_header_window_noise(self) -> None:
        text = """
Metric
Apple
Microsoft
Revenue
383.3
245.1
EBITDA
130.1
128.4
R&D Expense
31.4
29.5
Apple supply chain risk remains medium.
Microsoft Azure remains a growth engine.
"""
        merged = merge_per_company_metric_hints(text, ["Apple", "Microsoft"])
        self.assertNotEqual(merged["Apple"]["revenue"], merged["Microsoft"]["revenue"])
        self.assertEqual(merged["Microsoft"]["revenue"], 245.1)


if __name__ == "__main__":
    unittest.main()
