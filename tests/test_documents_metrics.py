from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.documents import (
    _first_metric_number,
    detect_statement_scale,
    extract_metric_hints_for_company,
    normalize_metric_hints_to_billion_usd,
)
from lumenfin.tools import summarize_document_context


class DocumentMetricIsolationTestCase(unittest.TestCase):
    def test_extract_metric_hints_for_company_does_not_leak_previous_company_line(self) -> None:
        text = (
            "Microsoft revenue 500 billion USD.\n"
            "Apple management discussion and outlook.\n"
            "No explicit Apple revenue number in this section.\n"
        )
        hints = extract_metric_hints_for_company(text, "Apple")
        self.assertEqual(hints, {})

    def test_summarize_document_context_prefers_company_scoped_metric_hints(self) -> None:
        docs = [
            {
                "document_id": "peer-1",
                "filename": "peer.pdf",
                "excerpt": "Microsoft and Apple peer review.",
                "detected_companies": ["Microsoft", "Apple"],
                "metric_hints": {"revenue": 500.0},
                "per_company_metric_hints": {
                    "Microsoft": {"revenue": 500.0},
                    "Apple": {"revenue": 300.0},
                },
            }
        ]
        summary = summarize_document_context(docs, "Apple")
        self.assertEqual(summary["metric_hints"]["revenue"], 300.0)


class MetricNumberParsingTestCase(unittest.TestCase):
    def test_first_metric_number_skips_percentage_and_uses_absolute_value(self) -> None:
        context = " revenue grew 5% to 130 billion USD in FY2025."
        value = _first_metric_number(context)
        self.assertEqual(value, 130.0)

    def test_first_metric_number_scales_million_to_billion_units(self) -> None:
        context = " revenue reached 500 million USD."
        value = _first_metric_number(context)
        self.assertEqual(value, 0.5)

    def test_first_metric_number_scales_unitless_sec_millions(self) -> None:
        context = " total revenue 245,122 "
        value = _first_metric_number(context)
        self.assertEqual(value, 245.1)

    def test_detect_statement_scale_in_millions(self) -> None:
        self.assertEqual(
            detect_statement_scale("Consolidated Statements of Income (In millions)"),
            "million",
        )

    def test_normalize_hints_scales_sec_millions_and_rejects_absurd(self) -> None:
        hints = normalize_metric_hints_to_billion_usd(
            {"revenue": 245122.0, "operating_income": 109433.0, "r_and_d": 29510.0},
            text="(In millions)",
        )
        self.assertAlmostEqual(hints["revenue"], 245.122, places=3)
        self.assertAlmostEqual(hints["operating_income"], 109.433, places=3)
        self.assertAlmostEqual(hints["r_and_d"], 29.51, places=2)

    def test_normalize_hints_keeps_already_billion_scale(self) -> None:
        hints = normalize_metric_hints_to_billion_usd(
            {"revenue": 391.0, "ebitda": 134.7},
            text="Apple revenue was 391.0 billion USD.",
        )
        self.assertEqual(hints["revenue"], 391.0)
        self.assertEqual(hints["ebitda"], 134.7)

    def test_normalize_hints_drops_still_implausible_revenue(self) -> None:
        hints = normalize_metric_hints_to_billion_usd(
            {"revenue": 5_000_000_000.0},
            text="",
        )
        self.assertNotIn("revenue", hints)

    def test_normalize_hints_drops_barely_over_ceiling_without_false_rescue(self) -> None:
        hints = normalize_metric_hints_to_billion_usd(
            {"revenue": 999.0},
            text="NVIDIA FY2025 revenue was 999 billion USD.",
        )
        self.assertNotIn("revenue", hints)


if __name__ == "__main__":
    unittest.main()
