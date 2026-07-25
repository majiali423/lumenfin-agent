from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.documents import _first_metric_number, extract_metric_hints_for_company
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



if __name__ == "__main__":
    unittest.main()
