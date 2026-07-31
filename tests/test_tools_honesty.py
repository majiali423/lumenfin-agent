from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.tools import (
    analyze_sentiment,
    quotes_are_weak_for_llm,
    retrieve_company_payload,
    summarize_document_context,
    validate_report,
)


class ToolsHonestyTestCase(unittest.TestCase):
    def test_narrative_only_upload_is_not_document_extracted(self) -> None:
        docs = [
            {
                "detected_companies": ["Oracle"],
                "metric_hints": {},
                "excerpt": "Oracle strategy narrative without tables.",
                "text": "Oracle strategy narrative without tables.",
            }
        ]
        payload = retrieve_company_payload(
            "Oracle",
            document_contexts=docs,
            allow_sample_data=False,
            prefer_uploaded_only=True,
        )
        self.assertEqual(payload["structured_source"], "none")
        self.assertFalse(payload.get("market_data"))
        self.assertEqual(
            payload["earnings_call_quotes"],
            ["Oracle strategy narrative without tables."],
        )
        self.assertNotIn("operating_income", payload.get("market_data") or {})

    def test_does_not_invent_operating_income_from_ebitda(self) -> None:
        docs = [
            {
                "detected_companies": ["Apple"],
                "metric_hints": {"revenue": 400.0, "ebitda": 120.0},
                "metric_hint_meta": {
                    "revenue": {
                        "normalized_value": 400.0,
                        "normalized_unit": "billion_usd",
                        "currency": "USD",
                        "confidence": "high",
                        "normalization_source": "table_caption",
                        "is_normalized": True,
                    },
                    "ebitda": {
                        "normalized_value": 120.0,
                        "normalized_unit": "billion_usd",
                        "currency": "USD",
                        "confidence": "high",
                        "normalization_source": "table_caption",
                        "is_normalized": True,
                    },
                },
                "excerpt": "Apple revenue 400 EBITDA 120.",
                "text": "Apple revenue 400 EBITDA 120.",
            }
        ]
        payload = retrieve_company_payload("Apple", document_contexts=docs, allow_sample_data=False)
        self.assertEqual(payload["structured_source"], "document_extracted")
        self.assertEqual(payload["market_data"]["revenue"], 400.0)
        self.assertEqual(payload["market_data"]["ebitda"], 120.0)
        self.assertIsNone(payload["market_data"].get("operating_income"))

    def test_empty_sentiment_is_neutral_not_bullish(self) -> None:
        result = analyze_sentiment([])
        self.assertEqual(result["label"], "neutral")
        self.assertEqual(result["positive_hits"], 0)
        self.assertEqual(result["caution_hits"], 0)

    def test_chinese_placeholder_is_weak_quote(self) -> None:
        self.assertTrue(quotes_are_weak_for_llm(["文档已上传，请基于 PDF 内容进行分析。"]))

    def test_summarize_skips_docs_without_company_detection(self) -> None:
        docs = [
            {
                "document_id": "orphan",
                "filename": "orphan.pdf",
                "excerpt": "Generic filing text.",
                "detected_companies": [],
                "metric_hints": {"revenue": 999.0},
            }
        ]
        summary = summarize_document_context(docs, "Apple")
        self.assertEqual(summary["source_documents"], [])
        self.assertEqual(summary["metric_hints"], {})

    def test_validate_report_accepts_english_markers(self) -> None:
        report = "Summary.\nData sources: SEC.\nDisclaimer: not investment advice."
        self.assertEqual(validate_report(report), [])


if __name__ == "__main__":
    unittest.main()
