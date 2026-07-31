"""Financial Grounding Layer — issuer SEC gap-fill for AST-computable facts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.tools import (
    _merge_document_market_data_with_live,
    has_computable_fundamentals,
    retrieve_company_payload,
)


class FinancialGroundingMergeTests(unittest.TestCase):
    def test_document_wins_overlap_live_fills_gaps(self) -> None:
        merged, filled = _merge_document_market_data_with_live(
            {"revenue": 10.0},
            {"revenue": 99.0, "ebitda": 4.0, "r_and_d": 1.0, "operating_income": 3.0},
        )
        self.assertEqual(merged["revenue"], 10.0)
        self.assertEqual(merged["ebitda"], 4.0)
        self.assertEqual(set(filled), {"ebitda", "r_and_d", "operating_income"})
        self.assertTrue(has_computable_fundamentals({"market_data": merged}))


class FinancialGroundingRetrieveTests(unittest.TestCase):
    def test_partial_upload_gap_fills_from_issuer_sec(self) -> None:
        """Partial PDF hints must not block issuer SEC companyfacts fill."""
        docs = [
            {
                "detected_companies": ["NVIDIA"],
                "metric_hints": {"revenue": 10.0},  # incomplete AST set
                "excerpt": "NVIDIA filing excerpt.",
                "text": "NVIDIA Form 10-K narrative.",
            }
        ]
        sec = {
            "market_data": {
                "revenue": 130.5,
                "ebitda": 86.0,
                "operating_income": 81.5,
                "r_and_d": 12.9,
            },
            "structured_source": "sec_companyfacts",
            "supply_chain": {"risk_level": "unknown", "signals": []},
            "earnings_call_quotes": [],
            "fundamentals_meta": {"provider": "sec_edgar", "symbol": "NVDA"},
        }
        with patch(
            "lumenfin.sec_fundamentals.fetch_sec_companyfacts_fundamentals",
            return_value=sec,
        ) as mock_sec:
            payload = retrieve_company_payload(
                "NVIDIA",
                document_contexts=docs,
                allow_sample_data=False,
                ticker="NVDA",
                fetch_sec_fundamentals=True,
                fetch_live_fundamentals=False,
            )
        mock_sec.assert_called_once()
        self.assertEqual(payload["structured_source"], "sec_companyfacts")
        self.assertEqual(payload["market_data"]["revenue"], 10.0)  # document wins
        self.assertEqual(payload["market_data"]["ebitda"], 86.0)
        self.assertTrue(has_computable_fundamentals(payload))
        meta = payload["fundamentals_meta"]
        self.assertTrue(meta.get("live_fallback_used"))
        self.assertEqual(meta.get("grounding_layer"), "issuer_sec_gap_fill")
        self.assertIn("ebitda", meta.get("sec_filled_keys") or [])

    def test_complete_document_still_skips_sec(self) -> None:
        docs = [
            {
                "detected_companies": ["NVIDIA"],
                "metric_hints": {"revenue": 130.5, "ebitda": 111.0, "r_and_d": 22.0},
                "excerpt": "full",
                "text": "full",
            }
        ]
        with patch(
            "lumenfin.sec_fundamentals.fetch_sec_companyfacts_fundamentals",
            return_value={"market_data": {"revenue": 1.0}},
        ) as mock_sec:
            payload = retrieve_company_payload(
                "NVIDIA",
                document_contexts=docs,
                allow_sample_data=False,
                fetch_sec_fundamentals=True,
            )
        mock_sec.assert_not_called()
        self.assertEqual(payload["structured_source"], "document_extracted")
        self.assertEqual(payload["fundamentals_meta"].get("grounding_layer"), "document_ast_complete")
        self.assertEqual(payload["market_data"]["revenue"], 130.5)

    def test_document_complete_tags_fiscal_year_from_filename(self) -> None:
        docs = [
            {
                "detected_companies": ["Microsoft"],
                "filename": "msft_fy2024_10k_long_excerpt.pdf",
                "metric_hints": {
                    "revenue": 245.1,
                    "operating_income": 109.4,
                    "r_and_d": 29.5,
                },
                "excerpt": "Microsoft revenue",
                "text": "Microsoft revenue operating income r_and_d",
            }
        ]
        payload = retrieve_company_payload(
            "Microsoft",
            document_contexts=docs,
            allow_sample_data=False,
            prefer_fiscal_year=2024,
        )
        meta = payload["fundamentals_meta"]
        self.assertEqual(meta.get("grounding_layer"), "document_ast_complete")
        self.assertEqual(meta.get("fiscal_year"), 2024)
        self.assertEqual(meta.get("fiscal_year_source"), "upload_filename")
        self.assertEqual(meta.get("period_alignment"), "exact")
        self.assertEqual(meta.get("requested_fiscal_year"), 2024)
        self.assertNotEqual(meta.get("period_end"), "2024-06-30")
        self.assertNotEqual(meta.get("period_end_source"), "issuer_convention_hint")
        self.assertIn("June", str(meta.get("fiscal_calendar_hint") or ""))
        self.assertEqual(meta.get("fiscal_calendar_hint_source"), "issuer_convention")

    def test_document_complete_assumes_fy_from_query_when_unlabeled(self) -> None:
        docs = [
            {
                "detected_companies": ["Microsoft"],
                "filename": "msft_upload.pdf",
                "metric_hints": {
                    "revenue": 245.1,
                    "operating_income": 109.4,
                    "r_and_d": 29.5,
                },
                "excerpt": "tables",
                "text": "tables",
            }
        ]
        payload = retrieve_company_payload(
            "Microsoft",
            document_contexts=docs,
            allow_sample_data=False,
            prefer_fiscal_year=2024,
        )
        meta = payload["fundamentals_meta"]
        self.assertEqual(meta.get("fiscal_year"), 2024)
        self.assertEqual(meta.get("fiscal_year_source"), "query")
        self.assertEqual(meta.get("period_alignment"), "assumed_from_query")

    def test_prefer_uploaded_only_blocks_partial_gap_fill(self) -> None:
        docs = [
            {
                "detected_companies": ["NVIDIA"],
                "metric_hints": {"revenue": 10.0},
                "excerpt": "partial",
                "text": "partial",
            }
        ]
        with patch(
            "lumenfin.sec_fundamentals.fetch_sec_companyfacts_fundamentals",
            return_value={"market_data": {"revenue": 1.0, "ebitda": 0.4}},
        ) as mock_sec:
            payload = retrieve_company_payload(
                "NVIDIA",
                document_contexts=docs,
                allow_sample_data=True,
                fetch_sec_fundamentals=True,
                prefer_uploaded_only=True,
            )
        mock_sec.assert_not_called()
        self.assertEqual(payload.get("structured_source"), "none")
        self.assertFalse(has_computable_fundamentals(payload))
        self.assertEqual(
            payload["fundamentals_meta"].get("grounding_layer"),
            "prefer_uploaded_only_refused",
        )


if __name__ == "__main__":
    unittest.main()
