"""Tests for SEC HTML table facts and consolidated ranking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.chunking import chunk_document
from lumenfin.rag.hybrid_retriever import _hits_from_scored_chunks
from lumenfin.sec_html import parse_sec_html_document, tables_to_financial_facts


class SecHtmlFactRankingTestCase(unittest.TestCase):
    def test_tables_to_facts_marks_consolidated_total(self) -> None:
        tables = [
            {
                "rows": [
                    ["", "2024", "2023"],
                    ["Total net sales", "391,035", "383,285"],
                    ["iPhone", "201,183", "200,583"],
                ]
            }
        ]
        facts = tables_to_financial_facts(
            tables,
            issuers=["Apple"],
            document_id="aapl",
            filename="aapl.htm",
        )
        revenues = [f for f in facts if (f.get("financial_fact") or {}).get("metric") == "revenue"]
        self.assertTrue(revenues)
        scopes = {(f.get("financial_fact") or {}).get("scope") for f in revenues}
        self.assertIn("consolidated", scopes)

    def test_ranking_prefers_consolidated_over_segment(self) -> None:
        document = {
            "document_id": "aapl",
            "filename": "aapl.htm",
            "detected_companies": ["Apple"],
            "issuer_companies": ["Apple"],
            "pages": [
                "Consolidated Statements of Operations\n"
                "Total net sales 2024 were 391035 million.\n"
                "Net sales by category: iPhone net sales 201183 million."
            ],
            "tables": [
                {
                    "rows": [
                        ["", "2024"],
                        ["Total net sales", "391,035"],
                        ["iPhone", "201,183"],
                    ]
                }
            ],
        }
        hits = _hits_from_scored_chunks(
            chunk_document(document),
            company="Apple",
            query="Apple FY2024 revenue / total net sales",
            top_k=3,
        )
        self.assertTrue(hits)
        top = hits[0].get("financial_fact") or {}
        self.assertEqual(top.get("metric"), "revenue")
        self.assertIn(str(top.get("value")), {"391035", "391,035"})
        self.assertNotEqual(str(top.get("value")).replace(",", ""), "201183")

    def test_real_aapl_html_fixture_parses_tables(self) -> None:
        path = ROOT / "fixtures" / "e2e_real" / "aapl-20240928.htm"
        if not path.exists():
            self.skipTest("Apple HTML fixture missing")
        doc = parse_sec_html_document(path)
        self.assertEqual(doc.get("detected_companies"), ["Apple"])
        self.assertGreater(int(doc.get("html_table_count") or 0), 10)
        self.assertEqual(doc.get("source_type"), "sec_html")


if __name__ == "__main__":
    unittest.main()
