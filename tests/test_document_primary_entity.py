"""Regression: primary issuer resolution vs 10-K peer mentions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.document_entity import resolve_document_entities
from lumenfin.documents import parse_pdf_document
from lumenfin.rag.chunking import chunk_document


class DocumentPrimaryEntityTestCase(unittest.TestCase):
    def test_nvidia_10k_body_mentions_do_not_become_issuers(self) -> None:
        text = """
        UNITED STATES SECURITIES AND EXCHANGE COMMISSION
        Form 10-K Annual Report for NVIDIA Corporation
        Item 1. Business
        We compete with AMD, Intel, Broadcom, and Microsoft in various markets.
        Customers include Amazon, Alphabet, and Tesla.
        Net revenue for fiscal year 2025 was $130,497 million.
        """
        pages = [text]
        entity = resolve_document_entities(
            text=text,
            pages=pages,
            filename="nvda_fy2025_10k_sec.pdf",
        )
        self.assertEqual(entity["detected_companies"], ["NVIDIA"])
        self.assertEqual(entity["issuer_companies"], ["NVIDIA"])
        self.assertEqual(entity["primary_company"]["name"], "NVIDIA")
        self.assertGreaterEqual(entity["primary_company"]["confidence"], 0.8)
        mentioned = set(entity["mentioned_companies"])
        self.assertTrue({"AMD", "Microsoft", "Amazon"} & mentioned)

    def test_peer_table_keeps_both_column_issuers(self) -> None:
        text = "\n".join(
            [
                "Consolidated Peer Fundamentals Table",
                "Metric Apple Microsoft",
                "Revenue 383.3 245.1",
                "EBITDA 130.1 128.4",
            ]
        )
        entity = resolve_document_entities(
            text=text,
            pages=[text],
            filename="apple_msft_fy2025_table.pdf",
        )
        self.assertEqual(set(entity["issuer_companies"]), {"Apple", "Microsoft"})

    def test_real_nvda_fixture_primary_is_nvidia_only(self) -> None:
        path = ROOT / "fixtures" / "e2e_real" / "nvda_fy2025_10k_sec.pdf"
        if not path.exists():
            self.skipTest("NVIDIA E2E fixture missing")
        parsed = parse_pdf_document(path)
        self.assertEqual(parsed["detected_companies"], ["NVIDIA"])
        self.assertEqual(parsed["primary_company"]["name"], "NVIDIA")
        self.assertNotIn("AMD", parsed["detected_companies"])
        self.assertNotIn("Amazon", parsed["detected_companies"])


class FinancialFactChunkTestCase(unittest.TestCase):
    def test_revenue_fact_chunk_contains_number(self) -> None:
        document = {
            "document_id": "aapl",
            "filename": "aapl_fy2024_10k_sec.pdf",
            "detected_companies": ["Apple"],
            "issuer_companies": ["Apple"],
            "pages": [
                "Apple Inc. Form 10-K\n"
                "Total net sales for 2024 were $391,035 million compared to 2023.\n"
                "Research and development expenses were $31,370 million."
            ],
        }
        chunks = chunk_document(document)
        fact_chunks = [c for c in chunks if c.get("financial_fact")]
        self.assertTrue(fact_chunks)
        revenue_facts = [
            c for c in fact_chunks if (c.get("financial_fact") or {}).get("metric") == "revenue"
        ]
        self.assertTrue(revenue_facts)
        blob = " ".join(c["text"] for c in revenue_facts)
        self.assertIn("391035", blob.replace(",", ""))
        self.assertTrue(all(c["companies"] == ["Apple"] for c in revenue_facts))


if __name__ == "__main__":
    unittest.main()
