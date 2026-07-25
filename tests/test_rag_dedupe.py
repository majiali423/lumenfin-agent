"""Tests for cross-company RAG hit dedupe."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.dedupe import dedupe_cross_company_rag_hits


class RagDedupeTestCase(unittest.TestCase):
    def test_shared_citation_prefers_mentioned_company(self) -> None:
        shared = {
            "chunk_id": "doc:p1:c0",
            "citation": "table.pdf#p1",
            "text": "Apple supply chain risk remains medium.",
            "companies": ["Apple", "Microsoft"],
            "score": 0.5,
        }
        evidence = {
            "Apple": [dict(shared)],
            "Microsoft": [dict(shared)],
        }
        deduped = dedupe_cross_company_rag_hits(evidence)
        self.assertEqual(len(deduped["Apple"]), 1)
        self.assertEqual(len(deduped["Microsoft"]), 0)

    def test_company_exclusive_chunks_always_kept(self) -> None:
        evidence = {
            "Apple": [
                {
                    "chunk_id": "a1",
                    "citation": "table.pdf#p1",
                    "text": "Apple Revenue: 383.3 (peer table).",
                    "companies": ["Apple"],
                    "score": 0.9,
                }
            ],
            "Microsoft": [
                {
                    "chunk_id": "m1",
                    "citation": "table.pdf#p1",
                    "text": "Microsoft Revenue: 245.1 (peer table).",
                    "companies": ["Microsoft"],
                    "score": 0.9,
                }
            ],
        }
        deduped = dedupe_cross_company_rag_hits(evidence)
        self.assertEqual(len(deduped["Apple"]), 1)
        self.assertEqual(len(deduped["Microsoft"]), 1)


if __name__ == "__main__":
    unittest.main()
