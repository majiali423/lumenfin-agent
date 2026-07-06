from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MCP_LAYER = ROOT / "mcp_layer"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(MCP_LAYER) not in sys.path:
    sys.path.insert(0, str(MCP_LAYER))

from adapters.doc_context import load_research_document_contexts, resolve_company_hint
from adapters.doc_search import search_research_documents


class McpDocSearchTestCase(unittest.TestCase):
    def test_load_research_document_contexts(self) -> None:
        contexts = load_research_document_contexts()
        self.assertGreaterEqual(len(contexts), 2)
        self.assertTrue(any("Apple" in context.get("detected_companies", []) for context in contexts))

    def test_resolve_company_hint(self) -> None:
        self.assertEqual(resolve_company_hint("Apple supply chain risk"), "Apple")

    def test_keyword_search_mode(self) -> None:
        payload = search_research_documents("supply chain risk", top_k=1, mode="keyword")
        self.assertEqual(payload["retrieval_mode"], "keyword")
        self.assertGreaterEqual(len(payload["hits"]), 1)


if __name__ == "__main__":
    unittest.main()
