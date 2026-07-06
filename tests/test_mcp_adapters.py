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

from adapters.doc_search import search_research_documents
from adapters.finance_db import query_company_metrics
from adapters.safe_calc import compute_ratio
from lumenfin.tools import safe_execute_formula


class McpAdapterTestCase(unittest.TestCase):
    def test_safe_calc_matches_lumenfin_core(self) -> None:
        variables = {"r_and_d_2025": 33.4, "revenue_2025": 412.0}
        formula = "r_and_d_2025 / revenue_2025"
        adapter = compute_ratio(formula, variables)
        core = safe_execute_formula(formula, variables)
        self.assertEqual(adapter["result"], core)
        self.assertEqual(adapter["engine"], "lumenfin.tools.safe_execute_formula")

    def test_finance_db_reads_sample_data(self) -> None:
        payload = query_company_metrics("NVIDIA", ["revenue_2025"])
        self.assertTrue(payload["found"])
        self.assertEqual(payload["source"], "lumenfin.sample_financial_data")
        self.assertEqual(payload["metrics"]["revenue_2025"]["value"], 130.5)

    def test_document_search_returns_hits(self) -> None:
        payload = search_research_documents("supply chain risk", top_k=1)
        self.assertGreaterEqual(len(payload["hits"]), 1)
        self.assertIn("doc", payload["hits"][0])


if __name__ == "__main__":
    unittest.main()
