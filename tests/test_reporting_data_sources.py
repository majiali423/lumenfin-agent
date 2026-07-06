from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.reporting import build_data_sources, build_run_manifest


class ReportingDataSourcesTestCase(unittest.TestCase):
    def test_sample_db_structured_source(self) -> None:
        sources = build_data_sources(
            {"companies": ["NVIDIA"], "document_contexts": []},
            llm_backend="local-fallback",
        )
        self.assertEqual(sources["structured"], "sample_db")
        self.assertEqual(sources["rag"], "skipped")
        self.assertFalse(sources["structured_uploaded"])

    def test_uploaded_json_structured_source(self) -> None:
        sources = build_data_sources(
            {
                "companies": ["CustomCo"],
                "document_contexts": [
                    {
                        "filename": "CustomCo_metrics.json",
                        "source_type": "structured_json",
                    }
                ],
            }
        )
        self.assertEqual(sources["structured"], "uploaded_json")
        self.assertTrue(sources["structured_uploaded"])

    def test_rag_milvus_when_hits(self) -> None:
        sources = build_data_sources(
            {
                "companies": ["NVIDIA"],
                "document_contexts": [{"filename": "nvda.pdf"}],
                "rag_evidence": {"NVIDIA": [{"page": 1}]},
                "rag_index_stats": {"chunks_indexed": 3},
            },
            rag_enabled=True,
        )
        self.assertEqual(sources["rag"], "milvus_hybrid")
        self.assertTrue(sources["pdf_uploaded"])

    def test_market_ok_when_price_present(self) -> None:
        sources = build_data_sources(
            {
                "market_snapshots": {
                    "NVIDIA": {"provider": "yahoo", "current_price": 120.5}
                }
            },
            market_provider="yahoo",
        )
        self.assertTrue(sources["market_ok"])
        self.assertEqual(sources["market"], "yahoo")

    def test_manifest_includes_data_sources(self) -> None:
        manifest = build_run_manifest(
            {"companies": ["NVIDIA"], "workflow_status": "completed"},
            thread_id="t1",
            llm_backend="local-fallback",
        )
        self.assertIn("data_sources", manifest)
        self.assertEqual(manifest["data_sources"]["structured"], "sample_db")


if __name__ == "__main__":
    unittest.main()
