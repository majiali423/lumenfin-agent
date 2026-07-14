from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.factory import resolve_milvus_uri
from lumenfin.tools import retrieve_company_payload


class PreferDocumentMetricsTestCase(unittest.TestCase):
    def test_document_metrics_win_over_sample_db(self) -> None:
        docs = [
            {
                "detected_companies": ["NVIDIA"],
                "metric_hints": {"revenue": 999.0, "ebitda": 111.0, "r_and_d": 22.0},
                "excerpt": "NVIDIA FY2025 revenue was 999 billion USD.",
                "text": "supply chain risk remains medium for packaging.",
            }
        ]
        payload = retrieve_company_payload("NVIDIA", document_contexts=docs, allow_sample_data=True)
        self.assertEqual(payload["structured_source"], "document_extracted")
        self.assertEqual(payload["market_data"]["revenue_2025"], 999.0)

    def test_sample_used_when_documents_lack_metrics(self) -> None:
        docs = [
            {
                "detected_companies": ["NVIDIA"],
                "metric_hints": {},
                "excerpt": "Narrative only.",
                "text": "Narrative only.",
            }
        ]
        payload = retrieve_company_payload("NVIDIA", document_contexts=docs, allow_sample_data=True)
        self.assertEqual(payload["structured_source"], "sample_db")
        self.assertIn("revenue_2025", payload["market_data"])


class MilvusIsolateTestCase(unittest.TestCase):
    def test_milvus_uri_isolates_by_pid_by_default(self) -> None:
        with mock.patch.dict(os.environ, {"MAS_MILVUS_ISOLATE": "true"}, clear=False):
            uri = resolve_milvus_uri("data/milvus_lite.db")
        self.assertIn(f"_p{os.getpid()}", uri)

    def test_milvus_uri_can_disable_isolate(self) -> None:
        with mock.patch.dict(os.environ, {"MAS_MILVUS_ISOLATE": "false"}, clear=False):
            uri = resolve_milvus_uri("data/milvus_lite.db")
        self.assertEqual(uri, "data/milvus_lite.db")


if __name__ == "__main__":
    unittest.main()
