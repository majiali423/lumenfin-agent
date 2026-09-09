"""RAG chunks must keep original PDF page numbers and unique chunk IDs."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.offline_env import apply_offline_env

apply_offline_env()

from lumenfin.api.app import create_app
from lumenfin.database import RagDocumentRepository
from lumenfin.document_ingest import parse_upload_documents
from lumenfin.documents import expand_document_page_contexts
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.rag.chunking import chunk_document
from lumenfin.rag.indexer import DocumentIndexer
from tests.support.fakes import FakeMarketDataClient
from tests.support.jobs import poll_job
from tests.test_graph_routing import build_test_config

GOLD = json.loads(
    (ROOT / "tests" / "fixtures" / "sec" / "nvda_multipage_oi_fy2024_gold.json").read_text(
        encoding="utf-8"
    )
)
PDF = ROOT / GOLD["source_file"]
QUERY = "Using uploaded files only, what is NVIDIA operating income from the filing facts?"


def _page_chunks(chunks: list[dict], page: int) -> list[dict]:
    return [item for item in chunks if int(item.get("page") or 0) == page]


class RagPageIdentityTestCase(unittest.TestCase):
    def test_full_file_and_expanded_pages_keep_original_page_numbers(self) -> None:
        full = {
            "document_id": "annual-report",
            "filename": "annual.pdf",
            "detected_companies": ["NVIDIA"],
            "pages": [
                "NVIDIA annual report FY2025.",
                "NVIDIA FY2024 operating income was 32.972 billion USD.",
            ],
        }
        expanded = expand_document_page_contexts([full])
        self.assertEqual([item.get("page") for item in expanded], [1, 2])
        reversed_pages = list(reversed(expanded))

        full_chunks = chunk_document(full)
        expanded_chunks = [chunk for doc in expanded for chunk in chunk_document(doc)]
        reversed_chunks = [chunk for doc in reversed_pages for chunk in chunk_document(doc)]

        for chunks in (full_chunks, expanded_chunks, reversed_chunks):
            ids = [item["chunk_id"] for item in chunks]
            self.assertEqual(len(ids), len(set(ids)), ids)
            page2 = _page_chunks(chunks, 2)
            self.assertTrue(page2, chunks)
            self.assertTrue(any("32.972" in str(item.get("text") or "") for item in page2))
            self.assertTrue(any(":p2:" in str(item.get("chunk_id") or "") for item in page2))
            self.assertFalse(any(int(item.get("page") or 0) == 2 and ":p1:" in str(item["chunk_id"]) for item in chunks))

    def test_blank_middle_page_does_not_compress_page_numbers(self) -> None:
        full = {
            "document_id": "annual-report",
            "filename": "annual.pdf",
            "detected_companies": ["NVIDIA"],
            "pages": [
                "NVIDIA annual report FY2025.",
                "   ",
                "NVIDIA FY2024 operating income was 32.972 billion USD.",
            ],
        }
        chunks = chunk_document(full)
        page3 = _page_chunks(chunks, 3)
        self.assertTrue(any("32.972" in str(item.get("text") or "") for item in page3), chunks)
        self.assertFalse(any("32.972" in str(item.get("text") or "") and int(item.get("page") or 0) == 2 for item in chunks))
        expanded = expand_document_page_contexts([full])
        self.assertEqual([item.get("page") for item in expanded], [1, 3])
        expanded_chunks = [chunk for doc in expanded for chunk in chunk_document(doc)]
        self.assertTrue(any(int(item.get("page") or 0) == 3 and "32.972" in str(item.get("text") or "") for item in expanded_chunks))

    def test_parse_upload_then_chunk_keeps_page2_identity(self) -> None:
        contexts = parse_upload_documents(PDF)
        self.assertEqual([item.get("page") for item in contexts], [1, 2])
        chunks = [chunk for context in contexts for chunk in chunk_document(context)]
        counts = Counter(item["chunk_id"] for item in chunks)
        self.assertFalse({key: value for key, value in counts.items() if value > 1})
        page2 = _page_chunks(chunks, 2)
        self.assertTrue(any("32.972" in str(item.get("text") or "") for item in page2))
        self.assertTrue(all(":p2:" in str(item["chunk_id"]) for item in page2))

    def test_indexer_sqlite_ready_and_duplicate_upload_does_not_add_chunks(self) -> None:
        root = ROOT / "test_artifacts" / f"rag-page-{uuid4().hex[:8]}"
        config = build_test_config(root)
        repository = RagDocumentRepository(config.database_url, db_path=config.db_path)
        indexer = DocumentIndexer(rag_store=None, repository=repository, tenant_id="test-tenant")
        first = indexer.index_file(PDF)
        self.assertEqual(first.get("status"), "ready", first.get("error"))
        self.assertGreaterEqual(int(first.get("chunk_count") or 0), 2)
        stored = repository.list_chunks(tenant_id="test-tenant", source_document_ids=[first["document_id"]])
        pages = {int(item.get("page") or 0) for item in stored}
        self.assertIn(1, pages)
        self.assertIn(2, pages)
        self.assertTrue(any("32.972" in str(item.get("text") or "") and int(item.get("page") or 0) == 2 for item in stored))
        ids = [item["chunk_id"] for item in stored]
        self.assertEqual(len(ids), len(set(ids)), ids)
        second = indexer.index_file(PDF)
        self.assertEqual(second.get("status"), "skipped_duplicate")
        again = repository.list_chunks(tenant_id="test-tenant", source_document_ids=[first["document_id"]])
        self.assertEqual(len(again), len(stored))
        repository.engine.dispose()


class MultipageRagUploadLoopTestCase(unittest.TestCase):
    def _run(self, *, rag_index_mode: str) -> tuple[dict, dict, list[dict]]:
        root = ROOT / "test_artifacts" / f"upload-rag-{rag_index_mode}-{uuid4().hex[:8]}"
        config = replace(
            build_test_config(root),
            api_key=None,
            rag_enabled=True,
            rag_index_mode=rag_index_mode,
            embedding_provider="deterministic",
        )
        app = create_app(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        with TestClient(app) as client:
            submitted = client.post(
                "/api/v1/jobs/upload",
                data={"query": QUERY, "thread_id": f"upload-rag-{rag_index_mode}", "export_artifacts": "false"},
                files={"files": (PDF.name, PDF.read_bytes(), "application/pdf")},
            )
            self.assertEqual(submitted.status_code, 202, submitted.text)
            public = poll_job(client, submitted.json()["job_id"])
        from lumenfin.database import JobRepository

        stored = JobRepository(config.database_url, db_path=config.db_path).get_job(
            submitted.json()["job_id"], tenant_id="test-tenant"
        )
        state = (stored or {}).get("result") or {}
        chunks = RagDocumentRepository(config.database_url, db_path=config.db_path).list_chunks(
            tenant_id="test-tenant"
        )
        return public, state, chunks

    def test_sync_on_run_multipage_upload_retrieves_page2_operating_income(self) -> None:
        public, state, chunks = self._run(rag_index_mode="sync_on_run")
        self._assert_page2_bound(public, state, chunks)

    def test_async_on_upload_multipage_indexes_page2_before_retrieval(self) -> None:
        public, state, chunks = self._run(rag_index_mode="async_on_upload")
        self._assert_page2_bound(public, state, chunks)
        self.assertTrue(chunks, "async_on_upload must persist chunks")
        stats = state.get("rag_index_stats") or {}
        self.assertTrue(stats.get("search_only"), stats)
        self.assertEqual(stats.get("index_status"), "ready_search_only")

    def _assert_page2_bound(self, public: dict, state: dict, chunks: list[dict]) -> None:
        self.assertEqual(public.get("status"), "completed", public.get("error_message"))
        nvda = (state.get("retrieved_docs") or {}).get("NVIDIA") or {}
        market = nvda.get("market_data") or {}
        self.assertAlmostEqual(float(market.get("operating_income") or 0), 32.972, places=3)
        prov = (nvda.get("fundamental_provenance") or {}).get("operating_income") or {}
        self.assertEqual(prov.get("period"), "FY2024")
        self.assertIn("#p2", str(prov.get("citation") or ""))
        report = str(state.get("final_report") or "")
        self.assertIn("32.97", report)
        self.assertIn("#p2", report)
        if chunks:
            self.assertTrue(
                any("32.972" in str(item.get("text") or "") and int(item.get("page") or 0) == 2 for item in chunks)
            )
            digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
            self.assertEqual(digest, GOLD["source_sha256"])


if __name__ == "__main__":
    unittest.main()
