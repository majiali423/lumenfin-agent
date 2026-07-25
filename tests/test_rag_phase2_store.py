"""Phase 2: Milvus server-mode helpers, filters, delete-before-write."""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.chunking import chunk_document
from lumenfin.rag.embeddings import DeterministicEmbeddingProvider
from lumenfin.rag.milvus_client import (
    build_vector_filter_expr,
    company_match_expr,
    is_milvus_server_uri,
    milvus_backend_kind,
    milvus_client_pool_stats,
    resolve_milvus_uri,
)
from lumenfin.rag.milvus_store import MilvusRAGStore


class MilvusUriHelperTestCase(unittest.TestCase):
    def test_server_uri_never_isolated(self) -> None:
        uri = "http://127.0.0.1:19530"
        self.assertTrue(is_milvus_server_uri(uri))
        self.assertEqual(milvus_backend_kind(uri), "milvus-server")
        self.assertEqual(resolve_milvus_uri(uri, isolate=True), uri)
        self.assertEqual(resolve_milvus_uri(uri, isolate=False), uri)

    def test_lite_uri_isolates_by_default(self) -> None:
        raw = "data/milvus_lite.db"
        resolved = resolve_milvus_uri(raw, isolate=True)
        self.assertIn(f"_p{os.getpid()}", resolved)
        self.assertEqual(resolve_milvus_uri(raw, isolate=False), raw)

    def test_company_and_tenant_filter_expr(self) -> None:
        expr = build_vector_filter_expr(
            tenant_id='t"1',
            source_document_ids=["doc-a", "doc-b"],
            companies=["Apple"],
        )
        self.assertIn('tenant_id == "t\\"1"', expr)
        self.assertIn("source_document_id in ", expr)
        self.assertIn("Apple", expr)
        self.assertIn(company_match_expr(["Apple"]), expr)


class DocumentReplaceAndFilterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "test_artifacts" / f"rag-p2-{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.uri = str(self.root / "rag.db")
        self.embedder = DeterministicEmbeddingProvider()
        self.store = MilvusRAGStore(self.uri, self.embedder, collection_name="phase2")

    def tearDown(self) -> None:
        self.store.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_replace_existing_removes_stale_pages(self) -> None:
        old_chunks = chunk_document(
            {
                "document_id": "doc-apple:v1",
                "filename": "apple.pdf",
                "detected_companies": ["Apple"],
                "pages": [
                    "Apple legacy page mentions obsolete margin figure 11 percent.",
                    "Apple supply chain note from prior filing.",
                ],
            }
        )
        self.store.index_chunks(
            old_chunks,
            tenant_id="tenant-a",
            source_document_id="doc-apple",
            content_hash="hash-old",
            replace_existing=True,
        )
        stale = self.store.vector_search(
            "obsolete margin figure",
            tenant_id="tenant-a",
            source_document_ids=["doc-apple"],
            companies=["Apple"],
            top_k=5,
        )
        self.assertTrue(any("obsolete" in hit["text"].lower() for hit in stale))

        new_chunks = chunk_document(
            {
                "document_id": "doc-apple:v2",
                "filename": "apple.pdf",
                "detected_companies": ["Apple"],
                "pages": [
                    "Apple refreshed filing focuses on services growth and EBITDA expansion.",
                    "Updated supply chain diversification progress for Apple.",
                ],
            }
        )
        stats = self.store.index_chunks(
            new_chunks,
            tenant_id="tenant-a",
            source_document_id="doc-apple",
            content_hash="hash-new",
            replace_existing=True,
        )
        self.assertGreaterEqual(int(stats.get("chunks_indexed") or 0), 1)

        after = self.store.vector_search(
            "obsolete margin figure",
            tenant_id="tenant-a",
            source_document_ids=["doc-apple"],
            companies=["Apple"],
            top_k=5,
        )
        self.assertFalse(any("obsolete" in hit["text"].lower() for hit in after))
        refreshed = self.store.vector_search(
            "services growth EBITDA",
            tenant_id="tenant-a",
            source_document_ids=["doc-apple"],
            companies=["Apple"],
            top_k=5,
        )
        self.assertTrue(any("services" in hit["text"].lower() for hit in refreshed))

    def test_company_filter_excludes_other_issuer(self) -> None:
        apple = chunk_document(
            {
                "document_id": "a1",
                "filename": "apple.md",
                "detected_companies": ["Apple"],
                "pages": ["Apple unique token ALPHA_SUPPLY_CHAIN_RISK in Asia."],
            }
        )
        msft = chunk_document(
            {
                "document_id": "m1",
                "filename": "msft.md",
                "detected_companies": ["Microsoft"],
                "pages": ["Microsoft unique token BETA_CLOUD_GROWTH and Azure demand."],
            }
        )
        self.store.index_chunks(
            apple,
            tenant_id="tenant-a",
            source_document_id="src-apple",
            replace_existing=True,
        )
        self.store.index_chunks(
            msft,
            tenant_id="tenant-a",
            source_document_id="src-msft",
            replace_existing=True,
        )
        apple_hits = self.store.vector_search(
            "ALPHA_SUPPLY_CHAIN_RISK BETA_CLOUD_GROWTH",
            tenant_id="tenant-a",
            companies=["Apple"],
            top_k=5,
        )
        self.assertTrue(apple_hits)
        self.assertTrue(all("Apple" in hit["companies"] for hit in apple_hits))
        self.assertFalse(any("Microsoft" in hit["companies"] for hit in apple_hits))

    def test_concurrent_searches_share_collection(self) -> None:
        chunks = chunk_document(
            {
                "document_id": "c1",
                "filename": "apple.md",
                "detected_companies": ["Apple"],
                "pages": [
                    "Apple revenue grew with strong services.",
                    "Apple supply chain remains a diligence focus.",
                ],
            }
        )
        self.store.index_chunks(
            chunks,
            tenant_id="tenant-a",
            source_document_id="src-1",
            replace_existing=True,
        )

        def _search(_: int) -> int:
            hits = self.store.vector_search(
                "Apple supply chain",
                tenant_id="tenant-a",
                source_document_ids=["src-1"],
                companies=["Apple"],
                top_k=3,
            )
            return len(hits)

        with ThreadPoolExecutor(max_workers=4) as pool:
            counts = list(pool.map(_search, range(8)))
        self.assertTrue(all(count >= 1 for count in counts))

    def test_health_reports_lite_backend(self) -> None:
        health = self.store.health()
        self.assertEqual(health["backend"], "milvus-lite")
        self.assertTrue(health["ready"])


class SharedClientPoolTestCase(unittest.TestCase):
    def test_server_uri_pool_stats_are_tracked(self) -> None:
        # We cannot assume a live Milvus server; only verify URI classification + pool API.
        self.assertEqual(milvus_backend_kind("https://milvus.example:19530"), "milvus-server")
        before = milvus_client_pool_stats()["size"]
        self.assertGreaterEqual(before, 0)


if __name__ == "__main__":
    unittest.main()
