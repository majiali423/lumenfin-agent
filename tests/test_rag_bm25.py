"""Native Milvus BM25 schema, retrieval, isolation, and degradation tests."""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.embeddings import DeterministicEmbeddingProvider
from lumenfin.rag.hybrid_retriever import HybridEvidenceRetriever
from lumenfin.rag.milvus_store import BM25_SCHEMA_VERSION, MilvusRAGStore


class _AlwaysFailEmbedder:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise TimeoutError("embedding timed out")


class _CountingEmbedder:
    def __init__(self, dimension: int = 384) -> None:
        self.inner = DeterministicEmbeddingProvider(dimension=dimension)
        self.dimension = dimension
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return self.inner.embed(texts)


def _chunk(
    *,
    chunk_id: str,
    text: str,
    company: str = "Apple",
    document_id: str = "doc-context",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "filename": f"{document_id}.md",
        "page": 1,
        "text": text,
        "companies": [company],
        "chunk_type": "financial_metric",
        "char_count": len(text),
    }


class MilvusBM25TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "test_artifacts" / f"bm25-{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.uri = str(self.root / "bm25.db")
        self.store = MilvusRAGStore(
            self.uri,
            DeterministicEmbeddingProvider(),
            collection_name="bm25_test",
        )

    def tearDown(self) -> None:
        self.store.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_schema_and_first_bm25_search_support_mixed_financial_terms(self) -> None:
        self.store.index_chunks(
            [
                _chunk(
                    chunk_id="relevant",
                    text="苹果 FY2025 营业利润率与研发强度提高，H100 GPU demand remained strong.",
                ),
                _chunk(
                    chunk_id="noise",
                    text="苹果办公家具和员工餐厅政策更新。",
                ),
            ],
            tenant_id="tenant-a",
            source_document_id="source-a",
        )

        hits = self.store.bm25_search(
            "苹果研发强度 H100 GPU",
            tenant_id="tenant-a",
            source_document_ids=["source-a"],
            top_k=2,
        )

        self.assertTrue(hits)
        self.assertEqual(hits[0]["chunk_id"], "relevant")
        self.assertEqual(hits[0]["retrieval_method"], "bm25")
        health = self.store.health()
        self.assertTrue(health["bm25_enabled"])
        self.assertEqual(health["schema_version"], BM25_SCHEMA_VERSION)

    def test_bm25_search_does_not_call_embedding_provider(self) -> None:
        self.store.close()
        counting = _CountingEmbedder()
        self.store = MilvusRAGStore(self.uri, counting, collection_name="bm25_counting")
        self.store.index_chunks(
            [_chunk(chunk_id="exact", text="Apple form 10-K contains AAPL EBITDA evidence.")],
            tenant_id="tenant-a",
            source_document_id="source-a",
        )
        calls_after_index = counting.calls

        hits = self.store.bm25_search(
            "AAPL 10-K EBITDA",
            tenant_id="tenant-a",
            source_document_ids=["source-a"],
        )

        self.assertTrue(hits)
        self.assertEqual(counting.calls, calls_after_index)

    def test_bm25_respects_tenant_and_source_filters(self) -> None:
        self.store.index_chunks(
            [_chunk(chunk_id="tenant-a", text="RARE_LIQUIDITY_TOKEN Apple evidence.")],
            tenant_id="tenant-a",
            source_document_id="source-a",
        )
        self.store.index_chunks(
            [
                _chunk(
                    chunk_id="tenant-b",
                    text="RARE_LIQUIDITY_TOKEN Microsoft confidential evidence.",
                    company="Microsoft",
                )
            ],
            tenant_id="tenant-b",
            source_document_id="source-b",
        )

        hits = self.store.bm25_search(
            "RARE_LIQUIDITY_TOKEN",
            tenant_id="tenant-a",
            source_document_ids=["source-a"],
            top_k=5,
        )

        self.assertEqual([hit["chunk_id"] for hit in hits], ["tenant-a"])

    def test_replace_removes_stale_bm25_terms(self) -> None:
        self.store.index_chunks(
            [_chunk(chunk_id="old", text="OBSOLETE_MARGIN_TOKEN old filing evidence.")],
            tenant_id="tenant-a",
            source_document_id="source-a",
        )
        self.assertTrue(
            self.store.bm25_search(
                "OBSOLETE_MARGIN_TOKEN",
                tenant_id="tenant-a",
                source_document_ids=["source-a"],
            )
        )

        self.store.index_chunks(
            [_chunk(chunk_id="new", text="CURRENT_SERVICES_TOKEN refreshed filing evidence.")],
            tenant_id="tenant-a",
            source_document_id="source-a",
            replace_existing=True,
        )

        stale = self.store.bm25_search(
            "OBSOLETE_MARGIN_TOKEN",
            tenant_id="tenant-a",
            source_document_ids=["source-a"],
        )
        current = self.store.bm25_search(
            "CURRENT_SERVICES_TOKEN",
            tenant_id="tenant-a",
            source_document_ids=["source-a"],
        )
        self.assertFalse(any(hit["chunk_id"] == "old" for hit in stale))
        self.assertFalse(any("OBSOLETE_MARGIN_TOKEN" in hit["text"] for hit in stale))
        self.assertEqual(current[0]["chunk_id"], "new")

    def test_embedding_failure_degrades_to_bm25_without_local_substitution(self) -> None:
        documents = [
            {
                "document_id": "apple-doc",
                "filename": "apple.md",
                "detected_companies": ["Apple"],
                "pages": ["Apple supply chain risk and H100 GPU demand evidence."],
            }
        ]
        self.store.index_documents(documents, session_id="session-a")
        self.store.embedder = _AlwaysFailEmbedder()
        self.store.clear_query_cache()
        retriever = HybridEvidenceRetriever(self.store, top_k=3)

        hits, meta = retriever.retrieve_for_company_with_meta(
            query="Apple H100 supply chain risk",
            company="Apple",
            session_id="session-a",
            document_contexts=documents,
        )

        self.assertTrue(hits)
        self.assertTrue(meta["degraded"])
        self.assertEqual(meta["mode"], "bm25_only_degraded")
        self.assertGreater(meta["bm25_hits"], 0)
        self.assertTrue(all(hit.get("rag_degraded") for hit in hits))

    def test_bm25_failure_keeps_dense_plus_local_fallback(self) -> None:
        documents = [
            {
                "document_id": "apple-doc",
                "filename": "apple.md",
                "detected_companies": ["Apple"],
                "pages": ["Apple supply chain risk and revenue evidence."],
            }
        ]
        self.store.index_documents(documents, session_id="session-a")

        def _fail_bm25(*args, **kwargs):
            raise RuntimeError("injected BM25 failure")

        self.store.bm25_search = _fail_bm25  # type: ignore[method-assign]
        retriever = HybridEvidenceRetriever(self.store, top_k=3)

        hits, meta = retriever.retrieve_for_company_with_meta(
            query="Apple supply chain risk",
            company="Apple",
            session_id="session-a",
            document_contexts=documents,
        )

        self.assertTrue(hits)
        self.assertTrue(meta["degraded"])
        self.assertEqual(meta["mode"], "hybrid_dense_lexical_fallback_rrf_degraded")
        self.assertEqual(meta["bm25_hits"], 0)
        self.assertGreater(meta["vector_hits"], 0)


class BM25SchemaCompatibilityTestCase(unittest.TestCase):
    def test_dense_only_collection_is_rejected_when_bm25_is_enabled(self) -> None:
        root = ROOT / "test_artifacts" / f"bm25-compat-{uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=True)
        uri = str(root / "legacy.db")
        legacy = MilvusRAGStore(
            uri,
            DeterministicEmbeddingProvider(),
            collection_name="legacy",
            bm25_enabled=False,
        )
        legacy.close()
        try:
            with self.assertRaisesRegex(RuntimeError, "not BM25-compatible"):
                MilvusRAGStore(
                    uri,
                    DeterministicEmbeddingProvider(),
                    collection_name="legacy",
                    bm25_enabled=True,
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
