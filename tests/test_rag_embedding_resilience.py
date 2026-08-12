"""Phase 3: embedding retry, query cache, vector→keyword degrade."""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.embeddings import (
    DeterministicEmbeddingProvider,
    ResilientEmbeddingProvider,
    build_embedding_provider,
)
from lumenfin.rag.hybrid_retriever import HybridEvidenceRetriever
from lumenfin.rag.milvus_store import MilvusRAGStore


class FlakyEmbedder:
    def __init__(self, *, fail_times: int, dimension: int = 384) -> None:
        self._inner = DeterministicEmbeddingProvider(dimension=dimension)
        self.fail_times = fail_times
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls <= self.fail_times:
            request = httpx.Request("POST", "https://example.test/embeddings")
            response = httpx.Response(429, request=request, text="Too Many Requests")
            raise httpx.HTTPStatusError("429", request=request, response=response)
        return self._inner.embed(texts)


class AlwaysFailEmbedder:
    def __init__(self, dimension: int = 384) -> None:
        self._inner = DeterministicEmbeddingProvider(dimension=dimension)
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise TimeoutError("embedding timed out")


class EmbeddingResilienceTestCase(unittest.TestCase):
    def test_resilient_retries_429_then_succeeds(self) -> None:
        flaky = FlakyEmbedder(fail_times=2)
        sleeps: list[float] = []
        provider = ResilientEmbeddingProvider(
            flaky,
            max_retries=3,
            backoff_seconds=0.25,
            sleep=sleeps.append,
            jitter_ratio=0.0,
        )
        vectors = provider.embed(["Apple supply chain risk"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(flaky.calls, 3)
        self.assertEqual(provider.last_attempts, 3)
        self.assertEqual(sleeps, [0.25, 0.5])

    def test_resilient_gives_up_on_persistent_timeout(self) -> None:
        failing = AlwaysFailEmbedder()
        provider = ResilientEmbeddingProvider(
            failing,
            max_retries=2,
            backoff_seconds=0.01,
            sleep=lambda _: None,
        )
        with self.assertRaises(TimeoutError):
            provider.embed(["query"])
        self.assertEqual(failing.calls, 2)
        self.assertIsNotNone(provider.last_error)

    def test_build_provider_wraps_dashscope_name_with_resilience(self) -> None:
        # Deterministic is not wrapped by default.
        plain = build_embedding_provider("deterministic", 384)
        self.assertIsInstance(plain, DeterministicEmbeddingProvider)
        wrapped = build_embedding_provider("deterministic", 384, resilient=True, max_retries=2)
        self.assertIsInstance(wrapped, ResilientEmbeddingProvider)


class QueryCacheAndDegradeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "test_artifacts" / f"rag-resilience-{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.uri = str(self.root / "rag.db")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_query_embedding_cached_across_searches(self) -> None:
        counter = DeterministicEmbeddingProvider()
        # Wrap with a counting shim
        class Counting:
            def __init__(self) -> None:
                self.inner = DeterministicEmbeddingProvider()
                self.calls = 0

            @property
            def dimension(self) -> int:
                return self.inner.dimension

            def embed(self, texts: list[str]) -> list[list[float]]:
                self.calls += 1
                return self.inner.embed(texts)

        counting = Counting()
        store = MilvusRAGStore(self.uri, counting, collection_name="cache_test")
        try:
            docs = [
                {
                    "document_id": "apple-1",
                    "filename": "apple.md",
                    "detected_companies": ["Apple"],
                    "pages": [
                        "Apple revenue grew with services strength.",
                        "Supply chain risk remains a key concern for Apple.",
                    ],
                }
            ]
            store.index_documents(docs, session_id="sess")
            index_calls = counting.calls
            store.prime_query_embedding("Apple supply chain risk")
            self.assertEqual(counting.calls, index_calls + 1)
            store.vector_search("Apple supply chain risk", session_id="sess", companies=["Apple"], top_k=2)
            store.vector_search("Apple supply chain risk", session_id="sess", companies=["Apple"], top_k=2)
            # Same query should not re-embed.
            self.assertEqual(counting.calls, index_calls + 1)
        finally:
            store.close()

    def test_vector_embed_failure_degrades_to_keyword(self) -> None:
        # Index with a healthy embedder, then swap to failing for query-time search.
        healthy = DeterministicEmbeddingProvider()
        store = MilvusRAGStore(self.uri, healthy, collection_name="degrade_test")
        documents = [
            {
                "document_id": "apple-1",
                "filename": "apple.md",
                "detected_companies": ["Apple"],
                "pages": [
                    "Apple revenue 400 billion EBITDA 120 billion.",
                    "Supply chain risk remains elevated for Apple operations.",
                ],
            }
        ]
        try:
            store.index_documents(documents, session_id="sess-deg")
            store.embedder = AlwaysFailEmbedder()
            store.clear_query_cache()
            retriever = HybridEvidenceRetriever(
                store,
                top_k=3,
                degrade_on_vector_error=True,
                min_score=0.05,
            )
            hits, meta = retriever.retrieve_for_company_with_meta(
                query="Apple supply chain risk assessment",
                company="Apple",
                session_id="sess-deg",
                document_contexts=documents,
            )
            self.assertTrue(meta["degraded"])
            self.assertEqual(meta["mode"], "bm25_only_degraded")
            self.assertGreater(meta["bm25_hits"], 0)
            self.assertGreaterEqual(len(hits), 1)
            self.assertTrue(all(hit.get("rag_degraded") for hit in hits))
            self.assertTrue(any("supply chain" in hit["text"].lower() for hit in hits))
        finally:
            store.close()

    def test_min_score_filters_weak_keyword_hits(self) -> None:
        store = MilvusRAGStore(
            self.uri,
            DeterministicEmbeddingProvider(),
            collection_name="score_test",
        )
        documents = [
            {
                "document_id": "misc-1",
                "filename": "misc.md",
                "detected_companies": ["Apple"],
                "pages": ["Unrelated boilerplate paragraph about office furniture."],
            }
        ]
        try:
            store.index_documents(documents, session_id="sess-score")
            # Force keyword-only by breaking vector path with degrade.
            store.embedder = AlwaysFailEmbedder()
            store.clear_query_cache()
            retriever = HybridEvidenceRetriever(
                store,
                top_k=3,
                degrade_on_vector_error=True,
                min_score=0.4,
            )
            hits, meta = retriever.retrieve_for_company_with_meta(
                query="Apple supply chain risk EBITDA margin",
                company="Apple",
                session_id="sess-score",
                document_contexts=documents,
            )
            self.assertTrue(meta["degraded"])
            self.assertEqual(hits, [])
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
