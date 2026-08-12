"""Offline contract, resilience, and integration tests for qwen3 rerank."""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.hybrid_retriever import HybridEvidenceRetriever
from lumenfin.rag.rerank import (
    DashScopeQwen3Reranker,
    FallbackReranker,
    LexicalReranker,
    build_reranker,
)
from scripts.run_rerank_eval import _qwen3_telemetry_complete, _telemetry_item


def _hits() -> list[dict]:
    return [
        {
            "chunk_id": "wrong-period",
            "text": "Apple FY2024 revenue was 391 billion dollars.",
            "chunk_type": "financial_metric",
            "score": 0.9,
            "retrieval_method": "hybrid_dense_bm25_rrf",
        },
        {
            "chunk_id": "correct-period",
            "text": "Apple FY2025 revenue was 416 billion dollars.",
            "chunk_type": "financial_metric",
            "score": 0.7,
            "retrieval_method": "hybrid_dense_bm25_rrf",
        },
        {
            "chunk_id": "wrong-company",
            "text": "Microsoft FY2025 revenue was 282 billion dollars.",
            "chunk_type": "financial_metric",
            "score": 0.8,
            "retrieval_method": "hybrid_dense_bm25_rrf",
        },
    ]


def _success_response(request: httpx.Request, indexes: list[int]) -> httpx.Response:
    results = [
        {"index": index, "relevance_score": round(0.95 - rank * 0.1, 4)}
        for rank, index in enumerate(indexes)
    ]
    return httpx.Response(
        200,
        request=request,
        json={
            "object": "list",
            "results": results,
            "model": "qwen3-rerank",
            "id": "req-test-1",
            "usage": {"total_tokens": 123},
        },
    )


class DashScopeQwen3RerankerTestCase(unittest.TestCase):
    def _provider(self, handler, **kwargs) -> DashScopeQwen3Reranker:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return DashScopeQwen3Reranker(
            api_key="test-key",
            base_url="https://workspace.example.test/compatible-api/v1",
            client=client,
            jitter_ratio=0.0,
            **kwargs,
        )

    def test_success_maps_indexes_without_returning_provider_documents(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content.decode("utf-8")))
            return _success_response(request, [1, 0])

        provider = self._provider(handler)
        ranked, meta = provider.rerank(
            "Apple FY2025 revenue",
            _hits(),
            top_k=2,
        )

        self.assertEqual([hit["chunk_id"] for hit in ranked], ["correct-period", "wrong-period"])
        self.assertEqual(captured["model"], "qwen3-rerank")
        self.assertEqual(captured["top_n"], 2)
        self.assertEqual(len(captured["documents"]), 3)
        self.assertNotIn("chunk_id", json.dumps(captured))
        self.assertEqual(meta["rerank_provider"], "dashscope")
        self.assertEqual(meta["rerank_attempts"], 1)
        self.assertEqual(meta["rerank_tokens"], 123)
        self.assertEqual(meta["rerank_request_id"], "req-test-1")
        self.assertFalse(meta["rerank_fallback"])
        self.assertTrue(all("+qwen3_rerank" in hit["retrieval_method"] for hit in ranked))

    def test_429_retries_once_then_succeeds(self) -> None:
        calls = 0
        sleeps: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, request=request, text="rate limited")
            return _success_response(request, [1])

        provider = self._provider(
            handler,
            max_attempts=2,
            backoff_seconds=0.01,
            sleep=sleeps.append,
        )
        ranked, meta = provider.rerank("Apple FY2025 revenue", _hits(), top_k=1)

        self.assertEqual(ranked[0]["chunk_id"], "correct-period")
        self.assertEqual(calls, 2)
        self.assertEqual(meta["rerank_attempts"], 2)
        self.assertEqual(sleeps, [0.01])

    def test_duplicate_index_is_rejected_and_lexical_fallback_is_used(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                request=request,
                json={
                    "results": [
                        {"index": 0, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.8},
                    ]
                },
            )

        fallback = FallbackReranker(self._provider(handler), LexicalReranker())
        ranked, meta = fallback.rerank("Apple FY2025 revenue", _hits(), top_k=2)

        self.assertEqual(len(ranked), 2)
        self.assertTrue(meta["rerank_fallback"])
        self.assertEqual(meta["rerank_error_type"], "invalid_response")
        self.assertEqual(meta["rerank_mode_suffix"], "lexical_rerank_fallback")
        self.assertTrue(
            all("+lexical_rerank_fallback" in hit["retrieval_method"] for hit in ranked)
        )

    def test_missing_key_falls_back_without_network_call(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _success_response(request, [0])

        primary = self._provider(handler)
        primary._api_key = ""  # Exercise runtime secret absence after construction.
        ranked, meta = FallbackReranker(primary).rerank(
            "Apple FY2025 revenue", _hits(), top_k=1
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(calls, 0)
        self.assertTrue(meta["rerank_fallback"])
        self.assertEqual(meta["rerank_attempts"], 0)
        self.assertEqual(meta["rerank_error_type"], "error")

    def test_timeout_retries_then_falls_back(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("timed out", request=request)

        primary = self._provider(
            handler,
            max_attempts=2,
            backoff_seconds=0.0,
            sleep=lambda _: None,
        )
        ranked, meta = FallbackReranker(primary).rerank(
            "Apple FY2025 revenue", _hits(), top_k=1
        )

        self.assertEqual(calls, 2)
        self.assertEqual(len(ranked), 1)
        self.assertTrue(meta["rerank_fallback"])
        self.assertEqual(meta["rerank_attempts"], 2)
        self.assertIn(meta["rerank_error_type"], {"timeout", "connection"})

    def test_oversized_candidate_set_falls_back_without_network_call(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return _success_response(request, [0])

        primary = self._provider(handler)
        oversized = [_hits()[0] for _ in range(501)]
        ranked, meta = FallbackReranker(primary).rerank(
            "Apple revenue", oversized, top_k=1
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(calls, 0)
        self.assertTrue(meta["rerank_fallback"])
        self.assertEqual(meta["rerank_attempts"], 0)

    def test_bulkhead_limits_two_parallel_calls_to_one_inflight(self) -> None:
        active = 0
        maximum = 0
        lock = threading.Lock()

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return _success_response(request, [1])

        provider = self._provider(handler, max_inflight=1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(provider.rerank, "Apple FY2025 revenue", _hits(), top_k=1)
                for _ in range(2)
            ]
            [future.result() for future in futures]
        self.assertEqual(maximum, 1)


class RerankIntegrationTestCase(unittest.TestCase):
    def test_eval_telemetry_artifact_is_complete_without_request_id_value(self) -> None:
        item = _telemetry_item(
            "case-1",
            {
                "rerank_requested_provider": "dashscope",
                "rerank_provider": "dashscope",
                "rerank_model": "qwen3-rerank",
                "rerank_latency_ms": 12.345,
                "rerank_attempts": 1,
                "rerank_tokens": 42,
                "rerank_fallback": False,
                "rerank_error_type": "",
                "rerank_request_id": "req-sensitive-value",
            },
        )

        self.assertTrue(_qwen3_telemetry_complete([item]))
        self.assertTrue(item["request_id_present"])
        self.assertNotIn("rerank_request_id", item)
        self.assertNotIn("req-sensitive-value", json.dumps(item))

    def test_eval_telemetry_gate_rejects_missing_request_id(self) -> None:
        item = _telemetry_item(
            "case-1",
            {
                "rerank_provider": "dashscope",
                "rerank_model": "qwen3-rerank",
                "rerank_latency_ms": 1.0,
                "rerank_attempts": 1,
                "rerank_tokens": 1,
            },
        )

        self.assertFalse(_qwen3_telemetry_complete([item]))

    def test_factory_default_remains_offline_lexical(self) -> None:
        self.assertIsInstance(build_reranker(""), LexicalReranker)

    def test_qwen3_factory_is_staged_behind_fallback(self) -> None:
        reranker = build_reranker(
            "qwen3",
            base_url="",
        )
        ranked, meta = reranker.rerank("Apple revenue", _hits(), top_k=1)
        self.assertEqual(len(ranked), 1)
        self.assertTrue(meta["rerank_fallback"])
        self.assertEqual(meta["rerank_requested_provider"], "dashscope")

    def test_hybrid_meta_names_lexical_fallback_and_marks_degraded(self) -> None:
        primary = DashScopeQwen3Reranker(api_key="", base_url="")
        retriever = HybridEvidenceRetriever(
            None,
            top_k=1,
            rerank_enabled=True,
            rerank_candidates=3,
            reranker=FallbackReranker(primary),
        )
        documents = [
            {
                "document_id": "apple",
                "filename": "apple.md",
                "detected_companies": ["Apple"],
                "pages": [
                    "Apple FY2024 revenue was 391 billion dollars.",
                    "Apple FY2025 revenue was 416 billion dollars.",
                ],
            }
        ]
        hits, meta = retriever.retrieve_for_company_with_meta(
            query="Apple FY2025 revenue",
            company="Apple",
            session_id="qwen3-test",
            document_contexts=documents,
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(
            meta["mode"],
            "lexical_fallback_only+lexical_rerank_fallback",
        )
        self.assertTrue(meta["degraded"])
        self.assertTrue(meta["rerank_fallback"])
        self.assertTrue(hits[0]["rag_degraded"])


if __name__ == "__main__":
    unittest.main()
