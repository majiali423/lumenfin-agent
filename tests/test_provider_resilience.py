"""Phase 3.3A provider resilience unit tests (deterministic, no live network)."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.llm import DeepSeekChatClient, LLMSettings, ResilientLLMClient
from lumenfin.provider_resilience import (
    DeadlineExceededError,
    InvalidProviderResponseError,
    ProviderCallContext,
    ProviderCallPolicy,
    call_with_policy,
    compute_backoff_seconds,
    extract_retry_after_seconds,
    summarize_provider_trace,
)
from lumenfin.provider_retry import call_with_transient_retry, extract_retry_after_seconds as extract_ra
from lumenfin.rag.embeddings import ResilientEmbeddingProvider


class PolicyRetryTestCase(unittest.TestCase):
    def test_max_attempts_is_total_physical_calls(self) -> None:
        calls = {"n": 0}

        def always_fail() -> str:
            calls["n"] += 1
            raise TimeoutError("boom")

        policy = ProviderCallPolicy(provider="t", operation="x", max_attempts=3, base_backoff_seconds=0.1)
        ctx = ProviderCallContext.create(rng=random.Random(0))
        ctx.sleep = lambda _: None
        with self.assertRaises(TimeoutError):
            call_with_policy(always_fail, policy=policy, context=ctx)
        self.assertEqual(calls["n"], 3)

    def test_retry_after_increases_backoff(self) -> None:
        response = httpx.Response(
            429,
            headers={"Retry-After": "7"},
            request=httpx.Request("GET", "https://example.test"),
        )
        exc = httpx.HTTPStatusError("limited", request=response.request, response=response)
        self.assertEqual(extract_retry_after_seconds(exc), 7.0)
        self.assertEqual(extract_ra(exc), 7.0)
        policy = ProviderCallPolicy(
            provider="t",
            operation="x",
            max_attempts=3,
            base_backoff_seconds=0.5,
            max_backoff_seconds=30,
            jitter_ratio=0.0,
        )
        delay = compute_backoff_seconds(
            attempt_index=0,
            policy=policy,
            retry_after_seconds=7.0,
            rng=random.Random(0),
        )
        self.assertEqual(delay, 7.0)

    def test_deadline_stops_retry(self) -> None:
        calls = {"n": 0}
        clock = {"t": 100.0}

        def fail() -> str:
            calls["n"] += 1
            raise TimeoutError("slow")

        policy = ProviderCallPolicy(provider="t", operation="x", max_attempts=5, base_backoff_seconds=1.0)
        ctx = ProviderCallContext(
            request_id="d1",
            deadline_monotonic=100.2,
            trace_sink=[],
            rng=random.Random(0),
            sleep=lambda _: None,
            now=lambda: clock["t"],
        )

        def advancing_sleep(seconds: float) -> None:
            clock["t"] += seconds

        ctx.sleep = advancing_sleep
        with self.assertRaises(DeadlineExceededError):
            call_with_policy(fail, policy=policy, context=ctx)
        self.assertLess(calls["n"], 5)

    def test_invalid_response_not_retried(self) -> None:
        calls = {"n": 0}

        def bad() -> str:
            calls["n"] += 1
            raise InvalidProviderResponseError("dim mismatch")

        policy = ProviderCallPolicy(provider="emb", operation="embed", max_attempts=3)
        ctx = ProviderCallContext.create()
        ctx.sleep = lambda _: None
        with self.assertRaises(InvalidProviderResponseError):
            call_with_policy(bad, policy=policy, context=ctx)
        self.assertEqual(calls["n"], 1)


class DeepSeekResilienceTestCase(unittest.TestCase):
    def test_shared_client_reuse_and_attempt_count(self) -> None:
        settings = LLMSettings(
            api_key="k",
            base_url="https://api.example.test",
            model="m",
            timeout_seconds=2.0,
            max_retries=3,
            retry_backoff_seconds=0.01,
        )
        mock_client = MagicMock()
        request = httpx.Request("POST", "https://api.example.test/chat/completions")
        responses = [
            httpx.Response(503, request=request),
            httpx.Response(503, request=request),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
                request=request,
            ),
        ]
        state = {"i": 0}

        def post(*args, **kwargs):
            idx = state["i"]
            state["i"] += 1
            response = responses[idx]
            if response.status_code >= 400:
                raise httpx.HTTPStatusError("err", request=request, response=response)
            return response

        mock_client.post.side_effect = post
        client = DeepSeekChatClient(settings, http_client=mock_client)
        ctx = ProviderCallContext.create(rng=random.Random(0))
        ctx.sleep = lambda _: None
        client.bind_call_context(ctx)
        text = client.chat("sys", "user")
        self.assertEqual(text, "ok")
        self.assertEqual(client.last_attempts, 3)
        self.assertEqual(mock_client.post.call_count, 3)

    def test_fallback_marks_degraded(self) -> None:
        primary = MagicMock()
        primary.backend_name = "deepseek"
        primary.model_name = "m"
        primary._usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
        primary.chat.side_effect = TimeoutError("upstream")
        primary.last_attempts = 3
        primary.last_trace = []
        primary.mark_usage_start = MagicMock()

        fallback = MagicMock()
        fallback.backend_name = "local-fallback"
        fallback.model_name = "local-fallback"
        fallback._usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
        fallback.chat.return_value = "synthetic"
        fallback.mark_usage_start = MagicMock()

        resilient = ResilientLLMClient(primary=primary, fallback=fallback, allow_fallback=True)
        text = resilient.chat("sys", "user")
        self.assertEqual(text, "synthetic")
        self.assertTrue(resilient.used_fallback)
        self.assertTrue(resilient.degraded)
        self.assertEqual(resilient.primary_error_class, "timeout")
        audit = resilient.fallback_audit()
        self.assertTrue(audit["degraded"])
        self.assertEqual(audit["error_class"], "timeout")


class EmbeddingOwnerTestCase(unittest.TestCase):
    def test_resilient_wrapper_is_single_retry_owner(self) -> None:
        inner = MagicMock()
        inner.dimension = 8
        inner._timeout = 1.0
        calls = {"n": 0}

        def embed(texts):
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("emb")
            return [[0.1] * 8 for _ in texts]

        inner.embed.side_effect = embed
        provider = ResilientEmbeddingProvider(inner, max_retries=3, backoff_seconds=0.01, sleep=lambda _: None)
        vectors = provider.embed(["a"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(calls["n"], 3)
        self.assertEqual(provider.last_attempts, 3)


class LegacyHelperCompatTestCase(unittest.TestCase):
    def test_call_with_transient_retry_still_total_attempts(self) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("t")
            return "ok"

        result = call_with_transient_retry(flaky, max_retries=3, sleep=lambda _: None)
        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 3)

    def test_trace_summary_ratio(self) -> None:
        events = [
            {"logical_call_id": "a", "attempt": 1, "status": "retry"},
            {"logical_call_id": "a", "attempt": 2, "status": "retry"},
            {"logical_call_id": "a", "attempt": 3, "status": "success", "latency_ms": 10},
        ]
        summary = summarize_provider_trace(events)
        self.assertEqual(summary["logical_provider_calls"], 1)
        self.assertEqual(summary["physical_provider_attempts"], 3)
        self.assertEqual(summary["retry_amplification_ratio"], 3.0)


if __name__ == "__main__":
    unittest.main()
