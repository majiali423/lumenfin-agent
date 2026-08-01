from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Lock, get_ident
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.finrun import export_finrun_state
from lumenfin.service import LumenFinAnalysisService
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class _BarrierTokenLLM(LocalFallbackLLMClient):
    model_name = "deepseek"

    def __init__(self, barrier: Barrier | None = None) -> None:
        super().__init__()
        self._barrier = barrier
        self._barrier_lock = Lock()
        self._barrier_threads: set[int] = set()

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        prompt = f"{system_prompt}\n{user_prompt}"
        if "NVIDIA" in prompt:
            prompt_tokens, completion_tokens = 101, 13
        elif "Apple" in prompt:
            prompt_tokens, completion_tokens = 11, 3
        else:
            prompt_tokens, completion_tokens = 7, 2
        content = LocalFallbackLLMClient().chat(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._add_usage(prompt_tokens, completion_tokens)
        if self._barrier is not None:
            thread_id = get_ident()
            with self._barrier_lock:
                should_wait = thread_id not in self._barrier_threads
                self._barrier_threads.add(thread_id)
            if should_wait:
                self._barrier.wait(timeout=10)
        return content


def _usage_projection(response: dict) -> dict:
    telemetry = response["result"]["run_telemetry"]
    return {
        "total_prompt_tokens": telemetry["total_prompt_tokens"],
        "total_completion_tokens": telemetry["total_completion_tokens"],
        "total_estimated_cost_usd": telemetry["total_estimated_cost_usd"],
        "node_spans": [
            {
                "step": span["step"],
                "prompt_tokens": span["prompt_tokens"],
                "completion_tokens": span["completion_tokens"],
                "estimated_cost_usd": span["estimated_cost_usd"],
            }
            for span in telemetry["node_spans"]
        ],
    }


class ServiceConcurrencyIsolationTestCase(unittest.TestCase):
    def test_concurrent_requests_match_serial_run_local_usage(self) -> None:
        requests = {
            "usage-apple": "Analyze Apple FY2025 profitability and R&D intensity.",
            "usage-nvidia": "Analyze NVIDIA FY2025 profitability and R&D intensity.",
        }

        def build_service(name: str, llm: LocalFallbackLLMClient) -> LumenFinAnalysisService:
            root = ROOT / "test_artifacts" / f"{name}-{uuid4().hex[:8]}"
            config = replace(build_test_config(root), rag_enabled=False)
            return LumenFinAnalysisService(
                config,
                llm_client=llm,
                market_data_client=FakeMarketDataClient(),
            )

        baselines: dict[str, dict] = {}
        for thread_id, query in requests.items():
            serial = build_service(f"serial-{thread_id}", _BarrierTokenLLM())
            baselines[thread_id] = _usage_projection(
                serial.analyze(query, thread_id=thread_id, export_artifacts=False)
            )

        concurrent = build_service("concurrent-usage", _BarrierTokenLLM(Barrier(2)))

        def run(item: tuple[str, str]) -> tuple[str, dict]:
            thread_id, query = item
            response = concurrent.analyze(query, thread_id=thread_id, export_artifacts=False)
            return thread_id, _usage_projection(response)

        with ThreadPoolExecutor(max_workers=2) as pool:
            actual = dict(pool.map(run, requests.items()))

        self.assertEqual(actual, baselines)
        for usage in actual.values():
            self.assertGreater(usage["total_prompt_tokens"], 0)
            self.assertGreater(usage["total_completion_tokens"], 0)
            self.assertGreater(usage["total_estimated_cost_usd"], 0)
            for span in usage["node_spans"]:
                self.assertGreaterEqual(span["prompt_tokens"], 0)
                self.assertGreaterEqual(span["completion_tokens"], 0)

    def test_concurrent_issuer_requests_keep_run_state_isolated(self) -> None:
        root = ROOT / "test_artifacts" / f"concurrency-{uuid4().hex[:8]}"
        config = replace(build_test_config(root), rag_enabled=False)
        service = LumenFinAnalysisService(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )

        first_system = service._system_for("probe-a")
        second_system = service._system_for("probe-b")
        self.assertIsNot(first_system, second_system)
        self.assertIsNot(first_system.reasoning_memory, second_system.reasoning_memory)
        self.assertIsNot(first_system.checkpointer, second_system.checkpointer)

        requests = {
            "nvidia-concurrent": "Analyze NVIDIA FY2025 profitability and R&D intensity.",
            "apple-concurrent": "Analyze Apple FY2025 profitability and R&D intensity.",
        }

        def run(item: tuple[str, str]) -> dict:
            thread_id, query = item
            return service.analyze(
                query=query,
                thread_id=thread_id,
                export_artifacts=False,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = dict(zip(requests, pool.map(run, requests.items()), strict=True))

        expectations = {
            "nvidia-concurrent": ("NVIDIA", "Apple"),
            "apple-concurrent": ("Apple", "NVIDIA"),
        }
        audit_objects: list[list[dict]] = []
        for thread_id, (expected, forbidden) in expectations.items():
            response = responses[thread_id]
            result = response["result"]
            self.assertEqual(result["query_plan"]["companies"], [expected])
            self.assertEqual(result["companies"], [expected])
            self.assertIn(expected, result["final_report"])
            self.assertNotIn(forbidden, result["final_report"])

            claims = result.get("verified_claims") or []
            self.assertTrue(claims)
            self.assertTrue(all(claim.get("entity") == expected for claim in claims))

            finrun = export_finrun_state(result)
            evidence_entities = {item.get("entity") for item in finrun["evidence"]}
            self.assertEqual(evidence_entities, {expected})

            audit = result.get("audit_log") or []
            self.assertTrue(audit)
            self.assertEqual(audit[0]["step"], "input_guardrail")
            audit_objects.append(audit)

        self.assertIsNot(audit_objects[0], audit_objects[1])
        self.assertEqual(
            service.get_checkpoint("nvidia-concurrent")["state"]["companies"],
            ["NVIDIA"],
        )
        self.assertEqual(
            service.get_checkpoint("apple-concurrent")["state"]["companies"],
            ["Apple"],
        )


if __name__ == "__main__":
    unittest.main()
