from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.checkpoint_store import CheckpointConflictError
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
    def _build_concurrent_service(self, name: str, llm: LocalFallbackLLMClient) -> LumenFinAnalysisService:
        root = ROOT / "test_artifacts" / f"{name}-{uuid4().hex[:8]}"
        config = replace(build_test_config(root), rag_enabled=False)
        return LumenFinAnalysisService(
            config,
            llm_client=llm,
            market_data_client=FakeMarketDataClient(),
        )

    def _assert_one_checkpoint_conflict(self, futures: list) -> dict:
        successes: list[dict] = []
        errors: list[Exception] = []
        for future in futures:
            try:
                successes.append(future.result())
            except Exception as exc:  # noqa: BLE001 - asserting public conflict type
                errors.append(exc)
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(type(errors[0]).__name__, "CheckpointConflictError")
        return successes[0]

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

    def test_same_thread_concurrent_analyze_rejects_one_stale_write(self) -> None:
        service = self._build_concurrent_service(
            "same-thread-analyze",
            _BarrierTokenLLM(Barrier(2)),
        )
        thread_id = "same-analyze"
        queries = [
            "Analyze Apple FY2025 profitability and R&D intensity.",
            "Analyze NVIDIA FY2025 profitability and R&D intensity.",
        ]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(service.analyze, query, thread_id, False)
                for query in queries
            ]
            winner = self._assert_one_checkpoint_conflict(futures)

        stored = service.get_checkpoint(thread_id)
        self.assertEqual(stored["query"], winner["query"])
        self.assertEqual(stored["state"]["companies"], winner["result"]["companies"])

    def test_analyze_and_clarify_from_same_revision_conflict(self) -> None:
        service = self._build_concurrent_service("analyze-clarify", LocalFallbackLLMClient())
        thread_id = "analyze-clarify"
        paused = service.analyze("Analyze profitability.", thread_id, False)
        self.assertEqual(paused["workflow_status"], "needs_clarification")
        service._llm_client = _BarrierTokenLLM(Barrier(2))

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    service.clarify,
                    thread_id,
                    {"company": "NVIDIA", "time_range": "FY2025"},
                    False,
                ),
                pool.submit(
                    service.analyze,
                    "Analyze Apple FY2025 profitability and R&D intensity.",
                    thread_id,
                    False,
                ),
            ]
            winner = self._assert_one_checkpoint_conflict(futures)

        stored = service.get_checkpoint(thread_id)
        self.assertEqual(stored["query"], winner["query"])
        self.assertEqual(stored["state"]["companies"], winner["result"]["companies"])

    def test_two_clarifications_from_same_revision_conflict(self) -> None:
        service = self._build_concurrent_service("double-clarify", LocalFallbackLLMClient())
        thread_id = "double-clarify"
        paused = service.analyze("Analyze profitability.", thread_id, False)
        self.assertEqual(paused["workflow_status"], "needs_clarification")
        service._llm_client = _BarrierTokenLLM(Barrier(2))
        clarifications = [
            {"company": "Apple", "time_range": "FY2025"},
            {"company": "NVIDIA", "time_range": "FY2025"},
        ]

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(service.clarify, thread_id, clarification, False)
                for clarification in clarifications
            ]
            winner = self._assert_one_checkpoint_conflict(futures)

        stored = service.get_checkpoint(thread_id)
        self.assertEqual(stored["state"]["companies"], winner["result"]["companies"])
        self.assertEqual(
            stored["state"].get("user_clarification"),
            winner["result"].get("user_clarification"),
        )
        self.assertEqual(len(stored["state"]["companies"]), 1)

    def test_job_conflict_is_failed_and_keeps_conflict_error(self) -> None:
        service = self._build_concurrent_service("job-conflict", LocalFallbackLLMClient())
        created = service.submit_job("Analyze NVIDIA FY2025.", thread_id="job-conflict-thread")
        conflict = CheckpointConflictError(
            "Checkpoint conflict for thread_id=job-conflict-thread: expected revision 1"
        )
        with patch.object(service, "analyze", side_effect=conflict):
            with self.assertRaises(CheckpointConflictError):
                service.run_job(
                    created["job_id"],
                    "Analyze NVIDIA FY2025.",
                    created["thread_id"],
                    export_artifacts=False,
                )

        job = service.get_job(created["job_id"], tenant_id=created["tenant_id"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("Checkpoint conflict", job["error_message"])
        self.assertIsNone(job["result"])

    def test_response_packages_its_committed_checkpoint_revision(self) -> None:
        service = self._build_concurrent_service(
            "response-checkpoint-snapshot",
            LocalFallbackLLMClient(),
        )
        thread_id = "response-checkpoint-snapshot"
        first_packaging = Event()
        release_first = Event()
        original_package = service._package_response

        def pause_first_package(*args, **kwargs):
            query = args[1]
            if "Apple" in query:
                first_packaging.set()
                self.assertTrue(release_first.wait(timeout=10))
            return original_package(*args, **kwargs)

        service._package_response = pause_first_package
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(
                service.analyze,
                "Analyze Apple FY2025 profitability and R&D intensity.",
                thread_id,
                False,
            )
            self.assertTrue(first_packaging.wait(timeout=10))
            second = service.analyze(
                "Analyze NVIDIA FY2025 profitability and R&D intensity.",
                thread_id,
                False,
            )
            release_first.set()
            first = first_future.result(timeout=10)

        self.assertEqual(first["checkpoint"]["revision"] + 1, second["checkpoint"]["revision"])
        self.assertEqual(first["checkpoint"]["state"]["companies"], first["result"]["companies"])
        self.assertEqual(first["checkpoint"]["state"]["companies"], ["Apple"])
        self.assertEqual(second["checkpoint"]["state"]["companies"], ["NVIDIA"])
        stored = service.get_checkpoint(thread_id)
        self.assertEqual(stored["revision"], second["checkpoint"]["revision"])
        self.assertEqual(stored["state"]["companies"], ["NVIDIA"])


if __name__ == "__main__":
    unittest.main()
