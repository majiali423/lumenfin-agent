from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
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


class ServiceConcurrencyIsolationTestCase(unittest.TestCase):
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
