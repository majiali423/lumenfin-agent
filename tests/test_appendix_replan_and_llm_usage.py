from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.graph import route_after_appendix_replan, route_after_quant, route_after_retrieval
from lumenfin.llm import LocalFallbackLLMClient, ResilientLLMClient
from tests.support.fakes import TimeoutLLMClient


class AppendixReplanRoutingTestCase(unittest.TestCase):
    def test_retrieval_routes_to_appendix_replan_on_replan_reason(self) -> None:
        state = {
            "fatal_data_gap": False,
            "replan_reason": "Appendix / evidence gap detected",
        }
        self.assertEqual(route_after_retrieval(state), "appendix_replan")

    def test_quant_routes_to_psychologist_even_if_legacy_replan_reason_present(self) -> None:
        # Partial quant no longer uses replan_reason; quant should continue downstream.
        state = {"replan_reason": "legacy appendix replan reason"}
        self.assertEqual(route_after_quant(state), "appendix_replan")

    def test_quant_routes_to_psychologist_on_partial_gap(self) -> None:
        state = {"partial_data_gap": True, "replan_reason": None}
        self.assertEqual(route_after_quant(state), "psychologist")

    def test_appendix_replan_routes_back_to_retrieval_until_degraded(self) -> None:
        self.assertEqual(route_after_appendix_replan({"degraded_mode": False}), "retrieval")
        self.assertEqual(route_after_appendix_replan({"degraded_mode": True}), "claim_binder")


class ResilientLLMUsageTestCase(unittest.TestCase):
    def test_usage_tracks_primary_success(self) -> None:
        primary = LocalFallbackLLMClient()
        client = ResilientLLMClient(primary=primary, fallback=LocalFallbackLLMClient())
        client.mark_usage_start()
        content = client.chat("system", "NVIDIA FY2025 executive summary in Chinese.")
        usage = client.usage_since_mark()
        self.assertIn("NVIDIA", content)
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)
        self.assertEqual(client.backend_name, "local-fallback")

    def test_usage_tracks_fallback_after_primary_timeout(self) -> None:
        client = ResilientLLMClient(primary=TimeoutLLMClient(), fallback=LocalFallbackLLMClient())
        client.mark_usage_start()
        content = client.chat("system", "NVIDIA FY2025 executive summary in Chinese.")
        usage = client.usage_since_mark()
        self.assertIn("NVIDIA", content)
        self.assertEqual(client.backend_name, "local-fallback")
        self.assertGreaterEqual(usage["prompt_tokens"], 1)
        self.assertGreaterEqual(usage["completion_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
