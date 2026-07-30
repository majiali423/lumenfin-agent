from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.agents import AgentRuntime
from lumenfin.graph import route_after_quant
from lumenfin.knowledge_store import InMemoryKnowledgeStore
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.memory import ReasoningMemory, SessionMemory
from lumenfin.tools import build_coverage_matrix, is_partial_compare_gap
from tests.support.fakes import FakeMarketDataClient


def _runtime() -> AgentRuntime:
    return AgentRuntime(
        session_memory=SessionMemory(),
        knowledge_memory=InMemoryKnowledgeStore(),
        reasoning_memory=ReasoningMemory(),
        llm_client=LocalFallbackLLMClient(),
        market_data_client=FakeMarketDataClient(),
        rag_enabled=False,
        data_mode="live",
        allow_sample_data=False,
    )


def _apple_sec_payload() -> dict:
    return {
        "market_data": {
            "revenue": 412.0,
            "ebitda": 141.2,
            "r_and_d": 33.4,
            "operating_income": 123.6,
        },
        "supply_chain": {"risk_level": "medium", "signals": []},
        "earnings_call_quotes": ["Apple management remains optimistic about services growth."],
        "structured_source": "sec_companyfacts",
    }


def _microsoft_narrative_payload() -> dict:
    return {
        "market_data": {},
        "supply_chain": {"risk_level": "unknown", "signals": []},
        "earnings_call_quotes": [
            "Microsoft leadership highlighted cloud demand and AI platform investments."
        ],
        "structured_source": "none",
    }


class PartialQuantTestCase(unittest.TestCase):
    def test_coverage_matrix_flags_partial_compare_at_retrieval(self) -> None:
        retrieved = {
            "Apple": _apple_sec_payload(),
            "Microsoft": _microsoft_narrative_payload(),
        }
        matrix = build_coverage_matrix(["Apple", "Microsoft"], retrieved)
        self.assertTrue(matrix["Apple"]["comparable"])
        self.assertFalse(matrix["Microsoft"]["comparable"])
        self.assertTrue(is_partial_compare_gap(["Apple", "Microsoft"], matrix))

    def test_quant_preserves_apple_when_microsoft_uncomputable(self) -> None:
        runtime = _runtime()
        state = {
            "companies": ["Apple", "Microsoft"],
            "retrieved_docs": {
                "Apple": _apple_sec_payload(),
                "Microsoft": _microsoft_narrative_payload(),
            },
            "market_snapshots": {
                "Apple": {"status": "ok", "current_price": 190.0, "trailing_pe": 28.0},
                "Microsoft": {"status": "failed", "current_price": None},
            },
            "audit_log": [],
            "run_telemetry": {},
        }
        update = runtime.quantitative_analyst(state)

        self.assertIn("Apple", update["financial_metrics"])
        self.assertNotIn("Microsoft", update["financial_metrics"])
        self.assertIn("ebitda_margin", update["financial_metrics"]["Apple"])
        self.assertTrue(update["partial_data_gap"])
        self.assertEqual(update["non_comparable_companies"], ["Microsoft"])
        self.assertIsNone(update.get("replan_reason"))
        self.assertEqual(update["peer_comparison"]["comparable_companies"], ["Apple"])
        self.assertEqual(
            update["peer_comparison"]["summary"],
            "Peer comparison is unavailable because only Apple has "
            "comparable structured ratio metrics in this run.",
        )

    def test_quant_does_not_route_to_appendix_replan_on_partial_gap(self) -> None:
        state = {
            "partial_data_gap": True,
            "replan_reason": None,
        }
        self.assertEqual(route_after_quant(state), "psychologist")


if __name__ == "__main__":
    unittest.main()
