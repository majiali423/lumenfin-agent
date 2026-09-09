from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.agents import AgentRuntime
from lumenfin.critic_checks import check_data_completeness
from lumenfin.eval.product_dev import build_catalog, catalog_items
from lumenfin.graph import route_after_bounded_repair, route_after_retrieval
from lumenfin.knowledge_store import InMemoryKnowledgeStore
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.memory import ReasoningMemory, SessionMemory
from lumenfin.bounded_repair import run_bounded_repair
from lumenfin.planning import build_query_plan
from lumenfin.task_spec import task_spec_from_plan
from tests.support.fakes import FakeMarketDataClient


def _runtime(*, gating: bool = True) -> AgentRuntime:
    return AgentRuntime(
        session_memory=SessionMemory(),
        knowledge_memory=InMemoryKnowledgeStore(),
        reasoning_memory=ReasoningMemory(),
        llm_client=LocalFallbackLLMClient(),
        market_data_client=FakeMarketDataClient(),
        rag_enabled=False,
        data_mode="live",
        allow_sample_data=False,
        fetch_live_fundamentals=False,
        fetch_sec_fundamentals=False,
        task_spec_gating=gating,
        profile_llm_max_attempts=0,
    )


def _risk_plan() -> dict:
    plan = build_query_plan("What is Apple supply-chain risk?", llm_client=None)
    return plan.to_dict()


def _narrative_payload(company: str) -> dict:
    return {
        "market_data": {},
        "supply_chain": {
            "risk_level": "medium",
            "signals": ["Greater China manufacturing concentration remains above internal target."],
        },
        "earnings_call_quotes": [],
        "structured_source": "none",
        "appendix": {},
        "fundamentals_meta": {},
        "provider_errors": [],
    }


class TaskSpecRoutingTestCase(unittest.TestCase):
    def test_document_evidence_still_runs_quant(self) -> None:
        spec = task_spec_from_plan(
            {
                "intent": "document_financial_diligence",
                "analysis_dimensions": ["document_evidence", "compliance"],
            }
        )
        self.assertFalse(spec.requires_ast_ratios)
        self.assertFalse(spec.skip_quant)
        spec = task_spec_from_plan(_risk_plan())
        self.assertEqual(spec.intent, "risk_compliance_review")
        self.assertFalse(spec.requires_ast_ratios)
        self.assertTrue(spec.skip_quant)

    def test_retrieval_routes_skip_quant_to_psychologist(self) -> None:
        state = {
            "fatal_data_gap": False,
            "replan_reason": None,
            "task_spec": task_spec_from_plan(_risk_plan()).to_dict(),
        }
        self.assertEqual(route_after_retrieval(state), "psychologist")

    def test_fatal_with_opt_in_repair_routes_to_bounded_repair(self) -> None:
        state = {
            "fatal_data_gap": True,
            "bounded_repair_enabled": True,
            "task_spec": {
                "intent": "financial_diligence",
                "dimensions": ["profitability"],
                "requires_ast_ratios": True,
                "skip_quant": False,
                "allow_risk_without_ratios": False,
            },
        }
        self.assertEqual(route_after_retrieval(state), "bounded_repair")
        self.assertEqual(route_after_bounded_repair({"fatal_data_gap": True, "task_spec": state["task_spec"]}), "claim_binder")

    def test_legacy_gating_marks_risk_without_ast_as_fatal(self) -> None:
        runtime = _runtime(gating=False)
        with patch("lumenfin.agents.retrieval.retrieve_company_payload", side_effect=lambda company, **_: _narrative_payload(company)):
            update = runtime.retrieval(
                {
                    "query": "What is Apple supply-chain risk?",
                    "companies": ["Apple"],
                    "query_plan": _risk_plan(),
                    "thread_id": "p4-legacy",
                    "document_contexts": [],
                    "target_symbols": {},
                    "run_telemetry": {},
                }
            )
        self.assertTrue(update["fatal_data_gap"])

    def test_taskspec_gating_does_not_fatal_risk_without_ast(self) -> None:
        runtime = _runtime(gating=True)
        with patch("lumenfin.agents.retrieval.retrieve_company_payload", side_effect=lambda company, **_: _narrative_payload(company)):
            update = runtime.retrieval(
                {
                    "query": "What is Apple supply-chain risk?",
                    "companies": ["Apple"],
                    "query_plan": _risk_plan(),
                    "thread_id": "p4-gating",
                    "document_contexts": [],
                    "target_symbols": {},
                    "run_telemetry": {},
                }
            )
        self.assertFalse(update["fatal_data_gap"])
        self.assertTrue(update["task_spec"]["skip_quant"])
        merged = {**update, "companies": ["Apple"]}
        self.assertEqual(route_after_retrieval(merged), "psychologist")
        self.assertNotIn("incomplete_data", str(update.get("workflow_status") or ""))

    def test_critic_skips_missing_quant_when_taskspec_skip_quant(self) -> None:
        violations = check_data_completeness(
            {
                "companies": ["Apple"],
                "financial_metrics": {},
                "sentiment_analysis": {"Apple": {"label": "neutral"}},
                "task_spec": task_spec_from_plan(_risk_plan()).to_dict(),
            }
        )
        self.assertFalse(any(item.code == "missing_quantitative_results" for item in violations))

    def test_catalog_splits_are_frozen(self) -> None:
        items = build_catalog()
        self.assertEqual(len(items), 36)
        self.assertEqual(len(catalog_items(split="dev")), 16)
        self.assertEqual(len(catalog_items(split="test")), 8)
        self.assertTrue(all(item["split"] == "test" for item in catalog_items(split="test")))

    def test_bounded_repair_fills_sample_then_clears_fatal(self) -> None:
        update = run_bounded_repair(
            {
                "retrieved_docs": {"Apple": {"market_data": {}, "structured_source": "none"}},
                "task_spec": {
                    "intent": "financial_diligence",
                    "dimensions": ["profitability"],
                    "requires_ast_ratios": True,
                    "skip_quant": False,
                    "allow_risk_without_ratios": False,
                    "allowed_tools": ["sample_fundamentals", "safe_ratio", "stop"],
                    "max_repair_steps": 2,
                    "max_tool_calls": 3,
                },
            },
            allow_sample_data=True,
        )
        self.assertFalse(update["fatal_data_gap"])
        self.assertEqual(update["retrieved_docs"]["Apple"]["structured_source"], "sample_db")
        self.assertTrue(any(event.get("tool") == "sample_fundamentals" for event in update["bounded_repair_trace"]))


if __name__ == "__main__":
    unittest.main()
