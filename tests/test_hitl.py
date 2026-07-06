from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin import LumenFinAgentSystem
from lumenfin.graph import route_after_query_planner
from lumenfin.llm import LocalFallbackLLMClient
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class HitlWorkflowTestCase(unittest.TestCase):
    def test_ambiguous_query_pauses_for_clarification(self) -> None:
        config = build_test_config(ROOT / "test_artifacts" / f"hitl-{uuid4().hex[:8]}")
        app = LumenFinAgentSystem(
            llm_client=LocalFallbackLLMClient(),
            app_config=config,
            market_data_client=FakeMarketDataClient(),
        )
        result = app.run("请分析供应链风险和研发投入。", thread_id="hitl-pause")

        self.assertEqual(result.get("workflow_status"), "needs_clarification")
        self.assertTrue(result.get("clarification_questions"))
        steps = [event["step"] for event in result.get("audit_log", [])]
        self.assertIn("await_clarification", steps)
        self.assertNotIn("supervisor", steps)

    def test_clarification_resume_completes_workflow(self) -> None:
        config = build_test_config(ROOT / "test_artifacts" / f"hitl-{uuid4().hex[:8]}")
        app = LumenFinAgentSystem(
            llm_client=LocalFallbackLLMClient(),
            app_config=config,
            market_data_client=FakeMarketDataClient(),
        )
        thread_id = "hitl-resume"
        paused = app.run("请分析供应链风险和研发投入。", thread_id=thread_id)
        self.assertEqual(paused.get("workflow_status"), "needs_clarification")

        resumed = app.resume_with_clarification(
            thread_id,
            {"company": "Apple", "time_range": "FY2025"},
        )
        self.assertEqual(resumed.get("workflow_status"), "completed")
        self.assertIn("final_report", resumed)
        self.assertIn("Apple", resumed["final_report"])

    def test_route_after_query_planner(self) -> None:
        self.assertEqual(
            route_after_query_planner({"missing_fields": ["company"], "user_clarification": {}}),
            "await_clarification",
        )
        self.assertEqual(
            route_after_query_planner({"missing_fields": ["company"], "user_clarification": {"company": "Apple"}}),
            "supervisor",
        )


if __name__ == "__main__":
    unittest.main()
