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
        # Partial clarification that leaves missing_fields should still pause.
        self.assertEqual(
            route_after_query_planner({"missing_fields": ["company"], "user_clarification": {"company": "Apple"}}),
            "await_clarification",
        )
        self.assertEqual(
            route_after_query_planner({"missing_fields": [], "user_clarification": {"company": "Apple"}}),
            "supervisor",
        )

    def test_clarify_updates_job_result_and_second_round_stays_paused(self) -> None:
        from lumenfin.service import LumenFinAnalysisService

        config = build_test_config(ROOT / "test_artifacts" / f"hitl-job-{uuid4().hex[:8]}")
        service = LumenFinAnalysisService(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        query = "请分析供应链风险和研发投入。"
        thread_id = f"hitl-job-{uuid4().hex[:8]}"
        tenant = config.rag_tenant_id
        paused = service.analyze(query, thread_id=thread_id, export_artifacts=False, tenant_id=tenant)
        self.assertEqual(paused["result"]["workflow_status"], "needs_clarification")
        job_id = f"job-{uuid4().hex[:8]}"
        service.repository.create_job(job_id, thread_id, query, tenant_id=tenant)
        service.repository.update_job_status(job_id, status="completed", result=paused["result"])
        still = service.clarify(
            thread_id,
            {"company": "Apple"},
            export_artifacts=False,
            tenant_id=tenant,
            job_id=job_id,
        )
        self.assertEqual(still["result"]["workflow_status"], "needs_clarification")
        after_partial = service.get_job(job_id, tenant_id=tenant)
        assert after_partial is not None
        self.assertEqual(after_partial["result"]["workflow_status"], "needs_clarification")
        done = service.clarify(
            thread_id,
            {"time_range": "FY2025"},
            export_artifacts=False,
            tenant_id=tenant,
            job_id=job_id,
        )
        self.assertEqual(done["result"]["workflow_status"], "completed")
        restored = service.get_job(job_id, tenant_id=tenant)
        assert restored is not None
        self.assertEqual(restored["result"]["workflow_status"], "completed")
        self.assertTrue(restored["result"].get("final_report"))

    def test_get_job_does_not_overlay_later_checkpoint_on_same_thread(self) -> None:
        from lumenfin.service import LumenFinAnalysisService

        config = build_test_config(ROOT / "test_artifacts" / f"hitl-overlay-{uuid4().hex[:8]}")
        service = LumenFinAnalysisService(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        tenant = config.rag_tenant_id
        thread_id = f"shared-{uuid4().hex[:8]}"
        apple_id = f"job-apple-{uuid4().hex[:6]}"
        service.repository.create_job(apple_id, thread_id, "Analyze Apple", tenant_id=tenant)
        service.repository.update_job_status(
            apple_id,
            status="completed",
            result={
                "query": "Analyze Apple",
                "workflow_status": "completed",
                "final_report": "Apple EBITDA is grounded.",
            },
        )
        service.checkpoint_repo.upsert(
            thread_id=thread_id,
            query="Analyze Microsoft",
            state={
                "query": "Analyze Microsoft",
                "workflow_status": "needs_clarification",
                "final_report": "",
                "clarification_questions": ["company?"],
            },
            llm_backend="test",
            expected_revision=0,
            tenant_id=tenant,
        )
        stored = service.get_job(apple_id, tenant_id=tenant)
        assert stored is not None
        self.assertEqual(stored["query"], "Analyze Apple")
        self.assertEqual(stored["result"]["query"], "Analyze Apple")
        self.assertEqual(stored["result"]["workflow_status"], "completed")
        self.assertEqual(stored["result"]["final_report"], "Apple EBITDA is grounded.")


if __name__ == "__main__":
    unittest.main()
