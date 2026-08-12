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
from lumenfin.planning import build_query_plan
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class CompanyUploadMismatchPlanningTestCase(unittest.TestCase):
    def test_softbank_query_with_apple_upload_pauses(self) -> None:
        docs = [{"detected_companies": ["Apple", "Microsoft"], "filename": "table.pdf"}]
        plan = build_query_plan(
            "Analyze SoftBank FY2024 profitability using the uploaded materials.",
            document_contexts=docs,
        )
        self.assertIn("company_upload_mismatch", plan.missing_fields)
        self.assertEqual(plan.companies, ["SoftBank"])
        self.assertEqual(plan.query_companies, ["SoftBank"])
        self.assertEqual(plan.upload_companies, ["Apple", "Microsoft"])
        self.assertTrue(any("不一致" in q for q in plan.clarification_questions))

    def test_clarification_scope_uploaded_resolves(self) -> None:
        docs = [{"detected_companies": ["Apple", "Microsoft"], "filename": "table.pdf"}]
        plan = build_query_plan(
            "Analyze SoftBank FY2024 profitability using the uploaded materials.",
            document_contexts=docs,
            user_clarification={"company_scope": "uploaded"},
        )
        self.assertNotIn("company_upload_mismatch", plan.missing_fields)
        self.assertEqual(set(plan.companies), {"Apple", "Microsoft"})
        self.assertEqual(plan.company_scope, "uploaded")

    def test_matching_query_and_upload_no_pause(self) -> None:
        docs = [{"detected_companies": ["Apple", "Microsoft"], "filename": "table.pdf"}]
        plan = build_query_plan(
            "Compare Apple and Microsoft FY2025 profitability using the uploaded table.",
            document_contexts=docs,
        )
        self.assertNotIn("company_upload_mismatch", plan.missing_fields)
        self.assertEqual(set(plan.companies), {"Apple", "Microsoft"})

    def test_upload_only_no_query_company_uses_upload(self) -> None:
        docs = [{"detected_companies": ["Apple", "Microsoft"], "filename": "table.pdf"}]
        plan = build_query_plan(
            "基于上传表格做 FY2025 盈利能力与研发强度对比。",
            document_contexts=docs,
        )
        self.assertNotIn("company_upload_mismatch", plan.missing_fields)
        self.assertEqual(set(plan.companies), {"Apple", "Microsoft"})
        self.assertEqual(plan.company_scope, "uploaded")


class CompanyUploadMismatchHitlTestCase(unittest.TestCase):
    def test_route_pauses_while_missing_fields_remain(self) -> None:
        self.assertEqual(
            route_after_query_planner({"missing_fields": ["company_upload_mismatch"]}),
            "await_clarification",
        )
        self.assertEqual(
            route_after_query_planner(
                {
                    "missing_fields": ["company_upload_mismatch"],
                    "user_clarification": {"notes": "still unsure"},
                }
            ),
            "await_clarification",
        )
        self.assertEqual(route_after_query_planner({"missing_fields": []}), "supervisor")

    def test_mismatch_workflow_pause_and_resume_with_scope(self) -> None:
        from dataclasses import replace

        config = replace(
            build_test_config(ROOT / "test_artifacts" / f"mismatch-{uuid4().hex[:8]}"),
            rag_enabled=False,
        )
        app = LumenFinAgentSystem(
            llm_client=LocalFallbackLLMClient(),
            app_config=config,
            market_data_client=FakeMarketDataClient(),
        )
        thread_id = "mismatch-softbank"
        docs = [
            {
                "detected_companies": ["Apple", "Microsoft"],
                "filename": "apple_msft.pdf",
                "text": "Apple revenue 383.3 EBITDA 130.1 R&D 31.4. Microsoft revenue 245.1 EBITDA 128.4 R&D 29.5.",
                "metric_hints": {"revenue": 383.3, "ebitda": 130.1, "r_and_d": 31.4},
                "per_company_metric_hints": {
                    "Apple": {"revenue": 383.3, "ebitda": 130.1, "r_and_d": 31.4},
                    "Microsoft": {"revenue": 245.1, "ebitda": 128.4, "r_and_d": 29.5},
                },
                "excerpt": "peer table",
                "source_type": "pdf",
            }
        ]
        paused = app.run(
            "Analyze SoftBank FY2024 profitability using the uploaded materials.",
            thread_id=thread_id,
            document_contexts=docs,
        )
        self.assertEqual(paused.get("workflow_status"), "needs_clarification")
        self.assertIn("company_upload_mismatch", paused.get("missing_fields") or [])
        self.assertNotIn("supervisor", [e["step"] for e in paused.get("audit_log", [])])

        resumed = app.resume_with_clarification(
            thread_id,
            {"company_scope": "uploaded", "time_range": "FY2025"},
        )
        self.assertIn(resumed.get("workflow_status"), {"completed", "incomplete_data"})
        self.assertEqual(set(resumed.get("companies") or []), {"Apple", "Microsoft"})


if __name__ == "__main__":
    unittest.main()
