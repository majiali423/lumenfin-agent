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
    def test_softbank_query_with_apple_upload_does_not_pause(self) -> None:
        docs = [{"detected_companies": ["Apple", "Microsoft"], "filename": "table.pdf"}]
        plan = build_query_plan(
            "Analyze SoftBank FY2024 profitability using the uploaded materials.",
            document_contexts=docs,
        )
        self.assertTrue(plan.evidence_company_gap)
        self.assertNotIn("company_upload_mismatch", plan.missing_fields)
        self.assertEqual(plan.companies, ["SoftBank"])
        self.assertEqual(plan.query_companies, ["SoftBank"])
        self.assertEqual(plan.upload_companies, ["Apple", "Microsoft"])
        self.assertFalse(plan.clarification_questions)

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

    def test_mismatch_runs_without_pause_and_does_not_steal_upload_issuer_numbers(self) -> None:
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
        result = app.run(
            "Analyze SoftBank FY2024 profitability using the uploaded materials.",
            thread_id=thread_id,
            document_contexts=docs,
        )
        self.assertNotEqual(result.get("workflow_status"), "needs_clarification")
        self.assertNotIn("company_upload_mismatch", result.get("missing_fields") or [])
        report = str(result.get("final_report") or "")
        self.assertNotIn("SoftBank Operating income is 383.3", report)
        self.assertNotIn("SoftBank Revenue is 383.3", report)
        self.assertEqual(result.get("companies"), ["SoftBank"])
        self.assertNotIn("383.3", str((result.get("verified_claims") or [])))


class NamedIssuerScopeGraphTestCase(unittest.TestCase):
    def _app(self, name: str):
        from dataclasses import replace

        config = replace(
            build_test_config(ROOT / "test_artifacts" / f"{name}-{uuid4().hex[:8]}"),
            rag_enabled=False,
        )
        return LumenFinAgentSystem(
            llm_client=LocalFallbackLLMClient(),
            app_config=config,
            market_data_client=FakeMarketDataClient(),
        )

    def test_apple_query_does_not_answer_with_nvidia_issuer_file(self) -> None:
        docs = [
            {
                "issuer_companies": ["NVIDIA"],
                "detected_companies": ["NVIDIA"],
                "filename": "nvda.pdf",
                "text": "NVIDIA FY2025 operating income was 81.453 billion USD. Revenue 130.5. R&D 12.9.",
                "excerpt": "NVIDIA FY2025 operating income was 81.453 billion USD.",
                "source_type": "pdf",
            }
        ]
        result = self._app("mismatch-aapl").run(
            "Using uploaded files only, what was Apple FY2025 operating income?",
            thread_id="apple-not-in-nvda",
            document_contexts=docs,
        )
        self.assertNotEqual(result.get("workflow_status"), "needs_clarification")
        self.assertEqual(result.get("companies"), ["Apple"])
        report = str(result.get("final_report") or "")
        self.assertNotIn("81.453", report)
        self.assertIn("Apple", report)
        self.assertNotRegex(report, r"请填写公司名称")
        self.assertRegex(
            report,
            r"do not contain statement evidence for Apple|not a substitute answer for Apple",
        )
        self.assertNotIn("lacked extractable revenue/EBITDA/R&D", report)
        apple = (result.get("retrieved_docs") or {}).get("Apple") or {}
        self.assertNotEqual(apple.get("structured_source"), "sample_db")
        self.assertFalse(any("81.453" in str(claim) for claim in (result.get("verified_claims") or [])))

    def test_peer_mention_without_apple_metrics_is_not_apple_evidence(self) -> None:
        docs = [
            {
                "issuer_companies": ["NVIDIA"],
                "detected_companies": ["NVIDIA", "Apple"],
                "filename": "nvda.pdf",
                "text": (
                    "NVIDIA FY2025 operating income was 81.453 billion USD. "
                    "The filing mentions Apple as a customer without Apple operating income."
                ),
                "excerpt": "NVIDIA FY2025 operating income was 81.453 billion USD.",
                "source_type": "pdf",
            }
        ]
        result = self._app("peer-mention").run(
            "Using uploaded files only, what was Apple FY2025 operating income?",
            thread_id="apple-peer-mention",
            document_contexts=docs,
        )
        self.assertEqual(result.get("companies"), ["Apple"])
        report = str(result.get("final_report") or "")
        self.assertNotIn("81.453", report)
        apple = (result.get("retrieved_docs") or {}).get("Apple") or {}
        self.assertIsNone(
            (apple.get("market_data") or {}).get("operating_income")
        )

    def test_partial_multi_company_keeps_available_issuer(self) -> None:
        docs = [
            {
                "issuer_companies": ["NVIDIA"],
                "detected_companies": ["NVIDIA"],
                "filename": "nvda.pdf",
                "text": "NVIDIA FY2025 operating income was 81.453 billion USD. Revenue 130.5. R&D 12.9.",
                "excerpt": "NVIDIA FY2025 operating income was 81.453 billion USD.",
                "source_type": "pdf",
            }
        ]
        result = self._app("partial-peers").run(
            "Using uploaded files only, what was Apple FY2025 operating income and NVIDIA FY2025 operating income?",
            thread_id="partial-aapl-nvda",
            document_contexts=docs,
        )
        self.assertEqual(set(result.get("companies") or []), {"Apple", "NVIDIA"})
        report = str(result.get("final_report") or "")
        self.assertIn("81.453", report)
        self.assertIn("NVIDIA", report)
        verified = result.get("verified_claims") or []
        self.assertTrue(
            any(
                (item.get("entity") if isinstance(item, dict) else getattr(item, "entity", None)) == "NVIDIA"
                and "operating_income"
                in str((item.get("metric_name") if isinstance(item, dict) else getattr(item, "metric_name", None)))
                for item in verified
            )
        )
        self.assertFalse(
            any(
                (item.get("entity") if isinstance(item, dict) else getattr(item, "entity", None)) == "Apple"
                and (item.get("verification") if isinstance(item, dict) else getattr(item, "verification", None))
                == "verified"
                and "operating_income"
                in str((item.get("metric_name") if isinstance(item, dict) else getattr(item, "metric_name", None)))
                for item in verified
            )
        )

    def test_bilateral_rd_keeps_nvidia_when_microsoft_is_named_first(self) -> None:
        docs = [
            {
                "issuer_companies": ["Microsoft"],
                "detected_companies": ["Microsoft"],
                "filename": "msft.pdf",
                "text": "Microsoft FY2024 research and development expense was 29.51 billion USD.",
                "excerpt": "Microsoft FY2024 research and development expense was 29.51 billion USD.",
                "source_type": "pdf",
            },
            {
                "issuer_companies": ["NVIDIA"],
                "detected_companies": ["NVIDIA"],
                "filename": "nvda.pdf",
                "text": "NVIDIA FY2025 research and development expense was 12.914 billion USD.",
                "excerpt": "NVIDIA FY2025 research and development expense was 12.914 billion USD.",
                "source_type": "pdf",
            },
        ]
        result = self._app("bilateral-rd").run(
            "Using uploaded files only, compare Microsoft FY2024 R&D expense with NVIDIA FY2025 R&D expense.",
            thread_id="msft-nvda-rd",
            document_contexts=docs,
        )
        report = str(result.get("final_report") or "")
        exec_summary = str(result.get("executive_summary") or "")
        self.assertIn("29.51", exec_summary)
        self.assertIn("12.914", exec_summary)
        self.assertIn("29.51", report)
        self.assertIn("12.914", report)
        self.assertNotIn("do not support a verified", exec_summary.lower())


if __name__ == "__main__":
    unittest.main()
