from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.planning import build_query_plan
from lumenfin.reporting import (
    build_metrics_csv_rows,
    effective_report_output_format,
    export_run_artifacts,
    format_next_actions,
    normalize_requested_output_format,
    write_metrics_csv,
)
from lumenfin import LumenFinAgentSystem
from lumenfin.llm import LocalFallbackLLMClient
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config
from uuid import uuid4


class OutputFormatContractTestCase(unittest.TestCase):
    def test_normalize_accepts_aliases_and_rejects_junk(self) -> None:
        self.assertEqual(normalize_requested_output_format("brief"), "executive_summary")
        self.assertEqual(normalize_requested_output_format("TABLE"), "table_summary")
        self.assertEqual(normalize_requested_output_format("research_report"), "research_report")
        self.assertIsNone(normalize_requested_output_format("foo"))
        self.assertIsNone(normalize_requested_output_format(""))
        self.assertIsNone(normalize_requested_output_format(None))

    def test_effective_mode_ignores_keyword_plan_output_format(self) -> None:
        # Even if planner stored "executive_summary" from keywords, report stays full
        # unless requested_output_format is set explicitly.
        state = {
            "query_plan": {"output_format": "executive_summary"},
            "query": "简版对比 Apple 和 Microsoft",
        }
        self.assertEqual(effective_report_output_format(state), "research_report")
        state["requested_output_format"] = "executive_summary"
        self.assertEqual(effective_report_output_format(state), "executive_summary")

    def test_keyword_still_detected_in_plan_but_not_used_for_trim(self) -> None:
        plan = build_query_plan("简版对比 Apple and Microsoft FY2024 profitability")
        self.assertEqual(plan.output_format, "executive_summary")
        self.assertEqual(
            effective_report_output_format({"query_plan": plan.to_dict()}),
            "research_report",
        )


class NextActionsAndCsvTestCase(unittest.TestCase):
    def test_next_actions_for_partial_gap(self) -> None:
        lines = format_next_actions(
            {
                "workflow_status": "completed",
                "partial_data_gap": True,
                "companies": ["Apple", "OpenAI"],
                "coverage_matrix": {
                    "Apple": {"comparable": True, "structured_source": "sample_db"},
                    "OpenAI": {"comparable": False, "structured_source": "none"},
                },
                "non_comparable_companies": ["OpenAI"],
                "data_gap_detail": "OpenAI missing AST inputs",
            }
        )
        text = "\n".join(lines)
        self.assertIn("Next Actions", text)
        self.assertIn("OpenAI", text)
        self.assertIn("Upload 10-K", text)

    def test_next_actions_empty_when_no_gap(self) -> None:
        self.assertEqual(
            format_next_actions({"workflow_status": "completed", "companies": ["Apple"]}),
            [],
        )

    def test_metrics_csv_rows_mirror_financial_metrics(self) -> None:
        rows = build_metrics_csv_rows(
            {
                "companies": ["Apple"],
                "workflow_status": "completed",
                "financial_metrics": {"Apple": {"ebitda_margin": 0.32}},
                "coverage_matrix": {"Apple": {"comparable": True, "structured_source": "sample_db"}},
            }
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["company"], "Apple")
        self.assertEqual(rows[0]["metric"], "ebitda_margin")
        self.assertEqual(rows[0]["value"], 0.32)
        self.assertEqual(rows[0]["comparable"], "yes")

    def test_export_writes_metrics_csv(self) -> None:
        result = {
            "workflow_status": "completed",
            "final_report": "# ok",
            "audit_log": [],
            "companies": ["Apple"],
            "financial_metrics": {"Apple": {"ebitda_margin": 0.3}},
            "coverage_matrix": {"Apple": {"comparable": True, "structured_source": "sample_db"}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = export_run_artifacts(result, Path(tmp), "t-brief")
            self.assertIn("metrics_csv_path", artifacts)
            path = Path(artifacts["metrics_csv_path"])
            self.assertTrue(path.exists())
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["metric"], "ebitda_margin")

    def test_write_metrics_csv_empty_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            write_metrics_csv(path, {"workflow_status": "incomplete_data", "companies": []})
            text = path.read_text(encoding="utf-8")
            self.assertIn("company,metric,value", text)


class ClarificationCopyTestCase(unittest.TestCase):
    def test_mismatch_question_is_short_and_optionized(self) -> None:
        plan = build_query_plan(
            "Analyze SoftBank FY2024 profitability using the uploaded materials.",
            document_contexts=[
                {"detected_companies": ["Apple", "Microsoft"], "filename": "table.pdf"}
            ],
        )
        self.assertIn("company_upload_mismatch", plan.missing_fields)
        joined = " ".join(plan.clarification_questions)
        self.assertIn("不一致", joined)
        self.assertIn("uploaded", joined)
        self.assertIn("query", joined)
        self.assertIn("both", joined)


class BriefReportEndToEndTestCase(unittest.TestCase):
    def test_explicit_brief_omits_thesis_keyword_alone_does_not(self) -> None:
        config = build_test_config(ROOT / "test_artifacts" / f"brief-{uuid4().hex[:8]}")
        system = LumenFinAgentSystem(
            llm_client=LocalFallbackLLMClient(),
            app_config=config,
            market_data_client=FakeMarketDataClient(),
        )
        query = "简版对比 Apple 与 Microsoft FY2024 profitability and R&D"
        full = system.run(query, thread_id=f"full-{uuid4().hex[:6]}")
        brief = system.run(
            query,
            thread_id=f"brief-{uuid4().hex[:6]}",
            output_format="executive_summary",
        )
        self.assertIn("Research Thesis & Positioning", full["final_report"])
        self.assertNotIn("Research Thesis & Positioning", brief["final_report"])
        self.assertNotIn("Strategic Analysis (SWOT)", brief["final_report"])
        self.assertIn("Financial Performance Analysis", brief["final_report"])
        self.assertIn("Brief Diligence", brief["final_report"])
        self.assertEqual(brief.get("requested_output_format"), "executive_summary")


if __name__ == "__main__":
    unittest.main()
