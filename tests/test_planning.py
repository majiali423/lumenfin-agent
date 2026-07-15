from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.planning import build_query_plan
from lumenfin.skills import SKILL_REGISTRY, get_skill_specs


class QueryPlanningTestCase(unittest.TestCase):
    def test_query_plan_extracts_companies_dimensions_and_skills(self) -> None:
        plan = build_query_plan(
            "Compare Apple and Microsoft 2025 profitability, R&D, supply chain risk, and compliance."
        )

        self.assertEqual(plan.intent, "comparative_financial_diligence")
        self.assertIn("Apple", plan.companies)
        self.assertIn("Microsoft", plan.companies)
        self.assertIn("profitability", plan.analysis_dimensions)
        self.assertIn("r_and_d", plan.analysis_dimensions)
        self.assertIn("supply_chain", plan.analysis_dimensions)
        self.assertIn("financial_ratios", plan.required_skills)
        self.assertIn("compliance_review", plan.required_skills)
        self.assertEqual(plan.missing_fields, [])

    def test_query_plan_marks_missing_company_without_blocking_workflow(self) -> None:
        plan = build_query_plan("Please analyze this business risk.")

        self.assertIn("company", plan.missing_fields)
        self.assertTrue(plan.clarification_questions)
        self.assertIn("company_identification", plan.required_skills)

    def test_query_plan_recognizes_private_company_without_ticker_hint(self) -> None:
        plan = build_query_plan("Analyze OpenAI FY2025 profitability using live fundamentals only.")

        self.assertEqual(plan.companies, ["OpenAI"])
        self.assertNotIn("company", plan.missing_fields)

    def test_skill_specs_are_registry_backed(self) -> None:
        specs = get_skill_specs(["financial_ratios", "unknown"])

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["name"], "financial_ratios")
        self.assertIn("financial_ratios", SKILL_REGISTRY)


if __name__ == "__main__":
    unittest.main()
