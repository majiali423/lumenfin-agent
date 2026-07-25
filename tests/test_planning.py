from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.planning import QueryStructureLLM, build_query_plan
from lumenfin.skills import SKILL_REGISTRY, get_skill_specs


class _StructureLLM:
    """Minimal stub that returns planner JSON for unit tests."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def chat(self, **kwargs: object) -> str:
        self.calls += 1
        return self.payload


class _SequenceLLM:
    """Return successive payloads across chat() calls (for retry tests)."""

    def __init__(self, payloads: list[str]) -> None:
        self.payloads = payloads
        self.calls = 0

    def chat(self, **kwargs: object) -> str:
        idx = min(self.calls, len(self.payloads) - 1)
        self.calls += 1
        return self.payloads[idx]


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

    def test_time_range_accepts_historical_years(self) -> None:
        plan = build_query_plan("Compare Apple and Microsoft 2018 profitability and R&D.")

        self.assertNotIn("time_range", plan.missing_fields)
        self.assertIn("Apple", plan.companies)

    def test_time_range_accepts_relative_and_range_phrases(self) -> None:
        cases = [
            "分析苹果这两年盈利能力",
            "对比微软最近三年研发投入",
            "Compare Apple profitability for 2015-2018",
            "Analyze Microsoft from 2015至2018",
        ]
        for query in cases:
            with self.subTest(query=query):
                plan = build_query_plan(query)
                self.assertNotIn("time_range", plan.missing_fields)

    def test_time_range_not_confused_by_clarify_substring(self) -> None:
        plan = build_query_plan("Clarify Apple profitability and supply chain risk.")

        self.assertIn("time_range", plan.missing_fields)

    def test_prefer_uploaded_only_from_english_phrasing(self) -> None:
        plan = build_query_plan(
            "Analyze Oracle FY2025 profitability using the uploaded note only.",
            document_contexts=[{"detected_companies": ["Oracle"], "filename": "note.pdf"}],
        )
        self.assertTrue(plan.prefer_uploaded_only)
        self.assertTrue(any("prefer_uploaded_only=true" in note for note in plan.planner_notes))

    def test_using_uploaded_without_only_stays_hybrid(self) -> None:
        plan = build_query_plan(
            "Compare Apple and Microsoft FY2025 using the uploaded consolidated table.",
            document_contexts=[{"detected_companies": ["Apple", "Microsoft"], "filename": "t.pdf"}],
        )
        self.assertFalse(plan.prefer_uploaded_only)

    def test_prefer_uploaded_only_from_chinese_phrasing(self) -> None:
        plan = build_query_plan(
            "仅用上传文件分析甲骨文盈利能力 FY2025",
            document_contexts=[{"detected_companies": ["Oracle"], "filename": "note.pdf"}],
        )
        self.assertTrue(plan.prefer_uploaded_only)

    def test_upload_without_only_phrasing_stays_hybrid(self) -> None:
        plan = build_query_plan(
            "Analyze Oracle FY2025 profitability and R&D.",
            document_contexts=[{"detected_companies": ["Oracle"], "filename": "note.pdf"}],
        )
        self.assertFalse(plan.prefer_uploaded_only)
        self.assertTrue(any("hybrid" in note for note in plan.planner_notes))

    def test_structured_llm_fills_long_tail_company_when_rules_miss(self) -> None:
        llm = _StructureLLM(
            '{"companies":["SoftBank"],"time_range":{"raw":"FY2024","has_time":true},'
            '"intent":"financial_diligence","dimensions":["profitability"],'
            '"confidence":{"companies":0.91,"time_range":0.88},'
            '"retrieval_query":"SoftBank FY2024 profitability analysis"}'
        )
        plan = build_query_plan(
            "Please diligence that Japanese conglomerate SoftBank Group for FY2024 margins.",
            llm_client=llm,  # type: ignore[arg-type]
        )
        # SoftBank is now in the alias map, so the rule layer may resolve it without LLM.
        self.assertIn("SoftBank", plan.companies)
        self.assertNotIn("company", plan.missing_fields)
        self.assertNotIn("time_range", plan.missing_fields)
        self.assertIn("SoftBank", plan.retrieval_query)

    def test_structured_llm_fills_unknown_issuer_when_alias_missing(self) -> None:
        llm = _StructureLLM(
            '{"companies":["Nomura Research Institute"],"time_range":{"raw":"FY2024","has_time":true},'
            '"intent":"financial_diligence","dimensions":["profitability"],'
            '"confidence":{"companies":0.93,"time_range":0.9},'
            '"retrieval_query":"Nomura Research Institute FY2024 profitability"}'
        )
        plan = build_query_plan(
            "Please diligence Nomura Research Institute for FY2024 margins.",
            llm_client=llm,  # type: ignore[arg-type]
        )
        self.assertEqual(llm.calls, 1)
        self.assertIn("Nomura Research Institute", plan.companies)

    def test_structured_llm_low_confidence_companies_are_ignored(self) -> None:
        llm = _StructureLLM(
            '{"companies":["TotallyFakeCorp"],"time_range":{"raw":"","has_time":false},'
            '"intent":"financial_diligence","dimensions":["profitability"],'
            '"confidence":{"companies":0.2,"time_range":0.1},'
            '"retrieval_query":"TotallyFakeCorp analysis"}'
        )
        plan = build_query_plan("Please analyze this business risk.", llm_client=llm)  # type: ignore[arg-type]
        self.assertNotIn("TotallyFakeCorp", plan.companies)
        self.assertIn("company", plan.missing_fields)

    def test_query_structure_pydantic_filters_unknown_dimensions(self) -> None:
        model = QueryStructureLLM.model_validate(
            {
                "companies": ["Apple"],
                "intent": "financial_diligence",
                "dimensions": ["profitability", "made_up_dim", "r_and_d"],
                "time_range": {"raw": "FY2025", "has_time": True},
                "confidence": {"companies": 0.9, "time_range": 0.9},
            }
        )
        self.assertEqual(model.dimensions, ["profitability", "r_and_d"])

    def test_structured_llm_retries_once_after_invalid_intent(self) -> None:
        bad = (
            '{"companies":["Nomura Research Institute"],"time_range":{"raw":"FY2024","has_time":true},'
            '"intent":"totally_invalid_intent","dimensions":["profitability"],'
            '"confidence":{"companies":0.93,"time_range":0.9},'
            '"retrieval_query":"Nomura Research Institute FY2024"}'
        )
        good = (
            '{"companies":["Nomura Research Institute"],"time_range":{"raw":"FY2024","has_time":true},'
            '"intent":"financial_diligence","dimensions":["profitability"],'
            '"confidence":{"companies":0.93,"time_range":0.9},'
            '"retrieval_query":"Nomura Research Institute FY2024 profitability"}'
        )
        llm = _SequenceLLM([bad, good])
        plan = build_query_plan(
            "Please diligence Nomura Research Institute for FY2024 margins.",
            llm_client=llm,  # type: ignore[arg-type]
        )
        self.assertEqual(llm.calls, 2)
        self.assertIn("Nomura Research Institute", plan.companies)

    def test_structured_llm_gives_up_after_two_invalid_payloads(self) -> None:
        bad = '{"intent":"not_a_real_intent","companies":["X"],"dimensions":[]}'
        llm = _SequenceLLM([bad, bad])
        plan = build_query_plan(
            "Please diligence Nomura Research Institute for FY2024 margins.",
            llm_client=llm,  # type: ignore[arg-type]
        )
        self.assertEqual(llm.calls, 2)
        # Rules cannot resolve this issuer; without valid LLM structure, company stays missing.
        self.assertNotIn("Nomura Research Institute", plan.companies)

    def test_skill_specs_are_registry_backed(self) -> None:
        specs = get_skill_specs(["financial_ratios", "unknown"])

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["name"], "financial_ratios")
        self.assertIn("financial_ratios", SKILL_REGISTRY)


if __name__ == "__main__":
    unittest.main()
