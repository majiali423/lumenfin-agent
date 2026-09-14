from __future__ import annotations

import unittest

from lumenfin.eval.baselines import run_input_from_task
from lumenfin.eval.contract_policy import (
    FAMILY_CHECK_MATRIX,
    SCORING_POLICY_VERSION,
    expected_subjects,
    missing_subjects_error,
    output_has_numeric_assertions,
)
from lumenfin.eval.gold_evaluator import score_task


def _fact_task() -> dict:
    return {
        "id": "unit-oi",
        "family": "financial_fact",
        "expected_action": "answer",
        "query": "Using uploaded files only, what is NVIDIA FY2025 operating income?",
        "required_facts": [{
            "entity": "NVIDIA",
            "metric": "operating_income",
            "period": "FY2025",
            "value": 81.453,
            "unit": "billion",
            "abs_tolerance": 0.002,
        }],
        "allowed_documents": [{"path": "excerpt.pdf", "document_id": "excerpt"}],
        "forbidden_claims": [],
    }


def _compare_task() -> dict:
    return {
        "id": "unit-compare",
        "family": "comparison_scope",
        "expected_action": "answer",
        "query": "Compare NVIDIA and Microsoft FY2025 operating income from the uploads.",
        "required_facts": [
            {
                "entity": "NVIDIA",
                "metric": "operating_income",
                "period": "FY2025",
                "value": 81.453,
                "unit": "billion",
                "abs_tolerance": 0.002,
            },
            {
                "entity": "Microsoft",
                "metric": "operating_income",
                "period": "FY2025",
                "value": 109.0,
                "unit": "billion",
                "abs_tolerance": 0.05,
            },
        ],
        "allowed_documents": [{"path": "excerpt.pdf", "document_id": "excerpt"}],
        "forbidden_claims": [],
    }


def _refuse_task() -> dict:
    return {
        "id": "unit-refuse",
        "family": "refuse_or_clarify",
        "expected_action": "refuse",
        "query": "What is Apple FY2025 operating income from the uploaded files?",
        "required_facts": [],
        "refuse": {"must_not_assert_values": [81.453]},
        "forbidden_claims": [],
    }


def _finrun(output: str, *, metrics: list | None = None, entities: list | None = None, evidence: list | None = None) -> dict:
    rows = metrics or []
    return {
        "schema_version": "1.0",
        "run_id": "unit-contract",
        "final_output": output,
        "entities": entities if entities is not None else [{"name": "NVIDIA"}],
        "steps": [],
        "metrics": rows,
        "evidence": evidence if evidence is not None else [
            {
                "entity": "NVIDIA",
                "citation": "excerpt.pdf#p1",
                "period": "FY2025",
                "text": output,
            }
        ],
        "market_data": [],
    }


class ContractPolicyTestCase(unittest.TestCase):
    def test_family_matrix_covers_six_families(self) -> None:
        self.assertEqual(len(FAMILY_CHECK_MATRIX), 6)
        self.assertIn("visible_supported_claims", FAMILY_CHECK_MATRIX["financial_fact"]["bench_shared"])
        self.assertIn("numeric_correctness", FAMILY_CHECK_MATRIX["ratio_calc"]["bench_internal"])

    def test_complete_single_fact(self) -> None:
        output = "NVIDIA FY2025 operating income was 81.453 billion USD [excerpt.pdf#p1]."
        scored = score_task(_fact_task(), final_output=output, system="b1_rag")
        self.assertEqual(scored["scoring_policy_version"], SCORING_POLICY_VERSION)
        self.assertTrue(scored["task_result"]["passed"])
        self.assertTrue(scored["diagnostic_pass"])
        self.assertEqual(scored["contract_result"]["status"], "not_applicable")
        self.assertTrue(scored["eval_acceptance_v1"])
        self.assertFalse(scored["formal_pass"])

    def test_right_number_wrong_unit_company_or_period(self) -> None:
        task = _fact_task()
        unit = score_task(task, final_output="NVIDIA FY2025 operating income was 81.453 million USD [excerpt.pdf#p1].")
        company = score_task(task, final_output="Microsoft FY2025 operating income was 81.453 billion USD [excerpt.pdf#p1].")
        period = score_task(task, final_output="NVIDIA FY2024 operating income was 81.453 billion USD [excerpt.pdf#p1].")
        self.assertFalse(unit["task_result"]["passed"])
        self.assertFalse(company["task_result"]["passed"])
        self.assertFalse(period["task_result"]["passed"])
        self.assertFalse(unit["diagnostic_pass"])

    def test_compare_two_companies_but_answer_one_fails_completeness(self) -> None:
        output = "NVIDIA FY2025 operating income was 81.453 billion USD [excerpt.pdf#p1]."
        scored = score_task(_compare_task(), final_output=output, system="b1_rag")
        self.assertEqual(scored["task_result"]["required_fact_coverage"]["hits"], 1)
        self.assertEqual(scored["task_result"]["required_fact_coverage"]["total"], 2)
        self.assertFalse(scored["passed"])
        self.assertTrue(any(item.startswith("missing_fact:Microsoft:") for item in scored["findings"]))
        self.assertIn("NVIDIA", expected_subjects(_compare_task()))
        self.assertIn("Microsoft", expected_subjects(_compare_task()))

    def test_correct_refuse_without_numbers_is_not_a_numeric_fail(self) -> None:
        output = (
            "The uploaded files do not provide Apple FY2025 operating income. "
            "I cannot answer with a number."
        )
        self.assertFalse(output_has_numeric_assertions(output))
        scored = score_task(_refuse_task(), final_output=output, system="b1_rag")
        self.assertTrue(scored["task_result"]["passed"], scored["findings"])
        self.assertTrue(scored["diagnostic_pass"])
        self.assertEqual(scored["contract_result"]["status"], "not_applicable")
        vsc = next(
            item for item in scored["contract_result"]["checks"]
            if item["name"] == "visible_supported_claims"
        )
        self.assertEqual(vsc["status"], "not_applicable")
        self.assertTrue(scored["eval_acceptance_v1"])

    def test_percent_sign_and_sentence_punctuation_are_numeric_assertions(self) -> None:
        self.assertTrue(
            output_has_numeric_assertions("I cannot answer. NVIDIA operating margin is 99.9%.")
        )
        self.assertTrue(output_has_numeric_assertions("Operating margin was 62.42%."))
        self.assertTrue(output_has_numeric_assertions("The margin is 99.9 percent."))
        self.assertTrue(output_has_numeric_assertions("Operating income was 81.453 billion."))
        self.assertFalse(
            output_has_numeric_assertions(
                "I cannot answer from aapl_fy2024_10k_extract.html page 1 (FY2024)."
            )
        )
        self.assertFalse(
            output_has_numeric_assertions("See nvda_fy2025_10k_excerpt.pdf#p2 filed in 2024.")
        )
        self.assertFalse(output_has_numeric_assertions("The uploaded files do not provide a number."))

    def test_refuse_with_unsupported_number_does_not_skip_checks(self) -> None:
        output = "I must refuse, but Apple FY2025 operating income was 81.453 billion USD."
        self.assertTrue(output_has_numeric_assertions(output))
        finrun = _finrun(output, metrics=[], entities=[{"name": "Apple"}], evidence=[])
        scored = score_task(
            _refuse_task(),
            final_output=output,
            finrun=finrun,
            system="b2_agent",
        )
        self.assertFalse(scored["task_result"]["passed"])
        self.assertTrue(
            "expected_refuse" in scored["findings"] or "forbidden_claim" in scored["findings"],
            scored["findings"],
        )
        self.assertIn(scored["contract_result"]["status"], {"failed", "unavailable", "error"})
        self.assertFalse(scored["eval_acceptance_v1"])

    def test_missing_expected_subjects_does_not_default_nvidia(self) -> None:
        task = {
            "id": "unit-missing-subjects",
            "family": "comparison_scope",
            "expected_action": "answer",
            "query": "Compare operating income for both issuers in the uploads.",
            "required_facts": [
                {"metric": "operating_income", "period": "FY2025", "value": 1.0, "unit": "billion"},
                {"metric": "operating_income", "period": "FY2025", "value": 2.0, "unit": "billion"},
            ],
        }
        self.assertEqual(expected_subjects(task), [])
        self.assertIsNotNone(missing_subjects_error(task))
        scored = score_task(
            task,
            final_output="Operating income figures are not complete.",
            system="b1_rag",
        )
        self.assertEqual(scored["contract_result"]["status"], "error")
        subjects = next(item for item in scored["contract_findings"] if item["name"] == "expected_subjects")
        self.assertEqual(subjects["status"], "error")
        self.assertTrue(any("refusing to default a subject" in str(item) for item in subjects["detail"]))
        self.assertFalse(scored["eval_acceptance_v1"])

    def test_required_bench_error_is_not_a_pass(self) -> None:
        output = "NVIDIA FY2025 operating income was 81.453 billion USD [excerpt.pdf#p1]."
        scored = score_task(
            _fact_task(),
            final_output=output,
            finrun={"schema_version": "1.0", "run_id": "broken", "final_output": output},
            system="b2_agent",
        )
        self.assertTrue(scored["diagnostic_pass"])
        self.assertTrue(scored["task_result"]["passed"])
        self.assertIn(scored["contract_result"]["status"], {"error", "unavailable"})
        self.assertFalse(scored["contract_result"]["passed"])
        self.assertFalse(scored["eval_acceptance_v1"])

    def test_equivalent_units_and_display_precision_still_pass_gold(self) -> None:
        billions = score_task(
            _fact_task(),
            final_output="NVIDIA FY2025 operating income was 81.453 billion USD [excerpt.pdf#p1].",
        )
        millions = score_task(
            _fact_task(),
            final_output="NVIDIA FY2025 operating income was 81,453 million USD [excerpt.pdf#p1].",
        )
        self.assertTrue(billions["passed"])
        self.assertTrue(millions["passed"])
        self.assertEqual(billions["diagnostic_pass"], millions["diagnostic_pass"])

    def test_scoring_labels_do_not_enter_b1_b2_run_input(self) -> None:
        task = _compare_task()
        baseline = run_input_from_task(task)
        mutated = dict(task)
        mutated["family"] = "refuse_or_clarify"
        mutated["expected_action"] = "refuse"
        mutated["scoring_eligibility"] = "diagnostic_only"
        mutated["required_facts"] = [{"entity": "Apple", "metric": "revenue", "value": 1}]
        mutated["scoring_contract"] = {"expected_entities": ["Apple"]}
        mutated["enabled_metrics"] = ["visible_supported_claims"]
        self.assertEqual(run_input_from_task(mutated), baseline)
        self.assertEqual(set(baseline), {"id", "query", "allowed_documents", "user_scope"})

    def test_both_internal_and_visible_wrong_gold_fails(self) -> None:
        output = "NVIDIA FY2025 operating income was 72.4 billion USD."
        metrics = [{
            "entity": "NVIDIA",
            "name": "operating_income",
            "value": 72.4,
            "unit": "billion",
            "period": "FY2025",
            "currency": "USD",
            "source": "excerpt.pdf#p1",
        }]
        scored = score_task(
            _fact_task(),
            final_output=output,
            finrun=_finrun(output, metrics=metrics),
            system="b2_agent",
        )
        self.assertFalse(scored["task_result"]["passed"])
        self.assertFalse(scored["diagnostic_pass"])
        contract_status = scored["contract_result"]["status"]
        if contract_status not in {"unavailable", "error"}:
            self.assertEqual(contract_status, "passed")
