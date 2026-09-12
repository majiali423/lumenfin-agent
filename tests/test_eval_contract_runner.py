from __future__ import annotations

import unittest

from lumenfin.eval.gold_evaluator import score_task


def _fact_task() -> dict:
    return {
        "id": "unit-oi-runner",
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


class ContractRunnerTestCase(unittest.TestCase):
    def test_shared_wrong_value_can_pass_export_while_gold_fails(self) -> None:
        from finagentbench.runner import evaluate_run  # noqa: F401

        output = "NVIDIA FY2025 operating income was 72.4 billion USD."
        finrun = {
            "schema_version": "1.0",
            "run_id": "unit-wrong-consistent",
            "final_output": output,
            "entities": [{"name": "NVIDIA"}],
            "steps": [],
            "metrics": [{
                "entity": "NVIDIA",
                "name": "operating_income",
                "value": 72.4,
                "unit": "billion",
                "period": "FY2025",
                "currency": "USD",
                "source": "excerpt.pdf#p1",
            }],
            "evidence": [{
                "entity": "NVIDIA",
                "citation": "excerpt.pdf#p1",
                "period": "FY2025",
                "text": output,
            }],
            "market_data": [],
        }
        scored = score_task(_fact_task(), final_output=output, finrun=finrun, system="b2_agent")
        self.assertEqual(scored["layer_a_finagentbench"].get("runner"), "finagentbench.evaluate_run")
        self.assertEqual(scored["contract_result"]["status"], "passed")
        self.assertFalse(scored["task_result"]["passed"])
        self.assertFalse(scored["eval_acceptance_v1"])
        self.assertEqual(scored["b2_internal_result"]["status"], "passed")

    def test_refuse_with_number_fails_visible_claims_via_runner(self) -> None:
        from finagentbench.runner import evaluate_run  # noqa: F401

        task = {
            "id": "unit-refuse-runner",
            "family": "refuse_or_clarify",
            "expected_action": "refuse",
            "query": "What is Apple FY2025 operating income from the uploaded files?",
            "required_facts": [],
            "refuse": {"must_not_assert_values": [81.453]},
        }
        output = "I must refuse, but Apple FY2025 operating income was 81.453 billion USD."
        finrun = {
            "schema_version": "1.0",
            "run_id": "unit-refuse-number",
            "final_output": output,
            "entities": [{"name": "Apple"}],
            "steps": [],
            "metrics": [],
            "evidence": [],
            "market_data": [],
        }
        scored = score_task(task, final_output=output, finrun=finrun, system="b2_agent")
        self.assertFalse(scored["passed"])
        self.assertEqual(scored["contract_result"]["status"], "failed")
        vsc = next(
            item for item in scored["contract_result"]["checks"]
            if item["name"] == "visible_supported_claims"
        )
        self.assertEqual(vsc["status"], "failed")
