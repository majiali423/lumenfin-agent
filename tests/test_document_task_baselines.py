from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

from scripts.offline_env import apply_offline_env

apply_offline_env()

from lumenfin.eval.baselines import b1_retrieval_plan, run_b1, run_b2, run_input_from_task, score_run
from lumenfin.eval.document_tasks import tasks_for


class BaselineSmokeTestCase(unittest.TestCase):
    def test_b1_and_b2_offline_do_not_read_gold_and_record_failures(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p21-narrative-refuse-oi")
        root = Path("test_artifacts") / f"baseline-{uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=True)
        b1 = score_run(task, run_b1(task, root=root / "b1", offline=True))
        b2 = score_run(task, run_b2(task, root=root / "b2", offline=True))
        self.assertEqual(b1["task_id"], task["id"])
        self.assertEqual(b2["task_id"], task["id"])
        self.assertNotIn("81.453", task["query"])
        self.assertIn("gold", b1)
        self.assertIn("gold", b2)
        self.assertIsNotNone(b1["gold"]["passed"] in {True, False})
        self.assertTrue(b1["system"].startswith("b1"))
        self.assertTrue(b2["system"].startswith("b2"))
        self.assertIsNone(b1.get("finrun"))
        self.assertFalse(b1["accuracy_eligible"])
        self.assertEqual(b1["result_class"], "offline_local_fallback")
        self.assertIn(b1["cost"]["status"], {"unknown_usage", "offline_no_billable_tokens", "unknown_price", "priced", "offline_no_billable_tokens"})

    def test_b1_plan_ignores_gold_and_keeps_all_query_companies(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if "compare NVIDIA" in item["query"] and "Microsoft" in item["query"])
        plan = b1_retrieval_plan(run_input_from_task(task))
        mutated = dict(task)
        mutated["required_facts"] = [{"entity": "Apple", "metric": "operating_income", "value": 1}]
        mutated["forbidden_claims"] = [{"entity": "Apple"}]
        mutated["family"] = "refuse_or_clarify"
        mutated["scoring_eligibility"] = "diagnostic_only"
        mutated.pop("refuse", None)
        after = b1_retrieval_plan(run_input_from_task(mutated))
        self.assertEqual(plan, after)
        self.assertEqual(run_input_from_task(task), run_input_from_task(mutated))
        self.assertIn("NVIDIA", plan["companies"])
        self.assertIn("Microsoft", plan["companies"])
        self.assertGreaterEqual(len(plan["companies"]), 2)
        self.assertNotIn("Apple", plan["companies"])
