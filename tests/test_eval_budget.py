from __future__ import annotations

import unittest

from lumenfin.eval.budget import (
    AUTH_TOKEN,
    EVAL_BUDGET,
    BudgetExhausted,
    EvalBudget,
    budget_packet,
)
from lumenfin.llm import LocalFallbackLLMClient, BaseLLMClient
from lumenfin.eval.budget import BudgetedLLMClient


class EvalBudgetTestCase(unittest.TestCase):
    def test_printer_and_runner_share_one_definition(self) -> None:
        packet = budget_packet(n_tasks=24)
        self.assertEqual(packet["id"], AUTH_TOKEN)
        self.assertEqual(packet["max_provider_requests"], EVAL_BUDGET["max_provider_requests"])
        self.assertEqual(packet["logical_calls_upper"], EVAL_BUDGET["logical_calls_upper"])
        self.assertEqual(packet["max_attempts_per_call"], 2)
        self.assertEqual(packet["b2_logical_calls_upper_per_task"], 19)
        self.assertEqual(packet["max_provider_requests"], (24 + 24 * 19) * 2)
        self.assertFalse(packet["pricing"]["fee_hard_cap_supported"])
        self.assertEqual(packet["model_id"], "deepseek-flash")

    def test_stops_new_requests_and_keeps_completed(self) -> None:
        spec = dict(EVAL_BUDGET)
        spec["max_provider_requests"] = 1
        budget = EvalBudget(spec=spec)
        client = BudgetedLLMClient(LocalFallbackLLMClient(), budget, spec)
        client.chat("sys", "hello", max_tokens=800)
        self.assertEqual(budget.logical_calls, 1)
        with self.assertRaises(BudgetExhausted):
            client.chat("sys", "again")
        self.assertTrue(budget.exhausted)
        snap = budget.snapshot()
        self.assertGreaterEqual(snap["provider_requests"], 1)
        self.assertTrue(snap["usd_unknown_until_usage"])

    def test_smoke_packet_keeps_full_http_cap(self) -> None:
        smoke = budget_packet(n_tasks=24, smoke=True)
        full = budget_packet(n_tasks=24, smoke=False)
        self.assertEqual(smoke["max_provider_requests"], full["max_provider_requests"])
        self.assertEqual(smoke["max_provider_requests"], 960)

    def test_combined_input_clip(self) -> None:
        from lumenfin.eval.budget import clip_combined_input

        system, user = clip_combined_input("s" * 8000, "u" * 8000, max_chars=12000)
        self.assertLessEqual(len(system) + len(user), 12000)
        self.assertLess(len(system), 8000)

    def test_fair_rerun_uses_remaining_not_a_new_cap(self) -> None:
        import tempfile
        from pathlib import Path

        from lumenfin.eval.budget import fair_rerun_budget_report

        spec = dict(EVAL_BUDGET)
        budget = EvalBudget(spec=spec, provider_requests=100, logical_calls=100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget_state.json"
            budget.save(path)
            other_out = Path(tmp) / "pilot_live_fair_v2.json"
            other_out.write_text("{}", encoding="utf-8")
            report = fair_rerun_budget_report(path)
            self.assertEqual(report["cap_http_total"], 960)
            self.assertEqual(report["used_http"], 100)
            self.assertEqual(report["remaining_http"], 860)
            self.assertFalse(report["new_quota_granted"])
            self.assertTrue(report["out_file_does_not_reset_cap"])
            self.assertEqual(report["fair_rerun"]["http_this_run_cannot_exceed_remaining"], 860)
            self.assertTrue(report["fair_rerun"]["expected_http"]["theoretical_upper_exceeds_remaining"])
            self.assertIn("Stop new provider HTTP", report["fair_rerun"]["on_exhaustion"])
            self.assertIn("剩余的 860", report["authorization_request"])
            self.assertIn("不是新的 960", report["authorization_request"])

    def test_budget_persists_across_load(self) -> None:
        import tempfile
        from pathlib import Path

        from lumenfin.eval.budget import load_eval_budget

        spec = dict(EVAL_BUDGET)
        budget = EvalBudget(spec=spec, provider_requests=17, logical_calls=9)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "budget_state.json"
            budget.save(path)
            loaded = load_eval_budget(path)
            self.assertEqual(loaded.provider_requests, 17)
            self.assertEqual(loaded.logical_calls, 9)
            self.assertEqual(loaded.max_provider_requests, 960)

    def test_http_hook_counts_failed_retry_before_post(self) -> None:
        class Flaky(BaseLLMClient):
            backend_name = "deepseek"
            model_name = "deepseek-flash"

            def __init__(self) -> None:
                super().__init__()
                self.settings = type("S", (), {"max_retries": 2})()
                self.posts = 0

            def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
                hook = getattr(self, "_before_provider_http", None)
                if callable(hook):
                    hook()
                self.posts += 1
                if self.posts == 1:
                    if callable(hook):
                        hook()
                    self.posts += 1
                    self.last_attempts = 2
                    raise RuntimeError("http 500")
                self.last_attempts = 1
                return "ok"

        spec = dict(EVAL_BUDGET)
        spec["max_provider_requests"] = 3
        budget = EvalBudget(spec=spec)
        client = BudgetedLLMClient(Flaky(), budget, spec)
        with self.assertRaises(RuntimeError):
            client.chat("sys", "hello")
        self.assertEqual(budget.provider_requests, 2)
        client.chat("sys", "retry-ok")
        self.assertEqual(budget.provider_requests, 3)

    def test_fork_shares_usage_totals(self) -> None:
        spec = dict(EVAL_BUDGET)
        spec["max_provider_requests"] = 10
        budget = EvalBudget(spec=spec)
        parent = BudgetedLLMClient(LocalFallbackLLMClient(), budget, spec)
        child = parent.fork_usage()
        child.chat("sys", "hello")
        self.assertGreater(parent._usage_totals["prompt_tokens"], 0)
        self.assertEqual(parent._usage_totals, child._usage_totals)
