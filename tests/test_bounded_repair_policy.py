from __future__ import annotations

import unittest

from lumenfin.bounded_repair import run_bounded_repair, sample_fill_allowed


def _state(**overrides: object) -> dict:
    payload = {
        "retrieved_docs": {"Apple": {"market_data": {}, "structured_source": "none"}},
        "companies": ["Apple"],
        "task_spec": {
            "intent": "financial_diligence",
            "dimensions": ["profitability"],
            "requires_ast_ratios": True,
            "skip_quant": False,
            "allow_risk_without_ratios": False,
            "allowed_tools": ["sample_fundamentals", "safe_ratio", "stop"],
            "max_repair_steps": 2,
            "max_tool_calls": 3,
        },
    }
    payload.update(overrides)
    return payload


class BoundedRepairPolicyTestCase(unittest.TestCase):
    def test_live_mode_does_not_fill_sample(self) -> None:
        allowed, reason = sample_fill_allowed(
            _state(data_mode="live"),
            "Apple",
            allow_sample_data=False,
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "sample_not_permitted")
        update = run_bounded_repair(_state(data_mode="live"), allow_sample_data=False)
        self.assertNotEqual(update["retrieved_docs"]["Apple"].get("structured_source"), "sample_db")
        self.assertTrue(update["fatal_data_gap"])

    def test_uploaded_only_does_not_fill_sample(self) -> None:
        state = _state(
            data_mode="demo",
            document_contexts=[{"filename": "10k.md", "text": "risk only", "page": 1}],
        )
        allowed, reason = sample_fill_allowed(state, "Apple", allow_sample_data=True)
        self.assertFalse(allowed)
        self.assertEqual(reason, "uploaded_only")
        update = run_bounded_repair(state, allow_sample_data=True)
        self.assertNotEqual(update["retrieved_docs"]["Apple"].get("structured_source"), "sample_db")

    def test_prefer_uploaded_only_without_contexts_does_not_fill_sample(self) -> None:
        state = _state(
            data_mode="demo",
            query="Analyze Apple FY2025 using uploaded files only",
            document_contexts=[],
            query_plan={"prefer_uploaded_only": True, "time_range": "FY2025"},
        )
        allowed, reason = sample_fill_allowed(state, "Apple", allow_sample_data=True)
        self.assertFalse(allowed)
        self.assertEqual(reason, "uploaded_only")
        update = run_bounded_repair(state, allow_sample_data=True)
        self.assertNotEqual(update["retrieved_docs"]["Apple"].get("structured_source"), "sample_db")
        self.assertTrue(update["fatal_data_gap"])

    def test_multi_year_compare_does_not_fill_single_sample_year(self) -> None:
        state = _state(
            data_mode="demo",
            query="Compare Apple FY2024 and FY2025 revenue",
            query_plan={"prefer_uploaded_only": False, "time_range": "FY2024-FY2025"},
        )
        allowed, reason = sample_fill_allowed(state, "Apple", allow_sample_data=True)
        self.assertFalse(allowed)
        self.assertEqual(reason, "period_mismatch")
        update = run_bounded_repair(state, allow_sample_data=True)
        self.assertNotEqual(update["retrieved_docs"]["Apple"].get("structured_source"), "sample_db")

    def test_wrong_year_does_not_fill_sample(self) -> None:
        state = _state(data_mode="demo", query="Analyze Apple FY2099 using uploads only")
        allowed, reason = sample_fill_allowed(state, "Apple", allow_sample_data=True)
        self.assertFalse(allowed)
        self.assertEqual(reason, "period_mismatch")
        update = run_bounded_repair(state, allow_sample_data=True)
        self.assertNotEqual(update["retrieved_docs"]["Apple"].get("structured_source"), "sample_db")
        self.assertTrue(any(event.get("reason") == "period_mismatch" for event in update["bounded_repair_trace"]))


if __name__ == "__main__":
    unittest.main()
