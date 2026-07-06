from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.evaluation import evaluate_run_state


class AgentEvaluationTestCase(unittest.TestCase):
    def test_complete_trace_scores_as_portfolio_ready(self) -> None:
        state = {
            "thread_id": "eval-test",
            "companies": ["Apple"],
            "audit_log": [
                {"step": "query_planner", "status": "ok", "detail": "planned query"},
                {"step": "supervisor", "status": "ok", "detail": "planned"},
                {"step": "retrieval", "status": "ok", "detail": "retrieved"},
                {"step": "quant", "status": "ok", "detail": "computed"},
                {"step": "psychologist", "status": "ok", "detail": "sentiment"},
                {"step": "critic", "status": "ok", "detail": "checked"},
                {"step": "synthesizer", "status": "ok", "detail": "reported"},
            ],
            "retrieved_docs": {"Apple": {"market_data": {"revenue_2025": 100.0}}},
            "financial_metrics": {"Apple": {"ebitda_margin": 0.3}},
            "risk_scores": {"Apple": {"financial_risk": 2.0}},
            "sentiment_analysis": {"Apple": {"label": "bullish"}},
            "final_report": (
                "# Report\nExecutive Summary\nFinancial Performance Analysis\nRisk\n"
                "Compliance\nMethodology\nDisclaimer\n" + ("detail " * 500)
            ),
            "audit_log_path": "unused",
        }

        result = evaluate_run_state(state)

        self.assertGreaterEqual(result.score, 90)
        self.assertEqual(result.grade, "production-ready demo")

    def test_missing_steps_are_detected(self) -> None:
        state = {"companies": ["Apple"], "audit_log": [], "final_report": ""}

        result = evaluate_run_state(state)

        self.assertLess(result.score, 60)
        self.assertIn("supervisor", result.checks["pipeline_completeness"]["missing_steps"])


if __name__ == "__main__":
    unittest.main()
