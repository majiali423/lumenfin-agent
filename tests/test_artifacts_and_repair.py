from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.artifacts import (
    RetrievalArtifact,
    RetrievalConfidence,
    RetrievalProvenance,
    Violation,
    score_retrieval_confidence,
)
from lumenfin.critic_checks import (
    check_data_completeness,
    check_report_compliance,
    check_retrieval_provenance,
    run_critic_checks,
)
from lumenfin.critic_repair import classify_critic_repair_target, classify_critic_violations
from lumenfin.repair_policies import REPAIR_POLICIES, resolve_repair_target


class RepairPolicyTestCase(unittest.TestCase):
    def test_quant_violation_wins_over_sentiment_when_both_present(self) -> None:
        violations = [
            Violation(code="missing_sentiment_analysis", severity="medium", message="sentiment gap"),
            Violation(code="missing_quantitative_results", severity="high", message="quant gap"),
        ]
        self.assertEqual(resolve_repair_target(violations), "quant")

    def test_repair_target_is_stable_when_message_text_changes(self) -> None:
        violations = [
            Violation(
                code="missing_quantitative_results",
                severity="high",
                message="Totally different human wording about quant failure.",
            )
        ]
        self.assertEqual(classify_critic_violations(violations), "quant")

    def test_legacy_string_wrapper_still_routes_quant_and_sentiment(self) -> None:
        self.assertEqual(
            classify_critic_repair_target(["Apple: missing quantitative results."]),
            "quant",
        )
        self.assertEqual(
            classify_critic_repair_target(["Apple: missing sentiment analysis."]),
            "psychologist",
        )

    def test_unknown_violation_falls_back_to_retrieval(self) -> None:
        violations = [Violation(code="unexpected_issue", severity="low", message="unknown")]
        self.assertEqual(resolve_repair_target(violations), "retrieval")

    def test_policy_table_covers_all_critic_codes(self) -> None:
        expected_codes = {
            "missing_quantitative_results",
            "missing_sentiment_analysis",
            "missing_structured_data",
            "low_retrieval_confidence",
            "missing_risk_disclaimer",
            "missing_data_provenance",
        }
        policy_codes = {policy.code for policy in REPAIR_POLICIES}
        self.assertTrue(expected_codes.issubset(policy_codes))


class CriticChecksTestCase(unittest.TestCase):
    def test_live_mode_fails_when_structured_source_is_none(self) -> None:
        state = {
            "data_mode": "live",
            "companies": ["Contoso"],
            "financial_metrics": {"Contoso": {"ebitda_margin": 0.2}},
            "sentiment_analysis": {"Contoso": {"label": "neutral"}},
            "retrieved_docs": {
                "Contoso": {
                    "structured_source": "none",
                    "provenance": {
                        "structured_source": "none",
                        "data_mode": "live",
                    },
                    "confidence": {"overall": 0.1},
                }
            },
        }
        violations = check_retrieval_provenance(state)
        codes = {item.code for item in violations}
        self.assertIn("missing_structured_data", codes)

    def test_demo_mode_does_not_require_live_structured_data(self) -> None:
        state = {
            "data_mode": "demo",
            "companies": ["Apple"],
            "retrieved_docs": {"Apple": {"structured_source": "sample_db"}},
        }
        self.assertEqual(check_retrieval_provenance(state), [])

    def test_report_compliance_detects_missing_english_disclaimer(self) -> None:
        violations = check_report_compliance("Executive summary only. No legal section.")
        codes = {item.code for item in violations}
        self.assertIn("missing_risk_disclaimer", codes)

    def test_report_compliance_accepts_bilingual_markers(self) -> None:
        report = (
            "Executive summary.\n"
            "Methodology and Data Sources: uploaded PDFs.\n"
            "Disclaimer: not investment advice."
        )
        self.assertEqual(check_report_compliance(report), [])

    def test_run_critic_checks_collects_multiple_issue_types(self) -> None:
        state = {
            "data_mode": "demo",
            "companies": ["Apple", "Microsoft"],
            "financial_metrics": {"Apple": {"ebitda_margin": 0.3}},
            "sentiment_analysis": {"Apple": {"label": "bullish"}},
            "report_sections": ["Executive summary only. No legal section or data-source notes."],
        }
        violations = run_critic_checks(state)
        codes = {item.code for item in violations}
        self.assertIn("missing_quantitative_results", codes)
        self.assertIn("missing_sentiment_analysis", codes)
        self.assertIn("missing_risk_disclaimer", codes)


class RetrievalArtifactTestCase(unittest.TestCase):
    def test_legacy_payload_preserves_provenance_and_confidence(self) -> None:
        artifact = RetrievalArtifact(
            company="Apple",
            market_data={"revenue_2025": 412.0},
            supply_chain={"risk_level": "medium", "signals": []},
            earnings_call_quotes=["quote"],
            source_documents=[{"filename": "apple.md", "excerpt": "revenue 412"}],
            market_snapshot={"provider": "fake", "status": "ok", "current_price": 180.0},
            profile="Apple profile.",
            rag_hits=[{"text": "chunk"}],
            provenance=RetrievalProvenance(
                structured_source="sample_db",
                market_provider="fake",
                market_status="ok",
                rag_enabled=True,
                rag_hit_count=1,
                document_count=0,
                data_mode="demo",
            ),
            confidence=RetrievalConfidence(overall=0.9, market_data=1.0, live_market=1.0, rag_coverage=0.33),
            structured_source="sample_db",
        )
        payload = artifact.to_legacy_payload()
        self.assertEqual(payload["market_data"]["revenue_2025"], 412.0)
        self.assertEqual(payload["provenance"]["structured_source"], "sample_db")
        self.assertAlmostEqual(payload["confidence"]["overall"], 0.9)

    def test_confidence_scores_penalize_failed_market_and_missing_fundamentals(self) -> None:
        confidence = score_retrieval_confidence(
            market_data={},
            live_market={"status": "failed"},
            rag_hits=[],
        )
        self.assertLess(confidence.overall, 0.2)
        self.assertEqual(confidence.market_data, 0.0)
        self.assertEqual(confidence.live_market, 0.0)


if __name__ == "__main__":
    unittest.main()
