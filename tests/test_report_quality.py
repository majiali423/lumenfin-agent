from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin import LumenFinAgentSystem
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.reporting import (
    build_analyst_executive_summary,
    filter_claims_for_brief,
    format_peer_metric_matrix,
    format_period_alignment_notice,
    parse_requested_fiscal_year,
    requested_fiscal_year_from_state,
)
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class ReportQualityHelpersTestCase(unittest.TestCase):
    def test_parse_requested_fiscal_year(self) -> None:
        self.assertEqual(parse_requested_fiscal_year("FY2024 profitability"), 2024)
        self.assertEqual(parse_requested_fiscal_year("Compare Apple 2024 vs Microsoft"), 2024)

    def test_period_alignment_flags_mismatch(self) -> None:
        state = {
            "query": "Compare Apple and Microsoft FY2024 profitability",
            "query_plan": {"time_range": "FY2024"},
            "companies": ["Apple", "Microsoft"],
            "retrieved_docs": {
                "Apple": {"fundamentals_meta": {"fiscal_year": 2025, "period_alignment": "fallback_latest"}},
                "Microsoft": {"fundamentals_meta": {"fiscal_year": 2025, "period_alignment": "fallback_latest"}},
            },
        }
        text = "\n".join(format_period_alignment_notice(state))
        self.assertIn("Period Alignment", text)
        self.assertIn("FALLBACK", text)
        self.assertIn("FY2024", text)

    def test_period_alignment_assumed_from_query_is_honest(self) -> None:
        state = {
            "query": "Microsoft FY2024 10-K excerpt",
            "query_plan": {"time_range": "FY2024"},
            "companies": ["Microsoft"],
            "retrieved_docs": {
                "Microsoft": {
                    "fundamentals_meta": {
                        "fiscal_year": 2024,
                        "period_alignment": "assumed_from_query",
                        "fiscal_year_source": "query",
                    }
                }
            },
        }
        text = "\n".join(format_period_alignment_notice(state))
        self.assertIn("FY2024", text)
        self.assertIn("assumed from query", text)
        self.assertIn("filing extract did not carry an explicit FY tag", text)
        self.assertNotIn("requested FY not found", text)

    def test_period_alignment_shows_period_end_and_calendar_note(self) -> None:
        state = {
            "query": "Compare Apple and Microsoft FY2024 profitability",
            "query_plan": {"time_range": "FY2024"},
            "companies": ["Apple", "Microsoft"],
            "retrieved_docs": {
                "Apple": {
                    "fundamentals_meta": {
                        "fiscal_year": 2024,
                        "period_end": "2024-09-28",
                        "period_alignment": "exact",
                    }
                },
                "Microsoft": {
                    "fundamentals_meta": {
                        "fiscal_year": 2024,
                        "period_end": "2024-06-30",
                        "period_alignment": "exact",
                    }
                },
            },
        }
        text = "\n".join(format_period_alignment_notice(state))
        self.assertIn("Period end", text)
        self.assertIn("2024-09-28", text)
        self.assertIn("2024-06-30", text)
        self.assertIn("Calendar note", text)
        self.assertIn("FY-label-aligned research comps", text)
        self.assertIn("exact match", text)

    def test_peer_matrix_marks_asymmetric_na(self) -> None:
        state = {
            "companies": ["Apple", "Microsoft"],
            "financial_metrics": {
                "Apple": {"ebitda_margin": 0.35, "operating_margin": 0.30, "r_and_d_intensity": 0.08},
                "Microsoft": {"operating_margin": 0.45, "r_and_d_intensity": 0.11},
            },
        }
        text = "\n".join(format_peer_metric_matrix(state))
        self.assertIn("Peer Metric Matrix", text)
        self.assertIn("n/a", text)
        self.assertIn("asymmetric", text)
        self.assertIn("FY-label research comps", text)
        summary = build_analyst_executive_summary(state, [], brief=True)
        self.assertIn("Comparison capsule", summary)
        self.assertIn("Operating Margin", summary)
        self.assertNotIn("Apple:", summary)  # capsule-only; no per-company claim dump
        self.assertIn("\n-", summary)  # multiline bullets, not one jammed paragraph

    def test_brief_filters_unknown_supply_and_thesis(self) -> None:
        class _C:
            def __init__(self, claim_type, metric_name, statement, entity="Apple"):
                self.claim_type = claim_type
                self.metric_name = metric_name
                self.statement = statement
                self.entity = entity

            def render_with_citation(self):
                return self.statement

        claims = [
            _C("numeric", "operating_margin", "Apple Operating margin is 32%."),
            _C("risk_conclusion", "supply_chain_risk", "Apple supply-chain risk signal is 'unknown'."),
            _C("investment_conclusion", None, "Apple supports a quality-screening research thesis."),
        ]
        kept = filter_claims_for_brief(claims)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].metric_name, "operating_margin")


class ReportQualityEndToEndTestCase(unittest.TestCase):
    def test_offline_compare_brief_has_capsule_matrix_no_profile_filler(self) -> None:
        config = build_test_config(ROOT / "test_artifacts" / f"rq-{uuid4().hex[:8]}")
        system = LumenFinAgentSystem(
            llm_client=LocalFallbackLLMClient(),
            app_config=config,
            market_data_client=FakeMarketDataClient(),
        )
        result = system.run(
            "Compare Apple and Microsoft FY2024 profitability and R&D intensity.",
            thread_id=f"rq-brief-{uuid4().hex[:6]}",
            output_format="executive_summary",
        )
        report = result.get("final_report") or ""
        self.assertIn("Period Alignment", report)
        self.assertIn("Peer Metric Matrix", report)
        self.assertIn("Comparison capsule", report)
        self.assertNotIn("Company Profiles & Business Overview", report)
        self.assertNotIn("No uploaded company profile document was provided", report)
        # Brief should not lead with unknown supply-chain / thesis filler in summary area.
        summary_block = report.split("## 1. Executive Summary", 1)[-1].split("## 3. Financial", 1)[0]
        self.assertNotIn("supply-chain risk signal is 'unknown'", summary_block)
        self.assertNotIn("quality-screening research thesis", summary_block)
        self.assertEqual(requested_fiscal_year_from_state(result), 2024)


if __name__ == "__main__":
    unittest.main()
