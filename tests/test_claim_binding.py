"""Unit tests for Claim → Evidence Binding."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.claims import (
    binding_summary,
    build_claims,
    filter_verified,
)


class ClaimBindingTests(unittest.TestCase):
    def _base_state(self) -> dict:
        return {
            "companies": ["NVIDIA"],
            "financial_metrics": {
                "NVIDIA": {
                    "ebitda_margin": 0.5,
                    "operating_margin": 0.4,
                    "r_and_d_intensity": 0.1,
                }
            },
            "retrieved_docs": {
                "NVIDIA": {
                    "structured_source": "sec_companyfacts",
                    "market_data": {
                        "revenue": 100.0,
                        "ebitda": 50.0,
                        "operating_income": 40.0,
                        "r_and_d": 10.0,
                    },
                    "fundamentals_meta": {"fiscal_year": 2025},
                    "supply_chain": {
                        "risk_level": "medium",
                        "signals": ["PDF mentions supply chain risk."],
                    },
                    "source_documents": [],
                }
            },
            "rag_evidence": {
                "NVIDIA": [
                    {
                        "citation": "nvda_fy2025_10k_sec.pdf#p12",
                        "text": "NVIDIA revenue was 100.0 billion USD and EBITDA was 50.0 billion.",
                        "source_type": "rag",
                    }
                ]
            },
            "risk_scores": {
                "NVIDIA": {
                    "financial_risk": 3.0,
                    "operational_risk": 4.0,
                    "market_risk": 5.0,
                    "supply_chain_risk": 5.5,
                }
            },
            "market_snapshots": {
                "NVIDIA": {
                    "current_price": 100.0,
                    "trailing_pe": 30.0,
                    "provider": "yahoo",
                    "fetched_at": "2026-01-01T00:00:00+00:00",
                }
            },
        }

    def test_numeric_and_risk_verified_with_citations(self) -> None:
        claims = build_claims(self._base_state())
        verified = filter_verified(claims)
        self.assertGreaterEqual(len(verified), 4)
        numeric = [c for c in verified if c.claim_type == "numeric"]
        self.assertTrue(any(c.metric_name == "ebitda_margin" for c in numeric))
        ebitda = next(c for c in numeric if c.metric_name == "ebitda_margin")
        self.assertTrue(ebitda.evidence_refs)
        # Prefer page-anchored RAG when numbers appear in hit text.
        self.assertIn("#p12", ebitda.primary_citation)
        risk = [c for c in verified if c.claim_type == "risk_conclusion"]
        self.assertTrue(risk)
        inv = [c for c in verified if c.claim_type == "investment_conclusion"]
        self.assertEqual(len(inv), 1)
        self.assertEqual(inv[0].verification, "verified")
        self.assertIn("[", inv[0].render_with_citation())

    def test_growth_rejected_without_multi_period(self) -> None:
        claims = build_claims(self._base_state())
        growth = [c for c in claims if c.claim_type == "growth"]
        self.assertEqual(len(growth), 1)
        self.assertEqual(growth[0].verification, "rejected")

    def test_investment_rejected_without_risk(self) -> None:
        state = self._base_state()
        state["risk_scores"] = {}
        state["retrieved_docs"]["NVIDIA"]["supply_chain"] = {}
        claims = build_claims(state)
        inv = [c for c in claims if c.claim_type == "investment_conclusion"]
        self.assertEqual(inv[0].verification, "rejected")

    def test_fail_closed_blocks_numeric_claims(self) -> None:
        state = self._base_state()
        state["fatal_data_gap"] = True
        state["retrieved_docs"]["NVIDIA"]["structured_source"] = "none"
        state["retrieved_docs"]["NVIDIA"]["market_data"] = {}
        state["financial_metrics"] = {}
        claims = build_claims(state)
        verified = filter_verified(claims)
        self.assertFalse(any(c.claim_type == "numeric" and c.verification == "verified" for c in claims))
        self.assertTrue(any(c.claim_type == "risk_conclusion" and c.verification == "verified" for c in verified))
        self.assertTrue(any(c.claim_type == "investment_conclusion" and c.verification == "rejected" for c in claims))


if __name__ == "__main__":
    unittest.main()
