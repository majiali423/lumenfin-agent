"""Adversarial reliability cases for unit normalization, claim binding, and period facts.

These tests encode the Phase-2 desired contracts. Several currently fail against HEAD
and are expected to turn green only after the corresponding fix commits.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.claims import build_claims, filter_verified
from lumenfin.documents import (
    _extract_metric_hints,
    _first_metric_number,
    detect_statement_scale,
    normalize_metric_hints_to_billion_usd,
)
from lumenfin.reporting import (
    annotate_upload_period_meta,
    format_period_alignment_notice,
    peer_period_end_span_days,
    period_end_for_company,
    suggest_period_end_hint,
)


def _hint_meta_available() -> bool:
    try:
        from lumenfin.documents import extract_metric_hint_meta  # noqa: F401

        return True
    except ImportError:
        return False


def _match_api_available() -> bool:
    try:
        from lumenfin.claims import match_numeric_evidence  # noqa: F401

        return True
    except ImportError:
        return False


class AdversarialUnitNormalizationTestCase(unittest.TestCase):
    def test_explicit_millions_245122(self) -> None:
        hints = normalize_metric_hints_to_billion_usd(
            {"revenue": 245122.0},
            text="Consolidated Statements of Income (In millions)",
        )
        self.assertAlmostEqual(hints["revenue"], 245.122, places=3)

    def test_explicit_thousands(self) -> None:
        hints = normalize_metric_hints_to_billion_usd(
            {"revenue": 245122000.0},
            text="(In thousands of U.S. dollars)",
        )
        self.assertAlmostEqual(hints["revenue"], 245.122, places=3)

    def test_explicit_billions_passthrough(self) -> None:
        hints = normalize_metric_hints_to_billion_usd(
            {"revenue": 245.122},
            text="(In billions)",
        )
        self.assertAlmostEqual(hints["revenue"], 245.122, places=3)

    def test_five_hundred_in_millions_is_half_billion(self) -> None:
        # Current bug: scale==million with value < 1000 skips scaling.
        hints = normalize_metric_hints_to_billion_usd(
            {"revenue": 500.0},
            text="(In millions)",
        )
        self.assertAlmostEqual(hints["revenue"], 0.5, places=4)

    def test_negative_operating_income_in_millions(self) -> None:
        hints = normalize_metric_hints_to_billion_usd(
            {"operating_income": -10500.0},
            text="(In millions)",
        )
        self.assertAlmostEqual(hints["operating_income"], -10.5, places=4)

    def test_unitless_large_number_is_not_high_confidence(self) -> None:
        self.assertTrue(
            _hint_meta_available(),
            "extract_metric_hint_meta not implemented yet",
        )
        from lumenfin.documents import extract_metric_hint_meta

        meta = extract_metric_hint_meta("Total revenue 245122", metric="revenue")
        self.assertIsNotNone(meta)
        self.assertNotEqual(meta.get("confidence"), "high")
        self.assertIn(meta.get("normalization_source"), {"inferred_million", "unitless", None, "none"})

    def test_unitless_boundary_999_1000_1001(self) -> None:
        self.assertTrue(_hint_meta_available(), "extract_metric_hint_meta not implemented yet")
        from lumenfin.documents import extract_metric_hint_meta

        for raw in (999.0, 1000.0, 1001.0):
            with self.subTest(raw=raw):
                meta = extract_metric_hint_meta(f"Revenue {raw:g}", metric="revenue")
                # Bare unitless numbers without amount cues may be skipped entirely, or
                # kept only as low-confidence inferred magnitudes — never high confidence.
                self.assertTrue(meta is None or meta.get("confidence") != "high")

    def test_already_normalized_not_double_scaled(self) -> None:
        # Re-entry must use explicit normalization metadata, not fractional shape.
        meta = {
            "revenue": {
                "raw_value": 245.122,
                "raw_scale": None,
                "currency": "USD",
                "normalized_value": 245.122,
                "normalized_unit": "billion_usd",
                "normalization_source": "provider_metadata",
                "confidence": "high",
                "is_normalized": True,
            }
        }
        once = normalize_metric_hints_to_billion_usd(
            {"revenue": 245.122},
            text="(In millions)",
            hint_meta=meta,
        )
        twice = normalize_metric_hints_to_billion_usd(once, text="(In millions)", hint_meta=meta)
        self.assertAlmostEqual(twice["revenue"], 245.122, places=3)

    def test_eur_millions_not_billion_usd(self) -> None:
        self.assertTrue(_hint_meta_available(), "extract_metric_hint_meta not implemented yet")
        from lumenfin.documents import extract_metric_hint_meta

        meta = extract_metric_hint_meta(
            "Revenue (In millions of EUR)\nRevenue 12000",
            metric="revenue",
        )
        self.assertIsNotNone(meta)
        self.assertNotEqual(meta.get("normalized_unit"), "billion_usd")

    def test_quarterly_revenue_not_auto_annual(self) -> None:
        self.assertTrue(_hint_meta_available(), "extract_metric_hint_meta not implemented yet")
        from lumenfin.documents import extract_metric_hint_meta

        meta = extract_metric_hint_meta(
            "Three months ended June 30\n(In millions)\nRevenue 61000",
            metric="revenue",
        )
        self.assertIsNotNone(meta)
        periodish = str(meta.get("period") or meta.get("period_hint") or "").lower()
        self.assertTrue(
            any(tok in periodish for tok in ("q", "quarter", "three months"))
            or meta.get("confidence") != "high",
            msg=f"quarterly amount must not silently become annual high-confidence: {meta}",
        )

    def test_distant_caption_still_scales_when_explicit(self) -> None:
        text = "(In millions)\n\n" + ("Discussion.\n" * 40) + "\nTotal revenue 245122\n"
        hints = _extract_metric_hints(text)
        self.assertIn("revenue", hints)
        self.assertAlmostEqual(hints["revenue"], 245.122, places=1)

    def test_multi_metric_does_not_steal_wrong_number(self) -> None:
        text = (
            "(In millions)\n"
            "Revenue 245122\n"
            "Research and development 29510\n"
            "Operating income 109433\n"
        )
        hints = _extract_metric_hints(text)
        self.assertAlmostEqual(hints["revenue"], 245.122, places=1)
        self.assertAlmostEqual(hints["r_and_d"], 29.51, places=1)
        self.assertAlmostEqual(hints["operating_income"], 109.433, places=1)

    def test_two_tables_different_scales_minimum_controllable(self) -> None:
        # Boundary: document-level scale detection is single-scale today.
        # Desired controllable behavior: prefer nearest caption; if unsupported,
        # metadata must not claim high confidence for both.
        text = (
            "Table A (In millions)\nRevenue 245122\n\n"
            "Table B (In thousands)\nRevenue 245122000\n"
        )
        if _hint_meta_available():
            from lumenfin.documents import extract_metric_hint_meta

            meta = extract_metric_hint_meta(text, metric="revenue")
            self.assertIsNotNone(meta)
            # Either resolves one table correctly or stays low-confidence.
            if meta.get("confidence") == "high":
                self.assertAlmostEqual(float(meta["normalized_value"]), 245.122, places=2)
        else:
            self.fail("extract_metric_hint_meta not implemented yet")


class AdversarialClaimBindingTestCase(unittest.TestCase):
    def _state(
        self,
        *,
        company: str = "Microsoft",
        market: dict | None = None,
        meta: dict | None = None,
        rag: list | None = None,
        metrics: dict | None = None,
        peer_company: str | None = None,
    ) -> dict:
        market = market or {
            "revenue": 245.122,
            "operating_income": 109.433,
            "r_and_d": 29.51,
            "ebitda": 120.0,
        }
        state = {
            "companies": [company] + ([peer_company] if peer_company else []),
            "financial_metrics": {
                company: metrics
                or {
                    "operating_margin": 0.446,
                    "ebitda_margin": 0.49,
                    "r_and_d_intensity": 0.12,
                }
            },
            "retrieved_docs": {
                company: {
                    "structured_source": "document_extracted",
                    "market_data": market,
                    "fundamentals_meta": meta
                    or {"fiscal_year": 2024, "period": "FY2024", "period_end": "2024-06-30"},
                    "source_documents": [],
                }
            },
            "rag_evidence": {company: rag or []},
            "risk_scores": {},
            "market_snapshots": {},
        }
        if peer_company:
            state["retrieved_docs"][peer_company] = {
                "structured_source": "sec_companyfacts",
                "market_data": {"revenue": 383.0, "operating_income": 120.0, "r_and_d": 30.0, "ebitda": 140.0},
                "fundamentals_meta": {"fiscal_year": 2024, "period": "FY2024"},
                "source_documents": [],
            }
            state["financial_metrics"][peer_company] = {
                "operating_margin": 0.31,
                "ebitda_margin": 0.36,
                "r_and_d_intensity": 0.08,
            }
            state["rag_evidence"][peer_company] = []
        return state

    def _claim(self, claims, metric_name: str):
        for claim in claims:
            if claim.metric_name == metric_name and claim.claim_type == "numeric":
                return claim
        self.fail(f"missing claim {metric_name}")

    def test_correct_number_wrong_metric_rejected(self) -> None:
        state = self._state(
            rag=[
                {
                    "citation": "msft.pdf#p3",
                    "source_type": "rag",
                    "period": "FY2024",
                    "text": "Research and development expense was 245.122 billion USD.",
                }
            ]
        )
        # Remove structured fund sentence path by emptying market rewrite? Keep market but
        # prefer RAG that has wrong metric label for revenue absolute.
        claim = self._claim(build_claims(state), "revenue")
        # Desired: RAG alone must not verify revenue from an R&D-labeled span.
        if claim.verification == "verified":
            cites = " ".join(r.citation for r in claim.evidence_refs)
            self.assertNotIn("#p3", cites)

    def test_same_value_revenue_and_rd_not_cross_bound(self) -> None:
        state = self._state(
            market={"revenue": 50.0, "r_and_d": 50.0, "operating_income": 20.0, "ebitda": 25.0},
            rag=[
                {
                    "citation": "x.pdf#p1",
                    "source_type": "rag",
                    "period": "FY2024",
                    "text": "Revenue was 50.0 billion USD. Later, R&D was 50.0 billion USD.",
                }
            ],
        )
        claims = build_claims(state)
        rev = self._claim(claims, "revenue")
        rd = self._claim(claims, "r_and_d")
        if _match_api_available():
            from lumenfin.claims import EvidenceRef, match_numeric_evidence

            rag = EvidenceRef(
                evidence_id="ev",
                entity="Microsoft",
                citation="x.pdf#p1",
                source_type="rag",
                text=state["rag_evidence"]["Microsoft"][0]["text"],
                period="FY2024",
            )
            m_rev = match_numeric_evidence(
                rag,
                entity="Microsoft",
                metric_name="revenue",
                value=50.0,
                unit="billion_usd",
                period="FY2024",
            )
            m_rd = match_numeric_evidence(
                rag,
                entity="Microsoft",
                metric_name="r_and_d",
                value=50.0,
                unit="billion_usd",
                period="FY2024",
            )
            self.assertTrue(m_rev.matched and m_rev.matched_metric)
            self.assertTrue(m_rd.matched and m_rd.matched_metric)
            self.assertNotEqual(m_rev.match_span, m_rd.match_span)
        else:
            self.assertEqual(rev.verification, "verified")
            self.assertEqual(rd.verification, "verified")
            self.fail("match_numeric_evidence not implemented yet")

    def test_period_mismatch_rejected(self) -> None:
        state = self._state(
            meta={"fiscal_year": 2024, "period": "FY2024"},
            rag=[
                {
                    "citation": "old.pdf#p2",
                    "source_type": "rag",
                    "period": "FY2023",
                    "text": "Revenue was 245.122 billion USD in FY2023.",
                }
            ],
        )
        # Drop fund evidence by using only rag-prefer path: still has fund pool.
        # Assert match API rejects period mismatch when available.
        self.assertTrue(_match_api_available(), "match_numeric_evidence not implemented yet")
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev",
            entity="Microsoft",
            citation="old.pdf#p2",
            source_type="rag",
            text="Revenue was 245.122 billion USD in FY2023.",
            period="FY2023",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)
        self.assertIn("period", match.reason.lower())

    def test_entity_mismatch_rejected(self) -> None:
        self.assertTrue(_match_api_available(), "match_numeric_evidence not implemented yet")
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev",
            entity="Apple",
            citation="aapl.pdf#p1",
            source_type="rag",
            text="Apple revenue was 383.0 billion USD.",
            period="FY2024",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="revenue",
            value=383.0,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)
        self.assertIn("entity", match.reason.lower())

    def test_ratio_cross_period_inputs_rejected(self) -> None:
        self.assertTrue(_match_api_available(), "match_numeric_evidence not implemented yet")
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev",
            entity="Microsoft",
            citation="mix.pdf#p1",
            source_type="rag",
            text="FY2024 operating income 109.433. FY2023 revenue 220.0.",
            period="FY2024",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="operating_margin",
            value=0.446,
            unit="ratio",
            period="FY2024",
            formula_inputs={"operating_income": 109.433, "revenue": 220.0},
        )
        self.assertFalse(match.matched)
        self.assertTrue(
            "period" in match.reason.lower() or "formula" in match.reason.lower()
        )

    def test_ratio_final_percent_alone_not_high_confidence(self) -> None:
        state = self._state(
            rag=[
                {
                    "citation": "m.pdf#p9",
                    "source_type": "rag",
                    "period": "FY2024",
                    "text": "Operating margin was 44.6%.",
                }
            ]
        )
        # Remove market so fund sentence lacks numbers? Keep market — fund has inputs.
        # Desired contract via match API on percent-only evidence:
        self.assertTrue(_match_api_available(), "match_numeric_evidence not implemented yet")
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev",
            entity="Microsoft",
            citation="m.pdf#p9",
            source_type="rag",
            text="Operating margin was 44.6%.",
            period="FY2024",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="operating_margin",
            value=0.446,
            unit="ratio",
            period="FY2024",
            formula_inputs={"operating_income": 109.433, "revenue": 245.122},
        )
        self.assertFalse(match.matched)
        self.assertIn("formula", match.reason.lower())

    def test_rag_number_without_rd_label_rejected(self) -> None:
        self.assertTrue(_match_api_available(), "match_numeric_evidence not implemented yet")
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev",
            entity="Microsoft",
            citation="m.pdf#p4",
            source_type="rag",
            text="The company recorded 29510 in the period.",
            period="FY2024",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="r_and_d",
            value=29.51,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)
        self.assertTrue(
            "metric" in match.reason.lower()
            or "label" in match.reason.lower()
            or "number_not_found" in match.reason.lower()
            or "unit" in match.reason.lower()
        )

    def test_rag_rd_millions_supports_billion_claim(self) -> None:
        self.assertTrue(_match_api_available(), "match_numeric_evidence not implemented yet")
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev",
            entity="Microsoft",
            citation="m.pdf#p4",
            source_type="rag",
            text="Research and development (In millions) 29510",
            period="FY2024",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="r_and_d",
            value=29.51,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertTrue(match.matched)
        self.assertTrue(match.matched_metric)
        self.assertIsNotNone(match.unit_conversion)

    def test_structured_fundamentals_field_bind(self) -> None:
        state = self._state(rag=[])
        claim = self._claim(build_claims(state), "revenue")
        self.assertEqual(claim.verification, "verified")

    def test_unit_ambiguous_not_auto_verified_via_match_api(self) -> None:
        self.assertTrue(_match_api_available(), "match_numeric_evidence not implemented yet")
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev",
            entity="Microsoft",
            citation="m.pdf#p5",
            source_type="rag",
            text="R&D was 29510.",
            period="FY2024",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="r_and_d",
            value=29.51,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)
        self.assertTrue(
            "unit" in match.reason.lower() or match.confidence in {"low", "none"}
        )


class AdversarialPeriodFactTestCase(unittest.TestCase):
    def test_sec_period_end_preserved(self) -> None:
        meta = annotate_upload_period_meta(
            {"fiscal_year": 2024, "period_end": "2024-06-30", "period_end_source": "sec_companyfacts"},
            company="Microsoft",
        )
        self.assertEqual(meta.get("period_end"), "2024-06-30")
        self.assertEqual(meta.get("period_end_source"), "sec_companyfacts")

    def test_upload_explicit_period_end_preserved(self) -> None:
        meta = annotate_upload_period_meta(
            {"fiscal_year": 2024, "period_end": "2024-09-28", "period_end_source": "upload_text"},
            company="Apple",
        )
        self.assertEqual(meta["period_end"], "2024-09-28")

    def test_microsoft_convention_does_not_mint_precise_date(self) -> None:
        hint, src = suggest_period_end_hint("Microsoft", 2024)
        self.assertIsNone(hint)
        self.assertTrue(src is None or "convention" in str(src))
        meta = annotate_upload_period_meta(
            {"fiscal_year": 2024},
            company="Microsoft",
            prefer_fiscal_year=2024,
        )
        period_end = str(meta.get("period_end") or "")
        self.assertNotEqual(period_end, "2024-06-30")
        self.assertNotEqual(meta.get("period_end_source"), "issuer_convention_hint")

    def test_nvidia_convention_does_not_mint_01_26(self) -> None:
        meta = annotate_upload_period_meta({"fiscal_year": 2025}, company="NVIDIA")
        self.assertNotEqual(str(meta.get("period_end") or ""), "2025-01-26")

    def test_assumed_from_query_not_exact_match(self) -> None:
        meta = annotate_upload_period_meta(
            {},
            company="Microsoft",
            prefer_fiscal_year=2024,
            document_contexts=[{"filename": "excerpt.pdf", "excerpt": "financial highlights"}],
        )
        self.assertEqual(meta.get("period_alignment"), "assumed_from_query")
        state = {
            "query": "Microsoft FY2024",
            "query_plan": {"time_range": "FY2024"},
            "companies": ["Microsoft"],
            "retrieved_docs": {"Microsoft": {"fundamentals_meta": meta}},
        }
        text = "\n".join(format_period_alignment_notice(state))
        self.assertIn("assumed", text.lower())
        self.assertNotIn("exact match", text.lower())

    def test_hint_plus_real_date_no_day_gap(self) -> None:
        state = {
            "companies": ["Apple", "Microsoft"],
            "retrieved_docs": {
                "Apple": {
                    "fundamentals_meta": {
                        "fiscal_year": 2024,
                        "period_end": "2024-09-28",
                        "period_end_source": "sec_companyfacts",
                    }
                },
                "Microsoft": {
                    "fundamentals_meta": {
                        "fiscal_year": 2024,
                        "fiscal_calendar_hint": "typically ends in late June",
                        "fiscal_calendar_hint_source": "issuer_convention",
                        # legacy mistaken precise date must be ignored without source fact
                        "period_end": "2024-06-30",
                        "period_end_source": "issuer_convention_hint",
                    }
                },
            },
        }
        # period_end_for_company must ignore convention-sourced dates.
        self.assertEqual(period_end_for_company(state, "Apple"), "2024-09-28")
        self.assertIsNone(period_end_for_company(state, "Microsoft"))
        self.assertIsNone(peer_period_end_span_days(state))

    def test_two_real_dates_compute_gap(self) -> None:
        state = {
            "companies": ["Apple", "Microsoft"],
            "retrieved_docs": {
                "Apple": {
                    "fundamentals_meta": {
                        "period_end": "2024-09-28",
                        "period_end_source": "sec_companyfacts",
                    }
                },
                "Microsoft": {
                    "fundamentals_meta": {
                        "period_end": "2024-06-30",
                        "period_end_source": "sec_companyfacts",
                    }
                },
            },
        }
        span = peer_period_end_span_days(state)
        self.assertIsNotNone(span)
        self.assertGreaterEqual(int(span), 90)

    def test_convention_only_in_calendar_note(self) -> None:
        state = {
            "query": "Microsoft FY2024",
            "query_plan": {"time_range": "FY2024"},
            "companies": ["Microsoft"],
            "retrieved_docs": {
                "Microsoft": {
                    "fundamentals_meta": {
                        "fiscal_year": 2024,
                        "period_alignment": "assumed_from_query",
                        "fiscal_calendar_hint": "typically ends in late June",
                        "fiscal_calendar_hint_source": "issuer_convention",
                    }
                }
            },
        }
        text = "\n".join(format_period_alignment_notice(state))
        self.assertIn("Calendar note", text)
        self.assertIn("late June", text)
        self.assertNotIn("2024-06-30", text)

    def test_period_end_requires_auditable_source(self) -> None:
        state = {
            "companies": ["Microsoft"],
            "retrieved_docs": {
                "Microsoft": {
                    "fundamentals_meta": {
                        "period_end": "2024-06-30",
                        # missing period_end_source
                    }
                }
            },
        }
        self.assertIsNone(period_end_for_company(state, "Microsoft"))


class Phase21ProvenancePromotionTestCase(unittest.TestCase):
    """Gaps that Phase 2.1 must close: provenance promotion and double-scale heuristics."""

    def test_fractional_explicit_million_scales(self) -> None:
        from lumenfin.documents import extract_metric_hint_meta, normalize_extracted_amount

        meta = extract_metric_hint_meta("(In millions)\nRevenue 500.5", metric="revenue")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["raw_value"], 500.5)
        self.assertEqual(meta["raw_scale"], "million")
        self.assertAlmostEqual(float(meta["normalized_value"]), 0.5005, places=6)
        self.assertEqual(meta["normalized_unit"], "billion_usd")
        self.assertEqual(meta["confidence"], "high")

        half = normalize_extracted_amount(
            0.5, raw_scale="million", currency="USD", normalization_source="inline_unit"
        )
        self.assertAlmostEqual(float(half.normalized_value or 0), 0.0005, places=6)
        thou = normalize_extracted_amount(
            500.25, raw_scale="thousand", currency="USD", normalization_source="table_caption"
        )
        self.assertAlmostEqual(float(thou.normalized_value or 0), 0.00050025, places=8)

    def test_double_scale_uses_metadata_not_fraction_shape(self) -> None:
        from dataclasses import asdict

        from lumenfin.documents import normalize_extracted_amount, normalize_metric_hints_to_billion_usd

        raw = normalize_extracted_amount(
            245.122, raw_scale="million", currency="USD", normalization_source="table_caption"
        )
        self.assertAlmostEqual(float(raw.normalized_value or 0), 0.245122, places=6)

        meta = asdict(raw)
        meta["normalized_value"] = 245.122
        meta["normalized_unit"] = "billion_usd"
        meta["is_normalized"] = True
        again = normalize_metric_hints_to_billion_usd(
            {"revenue": 245.122},
            text="(In millions)",
            hint_meta={"revenue": meta},
        )
        self.assertAlmostEqual(again["revenue"], 245.122, places=3)

    def test_unitless_low_confidence_not_computable_fundamentals(self) -> None:
        from lumenfin.documents import extract_metric_hint_meta
        from lumenfin.tools import _payload_from_documents, has_computable_fundamentals

        text = "Revenue 245122\nOperating income 109433\n"
        meta = extract_metric_hint_meta(text, metric="revenue")
        self.assertEqual(meta.get("confidence"), "low")
        docs = [
            {
                "detected_companies": ["Microsoft"],
                "filename": "unitless.pdf",
                "text": text,
                "excerpt": text,
                "metric_hints": {"revenue": 245.122, "operating_income": 109.433},
                "metric_hint_meta": {
                    "revenue": meta,
                    "operating_income": extract_metric_hint_meta(text, metric="operating_income"),
                },
            }
        ]
        payload = _payload_from_documents("Microsoft", docs, include_appendix=False)
        self.assertFalse(has_computable_fundamentals(payload))
        self.assertNotEqual(payload.get("structured_source"), "document_extracted")
        self.assertFalse(
            any(
                c.verification == "verified" and c.claim_type == "numeric"
                for c in build_claims(
                    {
                        "companies": ["Microsoft"],
                        "financial_metrics": {"Microsoft": {"operating_margin": 0.4}},
                        "retrieved_docs": {"Microsoft": payload},
                        "rag_evidence": {},
                        "risk_scores": {},
                        "market_snapshots": {},
                    }
                )
            )
        )

    def test_per_company_metric_hint_meta_retained(self) -> None:
        from lumenfin.documents import merge_per_company_metric_hint_meta, parse_pdf_document

        # Prefer explicit helper if available; otherwise parse_pdf contract.
        try:
            meta = merge_per_company_metric_hint_meta(
                "(In millions)\nMicrosoft\nRevenue 245122\nApple\nRevenue 383285\n",
                ["Microsoft", "Apple"],
            )
        except ImportError:
            self.fail("merge_per_company_metric_hint_meta not implemented yet")
        self.assertIn("Microsoft", meta)
        self.assertIn("Apple", meta)
        for company in ("Microsoft", "Apple"):
            rev = meta[company]["revenue"]
            for key in (
                "raw_value",
                "raw_scale",
                "currency",
                "normalized_value",
                "normalized_unit",
                "normalization_source",
                "confidence",
            ):
                self.assertIn(key, rev)


class Phase21StructuredBindingTestCase(unittest.TestCase):
    def test_structured_field_collision_rejects_wrong_metric(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev_fund_Microsoft_r_and_d_FY2024",
            entity="Microsoft",
            citation="lumenfin:sec_companyfacts:Microsoft:FY2024",
            source_type="sec_companyfacts",
            text="Microsoft revenue was 50.0 billion USD, R&D was 20.0 billion USD.",
            period="FY2024",
            metric_name="r_and_d",
            value=20.0,
            unit="billion_usd",
            confidence="high",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="r_and_d",
            value=50.0,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)
        self.assertIn("metric_value_mismatch", match.reason)

    def test_structured_same_value_binds_distinct_fields(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        rev = EvidenceRef(
            evidence_id="ev_fund_Microsoft_revenue_FY2024",
            entity="Microsoft",
            citation="lumenfin:sec_companyfacts:Microsoft:FY2024",
            source_type="sec_companyfacts",
            text="Microsoft revenue was 50.0 billion USD and R&D was 50.0 billion USD.",
            period="FY2024",
            metric_name="revenue",
            value=50.0,
            unit="billion_usd",
            confidence="high",
        )
        rd = EvidenceRef(
            evidence_id="ev_fund_Microsoft_r_and_d_FY2024",
            entity="Microsoft",
            citation="lumenfin:sec_companyfacts:Microsoft:FY2024",
            source_type="sec_companyfacts",
            text=rev.text,
            period="FY2024",
            metric_name="r_and_d",
            value=50.0,
            unit="billion_usd",
            confidence="high",
        )
        m_rev = match_numeric_evidence(
            rev, entity="Microsoft", metric_name="revenue", value=50.0, unit="billion_usd", period="FY2024"
        )
        m_rd = match_numeric_evidence(
            rd, entity="Microsoft", metric_name="r_and_d", value=50.0, unit="billion_usd", period="FY2024"
        )
        self.assertTrue(m_rev.matched)
        self.assertTrue(m_rd.matched)
        self.assertNotEqual(rev.evidence_id, rd.evidence_id)

    def test_structured_fields_beat_display_text(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        # Display text wrong, structured field correct → accept.
        good_field = EvidenceRef(
            evidence_id="ev_fund_Microsoft_revenue_FY2024",
            entity="Microsoft",
            citation="lumenfin:sec_companyfacts:Microsoft:FY2024",
            source_type="sec_companyfacts",
            text="Microsoft revenue was 999.0 billion USD.",
            period="FY2024",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            confidence="high",
        )
        self.assertTrue(
            match_numeric_evidence(
                good_field,
                entity="Microsoft",
                metric_name="revenue",
                value=245.122,
                unit="billion_usd",
                period="FY2024",
            ).matched
        )
        # Display text correct, structured field wrong → reject.
        bad_field = EvidenceRef(
            evidence_id="ev_fund_Microsoft_revenue_FY2024",
            entity="Microsoft",
            citation="lumenfin:sec_companyfacts:Microsoft:FY2024",
            source_type="sec_companyfacts",
            text="Microsoft revenue was 245.122 billion USD.",
            period="FY2024",
            metric_name="revenue",
            value=999.0,
            unit="billion_usd",
            confidence="high",
        )
        self.assertFalse(
            match_numeric_evidence(
                bad_field,
                entity="Microsoft",
                metric_name="revenue",
                value=245.122,
                unit="billion_usd",
                period="FY2024",
            ).matched
        )

    def test_period_specific_claim_rejects_unknown_period_rag_high_confidence(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        rag = EvidenceRef(
            evidence_id="ev_rag_1",
            entity="Microsoft",
            citation="m.pdf#p1",
            source_type="rag",
            text="Revenue was 245.122 billion USD.",
            period=None,
        )
        match = match_numeric_evidence(
            rag,
            entity="Microsoft",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched and match.confidence == "high")
        self.assertTrue(
            (not match.matched)
            or match.confidence in {"low", "medium", "none"}
            or "period" in match.reason.lower()
        )

    def test_formula_inputs_require_units(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        rag = EvidenceRef(
            evidence_id="ev_rag_2",
            entity="Microsoft",
            citation="m.pdf#p2",
            source_type="rag",
            text="Operating income 109.433\nRevenue 245.122",
            period="FY2024",
        )
        match = match_numeric_evidence(
            rag,
            entity="Microsoft",
            metric_name="operating_margin",
            value=0.446,
            unit="ratio",
            period="FY2024",
            formula_inputs={"operating_income": 109.433, "revenue": 245.122},
        )
        self.assertFalse(match.matched)
        self.assertIn("unit", match.reason.lower())


class Phase2FinalClosingTestCase(unittest.TestCase):
    """Remaining Phase 2 closure gaps: caller-hint backdoor, period gates, identifiers, tolerance."""

    def test_caller_hints_without_metadata_do_not_enter_ast(self) -> None:
        from lumenfin.claims import build_claims, filter_verified
        from lumenfin.tools import _payload_from_documents, has_computable_fundamentals

        docs = [
            {
                "detected_companies": ["Microsoft"],
                "metric_hints": {"revenue": 245.122, "operating_income": 109.433},
                "excerpt": "caller floats only",
                "text": "caller floats only",
            }
        ]
        payload = _payload_from_documents("Microsoft", docs, include_appendix=False)
        self.assertNotIn("revenue", payload.get("market_data") or {})
        self.assertNotEqual(payload.get("structured_source"), "document_extracted")
        self.assertFalse(has_computable_fundamentals(payload))
        state = {
            "companies": ["Microsoft"],
            "financial_metrics": {
                "Microsoft": {"operating_margin": 0.446},
            },
            "retrieved_docs": {"Microsoft": payload},
            "rag_evidence": {},
            "risk_scores": {},
            "market_snapshots": {},
        }
        verified = filter_verified(build_claims(state))
        self.assertFalse(
            any(c.metric_name in {"revenue", "operating_margin", "operating_income"} for c in verified)
        )

    def test_trusted_provider_metadata_enters_ast(self) -> None:
        from lumenfin.tools import _payload_from_documents, has_computable_fundamentals

        meta = {
            "revenue": {
                "normalized_value": 245.122,
                "normalized_unit": "billion_usd",
                "currency": "USD",
                "confidence": "high",
                "normalization_source": "provider_metadata",
                "provider": "sec_companyfacts",
                "period": "FY2024",
                "is_normalized": True,
            },
            "operating_income": {
                "normalized_value": 109.433,
                "normalized_unit": "billion_usd",
                "currency": "USD",
                "confidence": "high",
                "normalization_source": "provider_metadata",
                "provider": "sec_companyfacts",
                "period": "FY2024",
                "is_normalized": True,
            },
            "r_and_d": {
                "normalized_value": 29.5,
                "normalized_unit": "billion_usd",
                "currency": "USD",
                "confidence": "high",
                "normalization_source": "provider_metadata",
                "provider": "sec_companyfacts",
                "period": "FY2024",
                "is_normalized": True,
            },
        }
        docs = [
            {
                "detected_companies": ["Microsoft"],
                "metric_hints": {"revenue": 245.122, "operating_income": 109.433, "r_and_d": 29.5},
                "metric_hint_meta": meta,
                "excerpt": "provider",
                "text": "provider",
            }
        ]
        payload = _payload_from_documents("Microsoft", docs, include_appendix=False)
        self.assertAlmostEqual(payload["market_data"]["revenue"], 245.122, places=3)
        self.assertTrue(has_computable_fundamentals(payload))
        self.assertEqual(payload.get("structured_source"), "document_extracted")
        prov = payload.get("fundamental_provenance") or {}
        self.assertEqual((prov.get("revenue") or {}).get("provider"), "sec_companyfacts")
        self.assertEqual((prov.get("revenue") or {}).get("period"), "FY2024")

    def test_incomplete_provider_metadata_rejected(self) -> None:
        from lumenfin.tools import _payload_from_documents

        for missing in ("provider", "period", "currency", "normalized_unit", "confidence"):
            with self.subTest(missing=missing):
                meta = {
                    "normalized_value": 245.122,
                    "normalized_unit": "billion_usd",
                    "currency": "USD",
                    "confidence": "high",
                    "normalization_source": "provider_metadata",
                    "provider": "sec_companyfacts",
                    "period": "FY2024",
                    "is_normalized": True,
                }
                meta.pop(missing)
                docs = [
                    {
                        "detected_companies": ["Microsoft"],
                        "metric_hints": {"revenue": 245.122, "r_and_d": 29.5, "ebitda": 100.0},
                        "metric_hint_meta": {"revenue": meta},
                        "excerpt": "x",
                        "text": "x",
                    }
                ]
                payload = _payload_from_documents("Microsoft", docs, include_appendix=False)
                self.assertNotIn("revenue", payload.get("market_data") or {})

    def test_period_unknown_rag_not_matched_for_fy_claim(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        rag = EvidenceRef(
            evidence_id="ev_rag_period_unknown",
            entity="Microsoft",
            citation="m.pdf#p1",
            source_type="rag",
            text="Revenue was 245.122 billion USD.",
            period=None,
        )
        match = match_numeric_evidence(
            rag,
            entity="Microsoft",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)
        self.assertIn("period", match.reason.lower())

    def test_period_unknown_formula_rag_not_matched(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        rag = EvidenceRef(
            evidence_id="ev_rag_formula_unknown",
            entity="Microsoft",
            citation="m.pdf#p2",
            source_type="rag",
            text="Operating income was 109.433 billion USD. Revenue was 245.122 billion USD.",
            period=None,
        )
        match = match_numeric_evidence(
            rag,
            entity="Microsoft",
            metric_name="operating_margin",
            value=0.446,
            unit="ratio",
            period="FY2024",
            formula_inputs={"operating_income": 109.433, "revenue": 245.122},
        )
        self.assertFalse(match.matched)

    def test_formula_one_input_period_unknown_rejected(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        oi = EvidenceRef(
            evidence_id="ev_oi",
            entity="Microsoft",
            citation="a.pdf#p1",
            source_type="rag",
            text="Operating income was 109.433 billion USD for FY2024.",
            period="FY2024",
        )
        rev = EvidenceRef(
            evidence_id="ev_rev",
            entity="Microsoft",
            citation="b.pdf#p1",
            source_type="rag",
            text="Revenue was 245.122 billion USD.",
            period=None,
        )
        self.assertTrue(
            match_numeric_evidence(
                oi,
                entity="Microsoft",
                metric_name="operating_income",
                value=109.433,
                unit="billion_usd",
                period="FY2024",
            ).matched
        )
        self.assertFalse(
            match_numeric_evidence(
                rev,
                entity="Microsoft",
                metric_name="revenue",
                value=245.122,
                unit="billion_usd",
                period="FY2024",
            ).matched
        )

    def test_period_type_annual_alone_cannot_satisfy_fy(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        # period_type=annual (or legacy period="annual") must not prove FY2024.
        kwargs = dict(
            evidence_id="ev_annual_only",
            entity="Microsoft",
            citation="lumenfin:sec_companyfacts:Microsoft:annual:revenue",
            source_type="sec_companyfacts",
            text="Microsoft revenue was 245.122 billion USD.",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            confidence="high",
        )
        try:
            ref = EvidenceRef(period=None, period_type="annual", **kwargs)
        except TypeError:
            ref = EvidenceRef(period="annual", **kwargs)
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)
        self.assertIn("period", match.reason.lower())

    def test_period_and_period_type_together_ok(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        kwargs = dict(
            evidence_id="ev_fy_annual",
            entity="Microsoft",
            citation="lumenfin:sec_companyfacts:Microsoft:FY2024:revenue",
            source_type="sec_companyfacts",
            text="Microsoft revenue was 245.122 billion USD.",
            period="FY2024",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            confidence="high",
        )
        try:
            ref = EvidenceRef(period_type="annual", **kwargs)
        except TypeError:
            ref = EvidenceRef(**kwargs)
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertTrue(match.matched)

    def test_filing_identifiers_not_extracted_as_amounts(self) -> None:
        from lumenfin.documents import extract_metric_hint_meta

        for text in (
            "Revenue disclosures are included in Form 10-K.",
            "Revenue is discussed under Item 8.",
            "Revenue recognition follows ASC 606.",
            "Revenue appears in Note 12.",
            "Revenue is shown on Page 42.",
            "Revenue controls are covered by Section 404.",
            "Revenue from Form 10-Q filing.",
            "Revenue amendment 10-K/A notes.",
            "Revenue Item 1A risk factors.",
            "Revenue Form 8-K event.",
        ):
            with self.subTest(text=text):
                meta = extract_metric_hint_meta(text, metric="revenue")
                if meta is None:
                    continue
                raw = float(meta.get("raw_value") or -1)
                self.assertNotIn(
                    raw,
                    {8.0, 10.0, 12.0, 42.0, 404.0, 606.0},
                    msg=f"unexpected extraction from {text!r}: {meta}",
                )

    def test_real_amounts_still_extract_after_identifier_filter(self) -> None:
        from lumenfin.documents import extract_metric_hint_meta

        cases = (
            ("Revenue for fiscal 2024 was 245122 million USD.", 245.122),
            ("(In millions)\nRevenue 245122", 245.122),
            ("Revenue was $245.122 billion.", 245.122),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                meta = extract_metric_hint_meta(text, metric="revenue")
                self.assertIsNotNone(meta)
                self.assertAlmostEqual(float(meta["normalized_value"]), expected, places=3)
                self.assertEqual(meta["confidence"], "high")

    def test_structured_tolerance_rejects_nearby_but_distinct(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev_fund_Microsoft_revenue_FY2024",
            entity="Microsoft",
            citation="lumenfin:sec_companyfacts:Microsoft:FY2024:revenue",
            source_type="sec_companyfacts",
            text="Microsoft revenue was 245.122 billion USD.",
            period="FY2024",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            confidence="high",
        )
        self.assertTrue(
            match_numeric_evidence(
                ref,
                entity="Microsoft",
                metric_name="revenue",
                value=245.122,
                unit="billion_usd",
                period="FY2024",
            ).matched
        )
        self.assertFalse(
            match_numeric_evidence(
                ref,
                entity="Microsoft",
                metric_name="revenue",
                value=249.0,
                unit="billion_usd",
                period="FY2024",
            ).matched
        )

    def test_rag_text_tolerance_allows_small_ocr_error(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        rag = EvidenceRef(
            evidence_id="ev_rag_ocr",
            entity="Microsoft",
            citation="m.pdf#p3",
            source_type="rag",
            text="FY2024 Revenue was 245.1 billion USD.",
            period="FY2024",
        )
        match = match_numeric_evidence(
            rag,
            entity="Microsoft",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertTrue(match.matched)


class Phase2PeriodIdentityTestCase(unittest.TestCase):
    """Same-year period collisions, formula local conflicts, structured defaults."""

    def test_same_year_different_quarter_mismatch(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev_q3",
            entity="Microsoft",
            citation="m.pdf#p1",
            source_type="rag",
            text="Q3 2024 Revenue was 60.0 billion USD.",
            period="Q3 2024",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="revenue",
            value=60.0,
            unit="billion_usd",
            period="Q2 2024",
        )
        self.assertFalse(match.matched)
        self.assertIn("period_mismatch", match.reason)

    def test_fiscal_year_versus_quarter_mismatch(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        q = EvidenceRef(
            evidence_id="ev_q2",
            entity="Microsoft",
            citation="m.pdf#p1",
            source_type="rag",
            text="Q2 2024 Revenue was 60.0 billion USD.",
            period="Q2 2024",
        )
        fy = EvidenceRef(
            evidence_id="ev_fy",
            entity="Microsoft",
            citation="m.pdf#p2",
            source_type="rag",
            text="FY2024 Revenue was 245.122 billion USD.",
            period="FY2024",
        )
        self.assertFalse(
            match_numeric_evidence(
                q, entity="Microsoft", metric_name="revenue", value=60.0,
                unit="billion_usd", period="FY2024",
            ).matched
        )
        self.assertFalse(
            match_numeric_evidence(
                fy, entity="Microsoft", metric_name="revenue", value=245.122,
                unit="billion_usd", period="Q2 2024",
            ).matched
        )

    def test_same_quarter_exact_match(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev_q2",
            entity="Microsoft",
            citation="m.pdf#p1",
            source_type="rag",
            text="Q2 2024 Revenue was 60.0 billion USD.",
            period="Q2 2024",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="revenue",
            value=60.0,
            unit="billion_usd",
            period="Q2 2024",
        )
        self.assertTrue(match.matched)

    def test_evidence_period_overridden_by_conflicting_local_text(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev_conflict",
            entity="Microsoft",
            citation="m.pdf#p1",
            source_type="rag",
            text="Revenue was 245.122 billion USD for FY2023.",
            period="FY2024",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)
        self.assertIn("period", match.reason.lower())

    def test_formula_inputs_different_fy_rejected(self) -> None:
        from lumenfin.claims import EvidenceRef, build_claims, filter_verified

        state = {
            "companies": ["Microsoft"],
            "financial_metrics": {"Microsoft": {"operating_margin": 0.446}},
            "retrieved_docs": {
                "Microsoft": {
                    "structured_source": "sec_companyfacts",
                    "market_data": {
                        "revenue": 245.122,
                        "operating_income": 109.433,
                        "r_and_d": 29.5,
                    },
                    "fundamentals_meta": {"fiscal_year": 2024},
                    "fundamental_provenance": {
                        "operating_income": {
                            "source": "sec_companyfacts",
                            "confidence": "high",
                            "period": "FY2024",
                        },
                        "revenue": {
                            "source": "sec_companyfacts",
                            "confidence": "high",
                            "period": "FY2023",
                        },
                        "r_and_d": {
                            "source": "sec_companyfacts",
                            "confidence": "high",
                            "period": "FY2024",
                        },
                    },
                    "supply_chain": {"risk_level": "unknown", "signals": []},
                    "source_documents": [],
                }
            },
            "rag_evidence": {
                "Microsoft": [
                    {
                        "citation": "a.pdf#p1",
                        "source_type": "rag",
                        "period": "FY2024",
                        "text": "Operating income was 109.433 billion USD for FY2024.",
                    },
                    {
                        "citation": "b.pdf#p1",
                        "source_type": "rag",
                        "period": "FY2023",
                        "text": "Revenue was 245.122 billion USD for FY2023.",
                    },
                ]
            },
            "risk_scores": {},
            "market_snapshots": {},
        }
        claims = build_claims(state)
        margin = next(c for c in claims if c.metric_name == "operating_margin")
        self.assertNotEqual(margin.verification, "verified")
        self.assertTrue(
            "period" in margin.verify_reason.lower()
            or "formula_input_period" in margin.verify_reason.lower()
        )

        # Direct multi-input match API path
        from lumenfin.claims import match_numeric_evidence

        blended = EvidenceRef(
            evidence_id="ev_blend",
            entity="Microsoft",
            citation="blend.pdf#p1",
            source_type="rag",
            text=(
                "Operating income was 109.433 billion USD for FY2024. "
                "Revenue was 245.122 billion USD for FY2023."
            ),
            period="FY2024",
        )
        match = match_numeric_evidence(
            blended,
            entity="Microsoft",
            metric_name="operating_margin",
            value=0.446,
            unit="ratio",
            period="FY2024",
            formula_inputs={"operating_income": 109.433, "revenue": 245.122},
        )
        self.assertFalse(match.matched)
        self.assertIn("period", match.reason.lower())

    def test_formula_annual_and_quarter_mix_rejected(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        blended = EvidenceRef(
            evidence_id="ev_mix",
            entity="Microsoft",
            citation="mix.pdf#p1",
            source_type="rag",
            text=(
                "Operating income was 109.433 billion USD for FY2024. "
                "Revenue was 60.0 billion USD for Q2 2024."
            ),
            period="FY2024",
        )
        match = match_numeric_evidence(
            blended,
            entity="Microsoft",
            metric_name="operating_margin",
            value=0.446,
            unit="ratio",
            period="FY2024",
            formula_inputs={"operating_income": 109.433, "revenue": 60.0},
        )
        self.assertFalse(match.matched)
        self.assertIn("period", match.reason.lower())

    def test_period_type_annual_cannot_prove_fy(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev_annual",
            entity="Microsoft",
            citation="lumenfin:sec:Microsoft:annual:revenue",
            source_type="sec_companyfacts",
            text="Microsoft revenue was 245.122 billion USD.",
            period=None,
            period_type="annual",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            confidence="high",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)

    def test_provider_missing_is_normalized_not_promoted(self) -> None:
        from lumenfin.tools import _payload_from_documents

        meta = {
            "normalized_value": 245.122,
            "normalized_unit": "billion_usd",
            "currency": "USD",
            "confidence": "high",
            "normalization_source": "provider_metadata",
            "provider": "sec_companyfacts",
            "period": "FY2024",
            # intentionally omit is_normalized
        }
        docs = [
            {
                "detected_companies": ["Microsoft"],
                "metric_hints": {"revenue": 245.122, "r_and_d": 29.5, "ebitda": 100.0},
                "metric_hint_meta": {"revenue": meta},
                "excerpt": "x",
                "text": "x",
            }
        ]
        payload = _payload_from_documents("Microsoft", docs, include_appendix=False)
        self.assertNotIn("revenue", payload.get("market_data") or {})

    def test_structured_unit_missing_rejected(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev_fund_Microsoft_revenue_FY2024",
            entity="Microsoft",
            citation="lumenfin:sec_companyfacts:Microsoft:FY2024:revenue",
            source_type="sec_companyfacts",
            text="Microsoft revenue was 245.122 billion USD.",
            period="FY2024",
            metric_name="revenue",
            value=245.122,
            unit=None,
            confidence="high",
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)
        self.assertEqual(match.reason, "unit_missing")

    def test_structured_confidence_missing_rejected(self) -> None:
        from lumenfin.claims import EvidenceRef, match_numeric_evidence

        ref = EvidenceRef(
            evidence_id="ev_fund_Microsoft_revenue_FY2024",
            entity="Microsoft",
            citation="lumenfin:sec_companyfacts:Microsoft:FY2024:revenue",
            source_type="sec_companyfacts",
            text="Microsoft revenue was 245.122 billion USD.",
            period="FY2024",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            confidence=None,
        )
        match = match_numeric_evidence(
            ref,
            entity="Microsoft",
            metric_name="revenue",
            value=245.122,
            unit="billion_usd",
            period="FY2024",
        )
        self.assertFalse(match.matched)
        self.assertEqual(match.reason, "confidence_missing")


if __name__ == "__main__":
    unittest.main()
