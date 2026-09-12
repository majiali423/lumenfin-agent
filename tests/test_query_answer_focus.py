"""Offline regressions: answer the asked fact, keep company/period binding, skip extra nodes."""

from __future__ import annotations

import unittest

from lumenfin.claims.build import build_claims
from lumenfin.claims.models import filter_verified
from lumenfin.graph import route_after_quant, route_after_retrieval
from lumenfin.planning import build_query_plan
from lumenfin.query_focus import (
    detect_company_periods,
    detect_requested_metrics,
    format_billion_amount,
    query_has_relative_unanchored_period,
)
from lumenfin.quant_contract import has_requested_structured_facts, has_structured_amounts
from lumenfin.reporting import (
    annotate_upload_period_meta,
    build_analyst_executive_summary,
    requested_fiscal_year_from_state,
)
from lumenfin.task_spec import task_spec_from_plan
from lumenfin.tools import KNOWN_ALIASES, retrieve_company_payload


def _docs_for(*companies: str) -> list[dict]:
    return [{"detected_companies": list(companies), "filename": "upload.pdf", "text": "excerpt"}]


class QueryFocusTestCase(unittest.TestCase):
    def test_metrics_and_periods_from_varied_wording(self) -> None:
        self.assertEqual(
            detect_requested_metrics("What was Contoso FY2019 operating income?"),
            ["operating_income"],
        )
        self.assertEqual(
            detect_requested_metrics("Please report Fabrikam research and development expense"),
            ["r_and_d"],
        )
        self.assertNotIn(
            "operating_income",
            detect_requested_metrics("What was NVIDIA operating margin?"),
        )
        periods = detect_company_periods(
            "compare NVIDIA FY2025 operating income with Microsoft FY2024 operating income",
            ["NVIDIA", "Microsoft"],
            aliases=KNOWN_ALIASES,
        )
        self.assertEqual(periods["NVIDIA"], "FY2025")
        self.assertEqual(periods["Microsoft"], "FY2024")
        p15 = detect_company_periods(
            "Using uploaded files only, compare Microsoft FY2024 R&D expense with NVIDIA FY2025 R&D expense.",
            ["Microsoft", "NVIDIA"],
            aliases=KNOWN_ALIASES,
        )
        self.assertEqual(p15["Microsoft"], "FY2024")
        self.assertEqual(p15["NVIDIA"], "FY2025")

    def test_format_keeps_source_million_precision(self) -> None:
        self.assertEqual(format_billion_amount(245.122), "245.122")
        self.assertEqual(format_billion_amount(81.453), "81.453")
        self.assertEqual(format_billion_amount(29.51), "29.51")


class SimpleFactTaskSpecTestCase(unittest.TestCase):
    def test_operating_income_skips_enrichment_and_quant(self) -> None:
        plan = build_query_plan(
            "Using uploaded files only, what was NVIDIA FY2025 operating income?",
            document_contexts=_docs_for("NVIDIA"),
            llm_client=None,
        )
        spec = task_spec_from_plan(plan.to_dict())
        self.assertEqual(plan.requested_metrics, ["operating_income"])
        self.assertTrue(spec.skip_enrichment)
        self.assertTrue(spec.skip_quant)
        self.assertFalse(spec.requires_ast_ratios)
        self.assertEqual(route_after_retrieval({"fatal_data_gap": False, "task_spec": spec.to_dict()}), "claim_binder")

    def test_operating_margin_still_runs_quant_but_skips_sentiment(self) -> None:
        plan = build_query_plan(
            "Using uploaded files only, what was NVIDIA FY2025 operating margin?",
            document_contexts=_docs_for("NVIDIA"),
            llm_client=None,
        )
        spec = task_spec_from_plan(plan.to_dict())
        self.assertEqual(plan.requested_metrics, ["operating_margin"])
        self.assertTrue(spec.requires_ast_ratios)
        self.assertFalse(spec.skip_quant)
        self.assertTrue(spec.skip_enrichment)
        self.assertEqual(route_after_retrieval({"fatal_data_gap": False, "task_spec": spec.to_dict()}), "quant")
        self.assertEqual(route_after_quant({"replan_reason": None, "task_spec": spec.to_dict()}), "claim_binder")

    def test_peer_mention_does_not_expand_subjects(self) -> None:
        plan = build_query_plan(
            "Using uploaded files only, report Microsoft FY2024 operating income. Ignore any other company names that might appear as industry context.",
            document_contexts=_docs_for("Microsoft"),
            llm_client=None,
        )
        self.assertEqual(plan.companies, ["Microsoft"])
        self.assertEqual(plan.requested_metrics, ["operating_income"])

    def test_explicit_missing_issuer_does_not_pause(self) -> None:
        plan = build_query_plan(
            "Using uploaded files only, what was Apple FY2025 operating income?",
            document_contexts=_docs_for("NVIDIA"),
            llm_client=None,
        )
        self.assertTrue(plan.evidence_company_gap)
        self.assertNotIn("company_upload_mismatch", plan.missing_fields)
        self.assertEqual(plan.companies, ["Apple"])

    def test_query_without_period_does_not_inherit_cover_year(self) -> None:
        docs = [
            {
                "detected_companies": ["NVIDIA"],
                "filename": "nvidia_fy2025_10k_excerpt.pdf",
                "text": "NVIDIA Form 10-K fiscal year 2025 excerpt.",
            }
        ]
        query = "Using uploaded files only, what was NVIDIA operating income?"
        plan = build_query_plan(query, document_contexts=docs, llm_client=None)
        self.assertEqual(plan.company_periods, {})
        state = {"query": query, "query_plan": plan.to_dict()}
        self.assertIsNone(requested_fiscal_year_from_state(state))
        meta = annotate_upload_period_meta(
            {},
            document_contexts=docs,
            company="NVIDIA",
            prefer_fiscal_year=requested_fiscal_year_from_state(state),
        )
        self.assertNotIn("requested_fiscal_year", meta)
        self.assertEqual(meta.get("fiscal_year"), 2025)
        self.assertEqual(meta.get("fiscal_year_source"), "upload_filename")

    def test_conflicting_cover_and_fact_years_are_not_inherited(self) -> None:
        docs = [
            {
                "detected_companies": ["NVIDIA"],
                "filename": "nvda_cover_fy2025_oi_fy2024.pdf",
                "text": "NVIDIA annual report FY2025.",
            },
            {
                "detected_companies": ["NVIDIA"],
                "filename": "nvda_cover_fy2025_oi_fy2024.pdf",
                "text": "NVIDIA FY2024 operating income was 32.972 billion USD.",
                "page": 2,
            },
        ]
        meta = annotate_upload_period_meta(
            {},
            document_contexts=docs,
            company="NVIDIA",
            prefer_fiscal_year=None,
        )
        self.assertIsNone(meta.get("fiscal_year"))
        self.assertNotIn("requested_fiscal_year", meta)


class StructuredFactBindingTestCase(unittest.TestCase):
    def test_operating_income_without_revenue_is_still_answerable(self) -> None:
        payload = {"market_data": {"operating_income": 81.453}, "structured_source": "document_extracted"}
        self.assertTrue(has_structured_amounts(payload))
        self.assertTrue(has_requested_structured_facts(payload, ["operating_income"]))
        self.assertFalse(has_requested_structured_facts(payload, ["operating_margin"]))
        state = {
            "companies": ["NVIDIA"],
            "query_plan": {"requested_metrics": ["operating_income"], "intent": "document_financial_diligence"},
            "task_spec": task_spec_from_plan(
                {"intent": "document_financial_diligence", "requested_metrics": ["operating_income"]}
            ).to_dict(),
            "retrieved_docs": {
                "NVIDIA": {
                    "market_data": {"operating_income": 81.453},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2025},
                    "confidence": {"overall": 0.9},
                }
            },
            "financial_metrics": {"NVIDIA": {"operating_income": 81.453}},
            "rag_evidence": {
                "NVIDIA": [
                    {
                        "text": "NVIDIA FY2025 operating income was 81.453 billion USD.",
                        "citation": "nvda.pdf#p1",
                        "page": 1,
                    }
                ]
            },
        }
        verified = filter_verified(build_claims(state))
        names = {claim.metric_name for claim in verified if claim.claim_type == "numeric"}
        self.assertIn("operating_income", names)
        self.assertNotIn("operating_margin", names)
        summary = build_analyst_executive_summary(state, verified)
        self.assertIn("81.453", summary)
        self.assertIn("operating income", summary.lower())
        self.assertNotIn("Operating margin", summary)

    def test_margin_without_revenue_is_not_invented(self) -> None:
        payload = {"market_data": {"operating_income": 81.453}, "structured_source": "document_extracted"}
        self.assertFalse(has_requested_structured_facts(payload, ["operating_margin"]))
        state = {
            "companies": ["NVIDIA"],
            "query_plan": {"requested_metrics": ["operating_margin"]},
            "task_spec": task_spec_from_plan(
                {"intent": "document_financial_diligence", "requested_metrics": ["operating_margin"]}
            ).to_dict(),
            "retrieved_docs": {
                "NVIDIA": {
                    "market_data": {"operating_income": 81.453},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2025},
                }
            },
            "financial_metrics": {"NVIDIA": {"operating_income": 81.453}},
        }
        verified = filter_verified(build_claims(state))
        self.assertFalse(any(claim.metric_name == "operating_margin" for claim in verified))

    def test_multi_company_periods_stay_separate(self) -> None:
        state = {
            "companies": ["NVIDIA", "Microsoft"],
            "query_plan": {
                "requested_metrics": ["operating_income"],
                "company_periods": {"NVIDIA": "FY2025", "Microsoft": "FY2024"},
            },
            "task_spec": task_spec_from_plan(
                {"intent": "comparative_financial_diligence", "requested_metrics": ["operating_income"]}
            ).to_dict(),
            "retrieved_docs": {
                "NVIDIA": {
                    "market_data": {"operating_income": 81.453},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2025},
                    "rag_hits": [],
                },
                "Microsoft": {
                    "market_data": {"operating_income": 109.433},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2024},
                },
            },
            "financial_metrics": {
                "NVIDIA": {"operating_income": 81.453},
                "Microsoft": {"operating_income": 109.433},
            },
            "rag_evidence": {
                "NVIDIA": [{"text": "NVIDIA FY2025 operating income was 81.453 billion USD.", "citation": "nvda.pdf#p1", "page": 1}],
                "Microsoft": [{"text": "Microsoft FY2024 operating income was 109.433 billion USD.", "citation": "msft.pdf#p1", "page": 1}],
            },
        }
        verified = filter_verified(build_claims(state))
        by_co = {claim.entity: claim for claim in verified if claim.metric_name == "operating_income"}
        self.assertIn("NVIDIA", by_co)
        self.assertIn("Microsoft", by_co)
        self.assertEqual(by_co["NVIDIA"].period, "FY2025")
        self.assertEqual(by_co["Microsoft"].period, "FY2024")
        summary = build_analyst_executive_summary(state, verified)
        self.assertIn("81.453", summary)
        self.assertIn("109.433", summary)
        self.assertIn("nvda.pdf#p1", summary)
        self.assertIn("msft.pdf#p1", summary)

    def test_partial_company_gap_keeps_available_fact(self) -> None:
        state = {
            "companies": ["Microsoft", "Contoso"],
            "query_plan": {"requested_metrics": ["operating_income"]},
            "task_spec": task_spec_from_plan(
                {"intent": "comparative_financial_diligence", "requested_metrics": ["operating_income"]}
            ).to_dict(),
            "retrieved_docs": {
                "Microsoft": {
                    "market_data": {"operating_income": 109.433},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2024},
                },
                "Contoso": {"market_data": {}, "structured_source": "none"},
            },
            "financial_metrics": {"Microsoft": {"operating_income": 109.433}},
            "rag_evidence": {
                "Microsoft": [{"text": "Microsoft FY2024 operating income was 109.433 billion USD.", "citation": "msft.pdf#p1", "page": 1}],
            },
        }
        verified = filter_verified(build_claims(state))
        self.assertTrue(any(claim.entity == "Microsoft" and claim.metric_name == "operating_income" for claim in verified))
        self.assertFalse(any(claim.entity == "Contoso" and claim.metric_name == "operating_income" and claim.verification == "verified" for claim in verified))


class SkipEnrichmentEvidenceBoundaryTestCase(unittest.TestCase):
    def _spec(self):
        spec = task_spec_from_plan(
            {"intent": "document_financial_diligence", "requested_metrics": ["operating_income"]}
        )
        self.assertTrue(spec.skip_enrichment)
        self.assertEqual(route_after_retrieval({"fatal_data_gap": False, "task_spec": spec.to_dict()}), "claim_binder")
        return spec

    def _state(self, *, rag: list[dict] | None) -> dict:
        spec = self._spec()
        return {
            "companies": ["NVIDIA"],
            "query_plan": {"requested_metrics": ["operating_income"]},
            "task_spec": spec.to_dict(),
            "retrieved_docs": {
                "NVIDIA": {
                    "market_data": {"operating_income": 81.453},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2025},
                    "fundamental_provenance": {},
                }
            },
            "financial_metrics": {"NVIDIA": {"operating_income": 81.453}},
            "rag_evidence": {"NVIDIA": rag or []},
        }

    def _operating_income(self, state: dict):
        claims = build_claims(state)
        found = [claim for claim in claims if claim.metric_name == "operating_income"]
        self.assertTrue(found)
        return found[0]

    def test_wrong_company_excerpt_is_not_verified(self) -> None:
        claim = self._operating_income(
            self._state(
                rag=[
                    {
                        "text": "Microsoft FY2025 operating income was 81.453 billion USD.",
                        "citation": "msft.pdf#p1",
                        "page": 1,
                        "period": "FY2025",
                        "period_source": "document_text",
                        "period_alignment": "exact",
                    }
                ]
            )
        )
        self.assertNotEqual(claim.verification, "verified")

    def test_wrong_period_excerpt_is_not_verified(self) -> None:
        claim = self._operating_income(
            self._state(
                rag=[
                    {
                        "text": "NVIDIA FY2024 operating income was 81.453 billion USD.",
                        "citation": "nvda.pdf#p1",
                        "page": 1,
                        "period": "FY2024",
                        "period_source": "document_text",
                        "period_alignment": "exact",
                    }
                ]
            )
        )
        self.assertNotEqual(claim.verification, "verified")

    def test_excerpt_without_the_number_is_not_verified(self) -> None:
        claim = self._operating_income(
            self._state(
                rag=[
                    {
                        "text": "NVIDIA FY2025 operating income increased year over year.",
                        "citation": "nvda.pdf#p1",
                        "page": 1,
                        "period": "FY2025",
                        "period_source": "document_text",
                        "period_alignment": "exact",
                    }
                ]
            )
        )
        self.assertNotEqual(claim.verification, "verified")

    def test_missing_source_is_not_verified(self) -> None:
        claim = self._operating_income(self._state(rag=[]))
        self.assertNotEqual(claim.verification, "verified")

    def test_complex_supply_chain_query_still_runs_quant(self) -> None:
        plan = build_query_plan(
            "对比分析 Apple 与 Microsoft 2025 年供应链风险和研发投入",
            document_contexts=_docs_for("Apple", "Microsoft"),
            llm_client=None,
        )
        spec = task_spec_from_plan(plan.to_dict())
        self.assertFalse(spec.skip_enrichment)
        self.assertFalse(spec.skip_quant)
        self.assertEqual(route_after_retrieval({"fatal_data_gap": False, "task_spec": spec.to_dict()}), "quant")
        self.assertEqual(route_after_quant({"replan_reason": None, "task_spec": spec.to_dict()}), "psychologist")


class SubjectAndPeriodCorrectnessTestCase(unittest.TestCase):
    def test_relative_last_year_is_not_a_requested_fiscal_year(self) -> None:
        query = "Using uploaded files only, what was operating income last year?"
        self.assertTrue(query_has_relative_unanchored_period(query))
        plan = build_query_plan(
            query,
            document_contexts=[{"detected_companies": ["NVIDIA"], "filename": "nvda_fy2025.pdf"}],
            llm_client=None,
        )
        self.assertIsNone(requested_fiscal_year_from_state({"query": query, "query_plan": plan.to_dict()}))
        self.assertTrue(any("unanchored" in note.lower() for note in plan.planner_notes))

    def test_out_of_scope_verified_fact_is_not_the_answer(self) -> None:
        from lumenfin.claims.models import Claim

        state = {
            "companies": ["Apple"],
            "query": "Using uploaded files only, what was Apple FY2025 operating income?",
            "query_plan": {
                "requested_metrics": ["operating_income"],
                "query_companies": ["Apple"],
                "upload_companies": ["NVIDIA"],
                "evidence_company_gap": True,
                "company_scope": "mismatch",
            },
            "retrieved_docs": {"Apple": {"market_data": {}, "structured_source": "none"}},
        }
        nvidia_claim = Claim(
            claim_id="cl_abs_NVIDIA_operating_income",
            entity="NVIDIA",
            claim_type="numeric",
            statement="NVIDIA Operating income is 81.453 billion USD for FY2025.",
            value=81.453,
            unit="billion_usd",
            period="FY2025",
            metric_name="operating_income",
            verification="verified",
        )
        summary = build_analyst_executive_summary(state, [nvidia_claim])
        self.assertNotIn("81.453", summary)
        self.assertIn("Apple", summary)
        self.assertIn("NVIDIA", summary)

    def test_requested_year_mismatch_is_not_verified_as_the_answer(self) -> None:
        state = {
            "companies": ["NVIDIA"],
            "query": "Using uploaded files only, what was NVIDIA FY2025 operating income?",
            "query_plan": {
                "requested_metrics": ["operating_income"],
                "time_range": "FY2025",
                "company_periods": {"NVIDIA": "FY2025"},
            },
            "task_spec": task_spec_from_plan(
                {"intent": "document_financial_diligence", "requested_metrics": ["operating_income"]}
            ).to_dict(),
            "retrieved_docs": {
                "NVIDIA": {
                    "market_data": {"operating_income": 32.972},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2024},
                    "fundamental_provenance": {
                        "operating_income": {
                            "source": "document_extracted",
                            "period": "FY2024",
                            "period_source": "document_text",
                            "period_alignment": "exact",
                        }
                    },
                    "confidence": {"overall": 0.9},
                }
            },
            "financial_metrics": {"NVIDIA": {"operating_income": 32.972}},
            "rag_evidence": {
                "NVIDIA": [
                    {
                        "text": "NVIDIA FY2024 operating income was 32.972 billion USD.",
                        "citation": "nvda.pdf#p2",
                        "page": 2,
                        "period": "FY2024",
                        "period_source": "document_text",
                        "period_alignment": "exact",
                    }
                ]
            },
        }
        claims = build_claims(state)
        oi = [claim for claim in claims if claim.metric_name == "operating_income"]
        self.assertTrue(oi)
        self.assertNotEqual(oi[0].verification, "verified")
        summary = build_analyst_executive_summary(state, filter_verified(claims))
        self.assertIn("FY2024", summary)
        self.assertIn("FY2025", summary)
        self.assertNotIn("is 32.972 billion USD for FY2025", summary)

    def test_single_field_period_without_requested_year_stays_labeled(self) -> None:
        payload = retrieve_company_payload(
            "NVIDIA",
            document_contexts=[
                {
                    "detected_companies": ["NVIDIA"],
                    "filename": "nvda_cover_fy2025.pdf",
                    "text": "NVIDIA annual report FY2025",
                    "page": 1,
                    "citation": "nvda_cover_fy2025.pdf#p1",
                },
                {
                    "detected_companies": ["NVIDIA"],
                    "filename": "nvda_cover_fy2025.pdf",
                    "text": "NVIDIA FY2024 operating income was 32.972 billion USD.",
                    "page": 2,
                    "citation": "nvda_cover_fy2025.pdf#p2",
                },
            ],
            allow_sample_data=False,
            prefer_uploaded_only=True,
        )
        self.assertEqual(payload.get("structured_source"), "document_extracted")
        self.assertAlmostEqual(float(payload["market_data"]["operating_income"]), 32.972, places=3)
        self.assertEqual(payload["fundamental_provenance"]["operating_income"]["period"], "FY2024")
        self.assertNotEqual(payload["fundamental_provenance"]["operating_income"]["period"], "FY2025")

    def test_existing_labeled_fact_still_verifies(self) -> None:
        state = {
            "companies": ["NVIDIA"],
            "query": "Using uploaded files only, what was NVIDIA FY2025 operating income?",
            "query_plan": {"requested_metrics": ["operating_income"], "time_range": "FY2025"},
            "task_spec": task_spec_from_plan(
                {"intent": "document_financial_diligence", "requested_metrics": ["operating_income"]}
            ).to_dict(),
            "retrieved_docs": {
                "NVIDIA": {
                    "market_data": {"operating_income": 81.453},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2025},
                    "fundamental_provenance": {
                        "operating_income": {
                            "source": "document_extracted",
                            "period": "FY2025",
                            "period_source": "document_text",
                            "period_alignment": "exact",
                        }
                    },
                    "confidence": {"overall": 0.9},
                }
            },
            "financial_metrics": {"NVIDIA": {"operating_income": 81.453}},
            "rag_evidence": {
                "NVIDIA": [
                    {
                        "text": "NVIDIA FY2025 operating income was 81.453 billion USD.",
                        "citation": "nvda.pdf#p1",
                        "page": 1,
                        "period": "FY2025",
                        "period_source": "document_text",
                        "period_alignment": "exact",
                    }
                ]
            },
        }
        verified = filter_verified(build_claims(state))
        self.assertTrue(any(claim.metric_name == "operating_income" for claim in verified))
        summary = build_analyst_executive_summary(state, verified)
        self.assertIn("81.453", summary)
        self.assertIn("FY2025", summary)

    def test_unlabeled_field_amount_is_not_bound_to_cover_year(self) -> None:
        state = {
            "companies": ["NVIDIA"],
            "query": "Using uploaded files only, what is NVIDIA operating income and for which fiscal year is it stated?",
            "query_plan": {"requested_metrics": ["operating_income"]},
            "task_spec": task_spec_from_plan(
                {"intent": "document_financial_diligence", "requested_metrics": ["operating_income"]}
            ).to_dict(),
            "retrieved_docs": {
                "NVIDIA": {
                    "market_data": {"operating_income": 32.972},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2025, "fiscal_year_source": "upload_filename"},
                    "fundamental_provenance": {
                        "operating_income": {
                            "source": "document_extracted",
                            "period": None,
                            "period_source": "unknown",
                            "period_alignment": "unknown",
                            "citation": "nvda.pdf#p2",
                        }
                    },
                    "confidence": {"overall": 0.9},
                }
            },
            "financial_metrics": {"NVIDIA": {"operating_income": 32.972}},
            "rag_evidence": {
                "NVIDIA": [
                    {
                        "text": "operating income was 32.972 billion USD",
                        "citation": "nvda.pdf#p2",
                        "page": 2,
                    }
                ]
            },
        }
        claims = build_claims(state)
        oi = [claim for claim in claims if claim.metric_name == "operating_income"]
        self.assertTrue(oi)
        self.assertEqual(oi[0].verification, "verified")
        self.assertEqual(oi[0].period, "unknown")
        summary = build_analyst_executive_summary(state, filter_verified(claims))
        self.assertIn("32.972", summary)
        self.assertRegex(summary.lower(), r"period not stated")
        self.assertNotIn("for FY2025", summary)
        self.assertNotIn("do not support a verified", summary.lower())

    def test_bilateral_rd_compare_keeps_both_issuers(self) -> None:
        query = (
            "Using uploaded files only, compare Microsoft FY2024 R&D expense with NVIDIA FY2025 R&D expense."
        )
        plan = build_query_plan(
            query,
            document_contexts=[
                {"issuer_companies": ["Microsoft"], "detected_companies": ["Microsoft"], "filename": "msft.pdf"},
                {"issuer_companies": ["NVIDIA"], "detected_companies": ["NVIDIA"], "filename": "nvda.pdf"},
            ],
            llm_client=None,
        )
        self.assertEqual(plan.company_periods.get("Microsoft"), "FY2024")
        self.assertEqual(plan.company_periods.get("NVIDIA"), "FY2025")
        state = {
            "companies": ["Microsoft", "NVIDIA"],
            "query": query,
            "query_plan": plan.to_dict(),
            "task_spec": task_spec_from_plan(plan.to_dict()).to_dict(),
            "retrieved_docs": {
                "Microsoft": {
                    "market_data": {"r_and_d": 29.51},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2024},
                    "fundamental_provenance": {
                        "r_and_d": {
                            "source": "document_extracted",
                            "period": "FY2024",
                            "period_source": "document_text",
                            "period_alignment": "exact",
                            "citation": "msft.pdf#p1",
                        }
                    },
                },
                "NVIDIA": {
                    "market_data": {"r_and_d": 12.914},
                    "structured_source": "document_extracted",
                    "fundamentals_meta": {"fiscal_year": 2025},
                    "fundamental_provenance": {
                        "r_and_d": {
                            "source": "document_extracted",
                            "period": "FY2025",
                            "period_source": "document_text",
                            "period_alignment": "exact",
                            "citation": "nvda.pdf#p1",
                        }
                    },
                },
            },
            "financial_metrics": {"Microsoft": {"r_and_d": 29.51}, "NVIDIA": {"r_and_d": 12.914}},
            "rag_evidence": {
                "Microsoft": [
                    {"text": "Microsoft FY2024 R&D expense was 29.51 billion USD.", "citation": "msft.pdf#p1", "page": 1}
                ],
                "NVIDIA": [
                    {"text": "NVIDIA FY2025 R&D expense was 12.914 billion USD.", "citation": "nvda.pdf#p1", "page": 1}
                ],
            },
        }
        verified = filter_verified(build_claims(state))
        by_co = {claim.entity: claim for claim in verified if claim.metric_name == "r_and_d"}
        self.assertIn("Microsoft", by_co)
        self.assertIn("NVIDIA", by_co)
        summary = build_analyst_executive_summary(state, verified)
        self.assertIn("29.51", summary)
        self.assertIn("12.914", summary)
        self.assertNotIn("do not support a verified", summary.lower())


if __name__ == "__main__":
    unittest.main()
