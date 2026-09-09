from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.offline_env import apply_offline_env

apply_offline_env()

from lumenfin.documents import expand_document_page_contexts, extract_metric_amounts, unique_statement_period
from lumenfin.reporting import resolve_provenance_home
from lumenfin.tools import _payload_from_documents, retrieve_company_payload


SHARED_ID = "annual-report"


def _doc(filename: str, page: int, text: str, *, document_id: str | None = None) -> dict:
    name = filename
    return {
        "document_id": document_id or f"{Path(filename).stem}-p{page}",
        "filename": name,
        "citation": f"{name}#p{page}",
        "page": page,
        "detected_companies": ["NVIDIA"],
        "excerpt": text,
        "text": text,
    }


def _annual(page: int, text: str) -> dict:
    return _doc("annual.pdf", page, text, document_id=SHARED_ID)


class FieldPeriodBindingTestCase(unittest.TestCase):
    def test_cover_year_does_not_stamp_other_page_amount(self) -> None:
        cover = _doc("annual.pdf", 1, "NVIDIA annual report FY2025")
        facts = _doc(
            "annual.pdf",
            2,
            "NVIDIA FY2024 operating income was 32.972 billion USD.",
        )
        payload = _payload_from_documents("NVIDIA", [cover, facts], include_appendix=False)
        market = payload.get("market_data") or {}
        self.assertAlmostEqual(float(market["operating_income"]), 32.972, places=3)
        prov = payload["fundamental_provenance"]["operating_income"]
        self.assertEqual(prov["period"], "FY2024")
        self.assertEqual(prov["period_source"], "document_text")
        self.assertEqual(prov["period_alignment"], "exact")
        self.assertIn("#p2", str(prov.get("citation") or ""))
        self.assertNotIn("FY2025", str(prov.get("period") or ""))
        self.assertTrue(str(prov.get("source_record_id") or "").startswith("document:"))
        self.assertIn(":p2:operating_income", str(prov.get("source_record_id")))

        retrieved = retrieve_company_payload(
            "NVIDIA",
            document_contexts=[cover, facts],
            allow_sample_data=False,
            prefer_uploaded_only=True,
        )
        again = retrieved["fundamental_provenance"]["operating_income"]
        self.assertEqual(again["period"], "FY2024")
        self.assertIn("#p2", str(again.get("citation") or ""))

    def test_multi_year_table_is_ambiguous_not_exact(self) -> None:
        text = "FY2024    FY2025\nOperating income  32.972  81.453\n"
        amounts = extract_metric_amounts(text)
        row = amounts["operating_income"]
        self.assertIn(row.period_source, {"ambiguous", None})
        self.assertNotEqual(row.period_alignment, "exact")
        payload = _payload_from_documents(
            "NVIDIA",
            [_doc("table.pdf", 1, text)],
            include_appendix=False,
        )
        prov = (payload.get("fundamental_provenance") or {}).get("operating_income") or {}
        if prov:
            self.assertNotEqual(prov.get("period_alignment"), "exact")
            self.assertNotEqual(prov.get("period_source"), "document_text")

    def test_two_files_keep_their_own_periods(self) -> None:
        fy24 = _doc(
            "nvda_fy2024.pdf",
            1,
            "NVIDIA FY2024 operating income was 32.972 billion USD.",
            document_id="fy24",
        )
        fy25 = _doc(
            "nvda_fy2025.pdf",
            1,
            "NVIDIA FY2025 revenue was 130.5 billion USD.",
            document_id="fy25",
        )
        payload = _payload_from_documents("NVIDIA", [fy24, fy25], include_appendix=False)
        oi = payload["fundamental_provenance"]["operating_income"]
        rev = payload["fundamental_provenance"]["revenue"]
        self.assertEqual(oi["period"], "FY2024")
        self.assertIn("nvda_fy2024.pdf", str(oi.get("citation")))
        self.assertEqual(rev["period"], "FY2025")
        self.assertIn("nvda_fy2025.pdf", str(rev.get("citation")))

    def test_missing_period_stays_unknown(self) -> None:
        docs = [_doc("notes.pdf", 1, "NVIDIA operating income was 32.972 billion USD.")]
        payload = _payload_from_documents("NVIDIA", docs, include_appendix=False)
        prov = payload["fundamental_provenance"]["operating_income"]
        self.assertTrue(not prov.get("period") or prov.get("period_alignment") in {None, "unknown"})
        self.assertNotEqual(prov.get("period_alignment"), "exact")
        self.assertNotEqual(prov.get("period_source"), "query")

    def test_unique_same_page_fiscal_year_ended_is_document_text(self) -> None:
        self.assertEqual(
            unique_statement_period("Fiscal year ended: 2025-01-26. Operating income 81453."),
            "FY2025",
        )
        docs = [
            _doc(
                "nvda_fy2025_10k_excerpt.pdf",
                1,
                "NVIDIA. Fiscal year ended: 2025-01-26. Operating income 81.453 billion USD.",
            )
        ]
        payload = _payload_from_documents("NVIDIA", docs, include_appendix=False)
        prov = payload["fundamental_provenance"]["operating_income"]
        self.assertEqual(prov["period"], "FY2025")
        self.assertEqual(prov["period_source"], "document_text")
        self.assertEqual(prov["period_alignment"], "exact")
        self.assertIn("#p1", str(prov.get("citation") or ""))

    def test_prefix_fy_before_metric_label_binds_exact_period(self) -> None:
        amounts = extract_metric_amounts("NVIDIA FY2024 operating income was 32.972 billion USD.")
        row = amounts["operating_income"]
        self.assertAlmostEqual(float(row.normalized_value or 0), 32.972, places=3)
        self.assertEqual(row.period, "FY2024")
        self.assertEqual(row.period_alignment, "exact")

    def test_same_document_id_explicit_field_period_keeps_fy2024(self) -> None:
        cover = _annual(1, "NVIDIA annual report FY2025.")
        facts = _annual(2, "NVIDIA FY2024 operating income was 32.972 billion USD.")
        for docs in ([cover, facts], [facts, cover]):
            payload = _payload_from_documents("NVIDIA", docs, include_appendix=False)
            market = payload.get("market_data") or {}
            self.assertAlmostEqual(float(market["operating_income"]), 32.972, places=3)
            prov = payload["fundamental_provenance"]["operating_income"]
            self.assertEqual(prov["period"], "FY2024", docs)
            self.assertEqual(prov["period_alignment"], "exact")
            self.assertEqual(prov["period_source"], "document_text")
            self.assertEqual(str(prov.get("citation")), "annual.pdf#p2")
            self.assertEqual(
                str(prov.get("source_record_id")),
                "document:annual-report:p2:operating_income",
            )
            self.assertEqual(prov.get("location_status"), "unique")

    def test_same_document_id_missing_period_does_not_inherit_cover(self) -> None:
        cover = _annual(1, "NVIDIA annual report FY2025.")
        facts = _annual(2, "NVIDIA operating income was 32.972 billion USD.")
        for docs in ([cover, facts], [facts, cover]):
            payload = _payload_from_documents("NVIDIA", docs, include_appendix=False)
            market = payload.get("market_data") or {}
            self.assertAlmostEqual(float(market["operating_income"]), 32.972, places=3)
            prov = payload["fundamental_provenance"]["operating_income"]
            self.assertTrue(not prov.get("period") or prov.get("period_alignment") in {None, "unknown"})
            self.assertNotEqual(prov.get("period_alignment"), "exact")
            self.assertNotEqual(prov.get("period"), "FY2025")
            self.assertEqual(str(prov.get("citation")), "annual.pdf#p2")
            self.assertEqual(
                str(prov.get("source_record_id")),
                "document:annual-report:p2:operating_income",
            )

    def test_concatenated_pages_expand_without_changing_document_id(self) -> None:
        blob = {
            "document_id": SHARED_ID,
            "filename": "annual.pdf",
            "detected_companies": ["NVIDIA"],
            "pages": [
                "NVIDIA annual report FY2025.",
                "NVIDIA FY2024 operating income was 32.972 billion USD.",
            ],
            "text": "NVIDIA annual report FY2025.\nNVIDIA FY2024 operating income was 32.972 billion USD.",
            "excerpt": "NVIDIA annual report FY2025.",
        }
        pages = expand_document_page_contexts([blob])
        self.assertEqual(len(pages), 2)
        self.assertEqual({item["document_id"] for item in pages}, {SHARED_ID})
        payload = _payload_from_documents("NVIDIA", [blob], include_appendix=False)
        prov = payload["fundamental_provenance"]["operating_income"]
        self.assertAlmostEqual(float(payload["market_data"]["operating_income"]), 32.972, places=3)
        self.assertEqual(prov["period"], "FY2024")
        self.assertEqual(prov["citation"], "annual.pdf#p2")
        self.assertEqual(prov["source_record_id"], "document:annual-report:p2:operating_income")

    def test_ambiguous_and_unresolved_locations_are_not_first_match(self) -> None:
        cover = _annual(1, "NVIDIA annual report FY2025.")
        facts = _annual(2, "NVIDIA FY2024 operating income was 32.972 billion USD.")
        duplicate = dict(facts)
        home, status = resolve_provenance_home(
            {
                "citation": "annual.pdf#p2",
                "source_record_id": "document:annual-report:p2:operating_income",
            },
            [cover, facts],
        )
        self.assertEqual(status, "unique")
        self.assertEqual(home["page"], 2)
        _, ambiguous = resolve_provenance_home(
            {
                "citation": "annual.pdf#p2",
                "source_record_id": "document:annual-report:p2:operating_income",
            },
            [facts, duplicate],
        )
        self.assertEqual(ambiguous, "ambiguous")
        _, unresolved = resolve_provenance_home(
            {
                "citation": "annual.pdf#p9",
                "source_record_id": "document:annual-report:p9:operating_income",
            },
            [cover, facts],
        )
        self.assertEqual(unresolved, "unresolved")
        _, conflict = resolve_provenance_home(
            {
                "citation": "annual.pdf#p2",
                "source_record_id": "document:annual-report:p1:operating_income",
            },
            [cover, facts],
        )
        self.assertEqual(conflict, "ambiguous")

    def test_graph_finrun_does_not_promote_cover_year_as_exact_fact(self) -> None:
        from dataclasses import replace
        from uuid import uuid4

        from lumenfin.finrun import export_finrun_state
        from lumenfin.llm import LocalFallbackLLMClient
        from lumenfin.service import LumenFinAnalysisService
        from tests.support.fakes import FakeMarketDataClient
        from tests.test_graph_routing import build_test_config

        root = ROOT / "test_artifacts" / f"period-graph-{uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=True)
        cover = root / "cover.txt"
        facts = root / "facts.txt"
        cover.write_text("NVIDIA annual report FY2025\n", encoding="utf-8")
        facts.write_text("NVIDIA FY2024 operating income was 32.972 billion USD.\n", encoding="utf-8")
        config = replace(build_test_config(root), rag_enabled=False)
        service = LumenFinAnalysisService(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        payload = service.analyze(
            query="Using uploaded files only, what is NVIDIA operating income from the filing facts?",
            thread_id="period-cover-vs-field",
            export_artifacts=False,
            document_paths=[str(cover), str(facts)],
        )
        state = payload["result"]
        nvda = (state.get("retrieved_docs") or {}).get("NVIDIA") or {}
        prov = (nvda.get("fundamental_provenance") or {}).get("operating_income") or {}
        self.assertEqual(prov.get("period"), "FY2024")
        self.assertNotEqual(prov.get("period"), "FY2025")
        self.assertEqual(prov.get("period_alignment"), "exact")
        citation = str(prov.get("citation") or "")
        self.assertTrue("facts" in citation.lower() or "#p" in citation.lower(), citation)
        finrun = export_finrun_state(state)
        oi_ev = next(
            (
                row
                for row in finrun.get("evidence") or []
                if row.get("metric") == "operating_income"
            ),
            None,
        )
        if oi_ev:
            self.assertNotEqual(str(oi_ev.get("period") or "").upper(), "FY2025")
            self.assertIn("2024", str(oi_ev.get("period") or "FY2024"))
        body = str(finrun.get("final_output") or state.get("final_report") or "")
        if "32.972" in body and "FY2025" in body and "operating income" in body.lower():
            from finagentbench.metrics.visible_supported_claims import visible_supported_claims

            case = {
                "expected_entities": ["NVIDIA"],
                "required_steps": [],
                "enabled_metrics": ["visible_supported_claims"],
                "scoring_version": "3",
                "execution_path": "product_workflow",
                "require_checkable_metrics": True,
                "require_visible_claim_citations": True,
                "numeric_tolerance": 0.01,
                "block_on_severity": ["high", "critical"],
            }
            state["execution_path"] = "product_workflow"
            scored = visible_supported_claims(export_finrun_state(state), case)
            self.assertFalse(scored.passed, [item.message for item in scored.findings])


if __name__ == "__main__":
    unittest.main()
