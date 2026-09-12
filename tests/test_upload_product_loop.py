"""Upload → job → graph → FinRun → strict visible gate (no sample backfill)."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from tests.support.finagentbench import finagentbench_root

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = finagentbench_root()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from scripts.offline_env import apply_offline_env

apply_offline_env()

from lumenfin.api.app import create_app
from lumenfin.finrun import export_finrun_state
from lumenfin.llm import LocalFallbackLLMClient
from tests.support.fakes import FakeMarketDataClient
from tests.support.jobs import poll_job
from tests.test_graph_routing import build_test_config

GOLD = json.loads(
    (ROOT / "tests" / "fixtures" / "sec" / "nvda_fy2025_operating_income_gold.json").read_text(
        encoding="utf-8"
    )
)
PDF = ROOT / GOLD["source_file"]
SPARSE = ROOT / "tests" / "fixtures" / "sec" / "minimal" / "nvda_narrative_only.txt"
QUERY = (
    "Using uploaded files only, what is NVIDIA FY2025 operating income "
    "from the filing facts?"
)
CASE = {
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


class UploadProductLoopTestCase(unittest.TestCase):
    def test_source_hash_matches_hand_authored_gold(self) -> None:
        digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
        self.assertEqual(digest, GOLD["source_sha256"])
        self.assertTrue(PDF.is_file())

    def test_upload_job_uses_filing_not_sample_and_strict_gate(self) -> None:
        from finagentbench.metrics.visible_supported_claims import visible_supported_claims
        from finagentbench.schema import validate_finrun

        root = ROOT / "test_artifacts" / f"upload-loop-{uuid4().hex[:8]}"
        config = replace(build_test_config(root), api_key=None, rag_enabled=True)
        app = create_app(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        with TestClient(app) as client:
            with PDF.open("rb") as handle:
                submitted = client.post(
                    "/api/v1/jobs/upload",
                    data={"query": QUERY, "thread_id": "upload-nvda-oi", "export_artifacts": "false"},
                    files={"files": (PDF.name, handle, "application/pdf")},
                )
            self.assertEqual(submitted.status_code, 202, submitted.text)
            job_id = submitted.json()["job_id"]
            public = poll_job(client, job_id)
            self.assertEqual(public.get("status"), "completed", public.get("error_message"))
            refresh = client.get(f"/api/v1/jobs/{job_id}")
            self.assertEqual(refresh.status_code, 200)
            self.assertEqual(refresh.json()["status"], "completed")

        from lumenfin.database import JobRepository

        stored = JobRepository(config.database_url, db_path=config.db_path).get_job(
            job_id, tenant_id="test-tenant"
        )
        self.assertIsNotNone(stored)
        state = (stored or {}).get("result") or {}
        payload = (state.get("retrieved_docs") or {}).get("NVIDIA") or {}
        market = payload.get("market_data") or {}
        value = market.get("operating_income")
        self.assertIsInstance(value, (int, float))
        self.assertAlmostEqual(float(value), GOLD["value_billion_usd"], places=2)
        self.assertNotAlmostEqual(float(value), 72.4, places=1)
        metrics = (state.get("financial_metrics") or {}).get("NVIDIA") or {}
        self.assertAlmostEqual(float(metrics.get("operating_income")), GOLD["value_billion_usd"], places=2)

        self.assertEqual(payload.get("structured_source"), "document_extracted")
        self.assertNotEqual(payload.get("structured_source"), "sample_db")
        meta = payload.get("fundamentals_meta") or {}
        self.assertFalse(meta.get("live_fallback_used"))
        self.assertTrue(meta.get("prefer_uploaded_only") or meta.get("upload_present"))
        body = str(state.get("final_report") or "")
        self.assertTrue(body.strip())
        self.assertRegex(body, r"81\.45")
        self.assertNotRegex(body, r"72\.4")

        state["execution_path"] = "product_workflow"
        finrun = export_finrun_state(state)
        validate_finrun(finrun)
        evidence_ids = [
            row.get("id")
            for row in finrun.get("evidence") or []
            if row.get("metric") == "operating_income"
        ]
        self.assertTrue(evidence_ids)
        oi = next(item for item in finrun["metrics"] if item["name"] == "operating_income")
        self.assertTrue(set(oi.get("evidence_ids") or {}).intersection(evidence_ids))

        clean = visible_supported_claims(finrun, CASE)
        self.assertTrue(clean.passed, [item.message for item in clean.findings])

        number_mut = dict(finrun)
        number_mut["final_output"] = re.sub(
            r"(operating income is )(\d+\.\d+)",
            r"\g<1>99.99",
            str(finrun.get("final_output") or ""),
            count=1,
            flags=re.IGNORECASE,
        )
        self.assertNotEqual(number_mut["final_output"], finrun.get("final_output"))
        self.assertFalse(visible_supported_claims(number_mut, CASE).passed)

        cite = next(
            str(row["citation"])
            for row in finrun["evidence"]
            if row.get("metric") == "operating_income" and row.get("citation")
        )
        swapped = dict(finrun)
        swapped["final_output"] = str(finrun.get("final_output") or "").replace(cite, "forged.md#p99", 1)
        self.assertFalse(visible_supported_claims(swapped, CASE).passed)

    def test_sparse_upload_does_not_fill_sample_operating_income(self) -> None:
        root = ROOT / "test_artifacts" / f"upload-sparse-{uuid4().hex[:8]}"
        config = replace(build_test_config(root), api_key=None, rag_enabled=False)
        app = create_app(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        with TestClient(app) as client:
            submitted = client.post(
                "/api/v1/jobs/upload",
                data={"query": QUERY, "thread_id": "upload-nvda-sparse", "export_artifacts": "false"},
                files={"files": (SPARSE.name, SPARSE.read_bytes(), "text/plain")},
            )
            self.assertEqual(submitted.status_code, 202, submitted.text)
            public = poll_job(client, submitted.json()["job_id"])
        result = public.get("result") or {}
        metrics = (result.get("financial_metrics") or {}).get("NVIDIA") or {}
        self.assertNotAlmostEqual(float(metrics.get("operating_income") or 0), GOLD["value_billion_usd"], places=2)
        report = str(result.get("final_report") or public.get("error_message") or "")
        lowered = report.lower()
        self.assertNotAlmostEqual(float(metrics.get("operating_income") or 0), 72.4, places=1)
        self.assertTrue(
            metrics.get("operating_income") in {None, "", 0, 0.0}
            or "operating_income" not in metrics
            or abs(float(metrics.get("operating_income") or 0)) < 1e-9,
            metrics,
        )
        from lumenfin.database import JobRepository

        stored = JobRepository(config.database_url, db_path=config.db_path).get_job(
            submitted.json()["job_id"], tenant_id="test-tenant"
        )
        nvda = (((stored or {}).get("result") or {}).get("retrieved_docs") or {}).get("NVIDIA") or {}
        self.assertNotEqual(nvda.get("structured_source"), "sample_db")
        self.assertNotIn("operating_income", nvda.get("market_data") or {})
        meta = nvda.get("fundamentals_meta") or {}
        self.assertNotEqual(meta.get("grounding_layer"), "document_ast_complete")
        self.assertTrue(
            meta.get("prefer_uploaded_only")
            or meta.get("grounding_layer") == "prefer_uploaded_only_refused"
            or public.get("fatal_data_gap")
            or result.get("fatal_data_gap"),
            meta,
        )
        gap_codes = {
            public.get("fatal_data_gap"),
            result.get("fatal_data_gap"),
            result.get("incomplete_data"),
            result.get("degraded_mode"),
        }
        self.assertTrue(
            any(gap_codes)
            or "incomplete" in lowered
            or "missing" in lowered
            or "not bound" in lowered
            or "fatal_data_gap" in lowered
            or "prefer_uploaded_only" in lowered,
            report[:800],
        )
        self.assertNotRegex(report, r"72\.4")
        self.assertFalse("81.453" in report and "operating income" in lowered)


class MultipageUploadLoopTestCase(unittest.TestCase):
    """One PDF, two pages, shared document_id — upload → graph → FinRun → v3 scorer."""

    POS = json.loads(
        (ROOT / "tests" / "fixtures" / "sec" / "nvda_multipage_oi_fy2024_gold.json").read_text(
            encoding="utf-8"
        )
    )
    NEG = json.loads(
        (ROOT / "tests" / "fixtures" / "sec" / "nvda_multipage_oi_unspecified_gold.json").read_text(
            encoding="utf-8"
        )
    )
    QUERY = "Using uploaded files only, what is NVIDIA operating income from the filing facts?"

    def _run_upload(self, pdf: Path, thread_id: str) -> tuple[dict, dict]:
        root = ROOT / "test_artifacts" / f"upload-multipage-{uuid4().hex[:8]}"
        config = replace(build_test_config(root), api_key=None, rag_enabled=False)
        app = create_app(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        with TestClient(app) as client:
            submitted = client.post(
                "/api/v1/jobs/upload",
                data={"query": self.QUERY, "thread_id": thread_id, "export_artifacts": "false"},
                files={"files": (pdf.name, pdf.read_bytes(), "application/pdf")},
            )
            self.assertEqual(submitted.status_code, 202, submitted.text)
            public = poll_job(client, submitted.json()["job_id"])
        from lumenfin.database import JobRepository

        stored = JobRepository(config.database_url, db_path=config.db_path).get_job(
            submitted.json()["job_id"], tenant_id="test-tenant"
        )
        self.assertIsNotNone(stored)
        return public, (stored or {}).get("result") or {}

    def test_positive_pdf_hash_and_page_period_match_independent_gold(self) -> None:
        pdf = ROOT / self.POS["source_file"]
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        self.assertEqual(digest, self.POS["source_sha256"])
        import fitz

        opened = fitz.open(pdf)
        self.assertEqual(opened.page_count, 2)
        self.assertIn("FY2025", opened[0].get_text("text"))
        self.assertIn("FY2024", opened[1].get_text("text"))
        self.assertIn("32.972", opened[1].get_text("text"))
        opened.close()

    def test_multipage_upload_binds_fy2024_page2_and_strict_gate(self) -> None:
        from finagentbench.metrics.visible_supported_claims import visible_supported_claims
        from finagentbench.schema import validate_finrun

        pdf = ROOT / self.POS["source_file"]
        public, state = self._run_upload(pdf, "upload-nvda-multipage-pos")
        self.assertEqual(public.get("status"), "completed", public.get("error_message"))
        nvda = (state.get("retrieved_docs") or {}).get("NVIDIA") or {}
        market = nvda.get("market_data") or {}
        self.assertAlmostEqual(float(market["operating_income"]), self.POS["value_billion_usd"], places=3)
        prov = (nvda.get("fundamental_provenance") or {}).get("operating_income") or {}
        self.assertEqual(prov.get("period"), "FY2024")
        self.assertEqual(prov.get("period_alignment"), "exact")
        self.assertIn("#p2", str(prov.get("citation") or ""))
        self.assertIn(":p2:operating_income", str(prov.get("source_record_id") or ""))
        self.assertNotEqual(nvda.get("structured_source"), "sample_db")

        state["execution_path"] = "product_workflow"
        finrun = export_finrun_state(state)
        validate_finrun(finrun)
        oi_ev = next(
            (
                row
                for row in finrun.get("evidence") or []
                if row.get("metric") == "operating_income" and row.get("entity") in {None, "NVIDIA", "nvidia"}
            ),
            None,
        )
        if oi_ev is None:
            oi_ev = next(
                (row for row in finrun.get("evidence") or [] if row.get("metric") == "operating_income"),
                None,
            )
        self.assertIsNotNone(oi_ev, "answerable positive must export operating_income evidence")
        assert oi_ev is not None
        self.assertIn("2024", str(oi_ev.get("period") or ""))
        self.assertNotEqual(str(oi_ev.get("period") or "").upper(), "FY2025")
        cite = str(oi_ev.get("citation") or prov.get("citation") or "")
        self.assertIn("#p2", cite)

        clean = visible_supported_claims(finrun, CASE)
        self.assertTrue(clean.passed, [item.message for item in clean.findings])

        body = str(finrun.get("final_output") or "")
        oi_period = re.compile(
            r"(NVIDIA Operating income is 32\.97\d* billion USD for )FY2024",
            re.I,
        )
        self.assertRegex(body, oi_period, body[-2000:])
        period_lie = dict(finrun)
        period_lie["final_output"] = oi_period.sub(r"\1FY2025", body, count=1)
        period_result = visible_supported_claims(period_lie, CASE)
        self.assertFalse(period_result.passed, [item.message for item in period_result.findings])

        oi_cite = re.compile(
            r"(NVIDIA Operating income is 32\.97\d* billion USD for FY2024(?: \([^)]+\))? \[)([^\]]+#p)2(\])",
            re.I,
        )
        self.assertRegex(body, oi_cite, body[-2000:])
        cite_lie = dict(finrun)
        cite_lie["final_output"] = oi_cite.sub(r"\1\g<2>1\3", body, count=1)
        cite_result = visible_supported_claims(cite_lie, CASE)
        self.assertFalse(cite_result.passed, [item.message for item in cite_result.findings])

    def test_multipage_missing_period_does_not_inherit_cover_fy(self) -> None:
        from finagentbench.metrics.visible_supported_claims import visible_supported_claims

        pdf = ROOT / self.NEG["source_file"]
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        self.assertEqual(digest, self.NEG["source_sha256"])
        public, state = self._run_upload(pdf, "upload-nvda-multipage-neg")
        nvda = (state.get("retrieved_docs") or {}).get("NVIDIA") or {}
        market = nvda.get("market_data") or {}
        self.assertAlmostEqual(float(market.get("operating_income") or 0), 32.972, places=3)
        prov = (nvda.get("fundamental_provenance") or {}).get("operating_income") or {}
        self.assertNotEqual(prov.get("period_alignment"), "exact")
        self.assertNotEqual(prov.get("period"), "FY2025")
        self.assertIn("#p2", str(prov.get("citation") or ""))
        self.assertNotEqual(nvda.get("structured_source"), "sample_db")
        report = str(state.get("final_report") or public.get("error_message") or "")
        lowered = report.lower()
        exec_summary = str(state.get("executive_summary") or "")
        self.assertIn("32.972", exec_summary)
        self.assertRegex(exec_summary.lower(), r"period not stated|not stated next to|not noted in the source|unspecified period")
        self.assertNotRegex(exec_summary, r"Operating income is 32\.97\d* billion USD for FY2025", re.I)
        self.assertRegex(lowered, r"period not stated|not stated next to|not noted in the source|unspecified period")
        self.assertNotRegex(report, r"Operating income is 32\.97\d* billion USD for FY2025", re.I)
        state["execution_path"] = "product_workflow"
        finrun = export_finrun_state(state)
        oi_metric = next(
            (row for row in finrun.get("metrics") or [] if row.get("name") == "operating_income"),
            None,
        )
        self.assertIsNotNone(oi_metric)
        self.assertEqual(str(oi_metric.get("period") or ""), "unknown")
        oi_ev = next(
            (row for row in finrun.get("evidence") or [] if row.get("metric") == "operating_income"),
            None,
        )
        self.assertIsNotNone(oi_ev)
        self.assertEqual(str(oi_ev.get("period") or ""), "unknown")
        self.assertNotIn("latest operating income", str(oi_ev.get("text") or "").lower())
        cite = str(oi_ev.get("citation") or prov.get("citation") or "")
        self.assertIn("#p2", cite)

        clean = visible_supported_claims(finrun, CASE)
        self.assertTrue(clean.passed, [item.message for item in clean.findings])

        year_lie = dict(finrun)
        year_lie["final_output"] = (
            str(finrun.get("final_output") or report)
            + f"\nNVIDIA operating income is 32.972 billion USD for FY2025 [{cite}].\n"
        )
        year_result = visible_supported_claims(year_lie, CASE)
        self.assertFalse(year_result.passed, [item.message for item in year_result.findings])
        self.assertTrue(
            any(isinstance(item.target, dict) and item.target.get("code") == "wrong_period" for item in year_result.findings),
            [item.target for item in year_result.findings],
        )


if __name__ == "__main__":
    unittest.main()
