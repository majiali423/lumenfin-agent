"""Offline product workflow: graph → visible report → FinRun → product-quality gate."""

from __future__ import annotations

import copy
import json
import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

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

from lumenfin.finrun import export_finrun_state
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.service import LumenFinAnalysisService
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


GOLD = json.loads(
    (ROOT / "tests" / "fixtures" / "product_quality_gold_v1.json").read_text(encoding="utf-8")
)


class ProductQualityLoopTestCase(unittest.TestCase):
    def test_offline_graph_report_matches_independent_gold_and_blocks_prose_lie(self) -> None:
        from finagentbench.metrics.visible_supported_claims import visible_supported_claims
        from finagentbench.schema import validate_finrun

        root = ROOT / "test_artifacts" / f"product-quality-{uuid4().hex[:8]}"
        config = replace(build_test_config(root), rag_enabled=False)
        service = LumenFinAnalysisService(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        payload = service.analyze(
            query="Analyze Apple FY2025 EBITDA margin using available fundamentals.",
            thread_id="product-quality-offline",
            export_artifacts=False,
        )
        state = payload["result"]
        state["execution_path"] = "product_workflow"
        metric = (state.get("financial_metrics") or {}).get("Apple") or {}
        value = metric.get("ebitda_margin")
        expected = GOLD["expected_metric"]
        self.assertIsInstance(value, (int, float))
        self.assertGreaterEqual(float(value), expected["min"])
        self.assertLessEqual(float(value), expected["max"])

        finrun = export_finrun_state(state)
        validate_finrun(finrun)
        body = str(finrun.get("final_output") or "")
        self.assertTrue(body.strip())
        cite = next(
            str(row["citation"])
            for row in finrun["evidence"]
            if row.get("metric") in {"ebitda", "revenue"} and row.get("citation")
        )
        self.assertIn(cite, body)
        self.assertIn("EBITDA margin is", body)
        self.assertTrue(
            any(row.get("id") and row.get("metric") and row.get("source_record_id") for row in finrun.get("evidence") or []),
            "source fields must keep original record ids",
        )
        margin = next(item for item in finrun["metrics"] if item["name"] == "ebitda_margin")
        self.assertTrue(margin.get("evidence_ids"))
        revenue_ev = next(item for item in finrun["evidence"] if item.get("metric") == "revenue")
        self.assertIn(revenue_ev["id"], margin["evidence_ids"])
        self.assertTrue(str(revenue_ev.get("source_record_id") or "").startswith("sample_catalog:"))
        self.assertNotEqual(str(revenue_ev.get("source_record_id")), str(revenue_ev.get("value")))

        case = {
            "expected_entities": GOLD["expected_entities"],
            "required_steps": [],
            "enabled_metrics": ["visible_supported_claims"],
            "scoring_version": "3",
            "execution_path": "product_workflow",
            "require_checkable_metrics": True,
            "require_visible_claim_citations": True,
            "numeric_tolerance": 0.001,
            "block_on_severity": ["high", "critical"],
        }
        clean = visible_supported_claims(finrun, case)
        self.assertTrue(clean.passed, [item.message for item in clean.findings])

        number_mut = dict(finrun)
        number_body = re.sub(
            r"(EBITDA margin is )(\d+\.\d+)%",
            r"\g<1>99.99%",
            body,
            count=1,
            flags=re.IGNORECASE,
        )
        self.assertNotEqual(number_body, body)
        number_mut["final_output"] = number_body
        number_result = visible_supported_claims(number_mut, case)
        self.assertFalse(number_result.passed)
        self.assertTrue(any(item.target.get("code") == "wrong_number" for item in number_result.findings))

        swapped = dict(finrun)
        swapped["final_output"] = body.replace(cite, "forged.md#p99", 1)
        swapped_result = visible_supported_claims(swapped, case)
        self.assertFalse(swapped_result.passed)
        self.assertTrue(any(item.target.get("code") == "wrong_citation" for item in swapped_result.findings))

        deleted = dict(finrun)
        deleted["final_output"] = body.replace(f"[{cite}]", "", 1)
        self.assertFalse(visible_supported_claims(deleted, case).passed)

        evidence_mut = copy.deepcopy(finrun)
        for row in evidence_mut["evidence"]:
            if row.get("metric") == "revenue":
                row["currency"] = "EUR"
                break
        self.assertFalse(visible_supported_claims(evidence_mut, case).passed)

        pct = float(value) * 100.0
        table = (
            "\n| Company | FY2025 EBITDA margin |\n"
            "|---|---|\n"
            f"| Apple | {pct:.2f}% [{cite}] |\n"
        )
        with_table = dict(finrun)
        with_table["final_output"] = body + table
        self.assertTrue(visible_supported_claims(with_table, case).passed, "correct table must not fail the original report")
        wrong_table = dict(finrun)
        wrong_table["final_output"] = body + table.replace(f"{pct:.2f}%", "99.9%")
        table_result = visible_supported_claims(wrong_table, case)
        self.assertFalse(table_result.passed)
        self.assertTrue(
            any(item.target.get("code") == "wrong_number" and item.target.get("origin") == "table" for item in table_result.findings),
            [item.target for item in table_result.findings],
        )

        ledger_header = "| Entity | Type | Statement | Source |"
        if ledger_header in body:
            prefix, tail = body.split(ledger_header, 1)
            mutated_tail = re.sub(
                r"(Apple EBITDA margin is )(\d+\.\d+)%",
                r"\g<1>99.99%",
                tail,
                count=1,
                flags=re.IGNORECASE,
            )
            ledger_wrong = prefix + ledger_header + mutated_tail
            if "99.99%" not in mutated_tail:
                ledger_wrong = body + (
                    f"\n| Apple | Claim | Apple EBITDA margin is 99.99% for FY2025. | [{cite}] |\n"
                )
        else:
            ledger_wrong = body + (
                f"\n{ledger_header}\n|---|---|---|---|\n"
                f"| Apple | Claim | Apple EBITDA margin is 99.99% for FY2025. | [{cite}] |\n"
            )
        ledger_mut = dict(finrun)
        ledger_mut["final_output"] = ledger_wrong
        ledger_result = visible_supported_claims(ledger_mut, case)
        self.assertFalse(ledger_result.passed)
        self.assertTrue(
            any(item.target.get("origin") == "table" for item in ledger_result.findings),
            [item.target for item in ledger_result.findings],
        )
        shown = None
        match = re.search(r"EBITDA margin is (\d+\.\d+)%", body, flags=re.IGNORECASE)
        if match:
            shown = match.group(1)
        ledger_ok = dict(finrun)
        if shown:
            ledger_ok["final_output"] = body + (
                f"\n{ledger_header}\n|---|---|---|---|\n"
                f"| Apple | Claim | Apple EBITDA margin is {shown}% for FY2025. | [{cite}] |\n"
            )
            self.assertTrue(
                visible_supported_claims(ledger_ok, case).passed,
                "correct graph ledger must pass",
            )


if __name__ == "__main__":
    unittest.main()
