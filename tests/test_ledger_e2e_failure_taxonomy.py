from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout.ledger_e2e_taxonomy import (
    classify_e2e_case,
    gold_number_in_text,
    recommend_next_workstream,
)


def _load_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_e2e_failure_taxonomy.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_e2e_failure_taxonomy",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load taxonomy CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hit(chunk: str, document: str, text: str) -> dict:
    return {
        "chunk_id": chunk,
        "document_id": document,
        "text": text,
    }


class LedgerE2EFailureTaxonomyTests(unittest.TestCase):
    def test_gold_number_accepts_commas_and_million_scale(self) -> None:
        self.assertTrue(gold_number_in_text(26_017_000.0, "AR was 26,017,000."))
        self.assertTrue(gold_number_in_text(21_895_000.0, "Cash of 21.9 million"))

    def test_pool_miss_is_retrieval_leak(self) -> None:
        result = classify_e2e_case(
            pool_hits=[_hit("c1", "doc-other", "Revenue 12")],
            final_identity=[{"chunk_id": "c1", "document_id": "doc-other"}],
            qrels=[{"doc_id": "doc-gold", "relevance": 1}],
            gold_value=100.0,
            numeric_matched=False,
            abstain=True,
        )
        self.assertEqual(result["leak_class"], "retrieval_pool_miss")
        self.assertEqual(result["next_workstream"], "section_parent_retrieval")

    def test_unselected_gold_page_chunk_is_packing_gap(self) -> None:
        result = classify_e2e_case(
            pool_hits=[
                _hit("gold-a", "doc-gold", "Heading only"),
                _hit("gold-b", "doc-gold", "Accounts payable 1,223,000"),
                _hit("other", "doc-other", "Unrelated 9"),
            ],
            final_identity=[
                {"chunk_id": "gold-a", "document_id": "doc-gold"},
                {"chunk_id": "other", "document_id": "doc-other"},
            ],
            qrels=[{"doc_id": "doc-gold", "relevance": 2}],
            gold_value=1_223_000.0,
            numeric_matched=False,
            abstain=True,
        )
        self.assertTrue(result["hit_at_10"])
        self.assertEqual(result["leak_class"], "evidence_gap_unselected_chunk")

    def test_gold_page_without_number_is_absent_evidence(self) -> None:
        result = classify_e2e_case(
            pool_hits=[_hit("gold-a", "doc-gold", "Balance sheet discussion")],
            final_identity=[{"chunk_id": "gold-a", "document_id": "doc-gold"}],
            qrels=[{"doc_id": "doc-gold", "relevance": 1}],
            gold_value=73_273_000.0,
            numeric_matched=False,
            abstain=True,
        )
        self.assertEqual(result["leak_class"], "evidence_gap_number_absent")

    def test_number_in_context_with_abstain_is_generation_leak(self) -> None:
        result = classify_e2e_case(
            pool_hits=[_hit("gold-a", "doc-gold", "Capex was 560,000,000")],
            final_identity=[{"chunk_id": "gold-a", "document_id": "doc-gold"}],
            qrels=[{"doc_id": "doc-gold", "relevance": 1}],
            gold_value=560_000_000.0,
            numeric_matched=False,
            abstain=True,
        )
        self.assertEqual(result["leak_class"], "generation_abstain")
        self.assertEqual(
            result["next_workstream"],
            "generation_prompt_unseen_queries",
        )

    def test_recommend_parent_when_retrieval_dominates(self) -> None:
        self.assertEqual(
            recommend_next_workstream(
                {
                    "retrieval_pool_miss": 16,
                    "evidence_gap_number_absent": 12,
                    "generation_abstain": 5,
                    "ranking_top10_miss": 1,
                }
            ),
            "section_parent_retrieval",
        )

    def test_cli_rejects_allow_remote(self) -> None:
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            code = cli.main(
                [
                    "--parquet-path",
                    str(Path(tmp) / "missing.parquet"),
                    "--manifest",
                    str(Path(tmp) / "manifest.json"),
                    "--split-salt",
                    "salt",
                    "--candidate-dir",
                    tmp,
                    "--e2e-aggregate",
                    str(Path(tmp) / "agg.json"),
                    "--e2e-per-case",
                    str(Path(tmp) / "pc.jsonl"),
                    "--baseline-aggregate",
                    str(Path(tmp) / "base.json"),
                    "--baseline-per-case",
                    str(Path(tmp) / "base.jsonl"),
                    "--prerank-aggregate",
                    str(Path(tmp) / "pre.json"),
                    "--output-dir",
                    str(Path(tmp) / "out"),
                    "--allow-remote",
                ]
            )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
