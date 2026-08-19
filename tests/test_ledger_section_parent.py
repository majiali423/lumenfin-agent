from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_section_parent import (
    parent_page_index_unit,
    pool_hit,
    recommend_next,
    select_company_pages,
)
from lumenfin.eval.holdout.section_schema import SECTION_METADATA_UNAVAILABLE


def _load_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_section_parent.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_section_parent",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load section-parent CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sealer():
    path = ROOT / "scripts" / "seal_ledger_public_dev_section_parent.py"
    spec = importlib.util.spec_from_file_location(
        "seal_ledger_public_dev_section_parent",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load section-parent sealer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LedgerSectionParentTests(unittest.TestCase):
    def test_page_unit_does_not_infer_section_title(self) -> None:
        unit = parent_page_index_unit(
            {
                "document_id": "NYSE_MLR_2017/page_0004",
                "filename": "NYSE_MLR_2017.mmd",
                "pages": ["Item 7 Management Discussion 1,223,000"],
                "issuer_companies": ["nyse:mlr"],
                "ledger_page_zero": 4,
            }
        )
        self.assertEqual(unit["chunk_id"], "NYSE_MLR_2017/page_0004")
        self.assertEqual(unit["parent_chunk_id"], "NYSE_MLR_2017/page_0004")
        self.assertEqual(unit["section_title"], SECTION_METADATA_UNAVAILABLE)
        self.assertEqual(unit["chunk_type"], "eval_parent_page")
        self.assertIn("1,223,000", unit["text"])

    def test_multi_page_document_fails_closed(self) -> None:
        with self.assertRaisesRegex(HoldoutError, "exactly one page"):
            parent_page_index_unit(
                {
                    "document_id": "NYSE_MLR_2017/page_0004",
                    "filename": "NYSE_MLR_2017.mmd",
                    "pages": ["a", "b"],
                    "issuer_companies": ["nyse:mlr"],
                    "ledger_page_zero": 4,
                }
            )

    def test_selects_only_requested_companies(self) -> None:
        selected = select_company_pages(
            [
                {
                    "document_id": "NYSE_MLR_2017/page_0000",
                    "issuer_companies": ["nyse:mlr"],
                    "pages": ["one"],
                },
                {
                    "document_id": "AMEX_BRN_2017/page_0000",
                    "issuer_companies": ["amex:brn"],
                    "pages": ["two"],
                },
            ],
            ["nyse:mlr"],
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["document_id"], "NYSE_MLR_2017/page_0000")

    def test_pool_hit_uses_page_document_id(self) -> None:
        self.assertTrue(
            pool_hit(
                [{"document_id": "NYSE_MLR_2017/page_0004"}],
                [{"doc_id": "NYSE_MLR_2017/page_0004", "relevance": 1}],
            )
        )
        self.assertFalse(
            pool_hit(
                [{"document_id": "NYSE_MLR_2017/page_0005"}],
                [{"doc_id": "NYSE_MLR_2017/page_0004", "relevance": 1}],
            )
        )

    def test_no_lift_refuses_hybrid_embeddings(self) -> None:
        self.assertEqual(
            recommend_next(hybrid_pool_hits=148, parent_pool_hits=140, cases=200),
            "do_not_embed_page_parent_index",
        )
        self.assertEqual(
            recommend_next(hybrid_pool_hits=148, parent_pool_hits=160, cases=200),
            "hybrid_page_parent_index",
        )

    def test_sealer_rejects_non_tracked_output(self) -> None:
        sealer = _load_sealer()
        with tempfile.TemporaryDirectory() as tmp:
            code = sealer.main(
                [
                    "--aggregate",
                    str(Path(tmp) / "aggregate.json"),
                    "--per-case",
                    str(Path(tmp) / "per_case.jsonl"),
                    "--output",
                    str(Path(tmp) / "wrong.json"),
                ]
            )
        self.assertEqual(code, 2)

    def test_sealed_suffix_refuses_page_embeddings(self) -> None:
        path = (
            ROOT
            / "data"
            / "eval_rag"
            / "holdout"
            / "ledger_public_dev_section_parent_bm25_5x40.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["cases"], 200)
        self.assertEqual(payload["pool_hit_at_20"]["hybrid_chunk"], 148)
        self.assertEqual(payload["pool_hit_at_20"]["parent_page_bm25"], 103)
        self.assertEqual(
            payload["recommended_next_workstream"],
            "do_not_embed_page_parent_index",
        )
        self.assertEqual(payload["remote_calls"], 0)
        self.assertFalse(payload["product_accuracy_claim"])
        self.assertEqual(payload["financebench_phase4"], "NOT_RUN")

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
                    "--paired-aggregate",
                    str(Path(tmp) / "qwen3.json"),
                    "--paired-per-case",
                    str(Path(tmp) / "qwen3.jsonl"),
                    "--taxonomy-aggregate",
                    str(Path(tmp) / "tax.json"),
                    "--taxonomy-per-case",
                    str(Path(tmp) / "tax.jsonl"),
                    "--suffix-aggregate",
                    str(Path(tmp) / "suffix.json"),
                    "--e2e-aggregate",
                    str(Path(tmp) / "e2e.json"),
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
