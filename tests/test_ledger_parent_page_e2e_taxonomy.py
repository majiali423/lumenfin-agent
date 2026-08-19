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

from lumenfin.eval.holdout.ledger_e2e_taxonomy import classify_parent_page_generate_case
from lumenfin.eval.holdout.ledger_parent_return import (
    build_parent_page_hits,
    parent_prompt_char_cap,
)


def _load_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_parent_page_e2e_taxonomy.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_parent_page_e2e_taxonomy",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load parent-page taxonomy CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hit(chunk: str, document: str, text: str) -> dict:
    return {
        "chunk_id": chunk,
        "document_id": document,
        "text": text,
    }


class LedgerParentPageE2ETaxonomyTests(unittest.TestCase):
    def test_full_page_recovers_unselected_chunk(self) -> None:
        parent_hits = build_parent_page_hits(
            [
                {"chunk_id": "gold-a", "document_id": "NYSE_MLR_2017/page_0004"},
                {"chunk_id": "other", "document_id": "NYSE_MLR_2017/page_0005"},
            ],
            {
                "NYSE_MLR_2017/page_0004": "Accounts payable 1,223,000",
                "NYSE_MLR_2017/page_0005": "Unrelated narrative",
            },
        )
        result = classify_parent_page_generate_case(
            pool_hits=[
                _hit("gold-a", "NYSE_MLR_2017/page_0004", "Heading only"),
                _hit("gold-b", "NYSE_MLR_2017/page_0004", "Accounts payable 1,223,000"),
                _hit("other", "NYSE_MLR_2017/page_0005", "Unrelated 9"),
            ],
            final_identity=[
                {"chunk_id": "gold-a", "document_id": "NYSE_MLR_2017/page_0004"},
                {"chunk_id": "other", "document_id": "NYSE_MLR_2017/page_0005"},
            ],
            parent_hits=parent_hits,
            qrels=[{"doc_id": "NYSE_MLR_2017/page_0004", "relevance": 2}],
            gold_value=1_223_000.0,
            numeric_matched=False,
            abstain=True,
            parent_max_document_chars=parent_prompt_char_cap(parent_hits),
        )
        self.assertTrue(result["hit_at_10"])
        self.assertTrue(result["number_in_final_context"])
        self.assertEqual(result["leak_class"], "generation_abstain")

    def test_retrieved_page_without_digits_is_absent(self) -> None:
        parent_hits = build_parent_page_hits(
            [{"chunk_id": "gold-a", "document_id": "NYSE_MLR_2017/page_0004"}],
            {"NYSE_MLR_2017/page_0004": "Balance sheet discussion"},
        )
        result = classify_parent_page_generate_case(
            pool_hits=[
                _hit("gold-a", "NYSE_MLR_2017/page_0004", "Balance sheet discussion")
            ],
            final_identity=[
                {"chunk_id": "gold-a", "document_id": "NYSE_MLR_2017/page_0004"}
            ],
            parent_hits=parent_hits,
            qrels=[{"doc_id": "NYSE_MLR_2017/page_0004", "relevance": 1}],
            gold_value=73_273_000.0,
            numeric_matched=False,
            abstain=True,
            parent_max_document_chars=parent_prompt_char_cap(parent_hits),
        )
        self.assertEqual(result["leak_class"], "evidence_gap_number_absent")

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
                    "--parent-e2e-aggregate",
                    str(Path(tmp) / "parent.json"),
                    "--parent-e2e-per-case",
                    str(Path(tmp) / "parent.jsonl"),
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
