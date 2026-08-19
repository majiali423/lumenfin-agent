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

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_parent_probe import (
    attach_page_parent,
    neighbor_page_ids,
    pack_pages,
    recoverability,
)
from lumenfin.eval.holdout.section_schema import SECTION_METADATA_UNAVAILABLE


def _load_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_parent_pack_probe.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_parent_pack_probe",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load parent probe CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LedgerParentPackProbeTests(unittest.TestCase):
    def test_page_parent_does_not_infer_section_title(self) -> None:
        attached = attach_page_parent(
            {
                "chunk_id": "c0",
                "document_id": "AMEX_BRN_2017/page_0003",
                "text": "Item 7 Management Discussion 1,223,000",
            }
        )
        self.assertEqual(
            attached["parent_chunk_id"],
            "AMEX_BRN_2017/page_0003",
        )
        self.assertEqual(attached["section_title"], SECTION_METADATA_UNAVAILABLE)

    def test_neighbor_window_skips_negative_pages(self) -> None:
        self.assertEqual(
            neighbor_page_ids("AMEX_BRN_2017/page_0000", radius=1),
            ["AMEX_BRN_2017/page_0000", "AMEX_BRN_2017/page_0001"],
        )

    def test_retrieved_page_full_recovers_number_dropped_by_chunks(self) -> None:
        pages = {
            "NYSE_MLR_2017/page_0004": "Accounts payable were 1,223,000.",
            "NYSE_MLR_2017/page_0005": "Unrelated narrative.",
        }
        packed = pack_pages(
            ["NYSE_MLR_2017/page_0004"],
            pages,
            radius=0,
        )
        self.assertIn("1,223,000", packed)
        recovered = recoverability(
            gold_value=1_223_000.0,
            chunk_final_text="Heading only",
            retrieved_page_ids=["NYSE_MLR_2017/page_0004"],
            gold_page_ids=["NYSE_MLR_2017/page_0004"],
            page_by_id=pages,
        )
        self.assertFalse(recovered["recovered"]["chunk_final"])
        self.assertTrue(recovered["recovered"]["retrieved_page_full"])
        self.assertTrue(recovered["recovered"]["gold_page_full"])

    def test_missing_gold_page_fails_closed(self) -> None:
        with self.assertRaisesRegex(HoldoutError, "missing"):
            recoverability(
                gold_value=10.0,
                chunk_final_text="",
                retrieved_page_ids=["AMEX_BRN_2017/page_0001"],
                gold_page_ids=["AMEX_BRN_2017/page_0099"],
                page_by_id={"AMEX_BRN_2017/page_0001": "text"},
            )

    def test_recommend_parent_page_return_when_packing_recovers(self) -> None:
        cli = _load_cli()
        rows = [
            {
                "leak_class": "evidence_gap_number_absent",
                "recovered": {"retrieved_page_full": True, "gold_page_full": True},
            }
        ] * 10 + [
            {
                "leak_class": "retrieval_pool_miss",
                "recovered": {"retrieved_page_full": False, "gold_page_full": False},
            }
        ] * 5
        self.assertEqual(
            cli.recommend_next(rows),
            "retrieve_child_return_parent_page",
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
                    str(Path(tmp) / "e2e.json"),
                    "--e2e-per-case",
                    str(Path(tmp) / "e2e.jsonl"),
                    "--taxonomy-aggregate",
                    str(Path(tmp) / "tax.json"),
                    "--taxonomy-per-case",
                    str(Path(tmp) / "tax.jsonl"),
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
