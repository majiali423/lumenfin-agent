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
from lumenfin.eval.holdout.ledger_parent_return import (
    assert_disjoint_from_prefix,
    build_parent_page_hits,
    select_frozen_slice,
)
from lumenfin.eval.holdout.section_schema import SECTION_METADATA_UNAVAILABLE


def _load_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_parent_pack_suffix.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_parent_pack_suffix",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load suffix probe CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LedgerParentReturnTests(unittest.TestCase):
    def test_suffix_slice_skips_the_frozen_prefix(self) -> None:
        rows = [{"query_id": f"c0-q{index}"} for index in range(50)]
        plans = [{"query_ids": tuple(f"c0-q{index}" for index in range(50))}]
        selected = select_frozen_slice(rows, plans, start=10, count=40)
        self.assertEqual(selected[0]["query_id"], "c0-q10")
        self.assertEqual(selected[-1]["query_id"], "c0-q49")
        self.assertEqual(len(selected), 40)

    def test_overlap_with_prefix_fails_closed(self) -> None:
        with self.assertRaisesRegex(HoldoutError, "overlaps"):
            assert_disjoint_from_prefix(["a", "b"], ["b"])

    def test_parent_hits_use_full_page_and_page_id(self) -> None:
        hits = build_parent_page_hits(
            [
                {"chunk_id": "c0", "document_id": "NYSE_MLR_2017/page_0004"},
                {"chunk_id": "c1", "document_id": "NYSE_MLR_2017/page_0004"},
            ],
            {"NYSE_MLR_2017/page_0004": "Accounts payable 1,223,000"},
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["chunk_id"], "NYSE_MLR_2017/page_0004")
        self.assertEqual(hits[0]["parent_chunk_id"], "NYSE_MLR_2017/page_0004")
        self.assertEqual(hits[0]["section_title"], SECTION_METADATA_UNAVAILABLE)
        self.assertIn("1,223,000", hits[0]["text"])

    def test_missing_parent_page_fails_closed(self) -> None:
        with self.assertRaisesRegex(HoldoutError, "missing"):
            build_parent_page_hits(
                [{"chunk_id": "c0", "document_id": "NYSE_MLR_2017/page_0099"}],
                {"NYSE_MLR_2017/page_0000": "text"},
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
                    "--paired-aggregate",
                    str(Path(tmp) / "qwen3.json"),
                    "--paired-per-case",
                    str(Path(tmp) / "qwen3.jsonl"),
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
