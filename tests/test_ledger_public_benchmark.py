from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout import (
    PUBLIC_DEV,
    PUBLIC_HOLDOUT,
    HoldoutError,
    adapt_ledger_row,
    assign_ledger_company_split,
    build_ledger_split_manifest,
    iter_ledger_parquet_rows,
    ledger_snapshot_sha256,
)

REVISION = "a" * 40
ARTIFACT_SHA256 = "b" * 64
SPLIT_SALT = "lumenfin-ledger-public-v1"


def _load_cli():
    path = ROOT / "scripts" / "validate_ledger_public_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "validate_ledger_public_benchmark_cli", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load LEDGER validation CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    query_id: str = "SHW_accounts_receivable_2017",
    ticker: str = "SHW",
    year: int = 2017,
) -> dict:
    report_id = f"NYSE_{ticker}_{year}"
    return {
        "query_id": query_id,
        "query_text": f"What is accounts receivable for {ticker} in {year}?",
        "ticker": ticker,
        "exchange": "NYSE",
        "company_name": f"Company {ticker}",
        "industry": "Industrials",
        "year": year,
        "kpi": "accounts_receivable",
        "value": 123.0,
        "source": "edgar",
        "tag": "AccountsReceivableNetCurrent",
        "qrels": [
            {"doc_id": f"{report_id}/page_0002", "relevance": 2},
            {"doc_id": f"{report_id}/page_0005", "relevance": 0},
        ],
        "mmd_text": "cover\n<--- Page Split --->\nfinancial table",
    }


def _rows_covering_both_splits() -> list[dict]:
    by_role: dict[str, list[str]] = {PUBLIC_DEV: [], PUBLIC_HOLDOUT: []}
    index = 0
    while not all(by_role.values()):
        ticker = f"T{index:04d}"
        role = assign_ledger_company_split(
            f"nyse:{ticker}",
            salt=SPLIT_SALT,
        )
        by_role[role].append(ticker)
        index += 1
    dev = by_role[PUBLIC_DEV][0]
    holdout = by_role[PUBLIC_HOLDOUT][0]
    return [
        _row(query_id=f"{dev}_q1", ticker=dev),
        _row(query_id=f"{dev}_q2", ticker=dev, year=2018),
        _row(query_id=f"{holdout}_q1", ticker=holdout),
        _row(query_id=f"{holdout}_q2", ticker=holdout, year=2018),
    ]


class LedgerAdapterTests(unittest.TestCase):
    def test_official_fields_and_qrel_doc_ids_are_preserved(self) -> None:
        adapted = adapt_ledger_row(_row())
        self.assertEqual(adapted["query_id"], "SHW_accounts_receivable_2017")
        self.assertEqual(adapted["company_key"], "nyse:shw")
        self.assertEqual(adapted["positive_qrels"], 1)
        self.assertEqual(adapted["primary_qrels"], 1)
        self.assertEqual(adapted["page_count"], 2)
        self.assertEqual(
            adapted["qrels"][0]["doc_id"],
            "NYSE_SHW_2017/page_0002",
        )

    def test_invalid_qrels_fail_closed(self) -> None:
        cases = (
            [],
            [{"doc_id": "report/page_0001", "relevance": 0}],
            [{"doc_id": "report/page_0001", "relevance": 3}],
            [
                {"doc_id": "report/page_0001", "relevance": 2},
                {"doc_id": "report/page_0001", "relevance": 1},
            ],
        )
        for qrels in cases:
            with self.subTest(qrels=qrels):
                row = _row()
                row["qrels"] = qrels
                with self.assertRaises(HoldoutError):
                    adapt_ledger_row(row)

    def test_cross_company_qrel_fails_closed(self) -> None:
        row = _row()
        row["qrels"][0]["doc_id"] = "NYSE_OTHER_2017/page_0002"
        with self.assertRaisesRegex(HoldoutError, "exchange\\+ticker"):
            adapt_ledger_row(row)

    def test_missing_page_delimiters_fail_closed(self) -> None:
        row = _row()
        row["mmd_text"] = "one unaligned document"
        with self.assertRaisesRegex(HoldoutError, "page delimiters"):
            adapt_ledger_row(row)

    def test_empty_optional_tag_matches_official_schema(self) -> None:
        row = _row()
        row["tag"] = ""
        self.assertEqual(adapt_ledger_row(row)["tag"], "")


class LedgerSplitManifestTests(unittest.TestCase):
    def test_split_is_deterministic_and_company_disjoint(self) -> None:
        rows = _rows_covering_both_splits()
        first = build_ledger_split_manifest(
            rows,
            source_revision=REVISION,
            source_artifact_sha256=ARTIFACT_SHA256,
            salt=SPLIT_SALT,
        )
        second = build_ledger_split_manifest(
            list(reversed(rows)),
            source_revision=REVISION,
            source_artifact_sha256=ARTIFACT_SHA256,
            salt=SPLIT_SALT,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["rows"], 4)
        self.assertEqual(first["splits"][PUBLIC_DEV]["companies"], 1)
        self.assertEqual(first["splits"][PUBLIC_HOLDOUT]["companies"], 1)
        self.assertTrue(first["splits"][PUBLIC_DEV]["local_tuning_allowed"])
        self.assertFalse(
            first["splits"][PUBLIC_HOLDOUT]["local_tuning_allowed"]
        )
        self.assertTrue(
            first["splits"][PUBLIC_HOLDOUT]["held_out_from_local_tuning"]
        )

    def test_manifest_is_public_not_a_model_unseen_claim(self) -> None:
        manifest = build_ledger_split_manifest(
            _rows_covering_both_splits(),
            source_revision=REVISION,
            source_artifact_sha256=ARTIFACT_SHA256,
            salt=SPLIT_SALT,
        )
        serialized = json.dumps(manifest)
        self.assertTrue(manifest["public_benchmark"])
        self.assertEqual(
            manifest["foundation_model_training_exposure"], "unknown"
        )
        self.assertEqual(
            manifest["held_out_claim"], "public_company_disjoint_only"
        )
        self.assertFalse(manifest["product_accuracy_claim"])
        self.assertFalse(manifest["scoring_enabled"])
        self.assertEqual(manifest["remote_calls"], 0)
        self.assertNotIn("What is accounts receivable", serialized)
        self.assertNotIn("page_0002", serialized)
        self.assertNotIn("<--- Page Split --->", serialized)

    def test_bad_identity_and_duplicate_queries_fail_closed(self) -> None:
        rows = _rows_covering_both_splits()
        with self.assertRaisesRegex(HoldoutError, "40-character"):
            build_ledger_split_manifest(
                rows,
                source_revision="main",
                source_artifact_sha256=ARTIFACT_SHA256,
                salt=SPLIT_SALT,
            )
        with self.assertRaisesRegex(HoldoutError, "SHA256"):
            build_ledger_split_manifest(
                rows,
                source_revision=REVISION,
                source_artifact_sha256="bad",
                salt=SPLIT_SALT,
            )
        duplicate = list(rows) + [dict(rows[0])]
        with self.assertRaisesRegex(HoldoutError, "duplicate query_id"):
            build_ledger_split_manifest(
                duplicate,
                source_revision=REVISION,
                source_artifact_sha256=ARTIFACT_SHA256,
                salt=SPLIT_SALT,
            )


class LedgerParquetAndCliTests(unittest.TestCase):
    def _write_parquet(self, path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as parquet

        table = pa.Table.from_pylist(_rows_covering_both_splits())
        parquet.write_table(table, path)

    def test_local_parquet_iteration_and_snapshot_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.parquet"
            self._write_parquet(path)
            rows = list(iter_ledger_parquet_rows(path))
            self.assertEqual(len(rows), 4)
            self.assertRegex(ledger_snapshot_sha256(path), r"^[0-9a-f]{64}$")
            self.assertEqual(
                ledger_snapshot_sha256(path),
                ledger_snapshot_sha256(path),
            )

    def test_cli_writes_once_and_refuses_remote_or_overwrite(self) -> None:
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet_path = root / "ledger.parquet"
            manifest_path = root / "manifest.json"
            self._write_parquet(parquet_path)
            args = [
                "--parquet-path",
                str(parquet_path),
                "--source-revision",
                REVISION,
                "--split-salt",
                SPLIT_SALT,
                "--output-manifest",
                str(manifest_path),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args), 0)
                self.assertEqual(cli.main(args), 2)
                self.assertEqual(cli.main(args + ["--allow-remote"]), 2)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["scoring_enabled"])
            self.assertEqual(payload["remote_calls"], 0)


if __name__ == "__main__":
    unittest.main()
