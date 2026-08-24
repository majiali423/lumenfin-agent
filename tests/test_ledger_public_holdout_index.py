from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout.ledger import PUBLIC_DEV, PUBLIC_HOLDOUT, assign_ledger_company_split
from lumenfin.eval.ledger_public_holdout_index import (
    ALLOWED_COLUMNS,
    FORBIDDEN_COLUMNS,
    PRODUCT_COMMIT,
    PRODUCT_TAG,
    SPLIT_SALT,
    HoldoutIndexError,
    assert_clean_projection,
    assert_rc5_sources,
    budget_ok,
    estimate_from_documents,
    estimate_tokens,
    iter_projected_corpus_rows,
    materialize_documents,
    run_dry_run,
    scan_holdout_reports,
)


def _write_snapshot(path: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    from pyarrow import parquet

    path.mkdir(parents=True, exist_ok=True)
    parquet.write_table(pa.Table.from_pylist(rows), path / "0000.parquet")
    return path


def _row(ticker: str, year: int = 2017, extra_text: str = "") -> dict:
    report_id = f"NYSE_{ticker}_{year}"
    page = "operating income cash flow total assets revenue " + extra_text
    return {
        "query_id": f"{ticker}_secret_query",
        "query_text": f"SECRET QUERY for {ticker}",
        "ticker": ticker,
        "exchange": "NYSE",
        "company_name": f"Company {ticker}",
        "industry": "Industrials",
        "year": year,
        "kpi": "accounts_receivable",
        "value": 999.0,
        "source": "edgar",
        "tag": "secret-gold-tag",
        "qrels": [{"doc_id": f"{report_id}/page_0001", "relevance": 2}],
        "mmd_text": f"cover page\n<--- Page Split --->\n{page}\n<--- Page Split --->\nnotes",
    }


def _tickers_for_both_splits() -> tuple[str, str]:
    by_role = {PUBLIC_DEV: "", PUBLIC_HOLDOUT: ""}
    index = 0
    while not all(by_role.values()):
        ticker = f"H{index:04d}"
        role = assign_ledger_company_split(f"nyse:{ticker}", salt=SPLIT_SALT)
        if not by_role[role]:
            by_role[role] = ticker
        index += 1
    return by_role[PUBLIC_DEV], by_role[PUBLIC_HOLDOUT]


class ProjectionAndDryRunTests(unittest.TestCase):
    def test_forbidden_projection_fails_closed(self) -> None:
        with self.assertRaisesRegex(HoldoutIndexError, "forbidden"):
            assert_clean_projection(("exchange", "query_text"))
        with self.assertRaisesRegex(HoldoutIndexError, "forbidden"):
            assert_clean_projection(("value",))
        with self.assertRaisesRegex(HoldoutIndexError, "forbidden"):
            assert_clean_projection(("qrels",))

    def test_projected_rows_never_include_query_gold_or_qrels(self) -> None:
        dev, holdout = _tickers_for_both_splits()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            snapshot = _write_snapshot(
                Path(tmp) / "snap",
                [_row(dev), _row(holdout), _row(holdout, year=2018)],
            )
            rows = list(iter_projected_corpus_rows(snapshot))
            self.assertGreaterEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(set(row), set(ALLOWED_COLUMNS))
                for field in FORBIDDEN_COLUMNS:
                    self.assertNotIn(field, row)

    def test_holdout_corpus_is_company_split_not_query_selected(self) -> None:
        dev, holdout = _tickers_for_both_splits()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            snapshot = _write_snapshot(
                Path(tmp) / "snap",
                [_row(dev), _row(holdout), _row(holdout, year=2018)],
            )
            reports, audit, rows_seen = scan_holdout_reports(snapshot)
            self.assertEqual(rows_seen, 3)
            self.assertTrue(audit.projection_used)
            self.assertTrue(audit.corpus_document_text_accessed)
            self.assertFalse(audit.holdout_query_text_accessed)
            self.assertFalse(audit.holdout_gold_accessed)
            self.assertFalse(audit.holdout_qrels_accessed)
            documents, stats = materialize_documents(reports)
            self.assertEqual(stats["holdout_companies"], 1)
            self.assertEqual(stats["reports"], 2)
            self.assertGreaterEqual(stats["documents"], 4)
            expected_company = f"nyse:{holdout.casefold()}"
            self.assertTrue(
                all(item["issuer_companies"] == [expected_company] for item in documents)
            )
            estimate = estimate_from_documents(documents)
            self.assertGreater(estimate["chunks"], 0)
            self.assertTrue(budget_ok(estimate))
            dry = run_dry_run(
                repo_root=ROOT,
                snapshot=snapshot,
                official=False,
            )
            self.assertEqual(dry["status"], "BUDGET_OK")
            self.assertIs(dry["model_or_retrieval_tuning"], False)
            dumped = json.dumps(dry)
            self.assertNotIn("SECRET QUERY", dumped)
            self.assertNotIn("secret-gold-tag", dumped)
            self.assertNotIn("999.0", dumped)

    def test_rc5_sources_match_product_commit(self) -> None:
        hashes = assert_rc5_sources(repo_root=ROOT)
        self.assertEqual(len(hashes), 4)
        self.assertEqual(PRODUCT_TAG, "v0.1.0-rc.5")
        self.assertEqual(PRODUCT_COMMIT, "31e8680aa89636f1fd897d7aa5ed7ca86317bd73")

    def test_token_bounds_are_conservative(self) -> None:
        english = estimate_tokens("abcd" * 10)
        self.assertEqual(english["typical_qwen_style"], 10)
        self.assertEqual(english["conservative_mixed"], 20)
        self.assertEqual(english["hard_upper_char"], 40)
        chinese = estimate_tokens("收入利润")
        self.assertEqual(chinese["conservative_mixed"], 4)

    def test_blocked_index_seal_does_not_claim_a_built_index(self) -> None:
        payload = json.loads(
            (ROOT / "data" / "eval_rag" / "ledger_public_holdout_index_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["status"], "BLOCKED_AFTER_DRYRUN")
        self.assertIs(payload["holdout_consumed"], False)
        self.assertIs(payload["official_index_built"], False)
        self.assertIs(payload["phase_b_started"], False)
        v1 = json.loads(
            (ROOT / "data" / "eval_rag" / "ledger_public_holdout_e2e_result.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(v1["seal_status"], "BLOCKED_BEFORE_PREFLIGHT")
        self.assertIs(v1["holdout_consumed"], False)

    def test_cli_help_does_not_open_holdout(self) -> None:
        cli = ROOT / "scripts" / "run_ledger_public_holdout_index.py"
        completed = subprocess.run(
            [sys.executable, str(cli), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertNotIn("query_text", completed.stdout)
        self.assertNotIn("SECRET QUERY", completed.stdout)


if __name__ == "__main__":
    unittest.main()
