from __future__ import annotations

import contextlib
import hashlib
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
    build_ledger_public_dev_dataset,
    build_ledger_split_manifest,
    iter_ledger_parquet_rows,
    ledger_public_dev_qrel_audit,
    ledger_snapshot_sha256,
)

REVISION = "a" * 40
ARTIFACT_SHA256 = "b" * 64
SPLIT_SALT = "lumenfin-ledger-public-v1"
PUBLISHED_MANIFEST = (
    ROOT / "data" / "eval_rag" / "holdout" / "ledger_public_manifest.json"
)
PUBLISHED_BM25_BASELINE = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_bm25_baseline.json"
)


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


def _load_score_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_ranking.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_ranking_cli", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load LEDGER scoring CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sharded_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_bm25_sharded.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_bm25_sharded_cli", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load LEDGER sharded scoring CLI")
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
        "mmd_text": (
            "cover\n<--- Page Split --->\ncontents\n<--- Page Split --->\n"
            "accounts receivable financial table\n<--- Page Split --->\nnotes\n"
            "<--- Page Split --->\n"
            "appendix\n<--- Page Split --->\nnegative candidate"
        ),
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
        self.assertEqual(adapted["page_count"], 6)
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

    def test_same_company_cross_year_qrel_matches_official_benchmark(self) -> None:
        row = _row()
        row["qrels"][0]["doc_id"] = "NYSE_SHW_2018/page_0002"
        adapted = adapt_ledger_row(row)
        self.assertEqual(
            adapted["qrels"][0]["doc_id"],
            "NYSE_SHW_2018/page_0002",
        )

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
        self.assertNotIn("public_dev_offline_bm25_preflight_enabled", manifest)
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


class LedgerPublicDevCorpusTests(unittest.TestCase):
    def test_selects_only_dev_and_preserves_zero_based_page_ids(self) -> None:
        dev_row, _dev_two, holdout_row, _holdout_two = _rows_covering_both_splits()
        dev_row["mmd_text"] = (
            "cover\n<--- Page Split --->\n   \n<--- Page Split --->\n"
            "financial table\n<--- Page Split --->\nnotes\n<--- Page Split --->\n"
            "appendix\n<--- Page Split --->\nnegative candidate"
        )
        dev_row["qrels"].append(
            {
                "doc_id": f"NYSE_{dev_row['ticker']}_{dev_row['year']}/page_0001",
                "relevance": 0,
            }
        )
        dataset = build_ledger_public_dev_dataset(
            (row for row in (dev_row, holdout_row)),
            salt=SPLIT_SALT,
        )
        self.assertEqual(len(dataset.queries), 1)
        self.assertEqual(dataset.queries[0]["query_id"], dev_row["query_id"])
        page_ids = {
            document["ledger_doc_id"] for document in dataset.page_documents
        }
        report_id = f"NYSE_{dev_row['ticker']}_{dev_row['year']}"
        self.assertIn(f"{report_id}/page_0002", page_ids)
        self.assertNotIn(f"{report_id}/page_0001", page_ids)
        self.assertEqual(dataset.ignored_zero_qrels, 1)
        self.assertTrue(
            all(
                document["issuer_companies"]
                == [dataset.queries[0]["company_key"]]
                for document in dataset.page_documents
            )
        )

    def test_repeated_report_content_mismatch_fails_closed(self) -> None:
        dev_row = _rows_covering_both_splits()[0]
        repeated = dict(dev_row)
        repeated["query_id"] = "different-query"
        repeated["mmd_text"] += "\nchanged"
        with self.assertRaisesRegex(HoldoutError, "inconsistent"):
            build_ledger_public_dev_dataset(
                [dev_row, repeated],
                salt=SPLIT_SALT,
            )

    def test_missing_qrel_page_fails_closed(self) -> None:
        dev_row = _rows_covering_both_splits()[0]
        dev_row["qrels"] = [
            {
                "doc_id": f"NYSE_{dev_row['ticker']}_{dev_row['year']}/page_9999",
                "relevance": 2,
            }
        ]
        with self.assertRaisesRegex(HoldoutError, "invalid corpus pages"):
            build_ledger_public_dev_dataset([dev_row], salt=SPLIT_SALT)

    def test_missing_zero_relevance_page_must_be_a_real_blank(self) -> None:
        dev_row = _rows_covering_both_splits()[0]
        dev_row["qrels"][1] = {
            "doc_id": f"NYSE_{dev_row['ticker']}_{dev_row['year']}/page_9999",
            "relevance": 0,
        }
        with self.assertRaisesRegex(HoldoutError, "invalid_zero_qrels=1"):
            build_ledger_public_dev_dataset([dev_row], salt=SPLIT_SALT)

    def test_query_with_only_blank_positive_qrel_is_excluded(self) -> None:
        dev_row = _rows_covering_both_splits()[0]
        report_id = f"NYSE_{dev_row['ticker']}_{dev_row['year']}"
        dev_row["mmd_text"] = "cover\n<--- Page Split --->\n   "
        dev_row["qrels"] = [
            {"doc_id": f"{report_id}/page_0001", "relevance": 2}
        ]
        with self.assertRaisesRegex(HoldoutError, "no queries with reachable"):
            build_ledger_public_dev_dataset([dev_row], salt=SPLIT_SALT)

    def test_frozen_manifest_identity_mismatch_fails_closed(self) -> None:
        dev_row = _rows_covering_both_splits()[0]
        manifest = {
            "splits": {
                PUBLIC_DEV: {
                    "queries": 999,
                    "companies": 1,
                    "query_ids_sha256": "bad",
                    "company_keys_sha256": "bad",
                    "local_tuning_allowed": True,
                }
            }
        }
        with self.assertRaisesRegex(HoldoutError, "frozen manifest"):
            build_ledger_public_dev_dataset(
                [dev_row],
                salt=SPLIT_SALT,
                manifest=manifest,
            )


class LedgerParquetAndCliTests(unittest.TestCase):
    def _write_parquet(self, path: Path) -> None:
        import pyarrow as pa
        from pyarrow import parquet

        table = pa.Table.from_pylist(_rows_covering_both_splits())
        parquet.write_table(table, path)

    def test_local_parquet_iteration_and_snapshot_hash(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "ledger.parquet"
            self._write_parquet(path)
            rows = list(iter_ledger_parquet_rows(path))
            self.assertEqual(len(rows), 4)
            self.assertRegex(ledger_snapshot_sha256(path), r"^[0-9a-f]{64}$")
            self.assertEqual(
                ledger_snapshot_sha256(path),
                ledger_snapshot_sha256(path),
            )

    def test_preflight_selection_round_robins_companies(self) -> None:
        cli = _load_score_cli()
        dataset = cli.LedgerPublicDevDataset(
            queries=(
                {"query_id": "a1", "company_key": "a", "qrels": ()},
                {"query_id": "a2", "company_key": "a", "qrels": ()},
                {"query_id": "b1", "company_key": "b", "qrels": ()},
            ),
            page_documents=(
                {
                    "ledger_doc_id": "A/page_0000",
                    "filename": "A.mmd",
                    "issuer_companies": ["a"],
                },
                {
                    "ledger_doc_id": "B/page_0000",
                    "filename": "B.mmd",
                    "issuer_companies": ["b"],
                },
            ),
            companies=("a", "b"),
            reports=2,
        )
        selected = cli._subset_for_cases(dataset, 2)
        self.assertEqual(
            [query["query_id"] for query in selected.queries],
            ["a1", "b1"],
        )
        self.assertEqual(selected.companies, ("a", "b"))
        company_selected = cli._subset_for_company(
            dataset,
            company_key="A",
            max_cases=2,
        )
        self.assertEqual(
            [query["query_id"] for query in company_selected.queries],
            ["a1", "a2"],
        )
        self.assertEqual(company_selected.companies, ("a",))
        with self.assertRaisesRegex(HoldoutError, "company-sharded"):
            cli._subset_for_cases(dataset, 11)
        shard_zero = cli._subset_for_company_shard(
            dataset,
            shard_index=0,
            shard_count=2,
        )
        shard_one = cli._subset_for_company_shard(
            dataset,
            shard_index=1,
            shard_count=2,
        )
        self.assertEqual(shard_zero.companies, ("a",))
        self.assertEqual(shard_one.companies, ("b",))

    def test_hybrid_retrieval_uses_shared_top20_and_counts_query_call(self) -> None:
        cli = _load_score_cli()

        class Embedder:
            last_physical_calls = 1

        class Store:
            embedder = Embedder()
            last_query_embed_physical_calls = 1
            last_query_embed_cache_hit = False

            def __init__(self) -> None:
                self.calls: list[tuple[str, int, list[str]]] = []

            def bm25_search(self, _query, *, companies, top_k, **_kwargs):
                self.calls.append(("bm25", top_k, companies))
                return [
                    {"chunk_id": "bm25", "document_id": "doc-bm25", "score": 1.0},
                    {"chunk_id": "shared", "document_id": "doc-shared", "score": 0.9},
                ]

            def vector_search(self, _query, *, companies, top_k, **_kwargs):
                self.calls.append(("dense", top_k, companies))
                return [
                    {"chunk_id": "shared", "document_id": "doc-shared", "score": 1.0},
                    {"chunk_id": "dense", "document_id": "doc-dense", "score": 0.9},
                ]

        store = Store()
        hits, meta = cli._retrieve_candidates(
            store,
            query="question",
            company="nyse_shw",
            top_k=20,
            session_id="session",
            mode="hybrid",
        )
        self.assertEqual(
            store.calls,
            [
                ("bm25", 20, ["nyse_shw"]),
                ("dense", 20, ["nyse_shw"]),
            ],
        )
        self.assertEqual(len(hits), 3)
        self.assertEqual(meta["mode"], "hybrid_dense_bm25_rrf")
        self.assertEqual(meta["remote_calls"], 1)

        class CachedStore(Store):
            last_query_embed_physical_calls = 0
            last_query_embed_cache_hit = True

        _cached_hits, cached_meta = cli._retrieve_candidates(
            CachedStore(),
            query="question",
            company="nyse_shw",
            top_k=20,
            session_id="session",
            mode="hybrid",
        )
        self.assertEqual(cached_meta["remote_calls"], 0)
        self.assertTrue(cached_meta["query_embedding_cache_hit"])

    def test_hybrid_retrieval_fails_closed_without_dense_or_accounting(self) -> None:
        cli = _load_score_cli()

        class Store:
            class Embedder:
                last_physical_calls = 0

            embedder = Embedder()

            def bm25_search(self, *_args, **_kwargs):
                return [{"chunk_id": "bm25", "document_id": "doc", "score": 1.0}]

            def vector_search(self, *_args, **_kwargs):
                return []

        with self.assertRaisesRegex(HoldoutError, "physical-call accounting"):
            cli._retrieve_candidates(
                Store(),
                query="question",
                company="nyse_shw",
                top_k=20,
                session_id="session",
                mode="hybrid",
            )

        class EmptyDenseStore(Store):
            last_query_embed_physical_calls = 1

        with self.assertRaisesRegex(HoldoutError, "non-empty Dense and BM25"):
            cli._retrieve_candidates(
                EmptyDenseStore(),
                query="question",
                company="nyse_shw",
                top_k=20,
                session_id="session",
                mode="hybrid",
            )

        class FailedStore(Store):
            def vector_search(self, *_args, **_kwargs):
                raise RuntimeError("dense failed")

        with self.assertRaisesRegex(RuntimeError, "dense failed"):
            cli._retrieve_candidates(
                FailedStore(),
                query="question",
                company="nyse_shw",
                top_k=20,
                session_id="session",
                mode="hybrid",
            )

    def test_hybrid_cli_requires_explicit_remote_gate_and_one_case(self) -> None:
        cli = _load_score_cli()
        required = [
            "--parquet-path",
            "missing.parquet",
            "--manifest",
            "missing.json",
            "--split-salt",
            SPLIT_SALT,
            "--output-dir",
            "missing-output",
            "--mode",
            "hybrid",
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(required), 2)
            self.assertEqual(
                cli.main(
                    required
                    + [
                        "--embedding-provider",
                        "dashscope",
                        "--allow-remote",
                        "--max-cases",
                        "2",
                    ]
                ),
                2,
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
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = cli.main(args)
            self.assertEqual(status, 0, msg=output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args), 2)
                self.assertEqual(cli.main(args + ["--allow-remote"]), 2)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["scoring_enabled"])
            self.assertEqual(payload["remote_calls"], 0)

    def test_public_dev_bm25_preflight_is_local_and_isolated(self) -> None:
        cli = _load_score_cli()
        rows = _rows_covering_both_splits()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            parquet_path = root / "ledger.parquet"
            manifest_path = root / "manifest.json"
            output_dir = root / "run"
            import pyarrow as pa
            from pyarrow import parquet

            parquet.write_table(pa.Table.from_pylist(rows), parquet_path)
            manifest = build_ledger_split_manifest(
                rows,
                source_revision=REVISION,
                source_artifact_sha256=ledger_snapshot_sha256(parquet_path),
                salt=SPLIT_SALT,
            )
            audit_dataset = build_ledger_public_dev_dataset(
                rows,
                salt=SPLIT_SALT,
            )
            manifest["public_dev_corpus_audit"] = ledger_public_dev_qrel_audit(
                audit_dataset,
                source_queries=manifest["splits"][PUBLIC_DEV]["queries"],
            )
            manifest["public_dev_offline_bm25_preflight_enabled"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            fresh_cli = _load_score_cli()
            with self.assertRaisesRegex(HoldoutError, "tracked frozen"):
                fresh_cli._load_manifest(manifest_path)
            cli.TRACKED_MANIFEST_PATH = manifest_path.resolve()
            cli.PINNED_MANIFEST_CANONICAL_SHA256 = hashlib.sha256(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            pinned_hash = cli.PINNED_MANIFEST_CANONICAL_SHA256
            cli.PINNED_MANIFEST_CANONICAL_SHA256 = "0" * 64
            with self.assertRaisesRegex(HoldoutError, "identity has changed"):
                cli._load_manifest(manifest_path)
            cli.PINNED_MANIFEST_CANONICAL_SHA256 = pinned_hash
            args = [
                "--parquet-path",
                str(parquet_path),
                "--manifest",
                str(manifest_path),
                "--split-salt",
                SPLIT_SALT,
                "--output-dir",
                str(output_dir),
                "--max-cases",
                "1",
                "--batch-size",
                "2",
            ]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = cli.main(args)
            self.assertEqual(status, 0, msg=output.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args), 2)
                self.assertEqual(cli.main(args + ["--allow-remote"]), 2)
            aggregate = json.loads(
                (output_dir / "aggregate.json").read_text(encoding="utf-8")
            )
            self.assertEqual(aggregate["cases"], 1)
            self.assertEqual(aggregate["remote_calls"], 0)
            self.assertEqual(aggregate["qwen3_calls"], 0)
            self.assertEqual(aggregate["index"]["retrieval_mode"], "bm25")
            self.assertTrue((output_dir / "per_case.jsonl").is_file())

            failed_output = root / "failed-run"
            original_index_documents = cli._index_documents
            cli._index_documents = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                HoldoutError("synthetic index failure")
            )
            try:
                failed_args = list(args)
                failed_args[failed_args.index(str(output_dir))] = str(failed_output)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(failed_args), 2)
            finally:
                cli._index_documents = original_index_documents
            if failed_output.exists():
                self.assertTrue((failed_output / ".incomplete").is_file())
                self.assertFalse((failed_output / "aggregate.json").exists())
            retry_output = io.StringIO()
            with contextlib.redirect_stdout(retry_output):
                retry_status = cli.main(failed_args)
            self.assertEqual(retry_status, 0, msg=retry_output.getvalue())
            self.assertTrue((failed_output / "aggregate.json").is_file())


class LedgerShardedAggregationTests(unittest.TestCase):
    def _case_metric(self, query_id: str, hit: bool) -> dict:
        return {
            "case_id": query_id,
            "pool_size": 20,
            "final_size": 10,
            "gold_page_count": 1,
            "pool_hit": hit,
            "hit_at_5": float(hit),
            "hit_at_10": float(hit),
            "mrr": 1.0 if hit else 0.0,
            "ndcg_at_10": 1.0 if hit else 0.0,
            "unique_pages_top10": 10,
            "page_identity_coverage_top10": 1.0,
            "duplicate_page_occupancy_top10": 0.0,
            "failure_class": "hit_at_10" if hit else "gold_not_in_rerank_pool",
        }

    def _write_shard(self, root: Path, index: int, query_id: str) -> Path:
        shard = root / f"{index:03d}"
        shard.mkdir(parents=True)
        metric = self._case_metric(query_id, hit=index == 0)
        row = {
            "query_id": query_id,
            "arms": {"A_prod": metric, "R_page": dict(metric)},
        }
        per_case_text = json.dumps(row) + "\n"
        (shard / "per_case.jsonl").write_text(
            per_case_text,
            encoding="utf-8",
        )
        run_config = {"schema_version": "test"}
        run_config_sha256 = hashlib.sha256(
            json.dumps(
                run_config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        aggregate = {
            "dataset_snapshot_sha256": "snapshot",
            "source_manifest_sha256": "manifest",
            "run_config": run_config,
            "run_config_sha256": run_config_sha256,
            "per_case_sha256": hashlib.sha256(
                (shard / "per_case.jsonl").read_bytes()
            ).hexdigest(),
            "selection": {
                "strategy": "company_modulo_shard_v1",
                "company_shard_index": index,
                "company_shard_count": 2,
                "selected_cases": 1,
                "selected_companies": 1,
                "query_ids_sha256": hashlib.sha256(
                    query_id.encode("utf-8")
                ).hexdigest(),
                "company_keys_sha256": hashlib.sha256(
                    f"c{index}".encode()
                ).hexdigest(),
            },
            "call_accounting": {
                "retrieval_calls": 1,
                "retrieval_remote_calls": 0,
                "rerank_calls": 0,
                "rerank_attempts": 0,
                "rerank_fallbacks": 0,
                "remote_calls": 0,
            },
            "index": {
                "documents_indexed": 2,
                "chunks_indexed": 3,
                "embed_calls": 1,
                "companies": 1,
                "reports": 1,
            },
            "qrel_corpus_audit": {
                "scorable_queries": 2,
                "scorable_query_ids_sha256": hashlib.sha256(
                    b"q1\nq2"
                ).hexdigest(),
            },
            "qwen3_calls": 0,
        }
        (shard / "aggregate.json").write_text(
            json.dumps(aggregate),
            encoding="utf-8",
        )
        return shard

    def test_shard_aggregation_is_complete_and_remote_free(self) -> None:
        cli = _load_sharded_cli()
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            output = Path(tmp)
            shards_root = output / "shards"
            shards = [
                self._write_shard(shards_root, 0, "q1"),
                self._write_shard(shards_root, 1, "q2"),
            ]
            for index, shard in enumerate(shards):
                aggregate = json.loads(
                    (shard / "aggregate.json").read_text(encoding="utf-8")
                )
                cli._validate_completed_shard(
                    aggregate,
                    shard_index=index,
                    shard_count=2,
                    dataset_snapshot_sha256="snapshot",
                    expected_selection={
                        "selected_cases": 1,
                        "selected_companies": 1,
                        "query_ids_sha256": hashlib.sha256(
                            f"q{index + 1}".encode()
                        ).hexdigest(),
                        "company_keys_sha256": hashlib.sha256(
                            f"c{index}".encode()
                        ).hexdigest(),
                    },
                    run_config_sha256=aggregate["run_config_sha256"],
                    source_manifest_sha256="manifest",
                    expected_qrel_audit=aggregate["qrel_corpus_audit"],
                    per_case_sha256=aggregate["per_case_sha256"],
                )
            manifest = {
                "dataset_snapshot_sha256": "snapshot",
                "public_dev_corpus_audit": {
                    "scorable_queries": 2,
                    "scorable_query_ids_sha256": hashlib.sha256(
                        b"q1\nq2"
                    ).hexdigest(),
                },
                "splits": {PUBLIC_DEV: {"companies": 2}},
            }
            aggregate = cli._aggregate_shards(
                shards,
                manifest=manifest,
                output_dir=output,
            )
            self.assertEqual(aggregate["cases"], 2)
            self.assertEqual(aggregate["call_accounting"]["retrieval_calls"], 2)
            self.assertEqual(aggregate["call_accounting"]["remote_calls"], 0)
            self.assertEqual(aggregate["index"]["shards"], 2)
            self.assertEqual(aggregate["arms"]["A_prod"]["page_hit_at_10"], 0.5)
            self.assertEqual(
                aggregate["per_case_sha256"],
                hashlib.sha256((output / "per_case.jsonl").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                len((output / "per_case.jsonl").read_text().splitlines()),
                2,
            )

    def test_completed_shard_with_remote_calls_fails_closed(self) -> None:
        cli = _load_sharded_cli()
        aggregate = {
            "dataset_snapshot_sha256": "snapshot",
            "source_manifest_sha256": "manifest",
            "run_config": {"schema_version": "test"},
            "selection": {
                "strategy": "company_modulo_shard_v1",
                "company_shard_index": 0,
                "company_shard_count": 1,
            },
            "call_accounting": {
                "retrieval_calls": 1,
                "retrieval_remote_calls": 1,
                "remote_calls": 1,
            },
            "qrel_corpus_audit": {"scorable_queries": 1},
            "per_case_sha256": "per-case",
            "qwen3_calls": 0,
        }
        aggregate["run_config_sha256"] = cli.shard_cli._config_sha256(
            aggregate["run_config"]
        )
        with self.assertRaisesRegex(HoldoutError, "remote"):
            cli._validate_completed_shard(
                aggregate,
                shard_index=0,
                shard_count=1,
                dataset_snapshot_sha256="snapshot",
                expected_selection={},
                run_config_sha256=aggregate["run_config_sha256"],
                source_manifest_sha256="manifest",
                expected_qrel_audit={"scorable_queries": 1},
                per_case_sha256="per-case",
            )

    def test_shard_per_case_identity_cannot_be_replaced(self) -> None:
        cli = _load_sharded_cli()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            shard = self._write_shard(output / "shards", 0, "q1")
            row = json.loads(
                (shard / "per_case.jsonl").read_text(encoding="utf-8")
            )
            row["query_id"] = "unknown"
            for arm in row["arms"].values():
                arm["case_id"] = "unknown"
            altered_text = json.dumps(row) + "\n"
            (shard / "per_case.jsonl").write_text(
                altered_text,
                encoding="utf-8",
            )
            shard_aggregate = json.loads(
                (shard / "aggregate.json").read_text(encoding="utf-8")
            )
            shard_aggregate["per_case_sha256"] = hashlib.sha256(
                (shard / "per_case.jsonl").read_bytes()
            ).hexdigest()
            (shard / "aggregate.json").write_text(
                json.dumps(shard_aggregate),
                encoding="utf-8",
            )
            manifest = {
                "dataset_snapshot_sha256": "snapshot",
                "public_dev_corpus_audit": {
                    "scorable_queries": 1,
                    "scorable_query_ids_sha256": hashlib.sha256(
                        b"q1"
                    ).hexdigest(),
                },
                "splits": {PUBLIC_DEV: {"companies": 1}},
            }
            with self.assertRaisesRegex(HoldoutError, "per-case query identity"):
                cli._aggregate_shards(
                    [shard],
                    manifest=manifest,
                    output_dir=output,
                )

    def test_shard_metric_corruption_breaks_artifact_hash(self) -> None:
        cli = _load_sharded_cli()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            shard = self._write_shard(output / "shards", 0, "q1")
            row = json.loads(
                (shard / "per_case.jsonl").read_text(encoding="utf-8")
            )
            row["arms"]["A_prod"]["hit_at_10"] = 0.25
            (shard / "per_case.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )
            manifest = {
                "dataset_snapshot_sha256": "snapshot",
                "public_dev_corpus_audit": {
                    "scorable_queries": 1,
                    "scorable_query_ids_sha256": hashlib.sha256(
                        b"q1"
                    ).hexdigest(),
                },
                "splits": {PUBLIC_DEV: {"companies": 1}},
            }
            with self.assertRaisesRegex(HoldoutError, "per-case hash"):
                cli._aggregate_shards(
                    [shard],
                    manifest=manifest,
                    output_dir=output,
                )

    def test_shard_missing_accounting_field_fails_closed(self) -> None:
        cli = _load_sharded_cli()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            shard = self._write_shard(output / "shards", 0, "q1")
            aggregate = json.loads(
                (shard / "aggregate.json").read_text(encoding="utf-8")
            )
            aggregate["call_accounting"].pop("rerank_attempts")
            (shard / "aggregate.json").write_text(
                json.dumps(aggregate),
                encoding="utf-8",
            )
            manifest = {
                "dataset_snapshot_sha256": "snapshot",
                "public_dev_corpus_audit": {
                    "scorable_queries": 1,
                    "scorable_query_ids_sha256": hashlib.sha256(
                        b"q1"
                    ).hexdigest(),
                },
                "splits": {PUBLIC_DEV: {"companies": 1}},
            }
            with self.assertRaisesRegex(
                HoldoutError,
                "rerank_attempts is invalid",
            ):
                cli._aggregate_shards(
                    [shard],
                    manifest=manifest,
                    output_dir=output,
                )


class LedgerPublishedManifestTests(unittest.TestCase):
    def test_manifest_locks_public_company_disjoint_snapshot(self) -> None:
        payload = json.loads(PUBLISHED_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["dataset"]["source_revision"],
            "b7085dc6cb16b3ec8149a9baf6dd2d3416cf7619",
        )
        self.assertEqual(
            payload["dataset"]["source_artifact_sha256"],
            "405eb7c805db90258e4246651688b8d8bef89c77d4a4ce2cbcbf9e5fa4bfe9ad",
        )
        self.assertEqual(payload["rows"], 10000)
        self.assertEqual(payload["splits"][PUBLIC_DEV]["queries"], 7616)
        self.assertEqual(payload["splits"][PUBLIC_DEV]["companies"], 85)
        self.assertEqual(payload["splits"][PUBLIC_HOLDOUT]["queries"], 2384)
        self.assertEqual(payload["splits"][PUBLIC_HOLDOUT]["companies"], 26)
        self.assertFalse(payload["scoring_enabled"])
        self.assertTrue(payload["public_dev_offline_bm25_preflight_enabled"])
        self.assertEqual(
            payload["public_dev_corpus_audit"]["scorable_queries"],
            7615,
        )
        self.assertEqual(
            payload["public_dev_corpus_audit"]["scorable_query_ids_sha256"],
            "3dcfdab4b585834634c89d0b0d7a590862cdee19007960ca69366d4106c4ac75",
        )
        self.assertEqual(
            payload["public_dev_corpus_audit"][
                "excluded_queries_without_reachable_positive"
            ],
            1,
        )
        self.assertEqual(payload["remote_calls"], 0)

    def test_manifest_contains_identity_only(self) -> None:
        payload = json.loads(PUBLISHED_MANIFEST.read_text(encoding="utf-8"))
        serialized = json.dumps(payload)
        self.assertEqual(payload["foundation_model_training_exposure"], "unknown")
        self.assertFalse(payload["product_accuracy_claim"])
        self.assertNotIn("query_text", serialized)
        self.assertNotIn("mmd_text", serialized)
        self.assertNotIn("page_000", serialized)

    def test_sealed_bm25_baseline_is_complete_and_remote_free(self) -> None:
        payload = json.loads(PUBLISHED_BM25_BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(payload["cases"], 7615)
        self.assertEqual(payload["selection"]["selected_companies"], 85)
        self.assertEqual(payload["index"]["shards"], 17)
        self.assertFalse(payload["index"]["indexes_retained"])
        self.assertEqual(payload["call_accounting"]["retrieval_calls"], 7615)
        self.assertEqual(payload["call_accounting"]["remote_calls"], 0)
        self.assertEqual(payload["qwen3_calls"], 0)
        self.assertFalse(payload["primary_comparison_valid"])
        self.assertRegex(payload["per_case_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(payload["arms"]["A_prod"]["page_hit_at_10"], 0.3124)
        self.assertEqual(payload["arms"]["R_page"]["page_hit_at_10"], 0.3308)

    def test_sealed_bm25_baseline_contains_no_raw_corpus_content(self) -> None:
        payload = json.loads(PUBLISHED_BM25_BASELINE.read_text(encoding="utf-8"))
        serialized = json.dumps(payload)
        self.assertNotIn("query_text", serialized)
        self.assertNotIn("mmd_text", serialized)
        self.assertNotIn('"per_case":', serialized)


if __name__ == "__main__":
    unittest.main()
