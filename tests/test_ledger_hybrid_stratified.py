from __future__ import annotations

import hashlib
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

SEALED_RESULT = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_hybrid_stratified_5x50.json"
)


def _load_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_hybrid_stratified.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_hybrid_stratified",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stratified Hybrid CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric(query_id: str, *, hit: bool) -> dict:
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


def _row(query_id: str, *, hit: bool) -> dict:
    metric = _metric(query_id, hit=hit)
    return {
        "query_id": query_id,
        "arms": {"A_prod": metric, "R_page": dict(metric)},
    }


def _plan(cli, company: str, query_id: str) -> dict:
    return {
        "company_key": company,
        "company_key_sha256": cli._ids_sha256((company,)),
        "selected_cases": 1,
        "query_ids": (query_id,),
        "query_ids_sha256": cli._ids_sha256((query_id,)),
        "expected_query_http_calls": 1,
        "documents": 2,
        "reports": 1,
        "chunks": 3,
        "embed_chars": 100,
        "estimated_document_http_calls": 1,
    }


class LedgerHybridStratifiedTests(unittest.TestCase):
    def test_company_selection_evenly_spans_frozen_order(self) -> None:
        cli = _load_cli()
        companies = tuple(f"c{index:02d}" for index in range(85))
        self.assertEqual(
            cli.select_stratified_company_keys(companies, count=5),
            ("c00", "c21", "c42", "c63", "c84"),
        )
        with self.assertRaisesRegex(HoldoutError, "company-count"):
            cli.select_stratified_company_keys(companies, count=1)
        with self.assertRaisesRegex(HoldoutError, "company-count"):
            cli.select_stratified_company_keys(companies, count=6)

    def test_completed_company_validation_checks_identity_and_calls(self) -> None:
        cli = _load_cli()
        plan = _plan(cli, "c00", "q1")
        run_config = {"mode": "hybrid"}
        run_config_sha256 = cli.child_cli._config_sha256(run_config)
        audit = {"scorable_queries": 2}
        manifest = {"dataset_snapshot_sha256": "snapshot"}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            per_case = json.dumps(_row("q1", hit=True)) + "\n"
            (run_dir / "per_case.jsonl").write_text(
                per_case,
                encoding="utf-8",
            )
            aggregate = {
                "dataset_snapshot_sha256": "snapshot",
                "source_manifest_sha256": "manifest",
                "run_config": run_config,
                "run_config_sha256": run_config_sha256,
                "per_case_sha256": hashlib.sha256(
                    (run_dir / "per_case.jsonl").read_bytes()
                ).hexdigest(),
                "qrel_corpus_audit": audit,
                "selection": {
                    "strategy": "single_company_prefix_v1",
                    "selected_cases": 1,
                    "selected_companies": 1,
                    "query_ids_sha256": plan["query_ids_sha256"],
                    "company_keys_sha256": plan["company_key_sha256"],
                },
                "call_accounting": {
                    "retrieval_calls": 1,
                    "retrieval_remote_calls": 1,
                    "query_embedding_remote_calls": 1,
                    "document_embedding_remote_calls": 2,
                    "remote_calls": 3,
                    "rerank_calls": 0,
                    "rerank_attempts": 0,
                    "rerank_fallbacks": 0,
                },
                "remote_calls": 3,
                "qwen3_calls": 0,
                "index": {
                    "documents_indexed": 2,
                    "documents_in_scoped_corpus": 2,
                    "chunks_indexed": 3,
                    "embed_chars": 100,
                    "embed_physical_calls": 2,
                    "estimated_dashscope_http_calls": 1,
                    "reports": 1,
                    "companies": 1,
                },
            }
            (run_dir / "aggregate.json").write_text(
                json.dumps(aggregate),
                encoding="utf-8",
            )
            validated = cli._validate_company_run(
                run_dir,
                plan=plan,
                manifest=manifest,
                expected_run_config_sha256=run_config_sha256,
                source_manifest_sha256="manifest",
                expected_qrel_audit=audit,
            )
            self.assertEqual(validated["remote_calls"], 3)

            aggregate["call_accounting"]["remote_calls"] = 2
            (run_dir / "aggregate.json").write_text(
                json.dumps(aggregate),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(HoldoutError, "accounting mismatch"):
                cli._validate_company_run(
                    run_dir,
                    plan=plan,
                    manifest=manifest,
                    expected_run_config_sha256=run_config_sha256,
                    source_manifest_sha256="manifest",
                    expected_qrel_audit=audit,
                )

    def test_sealed_baseline_validation_is_strict_and_checks_arm_ids(self) -> None:
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            per_case_path = root / "baseline.jsonl"
            rows = [_row("q1", hit=True), _row("q2", hit=False)]
            per_case_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            run_config = {
                "embedding_provider": "deterministic",
                "retrieval_mode": "bm25",
                "evaluator_source_sha256": cli.child_cli._evaluator_source_sha256(),
            }
            qrel_audit = {"scorable_queries": 2}
            aggregate = {
                "cases": 2,
                "dataset_snapshot_sha256": "snapshot",
                "source_manifest_sha256": "manifest",
                "qrel_corpus_audit": qrel_audit,
                "remote_calls": 0,
                "qwen3_calls": 0,
                "primary_comparison_valid": False,
                "selection": {"selected_cases": 2, "selected_companies": 2},
                "index": {
                    "embedding_provider": "deterministic",
                    "retrieval_mode": "bm25",
                },
                "run_config": run_config,
                "run_config_sha256": cli.child_cli._config_sha256(run_config),
                "call_accounting": {
                    "retrieval_calls": 2,
                    "retrieval_remote_calls": 0,
                    "rerank_calls": 0,
                    "rerank_attempts": 0,
                    "rerank_fallbacks": 0,
                    "remote_calls": 0,
                },
                "per_case_sha256": hashlib.sha256(
                    per_case_path.read_bytes()
                ).hexdigest(),
            }
            aggregate_path = root / "baseline.json"
            aggregate_path.write_text(
                json.dumps(aggregate),
                encoding="utf-8",
            )
            old_path = cli.TRACKED_BASELINE_AGGREGATE
            old_hash = cli.PINNED_BASELINE_CANONICAL_SHA256
            old_cases = cli.EXPECTED_BASELINE_CASES
            cli.TRACKED_BASELINE_AGGREGATE = aggregate_path.resolve()
            cli.PINNED_BASELINE_CANONICAL_SHA256 = cli._config_sha256(
                aggregate
            )
            cli.EXPECTED_BASELINE_CASES = 2
            try:
                validated, validated_rows, _baseline_hash = (
                    cli._validate_sealed_baseline(
                    aggregate_path=aggregate_path,
                    per_case_path=per_case_path,
                    manifest={
                        "dataset_snapshot_sha256": "snapshot",
                        "splits": {"public_dev": {"companies": 2}},
                    },
                    source_manifest_sha256="manifest",
                    expected_qrel_audit=qrel_audit,
                    )
                )
                self.assertEqual(validated["cases"], 2)
                self.assertEqual(len(validated_rows), 2)
            finally:
                cli.TRACKED_BASELINE_AGGREGATE = old_path
                cli.PINNED_BASELINE_CANONICAL_SHA256 = old_hash
                cli.EXPECTED_BASELINE_CASES = old_cases

            rows[0]["arms"]["A_prod"]["case_id"] = "q2"
            per_case_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(HoldoutError, "arm identity"):
                cli._load_per_case(per_case_path)

    def test_aggregate_pairs_exact_baseline_queries_and_hashes_output(self) -> None:
        cli = _load_cli()
        plans = [_plan(cli, "c00", "q1"), _plan(cli, "c84", "q2")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dirs: list[Path] = []
            hybrid_rows = [_row("q1", hit=True), _row("q2", hit=True)]
            for index, row in enumerate(hybrid_rows):
                run_dir = root / f"run-{index}"
                run_dir.mkdir()
                (run_dir / "per_case.jsonl").write_text(
                    json.dumps(row) + "\n",
                    encoding="utf-8",
                )
                (run_dir / "aggregate.json").write_text(
                    json.dumps(
                        {
                            "run_config": {"mode": "hybrid"},
                            "run_config_sha256": "hybrid-config",
                            "call_accounting": {
                                "retrieval_calls": 1,
                                "retrieval_remote_calls": 1,
                                "rerank_calls": 0,
                                "rerank_attempts": 0,
                                "rerank_fallbacks": 0,
                                "document_embedding_remote_calls": 2,
                                "query_embedding_remote_calls": 1,
                                "remote_calls": 3,
                            },
                            "index": {
                                "documents_indexed": 2,
                                "chunks_indexed": 3,
                                "embed_calls": 1,
                                "embed_physical_calls": 2,
                                "embed_chars": 100,
                                "companies": 1,
                                "reports": 1,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                run_dirs.append(run_dir)

            baseline_rows = [_row("q1", hit=False), _row("q2", hit=True)]
            baseline_per_case = root / "baseline.jsonl"
            baseline_per_case.write_text(
                "".join(json.dumps(row) + "\n" for row in baseline_rows),
                encoding="utf-8",
            )
            baseline_aggregate = root / "baseline.json"
            baseline_aggregate.write_text(
                json.dumps(
                    {
                        "dataset_snapshot_sha256": "snapshot",
                        "per_case_sha256": hashlib.sha256(
                            baseline_per_case.read_bytes()
                        ).hexdigest(),
                        "run_config_sha256": "bm25-config",
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "output"
            output_dir.mkdir()
            aggregate = cli._aggregate(
                run_dirs,
                plans=plans,
                output_dir=output_dir,
                manifest={"dataset_snapshot_sha256": "snapshot"},
                baseline_aggregate=json.loads(
                    baseline_aggregate.read_text(encoding="utf-8")
                ),
                baseline_rows=baseline_rows,
                baseline_per_case_sha256=hashlib.sha256(
                    baseline_per_case.read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(aggregate["cases"], 2)
            self.assertEqual(aggregate["remote_calls"], 6)
            self.assertEqual(
                aggregate["comparison"]["paired_counts"]["A_prod"]["hit_at_10"],
                {"gain": 1, "loss": 0, "unchanged": 1},
            )
            self.assertEqual(
                aggregate["comparison"]["delta_hybrid_minus_bm25"]["A_prod"][
                    "page_hit_at_10"
                ],
                0.5,
            )
            self.assertEqual(
                aggregate["per_case_sha256"],
                hashlib.sha256(
                    (output_dir / "per_case.jsonl").read_bytes()
                ).hexdigest(),
            )

    def test_sealed_result_is_complete_redacted_and_source_bound(self) -> None:
        payload = json.loads(SEALED_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(payload["cases"], 250)
        self.assertEqual(payload["selection"]["companies"], 5)
        self.assertEqual(payload["selection"]["chunks"], 15829)
        self.assertEqual(payload["remote_calls"], 1851)
        self.assertEqual(payload["qwen3_calls"], 0)
        self.assertFalse(payload["primary_comparison_valid"])
        self.assertEqual(
            payload["comparison"]["delta_hybrid_minus_bm25"]["A_prod"][
                "pool_hit_rate"
            ],
            0.18,
        )
        self.assertEqual(
            payload["comparison"]["paired_counts"]["R_page"]["hit_at_10"],
            {"gain": 41, "loss": 4, "unchanged": 205},
        )
        cli = _load_cli()
        source_hash = hashlib.sha256(
            (
                ROOT / "scripts" / "run_ledger_public_dev_hybrid_stratified.py"
            )
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .encode()
        ).hexdigest()
        self.assertEqual(payload["orchestrator_source_sha256"], source_hash)
        self.assertEqual(
            payload["child_run_config"]["evaluator_source_sha256"],
            cli.child_cli._evaluator_source_sha256(),
        )
        self.assertRegex(payload["per_case_sha256"], r"^[0-9a-f]{64}$")
        serialized = json.dumps(payload)
        self.assertNotIn("query_text", serialized)
        self.assertNotIn("mmd_text", serialized)
        self.assertNotIn('"per_case":', serialized)


if __name__ == "__main__":
    unittest.main()
