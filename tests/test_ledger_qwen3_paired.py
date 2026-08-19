from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout import HoldoutError, LedgerPublicDevDataset

SEALED_RESULT = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_qwen3_paired_5x50.json"
)


def _load_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_qwen3_paired.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_qwen3_paired",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load paired Qwen3 CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hit(index: int, *, page: int | None = None) -> dict:
    number = index if page is None else page
    return {
        "chunk_id": f"chunk-{index}",
        "document_id": f"doc-{number}",
        "text": f"financial passage {index}",
        "companies": ["company-a"],
        "page": number,
        "retrieval_method": "hybrid_dense_bm25_rrf",
        "fusion_score": 1.0 / (index + 1),
    }


def _candidate(cli, *, query_id: str = "q1") -> dict:
    return cli._candidate_row(
        query={
            "query_id": query_id,
            "query_text": "What was revenue?",
            "company_key": "company-a",
        },
        hits=[_hit(index, page=(index + 1) // 2) for index in range(1, 21)],
    )


def _metric(query_id: str) -> dict:
    return {
        "case_id": query_id,
        "pool_size": 20,
        "final_size": 10,
        "gold_page_count": 1,
        "pool_hit": True,
        "hit_at_5": 1.0,
        "hit_at_10": 1.0,
        "mrr": 1.0,
        "ndcg_at_10": 1.0,
        "unique_pages_top10": 10,
        "page_identity_coverage_top10": 1.0,
        "duplicate_page_occupancy_top10": 0.0,
        "failure_class": "hit_at_10",
    }


class _FakeReranker:
    def __init__(self, *, fallback_arm: int = -1) -> None:
        self.calls = 0
        self.fallback_arm = fallback_arm

    def rerank(self, query: str, hits: list[dict], *, top_k: int):
        self.calls += 1
        return list(reversed(hits))[:top_k], {
            "rerank_attempts": 1,
            "rerank_tokens": 123,
            "rerank_fallback": self.calls == self.fallback_arm,
            "rerank_error_type": (
                "transient_provider_error"
                if self.calls == self.fallback_arm
                else ""
            ),
            "rerank_latency_ms": 4.5,
        }


class LedgerQwen3PairedTests(unittest.TestCase):
    def test_sealed_result_is_source_bound_redacted_and_complete(self) -> None:
        cli = _load_cli()
        payload = json.loads(SEALED_RESULT.read_text(encoding="utf-8"))
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(payload["schema_version"], cli.SCHEMA_VERSION)
        self.assertEqual(payload["cases"], 250)
        self.assertEqual(
            payload["reranker_source_sha256"],
            cli._reranker_source_sha256(),
        )
        self.assertRegex(payload["per_case_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(
            payload["candidate_manifest"][
                "candidate_set_identity_sha256"
            ],
            r"^[0-9a-f]{64}$",
        )
        self.assertNotIn('"query_text"', serialized)
        self.assertNotIn('"text"', serialized)
        calls = payload["call_accounting"]
        self.assertEqual(calls["candidate_embedding_remote_calls"], 1851)
        self.assertEqual(calls["qwen3_logical_calls"], 500)
        self.assertEqual(calls["qwen3_physical_attempts"], 500)
        self.assertEqual(calls["qwen3_tokens"], 2_392_888)
        self.assertEqual(calls["rerank_fallbacks"], 0)
        self.assertTrue(payload["primary_comparison_valid"])
        comparison = payload["comparison"]
        self.assertEqual(
            comparison["qwen3"]["A_prod"]["remote_calls"],
            250,
        )
        self.assertEqual(
            comparison["qwen3"]["R_page"]["remote_calls"],
            250,
        )
        self.assertEqual(
            comparison["delta_qwen3_minus_prerank"]["A_prod"][
                "page_hit_at_10"
            ],
            0.144,
        )
        self.assertEqual(
            comparison["delta_qwen3_minus_prerank"]["R_page"][
                "page_hit_at_10"
            ],
            0.136,
        )

    def test_candidate_identity_is_order_sensitive_and_cache_hash_binds_text(self) -> None:
        cli = _load_cli()
        first = _candidate(cli)
        reversed_row = cli._candidate_row(
            query={
                "query_id": "q1",
                "query_text": "What was revenue?",
                "company_key": "company-a",
            },
            hits=list(reversed(first["hits"])),
        )
        self.assertNotEqual(
            first["candidate_identity_sha256"],
            reversed_row["candidate_identity_sha256"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.jsonl"
            cli._atomic_jsonl(path, [first])
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            changed = json.loads(path.read_text(encoding="utf-8"))
            changed["hits"][0]["text"] = "changed passage"
            cli._atomic_jsonl(path, [changed])
            self.assertNotEqual(
                before,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_candidate_call_accounting_fails_closed(self) -> None:
        cli = _load_cli()
        selection = {
            "documents": 2,
            "chunks": 3,
            "embed_chars": 100,
            "estimated_document_http_calls": 1,
            "expected_query_http_calls_minimum": 1,
        }
        manifest = {
            "call_accounting": {
                "documents_indexed": 2,
                "chunks_indexed": 3,
                "embed_chars": 100,
                "document_embedding_remote_calls": 1,
                "query_embedding_remote_calls": 1,
                "remote_calls": 2,
            }
        }
        cli._validate_candidate_call_accounting(
            manifest,
            selection=selection,
        )
        manifest["call_accounting"]["remote_calls"] = 999
        with self.assertRaisesRegex(HoldoutError, "call accounting"):
            cli._validate_candidate_call_accounting(
                manifest,
                selection=selection,
            )

    def test_rerank_plan_counts_two_calls_per_case_and_actual_chars(self) -> None:
        cli = _load_cli()
        candidates = [_candidate(cli, query_id=f"q{index}") for index in range(3)]
        queries = {
            f"q{index}": {
                "query_id": f"q{index}",
                "query_text": "revenue",
                "company_key": "company-a",
            }
            for index in range(3)
        }
        plan = cli._rerank_plan(
            candidates,
            queries,
            max_document_chars=4000,
            max_attempts=2,
        )
        self.assertEqual(plan["qwen3_requests_without_retries"], 6)
        self.assertEqual(plan["qwen3_physical_attempts_ceiling"], 12)
        self.assertEqual(plan["document_slots"], 90)
        self.assertGreater(plan["request_chars"], 0)

    def test_remote_rerank_configuration_fails_before_provider_call(self) -> None:
        cli = _load_cli()
        settings = {
            "model": "qwen3-rerank",
            "_base_url": "https://example.invalid",
        }
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(HoldoutError, "credential"),
        ):
            cli._validate_remote_rerank_configuration(settings)
        with mock.patch.dict(
            "os.environ",
            {"DASHSCOPE_API_KEY": "test-key"},
            clear=True,
        ):
            with self.assertRaisesRegex(HoldoutError, "must be qwen3"):
                cli._validate_remote_rerank_configuration(
                    {**settings, "model": "another-model"}
                )
            with self.assertRaisesRegex(HoldoutError, "HTTPS"):
                cli._validate_remote_rerank_configuration(
                    {**settings, "_base_url": "http://example.invalid"}
                )
            with self.assertRaisesRegex(HoldoutError, "normalized"):
                cli._validate_remote_rerank_configuration(
                    {
                        **settings,
                        "instruct": "rank",
                        "timeout_seconds": 12.0,
                        "backoff_seconds": 0.25,
                        "max_attempts": 0,
                        "max_inflight": 2,
                        "max_document_chars": 4000,
                    }
                )
            with self.assertRaisesRegex(HoldoutError, "normalized"):
                cli._validate_remote_rerank_configuration(
                    {
                        **settings,
                        "instruct": "rank",
                        "timeout_seconds": float("nan"),
                        "backoff_seconds": 0.25,
                        "max_attempts": 2,
                        "max_inflight": 2,
                        "max_document_chars": 4000,
                    }
                )

    def test_all_candidate_query_identities_are_preflighted_together(self) -> None:
        cli = _load_cli()
        first = _candidate(cli, query_id="q1")
        second = _candidate(cli, query_id="q2")
        second["query_text_sha256"] = "wrong"
        queries = {
            "q1": {
                "query_text": "What was revenue?",
                "company_key": "company-a",
            },
            "q2": {
                "query_text": "What was revenue?",
                "company_key": "company-a",
            },
        }
        with self.assertRaisesRegex(HoldoutError, "query/company"):
            cli._validate_candidate_queries([first, second], queries)

    def test_frozen_plan_cannot_change_or_rewrite_after_rerank_starts(self) -> None:
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            cli._freeze_rerank_plan(output, {"request_chars": 100})
            cli._freeze_rerank_plan(output, {"request_chars": 100})
            with self.assertRaisesRegex(HoldoutError, "diverged"):
                cli._freeze_rerank_plan(output, {"request_chars": 101})
            (output / "per_case.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(HoldoutError, "has started"):
                cli._freeze_rerank_plan(output, {"request_chars": 100})

    def test_completed_rows_fail_closed_on_candidate_or_arm_divergence(self) -> None:
        cli = _load_cli()
        candidate = _candidate(cli)
        metric = _metric("q1")
        run_identity = "a" * 64
        reranked_arms = {}
        for arm in ("A_prod", "R_page"):
            pool = cli.prepare_rerank_pool(candidate["hits"], arm=arm)
            final = pool[:10]
            reranked_arms[arm] = {
                **dict(metric),
                "pool_identity_sha256": cli._hash_hits(pool),
                "final_identity_sha256": cli._hash_hits(final),
                "final_identity": cli._hit_identity(final),
                "rerank_attempts": 1,
                "rerank_tokens": 10,
                "rerank_fallback": False,
                "rerank_error_type": "",
                "rerank_latency_ms": 1.0,
            }
        row = {
            "query_id": "q1",
            "run_identity_sha256": run_identity,
            "shared_candidate_identity_sha256": candidate[
                "candidate_identity_sha256"
            ],
            "prerank_arms": {
                "A_prod": metric,
                "R_page": dict(metric),
            },
            "reranked_arms": reranked_arms,
        }
        row["row_sha256"] = cli._row_sha256(row)
        validated = cli._validate_completed_rows(
            [row],
            candidate_by_id={"q1": candidate},
            run_identity_sha256=run_identity,
        )
        self.assertEqual(set(validated), {"q1"})
        row["shared_candidate_identity_sha256"] = "wrong"
        with self.assertRaisesRegex(HoldoutError, "identity mismatch"):
            cli._validate_completed_rows(
                [row],
                candidate_by_id={"q1": candidate},
                run_identity_sha256=run_identity,
            )
        row["shared_candidate_identity_sha256"] = candidate[
            "candidate_identity_sha256"
        ]
        row["reranked_arms"]["R_page"]["case_id"] = "q2"
        row["row_sha256"] = cli._row_sha256(row)
        with self.assertRaisesRegex(HoldoutError, "arm identity"):
            cli._validate_completed_rows(
                [row],
                candidate_by_id={"q1": candidate},
                run_identity_sha256=run_identity,
            )
        row["reranked_arms"]["R_page"]["case_id"] = "q1"
        row["reranked_arms"]["R_page"]["rerank_tokens"] = 999
        with self.assertRaisesRegex(HoldoutError, "identity mismatch"):
            cli._validate_completed_rows(
                [row],
                candidate_by_id={"q1": candidate},
                run_identity_sha256=run_identity,
            )

    def test_one_case_uses_same_frozen_candidates_for_both_qwen3_arms(self) -> None:
        cli = _load_cli()
        query = {
            "query_id": "q1",
            "query_text": "What was revenue?",
            "company_key": "company-a",
            "qrels": [{"doc_id": "doc-10", "relevance": 1}],
        }
        dataset = LedgerPublicDevDataset(
            queries=(query,),
            page_documents=tuple(
                {"ledger_doc_id": f"doc-{index}"}
                for index in range(1, 11)
            ),
            companies=("company-a",),
            reports=1,
        )
        candidate = _candidate(cli)
        reranker = _FakeReranker()
        row = cli._run_one_case(
            query=query,
            candidate_row=candidate,
            dataset=dataset,
            reranker=reranker,
            run_identity_sha256="a" * 64,
        )
        self.assertEqual(reranker.calls, 2)
        self.assertEqual(
            row["shared_candidate_identity_sha256"],
            candidate["candidate_identity_sha256"],
        )
        self.assertEqual(
            row["reranked_arms"]["A_prod"]["pool_identity_sha256"],
            candidate["candidate_identity_sha256"],
        )
        self.assertNotEqual(
            row["reranked_arms"]["R_page"]["pool_identity_sha256"],
            candidate["candidate_identity_sha256"],
        )
        self.assertEqual(
            row["reranked_arms"]["A_prod"]["rerank_tokens"],
            123,
        )
        summary = cli._summary([row], "reranked_arms")
        self.assertEqual(summary["A_prod"]["remote_calls"], 1)
        self.assertEqual(summary["R_page"]["remote_calls"], 1)
        self.assertTrue(cli._all_qwen3_ok([row]))
        row["reranked_arms"]["A_prod"]["rerank_attempts"] = 0
        row["reranked_arms"]["R_page"]["rerank_attempts"] = 2
        self.assertFalse(cli._all_qwen3_ok([row]))

    def test_reranker_source_fingerprint_is_complete(self) -> None:
        cli = _load_cli()
        self.assertRegex(cli._reranker_source_sha256(), r"^[0-9a-f]{64}$")
        self.assertEqual(
            {
                path.relative_to(ROOT).as_posix()
                for path in cli._reranker_source_paths()
            },
            {
                "src/lumenfin/provider_retry.py",
                "src/lumenfin/provider_resilience.py",
                "src/lumenfin/rag/rerank.py",
                "src/lumenfin/eval/financebench/candidate_pool_ablation.py",
                "scripts/run_ledger_public_dev_qwen3_paired.py",
            },
        )

    def test_fallback_is_retained_in_per_case_record(self) -> None:
        cli = _load_cli()
        query = {
            "query_id": "q1",
            "query_text": "What was revenue?",
            "company_key": "company-a",
            "qrels": [{"doc_id": "doc-10", "relevance": 1}],
        }
        dataset = LedgerPublicDevDataset(
            queries=(query,),
            page_documents=tuple(
                {"ledger_doc_id": f"doc-{index}"}
                for index in range(1, 11)
            ),
            companies=("company-a",),
            reports=1,
        )
        row = cli._run_one_case(
            query=query,
            candidate_row=_candidate(cli),
            dataset=dataset,
            reranker=_FakeReranker(fallback_arm=2),
            run_identity_sha256="a" * 64,
        )
        self.assertFalse(
            row["reranked_arms"]["A_prod"]["rerank_fallback"]
        )
        self.assertTrue(
            row["reranked_arms"]["R_page"]["rerank_fallback"]
        )
        self.assertEqual(
            row["reranked_arms"]["R_page"]["rerank_error_type"],
            "transient_provider_error",
        )


if __name__ == "__main__":
    unittest.main()
