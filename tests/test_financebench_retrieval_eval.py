from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.financebench_fixtures import make_financebench_tree

from lumenfin.eval.financebench.reporting import (
    assert_no_secrets,
    compare_modes,
    completed_case_ids,
    environment_payload,
    read_jsonl,
    redact_mapping,
)
from lumenfin.eval.financebench.retrieval import RemoteEvalBlocked, retrieve_for_mode
from lumenfin.eval.financebench.runner import run_retrieval_eval
from lumenfin.eval.financebench.split import SplitError


class FakeRAGStore:
    """In-memory stand-in so mode isolation tests do not need Milvus/jieba."""

    bm25_enabled = True

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.kwargs: list[dict] = []

    def close(self) -> None:
        return None

    def index_documents(self, documents, session_id):
        return {
            "embed_calls": 1,
            "chunks_indexed": 2,
            "documents_indexed": len(documents),
        }

    def _hit(self, method: str) -> dict:
        return {
            "chunk_id": f"ACME_2022_10K:p2:c0:{method}",
            "document_id": "ACME_2022_10K",
            "filename": "ACME_2022_10K.pdf",
            "page": 2,
            "text": "ACME FY2022 capital expenditures were 1577 million USD on the cash flow statement.",
            "companies": ["Acme"],
            "chunk_type": "financial_metric",
            "score": 0.8,
            "retrieval_method": method,
            "citation": "ACME_2022_10K.pdf#p2",
        }

    def bm25_search(self, query, **kwargs):
        self.calls.append("bm25")
        self.kwargs.append({"method": "bm25", **kwargs})
        return [self._hit("bm25")]

    def vector_search(self, query, **kwargs):
        self.calls.append("vector")
        self.kwargs.append({"method": "vector", **kwargs})
        return [self._hit("vector")]


RESULT_KEYS = {
    "schema_version",
    "status",
    "environment",
    "summary",
    "breakdowns",
    "failures",
    "system",
    "synthetic_gate_disclaimer",
    "held_out_status",
}
CASE_KEYS = {
    "schema_version",
    "case_id",
    "financebench_id",
    "mode",
    "status",
    "company",
    "doc_name",
    "labels",
    "single_gold_page",
    "page_provenance_ok",
    "qrel_notes",
    "page",
    "chunk",
    "citations",
    "hits",
    "failure_class",
    "latency_ms",
    "degraded",
    "rerank_fallback",
    "rerank_tokens",
    "retrieval_methods",
    "effective_mode",
    "error_type",
}


class FinanceBenchRetrievalEvalTests(unittest.TestCase):
    def _run(self, root: Path, *, mode: str, resume: bool = False, limit: int | None = 2, store=None):
        out = root / "out" / mode
        fake = store or FakeRAGStore()
        with patch("lumenfin.eval.financebench.runner.build_eval_store", return_value=fake):
            results = run_retrieval_eval(
                dataset_dir=root / "src",
                output_dir=out,
                repo_root=ROOT,
                split="all",
                mode=mode,
                top_k=5,
                embedding_provider="deterministic",
                allow_remote=False,
                resume=resume,
                limit=limit,
                expected_questions=4,
                require_pdfs=True,
            )
        return results, fake

    def test_offline_modes_do_not_cross_wires(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_financebench_tree(root / "src")
            bm25_results, bm25_store = self._run(root, mode="bm25", limit=1)
            dense_results, dense_store = self._run(root, mode="dense", limit=1)
            hybrid_results, hybrid_store = self._run(root, mode="hybrid", limit=1)
            self.assertEqual(bm25_results["status"], "recorded")
            self.assertEqual(dense_results["status"], "recorded")
            self.assertEqual(hybrid_results["status"], "recorded")
            self.assertEqual(bm25_store.calls, ["bm25"])
            self.assertEqual(dense_store.calls, ["vector"])
            self.assertIn("bm25", hybrid_store.calls)
            self.assertIn("vector", hybrid_store.calls)
            hybrid_row = read_jsonl(root / "out" / "hybrid" / "per_case.jsonl")[0]
            self.assertTrue(any("rrf" in item or item in {"bm25", "vector"} for item in hybrid_row["retrieval_methods"]))
            self.assertFalse(any("qwen3" in item for item in hybrid_row["retrieval_methods"]))

    def test_remote_providers_are_blocked_by_default(self) -> None:
        with self.assertRaises(RemoteEvalBlocked):
            retrieve_for_mode(
                mode="hybrid-qwen3",
                store=None,  # type: ignore[arg-type]
                query="q",
                company="Acme",
                session_id="s",
                document_contexts=[],
                top_k=5,
                allow_remote=False,
            )
        with self.assertRaises(RemoteEvalBlocked):
            retrieve_for_mode(
                mode="dense",
                store=None,  # type: ignore[arg-type]
                query="q",
                company="Acme",
                session_id="s",
                document_contexts=[],
                top_k=5,
                embedding_provider="dashscope",
                allow_remote=False,
            )

    def test_qwen3_mode_does_not_silently_become_bm25_or_dense(self) -> None:
        store = FakeRAGStore()

        class StubReranker:
            provider_name = "dashscope"
            model_name = "qwen3-rerank"

            def rerank(self, query, hits, *, top_k):
                return hits[:top_k], {
                    "rerank_provider": "dashscope",
                    "rerank_model": "qwen3-rerank",
                    "rerank_fallback": False,
                    "rerank_tokens": 4,
                    "rerank_mode_suffix": "qwen3_rerank",
                }

        with patch(
            "lumenfin.eval.financebench.retrieval._qwen3_reranker",
            return_value=StubReranker(),
        ):
            hits, meta = retrieve_for_mode(
                mode="hybrid-qwen3",
                store=store,
                query="What is capex?",
                company="Acme",
                session_id="financebench-eval",
                document_contexts=[
                    {
                        "document_id": "ACME_2022_10K",
                        "filename": "ACME_2022_10K.pdf",
                        "detected_companies": ["Acme"],
                        "pages": ["ACME FY2022 capital expenditures were 1577 million USD."],
                    }
                ],
                top_k=5,
                allow_remote=True,
            )
        self.assertTrue(hits)
        self.assertIn("bm25", store.calls)
        self.assertIn("vector", store.calls)
        self.assertTrue(
            "qwen3" in str(meta.get("mode") or "")
            or meta.get("rerank_provider") == "dashscope"
        )
        self.assertFalse(meta.get("rerank_fallback"))

    def test_resume_skips_completed_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_financebench_tree(root / "src")
            first, _store = self._run(root, mode="bm25", limit=1)
            self.assertEqual(first["summary"]["cases"], 1)
            per_case = root / "out" / "bm25" / "per_case.jsonl"
            done = completed_case_ids(per_case, mode="bm25")
            self.assertEqual(len(done), 1)
            fake = FakeRAGStore()
            with patch("lumenfin.eval.financebench.runner.build_eval_store", return_value=fake):
                with patch(
                    "lumenfin.eval.financebench.runner.retrieve_for_mode",
                    side_effect=AssertionError("should not recompute completed case"),
                ) as mocked:
                    run_retrieval_eval(
                        dataset_dir=root / "src",
                        output_dir=root / "out" / "bm25",
                        repo_root=ROOT,
                        split="all",
                        mode="bm25",
                        top_k=5,
                        allow_remote=False,
                        resume=True,
                        limit=1,
                        expected_questions=4,
                        require_pdfs=True,
                    )
                    mocked.assert_not_called()
            self.assertEqual(len(read_jsonl(per_case)), 1)

    def test_json_markdown_schema_and_secret_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_financebench_tree(root / "src")
            results, _store = self._run(root, mode="dense", limit=1)
            self.assertTrue(RESULT_KEYS.issubset(results))
            rows = read_jsonl(root / "out" / "dense" / "per_case.jsonl")
            self.assertTrue(CASE_KEYS.issubset(rows[0]))
            serialized = json.dumps(results) + json.dumps(rows[0])
            self.assertNotIn("sk-", serialized)
            self.assertNotIn("evidence_text_full_page", serialized)
            markdown = (root / "out" / "dense" / "results.md").read_text(encoding="utf-8")
            self.assertIn("Hit@5", markdown)
            leaked = redact_mapping(
                {
                    "api_key": "sk-secretvalue",
                    "request_id": "req-123",
                    "note": "Authorization: Bearer sk-secretvalue",
                }
            )
            assert_no_secrets(leaked)
            self.assertEqual(leaked["api_key"], "[REDACTED]")
            env = environment_payload(
                repo_root=ROOT,
                dataset_hash="abc",
                split_manifest_hash="def",
                embedding_provider="deterministic",
                embedding_model="deterministic-hash",
                rerank_provider="none",
                rerank_model="",
                chunk_size=900,
                chunk_overlap=120,
                collection_name="financebench_eval",
                bm25_rrf_weight=1.1,
                top_k=10,
                mode="bm25",
                split="dev",
                remote_calls_enabled=False,
            )
            self.assertIn("lumenfin_commit", env)
            self.assertNotIn(os.getenv("DASHSCOPE_API_KEY") or "sk-not-present", json.dumps(env))

    def test_corpus_scope_does_not_pass_company_filter(self) -> None:
        store = FakeRAGStore()
        retrieve_for_mode(
            mode="bm25",
            store=store,
            query="capex",
            company="Acme",
            session_id="s",
            document_contexts=[],
            top_k=5,
            index_scope="corpus",
        )
        retrieve_for_mode(
            mode="dense",
            store=store,
            query="capex",
            company="Acme",
            session_id="s",
            document_contexts=[],
            top_k=5,
            index_scope="corpus",
        )
        self.assertIsNone(store.kwargs[0].get("companies"))
        self.assertIsNone(store.kwargs[1].get("companies"))

    def test_company_scope_does_not_return_other_company_documents(self) -> None:
        class LeakyStore(FakeRAGStore):
            def bm25_search(self, query, **kwargs):
                self.calls.append("bm25")
                self.kwargs.append({"method": "bm25", **kwargs})
                other = self._hit("bm25")
                other["companies"] = ["OtherCo"]
                other["document_id"] = "OTHER_2022_10K"
                other["chunk_id"] = "OTHER_2022_10K:p2:c0"
                return [other, self._hit("bm25")]

        store = LeakyStore()
        hits, meta = retrieve_for_mode(
            mode="bm25",
            store=store,
            query="capex",
            company="Acme",
            session_id="s",
            document_contexts=[],
            top_k=5,
            index_scope="company",
        )
        self.assertEqual(store.kwargs[0].get("companies"), ["Acme"])
        self.assertEqual([hit["document_id"] for hit in hits], ["ACME_2022_10K"])
        self.assertEqual(meta["index_scope"], "company")

    def test_manifest_marks_exposed_test_and_blocks_remote_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_financebench_tree(root / "src")
            fake = FakeRAGStore()
            with patch("lumenfin.eval.financebench.runner.build_eval_store", return_value=fake):
                results = run_retrieval_eval(
                    dataset_dir=root / "src",
                    output_dir=root / "out",
                    repo_root=ROOT,
                    split="test",
                    mode="bm25",
                    index_scope="corpus",
                    top_k=5,
                    embedding_provider="deterministic",
                    allow_remote=False,
                    expected_questions=4,
                    require_pdfs=True,
                )
            self.assertEqual(results["split_status"], "exposed_test")
            self.assertEqual(results["experiment_role"], "exploratory_baseline")
            self.assertEqual(results["environment"]["split_status"], "exposed_test")
            self.assertIn("ingestion", results)
            company = run_retrieval_eval
            with patch("lumenfin.eval.financebench.runner.build_eval_store", return_value=FakeRAGStore()):
                diagnostic = company(
                    dataset_dir=root / "src",
                    output_dir=root / "out-company",
                    repo_root=ROOT,
                    split="test",
                    mode="bm25",
                    index_scope="company",
                    top_k=5,
                    embedding_provider="deterministic",
                    expected_questions=4,
                    require_pdfs=True,
                )
            self.assertEqual(diagnostic["experiment_role"], "post_hoc_paired_diagnostic")

    def test_mode_all_writes_four_directories(self) -> None:
        class StubReranker:
            provider_name = "dashscope"
            model_name = "qwen3-rerank"

            def rerank(self, query, hits, *, top_k):
                return hits[:top_k], {
                    "rerank_provider": "dashscope",
                    "rerank_model": "qwen3-rerank",
                    "rerank_fallback": False,
                    "rerank_tokens": 2,
                    "rerank_mode_suffix": "qwen3_rerank",
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_financebench_tree(root / "src")
            fake = FakeRAGStore()
            with patch("lumenfin.eval.financebench.runner.build_eval_store", return_value=fake):
                with patch(
                    "lumenfin.eval.financebench.retrieval._qwen3_reranker",
                    return_value=StubReranker(),
                ):
                    results = run_retrieval_eval(
                        dataset_dir=root / "src",
                        output_dir=root / "out",
                        repo_root=ROOT,
                        split="all",
                        mode="all",
                        index_scope="corpus",
                        top_k=5,
                        embedding_provider="deterministic",
                        allow_remote=True,
                        limit=1,
                        expected_questions=4,
                        require_pdfs=True,
                    )
            self.assertEqual(results["mode"], "all")
            self.assertEqual(set(results["modes"]), {"bm25", "dense", "hybrid", "hybrid-qwen3"})
            for name in ("bm25", "dense", "hybrid", "hybrid-qwen3"):
                self.assertTrue((root / "out" / name / "per_case.jsonl").is_file())
            self.assertTrue((root / "out" / "ablation.json").is_file())

    def test_ablation_rank_movement_and_test_split_guard(self) -> None:
        comparison = compare_modes(
            {
                "bm25": [
                    {
                        "case_id": "fb-1",
                        "page": {
                            "first_relevant_rank": 4,
                            "mrr": 0.25,
                            "hit_at": {"1": 0, "3": 0, "5": 1, "10": 1},
                            "recall_at": {"1": 0, "3": 0, "5": 1, "10": 1},
                            "ndcg_at": {"5": 0.4, "10": 0.4},
                        },
                        "chunk": {
                            "mrr": 0.25,
                            "hit_at": {"5": 1, "10": 1, "20": 1},
                            "recall_at": {"5": 1, "10": 1, "20": 1},
                            "ndcg_at": {"10": 0.4},
                        },
                        "status": "ok",
                        "single_gold_page": True,
                    }
                ],
                "hybrid": [
                    {
                        "case_id": "fb-1",
                        "page": {
                            "first_relevant_rank": 1,
                            "mrr": 1.0,
                            "hit_at": {"1": 1, "3": 1, "5": 1, "10": 1},
                            "recall_at": {"1": 1, "3": 1, "5": 1, "10": 1},
                            "ndcg_at": {"5": 1.0, "10": 1.0},
                        },
                        "chunk": {
                            "mrr": 1.0,
                            "hit_at": {"5": 1, "10": 1, "20": 1},
                            "recall_at": {"5": 1, "10": 1, "20": 1},
                            "ndcg_at": {"10": 1.0},
                        },
                        "status": "ok",
                        "single_gold_page": True,
                    }
                ],
            }
        )
        self.assertEqual(comparison["improved"], 1)
        self.assertEqual(comparison["movements"][0]["kind"], "improved")
        with self.assertRaises(SplitError):
            run_retrieval_eval(
                dataset_dir=ROOT,
                output_dir=ROOT / "test_artifacts" / "should-not-write",
                repo_root=ROOT,
                split="test",
                mode="bm25",
                tuning=True,
                expected_questions=None,
            )


if __name__ == "__main__":
    unittest.main()
