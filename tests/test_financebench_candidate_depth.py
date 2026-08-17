from __future__ import annotations

import importlib.util
import json
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

from lumenfin.eval.financebench.candidate_depth import (
    CandidateDepthError,
    DIAGNOSTIC_CANDIDATE_K,
    assert_per_case_redacted,
    best_channel_rank,
    channel_recall_label,
    classify_depth_failure,
    copy_index_for_query,
    duplicate_page_occupancy,
    first_gold_rank,
    hit_at_depths,
    oracle_union_metrics,
    recommend_from_aggregate,
    require_fresh_output_dir,
    run_candidate_depth_diagnostic,
    score_case,
    unique_page_count,
    validate_candidate_depth_request,
)
from lumenfin.eval.financebench.index_inspect import (
    EXPECTED_CHUNKS,
    EXPECTED_DATASET_HASH,
    EXPECTED_DOCUMENTS,
    SOURCE_INDEX_CHUNKER,
    SOURCE_INDEX_COMMIT,
    IndexIncompatibleError,
    inspect_financebench_indexes,
    inspect_lite_index,
    require_compatible_index,
)
from lumenfin.eval.financebench.loader import load_financebench_dataset
from lumenfin.eval.financebench.reporting import sha256_file
from lumenfin.eval.financebench.retrieval import RemoteEvalBlocked
from lumenfin.eval.financebench.schema import DocumentInfo, EvidenceSpan, FinanceBenchQuestion
from lumenfin.eval.financebench.split import SplitError
from lumenfin.rag.embeddings import DashScopeEmbeddingProvider

REMOTE_PROBE_COUNT = 0


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "run_financebench_candidate_depth_cli",
        ROOT / "scripts" / "run_financebench_candidate_depth.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load candidate-depth CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _count_remote(*_args, **_kwargs):
    global REMOTE_PROBE_COUNT
    REMOTE_PROBE_COUNT += 1
    raise AssertionError("DashScope embedding must not be called in candidate-depth tests")


def _page_hit(doc: str, page: int, chunk_index: int = 0, text: str = "LEAKED_CHUNK_BODY") -> dict:
    return {
        "chunk_id": f"{doc}:p{page}:c{chunk_index}",
        "document_id": doc,
        "filename": f"{doc}.pdf",
        "page": page,
        "text": text,
        "companies": ["Acme"],
        "retrieval_method": "vector",
    }


def _question_multi_gold() -> FinanceBenchQuestion:
    return FinanceBenchQuestion(
        financebench_id="financebench_id_00001",
        case_id="fb-financebench_id_00001",
        question="SECRET_QUESTION_TEXT",
        answer="SECRET_ANSWER",
        justification="SECRET_JUSTIFICATION",
        question_type="metrics-generated",
        question_reasoning="Information extraction",
        domain_question_num="",
        company="Acme",
        doc_name="ACME_2022_10K",
        dataset_subset_label="OPEN_SOURCE",
        evidence=(
            EvidenceSpan(
                evidence_doc_name="ZERO_DOC_2022_10Q",
                evidence_page_num_zero=3,
                evidence_page_num_one=4,
                evidence_text="SECRET_EVIDENCE_TEXT",
            ),
            EvidenceSpan(
                evidence_doc_name="ACME_2022_10K",
                evidence_page_num_zero=1,
                evidence_page_num_one=2,
                evidence_text="SECRET_EVIDENCE_TEXT",
            ),
        ),
        document=DocumentInfo(doc_name="ACME_2022_10K", company="Acme", period="FY2022"),
    )


def _question(*, doc_name: str = "ACME_2022_10K", page_zero: int = 1) -> FinanceBenchQuestion:
    return FinanceBenchQuestion(
        financebench_id="financebench_id_00001",
        case_id="fb-financebench_id_00001",
        question="SECRET_QUESTION_TEXT",
        answer="SECRET_ANSWER",
        justification="SECRET_JUSTIFICATION",
        question_type="metrics-generated",
        question_reasoning="Information extraction",
        domain_question_num="",
        company="Acme",
        doc_name=doc_name,
        dataset_subset_label="OPEN_SOURCE",
        evidence=(
            EvidenceSpan(
                evidence_doc_name=doc_name,
                evidence_page_num_zero=page_zero,
                evidence_page_num_one=page_zero + 1,
                evidence_text="SECRET_EVIDENCE_TEXT",
            ),
        ),
        document=DocumentInfo(doc_name=doc_name, company="Acme", period="FY2022"),
    )


def _write_compatible_index(repo: Path, *, dataset_hash: str = EXPECTED_DATASET_HASH) -> Path:
    eval_root = repo / "outputs" / "financebench_eval_company"
    index_root = eval_root / "index-compat"
    collection = index_root / "eval.db" / "collections" / "financebench_eval"
    collection.mkdir(parents=True)
    schema_fields = [
        "id",
        "text",
        "vector",
        "sparse",
        "row_key",
        "session_id",
        "tenant_id",
        "source_document_id",
        "content_hash",
        "chunk_id",
        "document_id",
        "filename",
        "companies",
        "primary_company",
        "chunk_type",
        "page",
        "char_count",
        "financial_fact",
    ]
    fields = []
    for name in schema_fields:
        field = {"name": name, "dtype": "varchar"}
        if name == "vector":
            field = {"name": "vector", "dtype": "float_vector", "dim": 1024}
        if name == "page":
            field = {"name": "page", "dtype": "int64"}
        fields.append(field)
    (collection / "schema.json").write_text(json.dumps({"fields": fields}), encoding="utf-8")
    (collection / "manifest.json").write_text(
        json.dumps(
            {
                "current_seq": EXPECTED_CHUNKS,
                "active_wal_number": EXPECTED_DOCUMENTS,
                "index_specs": {
                    "vector": {"field_name": "vector", "index_type": "AUTOINDEX"},
                    "sparse": {"field_name": "sparse", "index_type": "SPARSE_INVERTED_INDEX"},
                },
            }
        ),
        encoding="utf-8",
    )
    env = {
        "dataset_hash": dataset_hash,
        "embedding_provider": "dashscope",
        "embedding_model": "text-embedding-v4",
        "embedding_dimension": 1024,
        "chunk_size": 900,
        "chunk_overlap": 120,
        "index_scope": "company",
        "bm25_rrf_weight": 1.1,
    }
    (eval_root / "hybrid").mkdir(parents=True)
    (eval_root / "hybrid" / "environment.json").write_text(json.dumps(env), encoding="utf-8")
    (eval_root / "results.json").write_text(
        json.dumps(
            {
                "ingestion": {
                    "document_count": EXPECTED_DOCUMENTS,
                    "chunks_created": EXPECTED_CHUNKS,
                    "zero_chunk_documents": ["JOHNSON_JOHNSON_2022Q4_EARNINGS"],
                }
            }
        ),
        encoding="utf-8",
    )
    return index_root / "eval.db"


class DepthStore:
    def __init__(self, bm25_hits, dense_hits) -> None:
        self.bm25_hits = bm25_hits
        self.dense_hits = dense_hits
        self.bm25_calls = 0
        self.vector_calls = 0
        self.indexed = 0

    def close(self) -> None:
        return None

    def index_documents(self, documents, session_id):
        self.indexed += len(documents)
        raise AssertionError("candidate-depth must not re-embed / index documents")

    def bm25_search(self, query, **kwargs):
        self.bm25_calls += 1
        self.assert_top_k(kwargs)
        return list(self.bm25_hits)

    def vector_search(self, query, **kwargs):
        self.vector_calls += 1
        self.assert_top_k(kwargs)
        return list(self.dense_hits)

    def assert_top_k(self, kwargs: dict) -> None:
        if int(kwargs.get("top_k") or 0) != DIAGNOSTIC_CANDIDATE_K:
            raise AssertionError(f"expected top_k={DIAGNOSTIC_CANDIDATE_K}, got {kwargs.get('top_k')}")


class FinanceBenchCandidateDepthTests(unittest.TestCase):
    def setUp(self) -> None:
        global REMOTE_PROBE_COUNT
        REMOTE_PROBE_COUNT = 0

    def tearDown(self) -> None:
        self.assertEqual(REMOTE_PROBE_COUNT, 0)

    def test_confirmation_dev_all_are_rejected(self) -> None:
        for split in ("confirmation", "dev", "all"):
            with self.assertRaises(SplitError):
                validate_candidate_depth_request(
                    split=split,
                    confirm_exposed_diagnostic=True,
                    allow_remote=True,
                    embedding_provider="dashscope",
                    output_dir=ROOT / "outputs" / "financebench_candidate_depth_test100",
                    repo_root=ROOT,
                )

    def test_missing_exposed_switch_is_rejected(self) -> None:
        with self.assertRaises(CandidateDepthError):
            validate_candidate_depth_request(
                split="test",
                confirm_exposed_diagnostic=False,
                allow_remote=True,
                embedding_provider="dashscope",
                output_dir=ROOT / "outputs" / "financebench_candidate_depth_test100",
                repo_root=ROOT,
            )

    def test_dashscope_without_allow_remote_is_rejected_before_index(self) -> None:
        with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
            with self.assertRaises(RemoteEvalBlocked):
                validate_candidate_depth_request(
                    split="test",
                    confirm_exposed_diagnostic=True,
                    allow_remote=False,
                    embedding_provider="dashscope",
                    output_dir=ROOT / "outputs" / "financebench_candidate_depth_test100",
                    repo_root=ROOT,
                )

    def test_copy_index_skips_lock_and_does_not_modify_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "original" / "eval.db"
            source.mkdir(parents=True)
            (source / "LOCK").write_text("lock", encoding="utf-8")
            (source / "keep.txt").write_text("ok", encoding="utf-8")
            copied = copy_index_for_query(source, Path(tmp) / "work")
            self.assertFalse((copied / "LOCK").exists())
            self.assertEqual((copied / "keep.txt").read_text(encoding="utf-8"), "ok")
            self.assertEqual((source / "LOCK").read_text(encoding="utf-8"), "lock")

    def test_cli_rejects_all_and_missing_remote_flag(self) -> None:
        cli = _load_cli()
        with patch.object(cli, "run_candidate_depth_diagnostic") as run:
            self.assertEqual(cli.main(["--split", "all", "--confirm-exposed-diagnostic", "--allow-remote"]), 2)
            self.assertEqual(cli.main(["--split", "test", "--confirm-exposed-diagnostic"]), 2)
        run.assert_not_called()

    def test_cli_rejects_confirmation_without_running(self) -> None:
        cli = _load_cli()
        with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
            with patch.object(cli, "run_candidate_depth_diagnostic") as run:
                code = cli.main(
                    [
                        "--split",
                        "confirmation",
                        "--confirm-exposed-diagnostic",
                        "--allow-remote",
                    ]
                )
        self.assertEqual(code, 2)
        run.assert_not_called()

    def test_incompatible_index_fails_before_remote_or_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "outputs" / "financebench_eval_company").mkdir(parents=True)
            with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
                with patch(
                    "lumenfin.documents.parse_pdf_document",
                    side_effect=AssertionError("PDF parse must not run"),
                ):
                    with patch(
                        "lumenfin.eval.financebench.candidate_depth.load_financebench_dataset",
                        side_effect=AssertionError("dataset load must not run"),
                    ):
                        with self.assertRaises(IndexIncompatibleError):
                            run_candidate_depth_diagnostic(
                                dataset_dir=repo,
                                output_dir=repo / "outputs" / "financebench_candidate_depth_test100",
                                repo_root=repo,
                                split="test",
                                confirm_exposed_diagnostic=True,
                                allow_remote=True,
                                embedding_provider="dashscope",
                                skip_index_copy=True,
                                require_clean_worktree=False,
                            )

    def test_first_gold_rank_and_hit_depths(self) -> None:
        gold = {("ACME_2022_10K", 2)}
        from lumenfin.eval.financebench.qrels import retrieved_page_keys

        hits = [_page_hit("ACME_2022_10K", page) for page in range(12, 0, -1)]
        hits.append(_page_hit("ACME_2022_10K", 2, chunk_index=9))
        unique = retrieved_page_keys(hits)
        self.assertEqual(first_gold_rank(unique, gold), 11)
        depth = hit_at_depths(unique, gold)
        self.assertEqual(depth["10"], 0.0)
        self.assertEqual(depth["20"], 1.0)
        self.assertEqual(depth["30"], 1.0)
        self.assertEqual(depth["50"], 1.0)
        self.assertEqual(
            classify_depth_failure(retrieved_pages=unique, gold_pages=gold),
            "gold_rank_11_20",
        )

    def test_page_dedup_occupancy(self) -> None:
        hits = [_page_hit("ACME_2022_10K", 2, chunk_index=index) for index in range(8)]
        hits.extend(_page_hit("ACME_2022_10K", 3, chunk_index=index) for index in range(2))
        self.assertEqual(unique_page_count(hits, k=10), 2)
        self.assertEqual(duplicate_page_occupancy(hits, k=10), 0.8)

    def test_channel_recall_and_wrong_period(self) -> None:
        gold = {("ACME_2022_10K", 2)}
        bm25 = [("ACME_2022_10K", 2)]
        dense = [("ACME_2021_10K", 4)]
        self.assertEqual(channel_recall_label(bm25_pages=bm25, dense_pages=dense, gold_pages=gold), "bm25_only")
        self.assertEqual(
            channel_recall_label(bm25_pages=dense, dense_pages=dense, gold_pages=gold),
            "neither",
        )
        documents = {
            "ACME_2022_10K": DocumentInfo(doc_name="ACME_2022_10K", company="Acme", period="FY2022"),
            "ACME_2021_10K": DocumentInfo(doc_name="ACME_2021_10K", company="Acme", period="FY2021"),
            "OTHER_2022_10K": DocumentInfo(doc_name="OTHER_2022_10K", company="Other", period="FY2022"),
        }
        self.assertEqual(
            classify_depth_failure(
                retrieved_pages=[("ACME_2021_10K", 4)],
                gold_pages=gold,
                documents=documents,
                gold_company="Acme",
            ),
            "wrong_period",
        )
        self.assertEqual(
            classify_depth_failure(
                retrieved_pages=[("OTHER_2022_10K", 4)],
                gold_pages=gold,
                documents=documents,
                gold_company="Acme",
            ),
            "wrong_document",
        )

    def test_zero_chunk_is_ingestion_failure(self) -> None:
        question = _question()
        row = score_case(
            question,
            {
                "bm25": [],
                "dense": [],
                "hybrid_rrf": [],
                "oracle_union": [],
            },
            documents={"ACME_2022_10K": DocumentInfo(doc_name="ACME_2022_10K", company="Acme", period="FY2022")},
            zero_chunk_names={"ACME_2022_10K"},
        )
        self.assertTrue(row["ingestion_failure"])
        self.assertTrue(row["affected_by_zero_chunk"])
        self.assertEqual(row["modes"]["hybrid_rrf"]["failure_class"], "ingestion_failure")

    def test_per_case_omits_question_evidence_and_chunk_text(self) -> None:
        question = _question()
        row = score_case(
            question,
            {
                "bm25": [_page_hit("ACME_2022_10K", 2)],
                "dense": [_page_hit("ACME_2022_10K", 9)],
                "hybrid_rrf": [_page_hit("ACME_2022_10K", 2)],
                "oracle_union": [_page_hit("ACME_2022_10K", 2), _page_hit("ACME_2022_10K", 9)],
            },
            documents={"ACME_2022_10K": DocumentInfo(doc_name="ACME_2022_10K", company="Acme", period="FY2022")},
            zero_chunk_names=set(),
        )
        blob = json.dumps(row)
        self.assertNotIn("SECRET_QUESTION_TEXT", blob)
        self.assertNotIn("SECRET_EVIDENCE_TEXT", blob)
        self.assertNotIn("LEAKED_CHUNK_BODY", blob)
        self.assertNotIn("SECRET_ANSWER", blob)
        assert_per_case_redacted(row)
        self.assertEqual(row["channel_recall"], "bm25_only")
        self.assertEqual(row["modes"]["hybrid_rrf"]["first_gold_rank"], 1)
        self.assertEqual(row["modes"]["hybrid_rrf"]["hit_at"]["10"], 1)
        self.assertNotIn("mrr_at_50", row["modes"]["oracle_union"])
        self.assertNotIn("first_gold_rank", row["modes"]["oracle_union"])
        self.assertEqual(row["best_channel_rank"], 1)

    def test_offline_run_writes_redacted_outputs_without_reembed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src = make_financebench_tree(repo / "src")
            questions, _documents, paths = load_financebench_dataset(
                src, expected_questions=4, require_pdfs=False
            )
            dataset_hash = sha256_file(paths.questions_path)
            eval_db = _write_compatible_index(repo, dataset_hash=dataset_hash)
            gold_doc = questions[0].doc_name
            gold_page = questions[0].evidence[0].evidence_page_num_one
            store = DepthStore(
                bm25_hits=[_page_hit(gold_doc, gold_page)],
                dense_hits=[_page_hit(gold_doc, gold_page + 3)],
            )
            out = repo / "outputs" / "financebench_candidate_depth_test100"
            inspection = {
                "compatible": True,
                "compatible_index": {
                    "compatible": True,
                    "uri": str(eval_db),
                    "dataset_hash": dataset_hash,
                    "collection_name": "financebench_eval",
                    "embedding_model": "text-embedding-v4",
                    "zero_chunk_documents": [],
                    "index_scope": "company",
                    "source_index_commit": SOURCE_INDEX_COMMIT,
                    "source_index_worktree_dirty": True,
                    "source_index_chunker": SOURCE_INDEX_CHUNKER,
                    "source_schema_sha256": sha256_file(
                        eval_db / "collections" / "financebench_eval" / "schema.json"
                    ),
                    "source_collection_manifest_sha256": sha256_file(
                        eval_db / "collections" / "financebench_eval" / "manifest.json"
                    ),
                    "index_not_current_chunker": True,
                },
                "opened_milvus_client": False,
                "modified_original_index": False,
            }
            with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
                with patch(
                    "lumenfin.documents.parse_pdf_document",
                    side_effect=AssertionError("PDF parse must not run"),
                ):
                    with patch(
                        "lumenfin.eval.financebench.retrieval._qwen3_reranker",
                        side_effect=AssertionError("Qwen3 must not run"),
                    ):
                        report = run_candidate_depth_diagnostic(
                            dataset_dir=src,
                            output_dir=out,
                            repo_root=repo,
                            split="test",
                            confirm_exposed_diagnostic=True,
                            allow_remote=False,
                            embedding_provider="deterministic",
                            expected_questions=4,
                            store=store,
                            index_inspection=inspection,
                            skip_index_copy=True,
                            require_clean_worktree=False,
                            worktree_dirty=False,
                        )
            self.assertEqual(store.indexed, 0)
            self.assertEqual(store.bm25_calls, report["summary"]["cases"])
            self.assertEqual(store.vector_calls, report["summary"]["cases"])
            self.assertEqual(report["qwen3_calls"], 0)
            self.assertEqual(report["chunk_reembed_calls"], 0)
            self.assertFalse(report["held_out"])
            self.assertFalse(report["product_accuracy_claim"])
            self.assertEqual(report["query_embedding_calls"], 0)
            rows = [
                json.loads(line)
                for line in (out / "per_case.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            blob = (out / "per_case.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("SECRET_", blob)
            self.assertNotIn("LEAKED_CHUNK_BODY", blob)
            for row in rows:
                assert_per_case_redacted(row)
            self.assertTrue((out / "summary.json").is_file())
            self.assertTrue((out / "results.md").is_file())
            self.assertTrue((out / "environment.json").is_file())
            self.assertIn("exposed_diagnostic_not_final", json.dumps(report["recommendations"]))
            self.assertEqual(eval_db.name, "eval.db")
            self.assertEqual(report["source_index"]["commit"], SOURCE_INDEX_COMMIT)
            self.assertTrue(report["source_index"]["worktree_dirty"])
            self.assertEqual(report["source_index"]["chunker"], SOURCE_INDEX_CHUNKER)
            self.assertTrue(report["source_index"]["not_current_chunker"])
            self.assertEqual(
                report["source_index"]["schema_sha256"],
                sha256_file(eval_db / "collections" / "financebench_eval" / "schema.json"),
            )
            self.assertEqual(
                report["source_index"]["collection_manifest_sha256"],
                sha256_file(eval_db / "collections" / "financebench_eval" / "manifest.json"),
            )
            self.assertEqual(report["environment"]["source_index_chunker"], SOURCE_INDEX_CHUNKER)
            self.assertTrue(report["environment"]["index_not_current_chunker"])
            self.assertFalse(report["diagnostic_code"]["worktree_dirty"])
            markdown = (out / "results.md").read_text(encoding="utf-8")
            self.assertIn("best_channel_rank", markdown)
            oracle_summary = (report["summary"].get("modes") or {}).get("oracle_union") or {}
            self.assertNotIn("mrr_at_50", oracle_summary)
            self.assertNotIn("mean_first_gold_rank_when_found", oracle_summary)
            self.assertNotIn("first_gold_rank", oracle_summary)
            self.assertNotRegex(markdown, r"(?im)^\|\s*oracle_union\s*\|")
            self.assertNotIn("wrong_period_in_top10", json.dumps(report["summary"]))
            self.assertIn("wrong_period_in_candidate_50", report["summary"])

    def test_file_inspect_accepts_matching_sidecar_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            eval_db = _write_compatible_index(repo)
            report = inspect_lite_index(
                eval_db,
                sidecar={
                    "eval_root": str(eval_db.parent.parent),
                    "index_scope": "company",
                    "environment": json.loads(
                        (repo / "outputs" / "financebench_eval_company" / "hybrid" / "environment.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                    "ingestion": {
                        "document_count": EXPECTED_DOCUMENTS,
                        "chunks_created": EXPECTED_CHUNKS,
                    },
                    "manifest": {},
                },
            )
            self.assertTrue(report["compatible"])
            self.assertEqual(report["section_metadata"], "NOT_AVAILABLE")
            self.assertEqual(report["source_index_commit"], SOURCE_INDEX_COMMIT)
            self.assertTrue(report["source_index_worktree_dirty"])
            self.assertEqual(report["source_index_chunker"], SOURCE_INDEX_CHUNKER)
            self.assertTrue(report["index_not_current_chunker"])
            self.assertTrue(report["source_schema_sha256"])
            self.assertTrue(report["source_collection_manifest_sha256"])
            discovered = inspect_financebench_indexes(repo)
            self.assertTrue(discovered["compatible"])
            selected = require_compatible_index(discovered)
            self.assertEqual(selected["chunks"], EXPECTED_CHUNKS)

    def test_file_inspect_rejects_wrong_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            eval_db = _write_compatible_index(repo, dataset_hash="deadbeef")
            discovered = inspect_financebench_indexes(repo)
            self.assertFalse(discovered["compatible"])
            self.assertIn("dataset_hash", discovered["candidates"][0]["mismatches"])
            self.assertEqual(eval_db.name, "eval.db")

    def test_historical_output_dir_is_rejected(self) -> None:
        with self.assertRaises(CandidateDepthError):
            validate_candidate_depth_request(
                split="test",
                confirm_exposed_diagnostic=True,
                allow_remote=True,
                embedding_provider="dashscope",
                output_dir=ROOT / "outputs" / "financebench_eval_company",
                repo_root=ROOT,
            )

    def test_recommendations_are_marked_diagnostic(self) -> None:
        recs = recommend_from_aggregate(
            {
                "cases": 100,
                "top10_misses": 40,
                "recoverable_11_30": 20,
                "top10_miss_depth": {
                    "gold_rank_11_20": 12,
                    "gold_rank_21_30": 8,
                    "gold_rank_31_50": 2,
                    "gold_not_in_top50": 5,
                },
                "channel_recall": {"bm25_only": 9, "dense_only": 6, "both": 40, "neither": 45},
                "rrf_worse_than_best_channel": 25,
            }
        )
        by_id = {item["id"]: item for item in recs}
        self.assertTrue(by_id["expand_candidate_pool_30"]["triggered"])
        self.assertTrue(by_id["keep_hybrid"]["triggered"])
        self.assertTrue(by_id["inspect_fusion_or_rerank_pool"]["triggered"])
        self.assertTrue(all(item["status"] == "exposed_diagnostic_not_final" for item in recs))
        self.assertTrue(all(item["held_out"] is False for item in recs))

    def test_existing_output_dir_is_refused_and_not_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "outputs" / "financebench_candidate_depth_test100"
            out.mkdir(parents=True)
            sentinel = out / "keep_me.txt"
            sentinel.write_text("alive", encoding="utf-8")
            nested = out / "nested"
            nested.mkdir()
            (nested / "also_keep.txt").write_text("stay", encoding="utf-8")
            with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
                with self.assertRaises(CandidateDepthError):
                    run_candidate_depth_diagnostic(
                        dataset_dir=Path(tmp),
                        output_dir=out,
                        repo_root=Path(tmp),
                        split="test",
                        confirm_exposed_diagnostic=True,
                        allow_remote=True,
                        embedding_provider="dashscope",
                        skip_index_copy=True,
                        require_clean_worktree=False,
                    )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "alive")
            self.assertEqual((nested / "also_keep.txt").read_text(encoding="utf-8"), "stay")
            self.assertTrue(out.is_dir())

    def test_copy_index_refuses_existing_dest_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "original" / "eval.db"
            source.mkdir(parents=True)
            (source / "keep.txt").write_text("source", encoding="utf-8")
            dest_parent = Path(tmp) / "work"
            dest_parent.mkdir()
            leftover = dest_parent / "leftover.txt"
            leftover.write_text("do-not-delete", encoding="utf-8")
            with self.assertRaises(CandidateDepthError):
                copy_index_for_query(source, dest_parent)
            self.assertEqual(leftover.read_text(encoding="utf-8"), "do-not-delete")
            self.assertEqual((source / "keep.txt").read_text(encoding="utf-8"), "source")
            self.assertFalse((dest_parent / "eval.db").exists())

    def test_locked_params_are_rejected_before_remote(self) -> None:
        mismatches = [
            {"candidate_k": 20},
            {"index_scope": "corpus"},
            {"embedding_dimension": 768},
            {"embedding_model": "text-embedding-v3"},
            {"bm25_rrf_weight": 1.0},
            {"dense_rrf_weight": 1.1},
        ]
        for kwargs in mismatches:
            with self.subTest(kwargs=kwargs):
                with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
                    with self.assertRaises(CandidateDepthError):
                        validate_candidate_depth_request(
                            split="test",
                            confirm_exposed_diagnostic=True,
                            allow_remote=True,
                            embedding_provider="dashscope",
                            output_dir=ROOT / "outputs" / "financebench_candidate_depth_test100",
                            repo_root=ROOT,
                            **kwargs,
                        )
                with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
                    with self.assertRaises(CandidateDepthError):
                        run_candidate_depth_diagnostic(
                            dataset_dir=ROOT,
                            output_dir=ROOT / "outputs" / "financebench_candidate_depth_test100",
                            repo_root=ROOT,
                            split="test",
                            confirm_exposed_diagnostic=True,
                            allow_remote=True,
                            embedding_provider="dashscope",
                            skip_index_copy=True,
                            require_clean_worktree=False,
                            **kwargs,
                        )

    def test_cli_rejects_unlocked_flags(self) -> None:
        cli = _load_cli()
        with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
            with patch.object(cli, "run_candidate_depth_diagnostic") as run:
                with self.assertRaises(SystemExit):
                    cli.main(
                        [
                            "--confirm-exposed-diagnostic",
                            "--allow-remote",
                            "--candidate-k",
                            "20",
                        ]
                    )
                with self.assertRaises(SystemExit):
                    cli.main(
                        [
                            "--confirm-exposed-diagnostic",
                            "--allow-remote",
                            "--index-scope",
                            "corpus",
                        ]
                    )
        run.assert_not_called()

    def test_oracle_union_has_no_mrr_or_concat_rank(self) -> None:
        gold = {("ACME_2022_10K", 2)}
        bm25_pages = [("ACME_2022_10K", page) for page in range(12, 2, -1)] + [("ACME_2022_10K", 2)]
        dense_pages = [("ACME_2022_10K", 2)]
        oracle = oracle_union_metrics(
            bm25_pages=bm25_pages,
            dense_pages=dense_pages,
            gold_pages=gold,
        )
        self.assertEqual(oracle["hit_at"]["10"], 1)
        self.assertEqual(oracle["coverage_at"]["10"], 1.0)
        self.assertNotIn("mrr_at_50", oracle)
        self.assertNotIn("first_gold_rank", oracle)
        self.assertNotIn("candidates", oracle)
        concat_first = first_gold_rank(bm25_pages + dense_pages, gold)
        self.assertEqual(concat_first, 11)
        self.assertNotEqual(oracle["hit_at"]["10"], 0)

        question = _question()
        bm25_hits = [_page_hit("ACME_2022_10K", page) for page in range(12, 2, -1)]
        bm25_hits.append(_page_hit("ACME_2022_10K", 2, chunk_index=99))
        row = score_case(
            question,
            {
                "bm25": bm25_hits,
                "dense": [_page_hit("ACME_2022_10K", 2)],
                "hybrid_rrf": bm25_hits,
                "oracle_union": bm25_hits + [_page_hit("ACME_2022_10K", 2)],
            },
            documents={"ACME_2022_10K": DocumentInfo(doc_name="ACME_2022_10K", company="Acme", period="FY2022")},
            zero_chunk_names=set(),
        )
        oracle_row = row["modes"]["oracle_union"]
        self.assertNotIn("mrr_at_50", oracle_row)
        self.assertNotIn("first_gold_rank", oracle_row)
        self.assertEqual(oracle_row["hit_at"]["10"], 1)
        self.assertEqual(row["best_channel_rank"], 1)
        self.assertEqual(row["modes"]["bm25"]["first_gold_rank"], 11)

    def test_best_channel_rank_is_min_of_found_channels(self) -> None:
        self.assertEqual(best_channel_rank(15, 3), 3)
        self.assertEqual(best_channel_rank(0, 8), 8)
        self.assertEqual(best_channel_rank(4, 0), 4)
        self.assertEqual(best_channel_rank(0, 0), 0)

    def test_partial_zero_chunk_is_not_ingestion_failure(self) -> None:
        question = _question_multi_gold()
        row = score_case(
            question,
            {
                "bm25": [_page_hit("ACME_2022_10K", 2)],
                "dense": [_page_hit("ACME_2022_10K", 2)],
                "hybrid_rrf": [_page_hit("ACME_2022_10K", 2)],
                "oracle_union": [_page_hit("ACME_2022_10K", 2)],
            },
            documents={
                "ACME_2022_10K": DocumentInfo(doc_name="ACME_2022_10K", company="Acme", period="FY2022"),
                "ZERO_DOC_2022_10Q": DocumentInfo(
                    doc_name="ZERO_DOC_2022_10Q", company="Acme", period="FY2022Q4"
                ),
            },
            zero_chunk_names={"ZERO_DOC_2022_10Q"},
        )
        self.assertTrue(row["affected_by_zero_chunk"])
        self.assertFalse(row["ingestion_failure"])
        self.assertEqual(row["modes"]["hybrid_rrf"]["failure_class"], "hit_at_10")
        self.assertEqual(row["modes"]["hybrid_rrf"]["hit_at"]["10"], 1)
        self.assertEqual(row["best_channel_rank"], 1)

    def test_dirty_worktree_is_rejected_before_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "outputs" / "financebench_candidate_depth_test100"
            with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
                with patch(
                    "lumenfin.eval.financebench.candidate_depth.copy_index_for_query",
                    side_effect=AssertionError("index copy must not run on dirty worktree"),
                ):
                    with self.assertRaises(CandidateDepthError):
                        run_candidate_depth_diagnostic(
                            dataset_dir=Path(tmp),
                            output_dir=out,
                            repo_root=Path(tmp),
                            split="test",
                            confirm_exposed_diagnostic=True,
                            allow_remote=True,
                            embedding_provider="dashscope",
                            skip_index_copy=True,
                            require_clean_worktree=True,
                            worktree_dirty=True,
                        )
            self.assertFalse(out.exists())

    def test_require_fresh_output_dir_allows_missing_or_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            empty = Path(tmp) / "empty"
            empty.mkdir()
            require_fresh_output_dir(missing)
            require_fresh_output_dir(empty)
            (empty / "file.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(CandidateDepthError):
                require_fresh_output_dir(empty)

    def test_remote_probe_count_stays_zero(self) -> None:
        self.assertEqual(REMOTE_PROBE_COUNT, 0)


if __name__ == "__main__":
    unittest.main()
