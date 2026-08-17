from __future__ import annotations

import hashlib
import importlib.util
import inspect
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

from tests.financebench_fixtures import make_financebench_tree, write_jsonl

from lumenfin.eval.financebench.candidate_pool_ablation import (
    ABLATION_OUTPUT_DIRNAME,
    ABLATION_PREFLIGHT_DIRNAME,
    ARM_SPECS,
    AblationError,
    EXPECTED_FOCUS_CASES,
    INDEX_COPY_DBNAME,
    INDEX_WORK_DIRNAME,
    build_locked_qwen3_reranker,
    construct_arm_pool,
    copy_index_for_query,
    expected_call_budget,
    is_locked_ablation_output_path,
    public_rerank_settings,
    run_candidate_pool_ablation,
    score_case_arms,
    snapshot_rerank_settings,
    validate_ablation_request,
    _focus_analysis,
    _pair_report,
)
from lumenfin.eval.financebench.constants import DEFAULT_RERANK_CANDIDATES, DEFAULT_TOP_K
from lumenfin.eval.financebench.frozen import PUBLISHED_CONFIG_HASH, load_frozen_config
from lumenfin.eval.financebench.index_inspect import (
    EXPECTED_CHUNKS,
    EXPECTED_DATASET_HASH,
    EXPECTED_DOCUMENTS,
    SOURCE_INDEX_CHUNKER,
    SOURCE_INDEX_COMMIT,
    SOURCE_INDEX_SESSION_ID,
    SOURCE_INDEX_TENANT_ID,
)
from lumenfin.eval.financebench.loader import load_financebench_dataset
from lumenfin.eval.financebench.reporting import sha256_file
from lumenfin.eval.financebench.retrieval import RemoteEvalBlocked
from lumenfin.eval.financebench.schema import DocumentInfo, EvidenceSpan, FinanceBenchQuestion
from lumenfin.eval.financebench.split import SplitError, assign_splits, questions_for_split
from lumenfin.rag.embeddings import DashScopeEmbeddingProvider
from lumenfin.rag.hybrid_retriever import HybridEvidenceRetriever
from lumenfin.rag.rerank import DEFAULT_RERANK_INSTRUCT

REMOTE_PROBE_COUNT = 0
LEAKY_RERANK_ERROR = (
    "Authorization: Bearer sk-testsecret123; "
    "url=https://dashscope.aliyuncs.com/compatible-mode/v1/reranks?api_key=sk-testsecret123; "
    'body={"results":[{"index":0,"relevance_score":0.99}],"id":"req-leaked"}'
)
LEAKY_BASE_URL_A = "https://example-rerank.invalid/api/v1"
LEAKY_BASE_URL_B = "https://other-rerank.invalid/v2?token=sk-testsecret123"
PRODUCTION_RAG_FILES = (
    "src/lumenfin/rag/hybrid_retriever.py",
    "src/lumenfin/rag/milvus_store.py",
    "src/lumenfin/rag/chunking.py",
    "src/lumenfin/agents/retrieval.py",
)


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "run_financebench_candidate_pool_ablation_cli",
        ROOT / "scripts" / "run_financebench_candidate_pool_ablation.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ablation CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _count_remote(*_args, **_kwargs):
    global REMOTE_PROBE_COUNT
    REMOTE_PROBE_COUNT += 1
    raise AssertionError("DashScope embedding must not be called in ablation tests")


def _hit(doc: str, page: int, chunk_index: int, text: str = "body") -> dict:
    return {
        "chunk_id": f"{doc}:p{page}:c{chunk_index}",
        "document_id": doc,
        "filename": f"{doc}.pdf",
        "page": page,
        "text": text,
        "companies": ["Acme"],
        "primary_company": "Acme",
    }


def _ranked_hits(*, n: int, gold_rank: int, doc: str = "ACME_2022_10K", gold_page: int = 2) -> list[dict]:
    rows: list[dict] = []
    for index in range(1, n + 1):
        page = gold_page if index == gold_rank else 200 + index
        rows.append(_hit(doc, page, index, text=f"chunk-{index}"))
    return rows


def _question() -> FinanceBenchQuestion:
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
                evidence_doc_name="ACME_2022_10K",
                evidence_page_num_zero=1,
                evidence_page_num_one=2,
                evidence_text="SECRET_EVIDENCE_TEXT",
            ),
        ),
        document=DocumentInfo(doc_name="ACME_2022_10K", company="Acme", period="FY2022"),
    )


def _write_compatible_index(repo: Path, *, dataset_hash: str = EXPECTED_DATASET_HASH) -> Path:
    eval_root = repo / "outputs" / "financebench_eval_company"
    index_root = eval_root / "index-compat"
    collection = index_root / "eval.db" / "collections" / "financebench_eval"
    collection.mkdir(parents=True)
    fields = [{"name": "id", "dtype": "varchar"}, {"name": "vector", "dtype": "float_vector", "dim": 1024}]
    for name in (
        "text",
        "sparse",
        "session_id",
        "tenant_id",
        "chunk_id",
        "document_id",
        "filename",
        "companies",
        "primary_company",
        "page",
    ):
        fields.append({"name": name, "dtype": "int64" if name == "page" else "varchar"})
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
                    "zero_chunk_documents": [],
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
        self.sessions: list[str] = []
        self.top_ks: list[int] = []

    def close(self) -> None:
        return None

    def index_documents(self, documents, session_id):
        raise AssertionError("ablation must not re-embed / index documents")

    def bm25_search(self, query, **kwargs):
        self.bm25_calls += 1
        self.sessions.append(str(kwargs.get("session_id") or ""))
        self.top_ks.append(int(kwargs.get("top_k") or 0))
        return list(self.bm25_hits)

    def vector_search(self, query, **kwargs):
        self.vector_calls += 1
        self.sessions.append(str(kwargs.get("session_id") or ""))
        self.top_ks.append(int(kwargs.get("top_k") or 0))
        return list(self.dense_hits)


class FakeIndexClient:
    def __init__(self) -> None:
        self.loaded = False
        self.load_calls = 0
        self.close_calls = 0

    def list_collections(self) -> list[str]:
        return ["financebench_eval"]

    def load_collection(self, collection_name: str) -> None:
        self.load_calls += 1
        self.loaded = True

    def release_collection(self, collection_name: str) -> None:
        self.loaded = False

    def get_collection_stats(self, _name: str) -> dict[str, int]:
        if not self.loaded:
            raise RuntimeError("released")
        return {"row_count": EXPECTED_CHUNKS}

    def query(self, collection_name, filter="", output_fields=None, limit=1):
        if not self.loaded:
            raise RuntimeError("released")
        return [
            {
                "session_id": SOURCE_INDEX_SESSION_ID,
                "tenant_id": SOURCE_INDEX_TENANT_ID,
                "companies": "3M",
                "primary_company": "3M",
                "document_id": "3M_2022_10K",
            }
        ]

    def close(self) -> None:
        self.close_calls += 1


class RecordingReranker:
    def __init__(self, *, fallback_on_k: int | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self.fallback_on_k = fallback_on_k

    def rerank(self, query, hits, *, top_k):
        self.calls.append((len(hits), int(top_k)))
        gold = [hit for hit in hits if int(hit.get("page") or 0) == 2]
        rest = [hit for hit in hits if int(hit.get("page") or 0) != 2]
        ranked = gold + rest
        fallback = self.fallback_on_k is not None and len(hits) == self.fallback_on_k
        meta = {
            "rerank_provider": "lexical" if fallback else "dashscope",
            "rerank_model": "qwen3-rerank",
            "rerank_latency_ms": 1.5,
            "rerank_tokens": 12 if not fallback else 0,
            "rerank_fallback": fallback,
            "rerank_error_type": "timeout" if fallback else "",
            "rerank_error": LEAKY_RERANK_ERROR,
        }
        return ranked[: int(top_k)], meta


def _make_n_question_tree(root: Path, *, n: int = 150) -> Path:
    questions = []
    for index in range(1, n + 1):
        questions.append(
            {
                "financebench_id": f"financebench_id_{index:05d}",
                "company": "Acme",
                "doc_name": "ACME_2022_10K",
                "question_type": "metrics-generated",
                "question_reasoning": "Information extraction",
                "question": f"What is metric {index} for Acme?",
                "answer": "$1577.00",
                "justification": "Direct extraction.",
                "dataset_subset_label": "OPEN_SOURCE",
                "evidence": [
                    {
                        "evidence_doc_name": "ACME_2022_10K",
                        "evidence_page_num": 1,
                        "evidence_text": "ACME FY2022 capital expenditures were 1577 million USD.",
                    }
                ],
            }
        )
    documents = [
        {
            "doc_name": "ACME_2022_10K",
            "company": "Acme",
            "doc_type": "10K",
            "period": "FY2022",
            "gics_sector": "Industrials",
            "ticker": "ACME",
        }
    ]
    write_jsonl(root / "data" / "financebench_open_source.jsonl", questions)
    write_jsonl(root / "data" / "financebench_document_information.jsonl", documents)
    return root


def _write_depth_v2_per_case(repo: Path, rows: list[dict] | None = None, *, raw: str | None = None) -> Path:
    path = repo / "outputs" / "financebench_candidate_depth_test100_v2" / "per_case.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
        return path
    write_jsonl(path, rows or [])
    return path


def _artifact_blob(out: Path) -> str:
    parts: list[str] = []
    if not out.exists():
        return ""
    for path in sorted(out.rglob("*")):
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _test_case_ids(src: Path, *, expected_questions: int) -> list[str]:
    questions, _documents, _paths = load_financebench_dataset(
        src, expected_questions=expected_questions, require_pdfs=False
    )
    assignment = assign_splits(questions)
    return [question.case_id for question in questions_for_split(questions, assignment, "test")]


def _depth_focus_rows(case_ids: list[str], *, n_focus: int) -> list[dict]:
    rows: list[dict] = []
    for index, case_id in enumerate(case_ids):
        failure = "gold_rank_11_20" if index < n_focus else "hit_at_10"
        rows.append(
            {
                "case_id": case_id,
                "modes": {"hybrid_rrf": {"failure_class": failure}},
            }
        )
    return rows


class FinanceBenchCandidatePoolAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        global REMOTE_PROBE_COUNT
        REMOTE_PROBE_COUNT = 0

    def tearDown(self) -> None:
        self.assertEqual(REMOTE_PROBE_COUNT, 0)

    def _compat(self, eval_db: Path, dataset_hash: str = EXPECTED_DATASET_HASH) -> dict:
        return {
            "compatible": True,
            "compatible_index": {
                "compatible": True,
                "uri": str(eval_db),
                "dataset_hash": dataset_hash,
                "collection_name": "financebench_eval",
                "chunks": EXPECTED_CHUNKS,
                "embedding_model": "text-embedding-v4",
                "index_scope": "company",
                "source_index_session_id": SOURCE_INDEX_SESSION_ID,
                "source_index_tenant_id": SOURCE_INDEX_TENANT_ID,
                "source_index_chunker": SOURCE_INDEX_CHUNKER,
                "source_index_commit": SOURCE_INDEX_COMMIT,
                "zero_chunk_documents": [],
            },
        }

    def _prepare_run(self, repo: Path, *, n_questions: int = 4):
        src = (
            make_financebench_tree(repo / "src")
            if n_questions == 4
            else _make_n_question_tree(repo / "src", n=n_questions)
        )
        _questions, _documents, paths = load_financebench_dataset(
            src, expected_questions=n_questions, require_pdfs=False
        )
        dataset_hash = sha256_file(paths.questions_path)
        eval_db = _write_compatible_index(repo, dataset_hash=dataset_hash)
        out = repo / "outputs" / ABLATION_OUTPUT_DIRNAME
        return src, dataset_hash, eval_db, out

    def _run(
        self,
        *,
        repo: Path,
        src: Path,
        eval_db: Path,
        dataset_hash: str,
        out: Path,
        store: DepthStore | None,
        reranker: RecordingReranker,
        expected_questions: int,
        resume: bool = False,
        stop_after: int | None = None,
        skip_index_copy: bool = True,
        index_query_client: FakeIndexClient | None = None,
    ) -> dict:
        with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
            return run_candidate_pool_ablation(
                dataset_dir=src,
                output_dir=out,
                repo_root=repo,
                split="test",
                confirm_exposed_diagnostic=True,
                allow_remote=False,
                embedding_provider="deterministic",
                expected_questions=expected_questions,
                store=store,
                reranker=reranker,
                index_inspection=self._compat(eval_db, dataset_hash),
                skip_index_copy=skip_index_copy,
                require_clean_worktree=False,
                resume=resume,
                stop_after=stop_after,
                index_query_client=index_query_client,
            )

    def _assert_no_leaks(self, blob: str) -> None:
        for token in (
            "sk-testsecret123",
            "Authorization",
            "api_key=sk-",
            LEAKY_BASE_URL_A,
            LEAKY_BASE_URL_B,
            "dashscope.aliyuncs.com",
            '"results":[{"index":0',
            "req-leaked",
        ):
            self.assertNotIn(token, blob)

    def test_expected_call_budget_is_100_and_300(self) -> None:
        budget = expected_call_budget()
        self.assertEqual(budget["query_embedding_calls_expected"], 100)
        self.assertEqual(budget["qwen3_rerank_calls_expected"], 300)
        self.assertEqual(budget["chunk_reembed_calls_expected"], 0)

    def test_arm_pool_uses_top20_for_a_and_top50_for_bc(self) -> None:
        hits = _ranked_hits(n=50, gold_rank=25)
        pool_a = construct_arm_pool(bm25_hits=hits, dense_hits=hits, spec=ARM_SPECS["A"])
        pool_b = construct_arm_pool(bm25_hits=hits, dense_hits=hits, spec=ARM_SPECS["B"])
        pool_c = construct_arm_pool(bm25_hits=hits, dense_hits=hits, spec=ARM_SPECS["C"])
        self.assertEqual(len(pool_a), 20)
        self.assertEqual(len(pool_b), 20)
        self.assertEqual(len(pool_c), 30)
        gold_id = hits[24]["chunk_id"]
        self.assertNotIn(gold_id, {item["chunk_id"] for item in pool_a})
        self.assertNotIn(gold_id, {item["chunk_id"] for item in pool_b})
        self.assertIn(gold_id, {item["chunk_id"] for item in pool_c})

    def test_final_is_always_top10_and_c_can_rescue(self) -> None:
        hits = _ranked_hits(n=50, gold_rank=25)
        reranker = RecordingReranker()
        row = score_case_arms(
            _question(),
            bm25_hits=hits,
            dense_hits=hits,
            reranker=reranker,
            settings={
                "model": "qwen3-rerank",
                "instruct": DEFAULT_RERANK_INSTRUCT,
            },
            documents={},
            zero_chunk_names=set(),
            session_id=SOURCE_INDEX_SESSION_ID,
        )
        self.assertEqual(len(row["arms"]["A"]["final"]), 10)
        for name in ("A", "B", "C"):
            self.assertEqual(len(row["arms"][name]["final"]), 10)
            self.assertEqual(row["arms"][name]["final_k"], 10)
        self.assertEqual(reranker.calls, [(20, 10), (20, 10), (30, 10)])
        self.assertFalse(row["arms"]["A"]["gold_in_rerank_pool"])
        self.assertFalse(row["arms"]["B"]["gold_in_rerank_pool"])
        self.assertTrue(row["arms"]["C"]["gold_in_rerank_pool"])
        self.assertEqual(row["arms"]["A"]["scores"]["hit_at"]["10"], 0)
        self.assertEqual(row["arms"]["B"]["scores"]["hit_at"]["10"], 0)
        self.assertEqual(row["arms"]["C"]["scores"]["hit_at"]["10"], 1)
        self.assertEqual(row["hybrid_rrf50_first_gold_rank"], 25)
        self.assertTrue(row["in_rank_11_30_focus"])

    def test_fallback_is_recorded_per_arm(self) -> None:
        hits = _ranked_hits(n=50, gold_rank=1)
        reranker = RecordingReranker(fallback_on_k=30)
        row = score_case_arms(
            _question(),
            bm25_hits=hits,
            dense_hits=hits,
            reranker=reranker,
            settings={"model": "qwen3-rerank", "instruct": DEFAULT_RERANK_INSTRUCT},
            documents={},
            zero_chunk_names=set(),
            session_id=SOURCE_INDEX_SESSION_ID,
        )
        self.assertFalse(row["arms"]["A"]["rerank"]["fallback"])
        self.assertTrue(row["arms"]["C"]["rerank"]["fallback"])
        self.assertEqual(row["arms"]["C"]["rerank"]["error_type"], "timeout")
        self.assertFalse(row["arms"]["C"]["rerank"]["qwen3_ok"])
        self.assertTrue(row["arms"]["A"]["rerank"]["qwen3_ok"])
        self.assertNotIn("error", row["arms"]["C"]["rerank"])
        serialized = json.dumps(row)
        self.assertNotIn("rerank_error", serialized)
        self.assertNotIn("sk-testsecret123", serialized)

    def test_rank_11_30_rescue_counts(self) -> None:
        hits = _ranked_hits(n=50, gold_rank=25)
        row = score_case_arms(
            _question(),
            bm25_hits=hits,
            dense_hits=hits,
            reranker=RecordingReranker(),
            settings={"model": "qwen3-rerank", "instruct": DEFAULT_RERANK_INSTRUCT},
            documents={},
            zero_chunk_names=set(),
            session_id=SOURCE_INDEX_SESSION_ID,
        )
        focus = _focus_analysis([row], depth_ids=None)
        self.assertEqual(focus["focus_cases"], 1)
        self.assertEqual(focus["a_miss_b_hit"], 0)
        self.assertEqual(focus["a_miss_c_hit"], 1)
        self.assertEqual(focus["c_rescued_from_a_miss"], 1)

    def test_paired_stats_use_same_cases(self) -> None:
        hits = _ranked_hits(n=50, gold_rank=25)
        row = score_case_arms(
            _question(),
            bm25_hits=hits,
            dense_hits=hits,
            reranker=RecordingReranker(),
            settings={"model": "qwen3-rerank", "instruct": DEFAULT_RERANK_INSTRUCT},
            documents={},
            zero_chunk_names=set(),
            session_id=SOURCE_INDEX_SESSION_ID,
        )
        paired = _pair_report([row], baseline="A", candidate="C")
        self.assertEqual(paired["n"], 1)
        self.assertEqual(paired["delta_hit_at_10"], 1.0)
        self.assertIn("ci95_low", paired["paired_bootstrap"]["delta_hit_at_10"])
        self.assertEqual(paired["mcnemar"]["hit_at_10"]["candidate_only"], 1)
        self.assertEqual(paired["hit_at_10_movement"]["improved"], 1)

    def test_confirmation_dev_all_rejected(self) -> None:
        for split in ("confirmation", "dev", "all"):
            with self.assertRaises(SplitError):
                validate_ablation_request(
                    split=split,
                    confirm_exposed_diagnostic=True,
                    allow_remote=True,
                    embedding_provider="dashscope",
                    output_dir=ROOT / "outputs" / ABLATION_OUTPUT_DIRNAME,
                    repo_root=ROOT,
                )

    def test_candidate_depth_dirs_are_rejected(self) -> None:
        for name in (
            "financebench_candidate_depth_test100",
            "financebench_candidate_depth_test100_v2",
            "financebench_candidate_depth_test100_v2_preflight",
            "financebench_candidate_depth_test100_v2_preflight2",
        ):
            with self.assertRaises(AblationError):
                validate_ablation_request(
                    split="test",
                    confirm_exposed_diagnostic=True,
                    allow_remote=True,
                    embedding_provider="dashscope",
                    output_dir=ROOT / "outputs" / name,
                    repo_root=ROOT,
                )

    def test_full_run_one_dense_embed_three_reranks_and_no_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src = make_financebench_tree(repo / "src")
            questions, _documents, paths = load_financebench_dataset(
                src, expected_questions=4, require_pdfs=False
            )
            dataset_hash = sha256_file(paths.questions_path)
            eval_db = _write_compatible_index(repo, dataset_hash=dataset_hash)
            hits = _ranked_hits(n=50, gold_rank=1, gold_page=2)
            store = DepthStore(hits, hits)
            reranker = RecordingReranker()
            out = repo / "outputs" / ABLATION_OUTPUT_DIRNAME
            with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
                report = run_candidate_pool_ablation(
                    dataset_dir=src,
                    output_dir=out,
                    repo_root=repo,
                    split="test",
                    confirm_exposed_diagnostic=True,
                    allow_remote=False,
                    embedding_provider="deterministic",
                    expected_questions=4,
                    store=store,
                    reranker=reranker,
                    index_inspection=self._compat(eval_db, dataset_hash),
                    skip_index_copy=True,
                    require_clean_worktree=False,
                )
            n_cases = report["summary"]["cases"]
            self.assertGreaterEqual(n_cases, 1)
            self.assertEqual(store.vector_calls, n_cases)
            self.assertEqual(store.bm25_calls, n_cases)
            self.assertEqual(len(reranker.calls), n_cases * 3)
            self.assertEqual(set(store.sessions), {SOURCE_INDEX_SESSION_ID})
            self.assertEqual(set(store.top_ks), {50})
            self.assertTrue((out / "summary.json").is_file())
            self.assertTrue((out / "paired.json").is_file())
            self.assertTrue((out / "per_case.jsonl").is_file())
            self.assertTrue((out / "failures.jsonl").is_file())
            self.assertTrue((out / "results.md").is_file())
            self.assertTrue((out / "manifest.json").is_file())
            self.assertTrue((out / "environment.json").is_file())
            leaked = (out / "per_case.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("SECRET_QUESTION", leaked)
            self.assertNotIn("SECRET_EVIDENCE", leaked)
            self.assertNotIn("What is the FY2022", leaked)
            self.assertEqual(report["chunk_reembed_calls"], 0)
            self.assertEqual(report["experiment_role"], "exposed_test_100_post_hoc_ablation")
            self.assertEqual(report["query_embedding_calls_total"], n_cases)
            self.assertEqual(report["query_embedding_calls_this_invocation"], n_cases)
            self.assertEqual(report["qwen3_calls_total"], n_cases * 3)
            self.assertEqual(report["qwen3_calls_this_invocation"], n_cases * 3)
            self.assertTrue(report["primary_comparison_valid"])
            self.assertEqual(report["all_arms_qwen3_ok_cases"], n_cases)
            self.assertEqual(report["billing_semantics"], "at_least_once")
            self.assertFalse(report["exactly_once"])
            env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
            for item in env.get("credential_sources") or []:
                self.assertEqual(set(item), {"key", "source"})
            case_row = json.loads((out / "per_case.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("rerank_latency_ms", case_row["arms"]["A"]["rerank"])
            self.assertNotIn("latency_ms", case_row["arms"]["A"]["rerank"])
            self.assertIn("channel_retrieval_latency_ms", case_row)
            self._assert_no_leaks(_artifact_blob(out))

    def test_preflight_does_not_embed_or_rerank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            eval_db = _write_compatible_index(repo)
            out = repo / "outputs" / "financebench_candidate_pool_ablation_test100_preflight"
            reranker = RecordingReranker()
            store = DepthStore([], [])
            with patch.object(DashScopeEmbeddingProvider, "embed", side_effect=_count_remote):
                report = run_candidate_pool_ablation(
                    dataset_dir=repo,
                    output_dir=out,
                    repo_root=repo,
                    split="test",
                    confirm_exposed_diagnostic=True,
                    allow_remote=False,
                    embedding_provider="dashscope",
                    skip_index_copy=True,
                    store=store,
                    reranker=reranker,
                    index_inspection=self._compat(eval_db),
                    index_query_client=FakeIndexClient(),
                    require_clean_worktree=False,
                    preflight_only=True,
                )
            self.assertEqual(report["status"], "PREFLIGHT_OK")
            self.assertEqual(report["query_embedding_calls"], 0)
            self.assertEqual(report["query_embedding_calls_total"], 0)
            self.assertEqual(report["query_embedding_calls_this_invocation"], 0)
            self.assertEqual(report["query_embedding_calls_expected"], 100)
            self.assertEqual(report["qwen3_calls"], 0)
            self.assertEqual(report["qwen3_calls_total"], 0)
            self.assertEqual(report["qwen3_calls_this_invocation"], 0)
            self.assertEqual(report["qwen3_rerank_calls_expected"], 300)
            self.assertEqual(report["chunk_reembed_calls_expected"], 0)
            self.assertEqual(store.vector_calls, 0)
            self.assertEqual(reranker.calls, [])
            self.assertTrue((out / "preflight.json").is_file())
            self.assertFalse((out / "summary.json").is_file())
            for item in report["credential_sources"]:
                self.assertEqual(set(item), {"key", "source"})

    def test_resume_does_not_repeat_remote_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src = make_financebench_tree(repo / "src")
            questions, _documents, paths = load_financebench_dataset(
                src, expected_questions=4, require_pdfs=False
            )
            dataset_hash = sha256_file(paths.questions_path)
            eval_db = _write_compatible_index(repo, dataset_hash=dataset_hash)
            hits = _ranked_hits(n=50, gold_rank=1)
            out = repo / "outputs" / ABLATION_OUTPUT_DIRNAME
            first_store = DepthStore(hits, hits)
            first_reranker = RecordingReranker()
            run_candidate_pool_ablation(
                dataset_dir=src,
                output_dir=out,
                repo_root=repo,
                split="test",
                confirm_exposed_diagnostic=True,
                allow_remote=False,
                embedding_provider="deterministic",
                expected_questions=4,
                store=first_store,
                reranker=first_reranker,
                index_inspection=self._compat(eval_db, dataset_hash),
                skip_index_copy=True,
                require_clean_worktree=False,
            )
            second_store = DepthStore(hits, hits)
            second_reranker = RecordingReranker()
            second = run_candidate_pool_ablation(
                dataset_dir=src,
                output_dir=out,
                repo_root=repo,
                split="test",
                confirm_exposed_diagnostic=True,
                allow_remote=False,
                embedding_provider="deterministic",
                expected_questions=4,
                store=second_store,
                reranker=second_reranker,
                index_inspection=self._compat(eval_db, dataset_hash),
                skip_index_copy=True,
                require_clean_worktree=False,
                resume=True,
            )
            self.assertEqual(second_store.vector_calls, 0)
            self.assertEqual(second_reranker.calls, [])
            n_cases = first_store.vector_calls
            self.assertEqual(second["query_embedding_calls_this_invocation"], 0)
            self.assertEqual(second["qwen3_calls_this_invocation"], 0)
            self.assertEqual(second["query_embedding_calls_total"], n_cases)
            self.assertEqual(second["qwen3_calls_total"], n_cases * 3)

    def test_existing_output_dir_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            out = repo / "outputs" / ABLATION_OUTPUT_DIRNAME
            out.mkdir(parents=True)
            sentinel = out / "summary.json"
            sentinel.write_text("DO_NOT_TOUCH", encoding="utf-8")
            eval_db = _write_compatible_index(repo)
            with self.assertRaises(AblationError):
                run_candidate_pool_ablation(
                    dataset_dir=repo,
                    output_dir=out,
                    repo_root=repo,
                    split="test",
                    confirm_exposed_diagnostic=True,
                    allow_remote=False,
                    embedding_provider="deterministic",
                    store=DepthStore([], []),
                    index_inspection=self._compat(eval_db),
                    skip_index_copy=True,
                    require_clean_worktree=False,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "DO_NOT_TOUCH")

    def test_cli_rejects_confirmation_and_missing_flags(self) -> None:
        cli = _load_cli()
        with patch.object(cli, "run_candidate_pool_ablation") as run:
            self.assertEqual(cli.main(["--split", "confirmation", "--confirm-exposed-diagnostic", "--allow-remote"]), 2)
            self.assertEqual(cli.main(["--split", "test", "--confirm-exposed-diagnostic"]), 2)
        run.assert_not_called()

    def test_production_defaults_unchanged(self) -> None:
        signature = inspect.signature(HybridEvidenceRetriever.__init__)
        self.assertEqual(signature.parameters["rerank_candidates"].default, 20)
        self.assertEqual(DEFAULT_RERANK_CANDIDATES, 20)
        self.assertEqual(DEFAULT_TOP_K, 10)
        source = Path(ROOT / "src" / "lumenfin" / "config.py").read_text(encoding="utf-8")
        self.assertIn('MAS_RAG_RERANK_CANDIDATES", "20"', source)
        self.assertIn('MAS_RAG_TOP_K", "5"', source)
        self.assertIn("qwen3-rerank", source)

    def test_frozen_config_hash_unchanged(self) -> None:
        cfg = load_frozen_config(ROOT / "data" / "eval_rag" / "financebench" / "frozen_config.json")
        self.assertEqual(cfg.config_hash, PUBLISHED_CONFIG_HASH)
        self.assertEqual(
            PUBLISHED_CONFIG_HASH,
            "18a483f604f3a5420264e746d9219e77e3c9bddbd91c5c50252025b40ccb1ee7",
        )

    def test_remote_probe_count_stays_zero(self) -> None:
        self.assertEqual(REMOTE_PROBE_COUNT, 0)

    def test_resume_50_then_100_accumulates_100_and_300(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo, n_questions=150)
            hits = _ranked_hits(n=50, gold_rank=1)
            first_store = DepthStore(hits, hits)
            first_reranker = RecordingReranker()
            first = self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=first_store,
                reranker=first_reranker,
                expected_questions=150,
                stop_after=50,
            )
            self.assertEqual(first["status"], "interrupted")
            self.assertEqual(first["query_embedding_calls_this_invocation"], 50)
            self.assertEqual(first["query_embedding_calls_total"], 50)
            self.assertEqual(first["qwen3_calls_this_invocation"], 150)
            self.assertEqual(first["qwen3_calls_total"], 150)
            second_store = DepthStore(hits, hits)
            second_reranker = RecordingReranker()
            second = self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=second_store,
                reranker=second_reranker,
                expected_questions=150,
                resume=True,
            )
            self.assertEqual(second["status"], "recorded")
            self.assertEqual(second_store.vector_calls, 50)
            self.assertEqual(len(second_reranker.calls), 150)
            self.assertEqual(second["query_embedding_calls_this_invocation"], 50)
            self.assertEqual(second["query_embedding_calls_total"], 100)
            self.assertEqual(second["qwen3_calls_this_invocation"], 150)
            self.assertEqual(second["qwen3_calls_total"], 300)
            self.assertEqual(second["query_embedding_calls_expected"], 100)
            self.assertEqual(second["qwen3_calls_expected"], 300)

    def test_resume_complete_100_makes_zero_new_remote_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo, n_questions=150)
            hits = _ranked_hits(n=50, gold_rank=1)
            first = self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=150,
            )
            self.assertEqual(first["query_embedding_calls_total"], 100)
            self.assertEqual(first["qwen3_calls_total"], 300)
            second_store = DepthStore(hits, hits)
            second_reranker = RecordingReranker()
            second = self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=second_store,
                reranker=second_reranker,
                expected_questions=150,
                resume=True,
            )
            self.assertEqual(second_store.vector_calls, 0)
            self.assertEqual(second_reranker.calls, [])
            self.assertEqual(second["query_embedding_calls_this_invocation"], 0)
            self.assertEqual(second["qwen3_calls_this_invocation"], 0)
            self.assertEqual(second["query_embedding_calls_total"], 100)
            self.assertEqual(second["qwen3_calls_total"], 300)

    def test_resume_mismatch_fails_before_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            hits = _ranked_hits(n=50, gold_rank=1)
            self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=4,
            )
            checkpoint = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
            checkpoint["complete_case_ids"] = ["fb-not-a-real-case"]
            checkpoint["completed_case_ids"] = ["fb-not-a-real-case"]
            checkpoint["completed_cases"] = 1
            (out / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
            second_store = DepthStore(hits, hits)
            second_reranker = RecordingReranker()
            with self.assertRaises(AblationError):
                self._run(
                    repo=repo,
                    src=src,
                    eval_db=eval_db,
                    dataset_hash=dataset_hash,
                    out=out,
                    store=second_store,
                    reranker=second_reranker,
                    expected_questions=4,
                    resume=True,
                )
            self.assertEqual(second_store.vector_calls, 0)
            self.assertEqual(second_reranker.calls, [])

    def test_fallback_is_not_valid_primary_qwen3_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo, n_questions=150)
            hits = _ranked_hits(n=50, gold_rank=1)
            report = self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(fallback_on_k=30),
                expected_questions=150,
            )
            self.assertFalse(report["primary_comparison_valid"])
            self.assertEqual(report["comparison_status"], "degraded_descriptive")
            self.assertEqual(report["paired"]["C_vs_A"]["n_descriptive"], 100)
            self.assertEqual(report["paired"]["C_vs_A"]["n"], 100)
            self.assertEqual(report["paired"]["C_vs_A"]["qwen3_ok_complete_cases"], 0)
            self.assertEqual(report["paired"]["B_vs_A"]["qwen3_ok_complete_cases"], 100)
            self.assertFalse(report["paired"]["B_vs_A"]["primary_comparison_valid"])
            markdown = (out / "results.md").read_text(encoding="utf-8")
            self.assertIn("degraded/descriptive", markdown)
            self.assertIn("not** a valid Qwen3 ablation conclusion", markdown)
            self.assertEqual(len(json.loads((out / "per_case.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])["arms"]), 3)

    def test_zero_fallback_paired_n_is_100(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo, n_questions=150)
            hits = _ranked_hits(n=50, gold_rank=1)
            report = self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=150,
            )
            self.assertTrue(report["primary_comparison_valid"])
            self.assertEqual(report["all_arms_qwen3_ok_cases"], 100)
            for key in ("B_vs_A", "C_vs_A", "C_vs_B"):
                pair = report["paired"][key]
                self.assertEqual(pair["n"], 100)
                self.assertEqual(pair["qwen3_ok_complete_cases"], 100)
                self.assertTrue(pair["primary_comparison_valid"])
                self.assertEqual(pair["comparison_status"], "qwen3_primary")

    def test_base_url_change_fails_resume_and_is_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            hits = _ranked_hits(n=50, gold_rank=1)
            with patch.dict(os.environ, {"DASHSCOPE_RERANK_BASE_URL": LEAKY_BASE_URL_A}):
                self._run(
                    repo=repo,
                    src=src,
                    eval_db=eval_db,
                    dataset_hash=dataset_hash,
                    out=out,
                    store=DepthStore(hits, hits),
                    reranker=RecordingReranker(),
                    expected_questions=4,
                )
            self._assert_no_leaks(_artifact_blob(out))
            second_store = DepthStore(hits, hits)
            with patch.dict(os.environ, {"DASHSCOPE_RERANK_BASE_URL": LEAKY_BASE_URL_B}):
                with self.assertRaises(AblationError) as raised:
                    self._run(
                        repo=repo,
                        src=src,
                        eval_db=eval_db,
                        dataset_hash=dataset_hash,
                        out=out,
                        store=second_store,
                        reranker=RecordingReranker(),
                        expected_questions=4,
                        resume=True,
                    )
            self.assertEqual(second_store.vector_calls, 0)
            self.assertNotIn(LEAKY_BASE_URL_A, str(raised.exception))
            self.assertNotIn(LEAKY_BASE_URL_B, str(raised.exception))
            self._assert_no_leaks(_artifact_blob(out))

    def test_reranker_uses_snapshot_not_later_env(self) -> None:
        with patch.dict(os.environ, {"DASHSCOPE_RERANK_BASE_URL": LEAKY_BASE_URL_A}):
            settings = snapshot_rerank_settings()
        with patch.dict(os.environ, {"DASHSCOPE_RERANK_BASE_URL": LEAKY_BASE_URL_B}):
            with patch(
                "lumenfin.eval.financebench.candidate_pool_ablation.build_reranker"
            ) as mock_build:
                mock_build.return_value = object()
                build_locked_qwen3_reranker(settings)
            self.assertEqual(mock_build.call_args.kwargs["base_url"], LEAKY_BASE_URL_A.rstrip("/"))
            later = snapshot_rerank_settings()
            self.assertNotEqual(
                public_rerank_settings(settings)["base_url_sha256"],
                public_rerank_settings(later)["base_url_sha256"],
            )

    def test_focus_file_hash_and_count_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            case_ids = _test_case_ids(src, expected_questions=4)
            focus_path = _write_depth_v2_per_case(repo, _depth_focus_rows(case_ids, n_focus=1))
            digest = sha256_file(focus_path)
            hits = _ranked_hits(n=50, gold_rank=1)
            report = self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=4,
            )
            focus = report["summary"]["rank_11_30"]
            self.assertEqual(focus["source_status"], "candidate_depth_v2")
            self.assertEqual(focus["per_case_sha256"], digest)
            self.assertEqual(focus["focus_case_count"], 1)
            expected_ids_hash = hashlib.sha256(
                json.dumps(sorted([case_ids[0]]), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            self.assertEqual(focus["focus_case_ids_sha256"], expected_ids_hash)
            env_cfg = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(env_cfg["config"]["focus"]["per_case_sha256"], digest)

    def test_corrupt_focus_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            _write_depth_v2_per_case(repo, raw="{not-json\n")
            store = DepthStore(_ranked_hits(n=50, gold_rank=1), _ranked_hits(n=50, gold_rank=1))
            with self.assertRaises(AblationError) as raised:
                self._run(
                    repo=repo,
                    src=src,
                    eval_db=eval_db,
                    dataset_hash=dataset_hash,
                    out=out,
                    store=store,
                    reranker=RecordingReranker(),
                    expected_questions=4,
                )
            self.assertIn("refusing silent recompute", str(raised.exception))
            self.assertEqual(store.vector_calls, 0)
            self.assertFalse((out / "summary.json").is_file())

    def test_cli_rejects_non_locked_output_dir(self) -> None:
        cli = _load_cli()
        with patch.object(cli, "run_candidate_pool_ablation") as run:
            self.assertEqual(
                cli.main(
                    [
                        "--split",
                        "test",
                        "--confirm-exposed-diagnostic",
                        "--allow-remote",
                        "--output-dir",
                        str(ROOT / "tmp" / "ablation-not-locked"),
                    ]
                ),
                2,
            )
        run.assert_not_called()
        with self.assertRaises(AblationError):
            validate_ablation_request(
                split="test",
                confirm_exposed_diagnostic=True,
                allow_remote=True,
                embedding_provider="dashscope",
                output_dir=ROOT / "tmp" / "ablation-not-locked",
                repo_root=ROOT,
                enforce_locked_output_dir=True,
            )

    def test_production_rag_files_do_not_reference_ablation(self) -> None:
        for rel in PRODUCTION_RAG_FILES:
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("candidate_pool_ablation", text)

    def test_locked_output_subdir_is_rejected(self) -> None:
        nested = ROOT / "outputs" / ABLATION_OUTPUT_DIRNAME / "another-run"
        self.assertFalse(is_locked_ablation_output_path(nested, repo_root=ROOT))
        with self.assertRaises(AblationError):
            validate_ablation_request(
                split="test",
                confirm_exposed_diagnostic=True,
                allow_remote=True,
                embedding_provider="dashscope",
                output_dir=nested,
                repo_root=ROOT,
                enforce_locked_output_dir=True,
            )

    def test_resume_reuses_existing_index_copy_without_injected_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            hits = _ranked_hits(n=50, gold_rank=1)
            stores: list[DepthStore] = []

            def _build_store(**_kwargs):
                store = DepthStore(hits, hits)
                stores.append(store)
                return store

            with patch(
                "lumenfin.eval.financebench.candidate_pool_ablation.copy_index_for_query",
                wraps=copy_index_for_query,
            ) as spy_copy:
                with patch(
                    "lumenfin.eval.financebench.retrieval.build_eval_store",
                    side_effect=_build_store,
                ):
                    first = self._run(
                        repo=repo,
                        src=src,
                        eval_db=eval_db,
                        dataset_hash=dataset_hash,
                        out=out,
                        store=None,
                        reranker=RecordingReranker(),
                        expected_questions=4,
                        skip_index_copy=False,
                        index_query_client=FakeIndexClient(),
                        stop_after=1,
                    )
                    self.assertEqual(first["status"], "interrupted")
                    self.assertTrue((out / INDEX_WORK_DIRNAME / INDEX_COPY_DBNAME).exists())
                    self.assertEqual(spy_copy.call_count, 1)
                    second = self._run(
                        repo=repo,
                        src=src,
                        eval_db=eval_db,
                        dataset_hash=dataset_hash,
                        out=out,
                        store=None,
                        reranker=RecordingReranker(),
                        expected_questions=4,
                        skip_index_copy=False,
                        index_query_client=FakeIndexClient(),
                        resume=True,
                    )
                    self.assertEqual(spy_copy.call_count, 1)
            self.assertEqual(second["status"], "recorded")
            self.assertGreaterEqual(len(stores), 2)
            self.assertGreater(stores[1].vector_calls, 0)

    def _progress_ids(self, out: Path, name: str) -> list[str]:
        payload = json.loads((out / name).read_text(encoding="utf-8"))
        return list(payload.get("complete_case_ids") or payload.get("completed_case_ids") or [])

    def _completed_case_ids(self, out: Path) -> list[str]:
        return [
            json.loads(line)["case_id"]
            for line in (out / "per_case.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _rewrite_progress_ids(self, out: Path, name: str, ids: list[str]) -> None:
        payload = json.loads((out / name).read_text(encoding="utf-8"))
        payload["complete_case_ids"] = list(ids)
        payload["completed_case_ids"] = list(ids)
        payload["completed_cases"] = len(ids)
        payload["cases"] = len(ids)
        payload["query_embedding_calls"] = len(ids)
        payload["query_embedding_calls_total"] = len(ids)
        payload["qwen3_calls"] = len(ids) * 3
        payload["qwen3_calls_total"] = len(ids) * 3
        (out / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _drop_progress_keys(self, out: Path, name: str, *keys: str) -> None:
        payload = json.loads((out / name).read_text(encoding="utf-8"))
        for key in keys:
            payload.pop(key, None)
        (out / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _patch_progress(self, out: Path, name: str, **updates: object) -> None:
        payload = json.loads((out / name).read_text(encoding="utf-8"))
        payload.update(updates)
        (out / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _assert_resume_fails_closed_before_work(
        self,
        *,
        repo: Path,
        src: Path,
        eval_db: Path,
        dataset_hash: str,
        out: Path,
        expected_questions: int = 4,
        substring: str | None = None,
    ) -> str:
        reranker = RecordingReranker()
        remote_before = REMOTE_PROBE_COUNT
        with patch(
            "lumenfin.eval.financebench.candidate_pool_ablation.copy_index_for_query",
        ) as spy_copy:
            with patch(
                "lumenfin.eval.financebench.retrieval.build_eval_store",
            ) as spy_store:
                with self.assertRaises(AblationError) as raised:
                    self._run(
                        repo=repo,
                        src=src,
                        eval_db=eval_db,
                        dataset_hash=dataset_hash,
                        out=out,
                        store=None,
                        reranker=reranker,
                        expected_questions=expected_questions,
                        skip_index_copy=False,
                        index_query_client=FakeIndexClient(),
                        resume=True,
                    )
        message = str(raised.exception)
        if substring is not None:
            self.assertIn(substring, message)
        self.assertEqual(spy_copy.call_count, 0)
        spy_store.assert_not_called()
        self.assertEqual(reranker.calls, [])
        self.assertEqual(REMOTE_PROBE_COUNT, remote_before)
        return message

    def test_resume_recovers_when_per_case_is_ahead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            hits = _ranked_hits(n=50, gold_rank=1)
            self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=4,
                stop_after=2,
            )
            per_case_ids = [
                json.loads(line)["case_id"]
                for line in (out / "per_case.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(per_case_ids), 2)
            self._rewrite_progress_ids(out, "checkpoint.json", per_case_ids[:1])
            self._rewrite_progress_ids(out, "manifest.json", per_case_ids[:1])
            second_store = DepthStore(hits, hits)
            second = self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=second_store,
                reranker=RecordingReranker(),
                expected_questions=4,
                resume=True,
            )
            recovered = set(self._progress_ids(out, "checkpoint.json"))
            self.assertTrue(set(per_case_ids).issubset(recovered))
            n_test = len(_test_case_ids(src, expected_questions=4))
            self.assertEqual(second_store.vector_calls, max(0, n_test - 2))
            self.assertEqual(second["query_embedding_calls_this_invocation"], second_store.vector_calls)

    def test_resume_recovers_when_manifest_lags_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            hits = _ranked_hits(n=50, gold_rank=1)
            self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=4,
                stop_after=2,
            )
            per_case_ids = [
                json.loads(line)["case_id"]
                for line in (out / "per_case.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self._rewrite_progress_ids(out, "manifest.json", per_case_ids[:1])
            second_store = DepthStore(hits, hits)
            second = self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=second_store,
                reranker=RecordingReranker(),
                expected_questions=4,
                resume=True,
            )
            manifest_ids = set(self._progress_ids(out, "manifest.json"))
            checkpoint_ids = set(self._progress_ids(out, "checkpoint.json"))
            self.assertEqual(manifest_ids, checkpoint_ids)
            self.assertTrue(set(per_case_ids).issubset(manifest_ids))
            self.assertIn(second["status"], {"recorded", "interrupted"})
            n_test = len(_test_case_ids(src, expected_questions=4))
            self.assertEqual(second_store.vector_calls, max(0, n_test - 2))
            self.assertGreaterEqual(len(manifest_ids), 2)

    def test_resume_refuses_divergent_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            hits = _ranked_hits(n=50, gold_rank=1)
            self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=4,
                stop_after=1,
            )
            per_case_ids = [
                json.loads(line)["case_id"]
                for line in (out / "per_case.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self._rewrite_progress_ids(out, "checkpoint.json", ["fb-not-in-per-case"])
            self._rewrite_progress_ids(out, "manifest.json", ["fb-not-in-per-case"])
            second_store = DepthStore(hits, hits)
            with self.assertRaises(AblationError) as raised:
                self._run(
                    repo=repo,
                    src=src,
                    eval_db=eval_db,
                    dataset_hash=dataset_hash,
                    out=out,
                    store=second_store,
                    reranker=RecordingReranker(),
                    expected_questions=4,
                    resume=True,
                )
            self.assertIn("diverge", str(raised.exception))
            self.assertEqual(second_store.vector_calls, 0)

    def test_resume_refuses_pairwise_divergent_manifest_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            hits = _ranked_hits(n=50, gold_rank=1)
            self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=4,
                stop_after=2,
            )
            per_case_ids = self._completed_case_ids(out)
            self.assertGreaterEqual(len(per_case_ids), 2)
            self._rewrite_progress_ids(out, "manifest.json", per_case_ids[:1])
            self._rewrite_progress_ids(out, "checkpoint.json", per_case_ids[1:2])
            self._assert_resume_fails_closed_before_work(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                substring="diverge",
            )

    def test_resume_refuses_missing_manifest_completed_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            hits = _ranked_hits(n=50, gold_rank=1)
            self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=4,
                stop_after=1,
            )
            self._drop_progress_keys(
                out, "manifest.json", "complete_case_ids", "completed_case_ids"
            )
            self._assert_resume_fails_closed_before_work(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                substring="missing completed case IDs",
            )

    def test_resume_refuses_checkpoint_completed_ids_not_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            hits = _ranked_hits(n=50, gold_rank=1)
            self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=4,
                stop_after=1,
            )
            self._patch_progress(
                out,
                "checkpoint.json",
                complete_case_ids="not-a-list",
                completed_case_ids="not-a-list",
            )
            self._assert_resume_fails_closed_before_work(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                substring="completed case IDs are invalid",
            )

    def test_resume_refuses_completed_cases_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            hits = _ranked_hits(n=50, gold_rank=1)
            self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=4,
                stop_after=1,
            )
            ids = self._progress_ids(out, "checkpoint.json")
            self._patch_progress(out, "checkpoint.json", completed_cases=len(ids) + 1)
            self._assert_resume_fails_closed_before_work(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                substring="completed_cases does not match",
            )

    def test_focus_99_cases_fails_closed(self) -> None:
        self._assert_focus_lock_fails(n_focus=25, drop_last=True)

    def test_focus_wrong_ids_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo, n_questions=150)
            case_ids = _test_case_ids(src, expected_questions=150)
            wrong = [f"fb-wrong-{index:05d}" for index in range(len(case_ids))]
            _write_depth_v2_per_case(repo, _depth_focus_rows(wrong, n_focus=EXPECTED_FOCUS_CASES))
            store = DepthStore(_ranked_hits(n=50, gold_rank=1), _ranked_hits(n=50, gold_rank=1))
            with self.assertRaises(AblationError) as raised:
                self._run(
                    repo=repo,
                    src=src,
                    eval_db=eval_db,
                    dataset_hash=dataset_hash,
                    out=out,
                    store=store,
                    reranker=RecordingReranker(),
                    expected_questions=150,
                )
            self.assertIn("do not match the current test split", str(raised.exception))
            self.assertEqual(store.vector_calls, 0)

    def test_focus_24_and_26_fail_closed(self) -> None:
        for n_focus in (24, 26):
            self._assert_focus_lock_fails(n_focus=n_focus)

    def _assert_focus_lock_fails(self, *, n_focus: int, drop_last: bool = False) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo, n_questions=150)
            case_ids = _test_case_ids(src, expected_questions=150)
            rows_ids = case_ids[:-1] if drop_last else case_ids
            _write_depth_v2_per_case(repo, _depth_focus_rows(rows_ids, n_focus=n_focus))
            store = DepthStore(_ranked_hits(n=50, gold_rank=1), _ranked_hits(n=50, gold_rank=1))
            with self.assertRaises(AblationError) as raised:
                self._run(
                    repo=repo,
                    src=src,
                    eval_db=eval_db,
                    dataset_hash=dataset_hash,
                    out=out,
                    store=store,
                    reranker=RecordingReranker(),
                    expected_questions=150,
                )
            self.assertIn("refusing silent recompute", str(raised.exception))
            self.assertEqual(store.vector_calls, 0)

    def test_resume_refuses_changed_focus_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src, dataset_hash, eval_db, out = self._prepare_run(repo)
            case_ids = _test_case_ids(src, expected_questions=4)
            _write_depth_v2_per_case(repo, _depth_focus_rows(case_ids, n_focus=1))
            hits = _ranked_hits(n=50, gold_rank=1)
            self._run(
                repo=repo,
                src=src,
                eval_db=eval_db,
                dataset_hash=dataset_hash,
                out=out,
                store=DepthStore(hits, hits),
                reranker=RecordingReranker(),
                expected_questions=4,
                stop_after=1,
            )
            path = repo / "outputs" / "financebench_candidate_depth_test100_v2" / "per_case.jsonl"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            second_store = DepthStore(hits, hits)
            with self.assertRaises(AblationError) as raised:
                self._run(
                    repo=repo,
                    src=src,
                    eval_db=eval_db,
                    dataset_hash=dataset_hash,
                    out=out,
                    store=second_store,
                    reranker=RecordingReranker(),
                    expected_questions=4,
                    resume=True,
                )
            self.assertIn("mismatch", str(raised.exception))
            self.assertEqual(second_store.vector_calls, 0)


if __name__ == "__main__":
    unittest.main()
