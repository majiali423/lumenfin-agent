"""Exposed test-100 candidate-pool / Qwen3 paired ablation (eval-only).

Compares three locked arms on the already-exposed test-100. Not held-out,
not product accuracy, and not a new FinanceBench score. Does not change
production retriever defaults or re-embed the historical chunk corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ...env_bootstrap import describe_credential_sources
from ...rag.hybrid_retriever import reciprocal_rank_fusion
from ...rag.rerank import DEFAULT_RERANK_INSTRUCT, build_reranker
from .candidate_depth import (
    CandidateDepthError,
    assert_per_case_redacted,
    copy_index_for_query,
    first_gold_rank,
    gold_pages_for_scoring,
    is_failed_preflight_output_path,
    is_first_attempt_output_path,
    is_scoring_output_path,
    require_clean_diagnostic_worktree,
    require_fresh_output_dir,
)
from .constants import (
    DEFAULT_BM25_RRF_WEIGHT,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_DASHSCOPE_EMBEDDING_DIM,
    EXPECTED_OPEN_SOURCE_QUESTIONS,
    REMOTE_EMBEDDING_PROVIDERS,
    SPLIT_TEST_SIZE,
)
from .index_inspect import (
    EXPECTED_CHUNKS,
    SOURCE_INDEX_CHUNKER,
    SOURCE_INDEX_COMMIT,
    SOURCE_INDEX_WORKTREE_DIRTY,
    IndexIncompatibleError,
    inspect_financebench_indexes,
    is_historical_output_path,
    require_compatible_index,
)
from .index_session import (
    DEFAULT_CANARY_COMPANIES,
    EMPTY_RETRIEVAL_FAIL_FAST,
    FAILED_PREFLIGHT_OUTPUT_DIRNAME,
    FIRST_ATTEMPT_OUTPUT_DIRNAME,
    FORBIDDEN_QUERY_SESSION_ID,
    LOCKED_OUTPUT_DIRNAME as DEPTH_SCORING_DIRNAME,
    LOCKED_PREFLIGHT_OUTPUT_DIRNAME as DEPTH_PREFLIGHT2_DIRNAME,
    SOURCE_INDEX_SESSION_ID,
    IndexSessionError,
    resolve_query_session_id,
    resolve_source_scope,
    verify_copied_index_scope,
)
from .loader import load_financebench_dataset, normalize_doc_name
from .metrics import hit_at_k, mean, mean_reciprocal_rank, ndcg_at_k, percentile_sorted, recall_at_k
from .paired import compare_paired_systems
from .qrels import gold_pages_for, retrieved_page_keys
from .reporting import (
    assert_no_secrets,
    environment_payload,
    read_jsonl,
    redact_mapping,
    sha256_file,
    write_json,
    write_jsonl,
)
from .retrieval import require_allow_remote
from .schema import DocumentInfo, FinanceBenchQuestion
from .split import SplitError, assign_splits, experiment_governance, questions_for_split

ABLATION_SCHEMA = "financebench_candidate_pool_ablation.v1"
EXPERIMENT_ROLE = "exposed_test_100_post_hoc_ablation"
LOCKED_SPLIT = "test"
LOCKED_INDEX_SCOPE = "company"
LOCKED_EMBEDDING_PROVIDER = "dashscope"
LOCKED_EMBEDDING_MODEL_NAME = "text-embedding-v4"
LOCKED_EMBEDDING_DIMENSION = DEFAULT_DASHSCOPE_EMBEDDING_DIM
LOCKED_DENSE_RRF_WEIGHT = 1.0
LOCKED_BM25_RRF_WEIGHT = DEFAULT_BM25_RRF_WEIGHT
LOCKED_CHANNEL_FETCH_K = 50
LOCKED_FINAL_K = 10
ABLATION_OUTPUT_DIRNAME = "financebench_candidate_pool_ablation_test100"
ABLATION_PREFLIGHT_DIRNAME = "financebench_candidate_pool_ablation_test100_preflight"
INDEX_WORK_DIRNAME = "_index_work"
INDEX_COPY_DBNAME = "eval.db"
CANDIDATE_DEPTH_V2_PER_CASE = Path("outputs") / "financebench_candidate_depth_test100_v2" / "per_case.jsonl"
FOCUS_FAILURE_CLASSES = frozenset({"gold_rank_11_20", "gold_rank_21_30"})
EXPECTED_FOCUS_CASES = 25
EXPECTED_QUERY_EMBEDDING_CALLS = SPLIT_TEST_SIZE
EXPECTED_QWEN3_CALLS = SPLIT_TEST_SIZE * 3
EXPECTED_CHUNK_REEMBED_CALLS = 0
BILLING_SEMANTICS = "at_least_once"
PUBLIC_RERANK_SETTING_KEYS = (
    "model",
    "instruct",
    "timeout_seconds",
    "max_attempts",
    "backoff_seconds",
    "max_inflight",
    "max_document_chars",
    "base_url_sha256",
)
_SAFE_ERROR_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_LEAK_RE = re.compile(r"https?://|\bsk-[A-Za-z0-9_-]+|Authorization|Bearer\s", re.IGNORECASE)
_PAIR_KEYS = (("B_vs_A", "A", "B"), ("C_vs_A", "A", "C"), ("C_vs_B", "B", "C"))


@dataclass(frozen=True)
class ArmSpec:
    name: str
    channel_k: int
    rrf_k: int
    rerank_k: int
    final_k: int = LOCKED_FINAL_K


ARM_SPECS: dict[str, ArmSpec] = {
    "A": ArmSpec(name="A", channel_k=20, rrf_k=20, rerank_k=20, final_k=LOCKED_FINAL_K),
    "B": ArmSpec(name="B", channel_k=50, rrf_k=20, rerank_k=20, final_k=LOCKED_FINAL_K),
    "C": ArmSpec(name="C", channel_k=50, rrf_k=30, rerank_k=30, final_k=LOCKED_FINAL_K),
}
ARM_ORDER = ("A", "B", "C")


class AblationError(CandidateDepthError):
    """Raised when the candidate-pool ablation request is invalid."""


class InvalidEmptyRetrievalError(AblationError):
    def __init__(self, message: str, *, query_embedding_calls: int = 0) -> None:
        super().__init__(message)
        self.query_embedding_calls = int(query_embedding_calls)


def _env_text(env: Mapping[str, str], key: str, default: str) -> str:
    raw = env.get(key)
    if raw is None:
        return default
    return str(raw)


def _env_number(env: Mapping[str, str], key: str, default: str, caster):
    raw = env.get(key, default)
    try:
        return caster(raw)
    except (TypeError, ValueError) as exc:
        raise AblationError(f"invalid {key}") from exc


def normalize_rerank_base_url(raw: str) -> str:
    return str(raw or "").strip().rstrip("/")


def rerank_base_url_sha256(raw: str) -> str:
    return hashlib.sha256(normalize_rerank_base_url(raw).encode("utf-8")).hexdigest()


def snapshot_rerank_settings(*, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Capture reranker settings once. Later env changes must not affect construction."""
    env = os.environ if environ is None else environ
    normalized = normalize_rerank_base_url(_env_text(env, "DASHSCOPE_RERANK_BASE_URL", ""))
    return {
        "model": (_env_text(env, "DASHSCOPE_RERANK_MODEL", "qwen3-rerank").strip() or "qwen3-rerank"),
        "instruct": (_env_text(env, "MAS_RAG_RERANK_INSTRUCT", DEFAULT_RERANK_INSTRUCT).strip() or DEFAULT_RERANK_INSTRUCT),
        "timeout_seconds": _env_number(env, "MAS_RAG_RERANK_TIMEOUT_SECONDS", "12", float),
        "max_attempts": _env_number(env, "MAS_RAG_RERANK_MAX_ATTEMPTS", "2", int),
        "backoff_seconds": _env_number(env, "MAS_RAG_RERANK_BACKOFF_SECONDS", "0.25", float),
        "max_inflight": _env_number(env, "MAS_RAG_RERANK_MAX_INFLIGHT_PER_PROCESS", "2", int),
        "max_document_chars": _env_number(env, "MAS_RAG_RERANK_MAX_DOCUMENT_CHARS", "4000", int),
        "base_url_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "_base_url": normalized,
    }


def public_rerank_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {key: settings[key] for key in PUBLIC_RERANK_SETTING_KEYS}


def resolved_rerank_settings(*, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    return public_rerank_settings(snapshot_rerank_settings(environ=environ))


def build_locked_qwen3_reranker(settings: Mapping[str, Any]) -> Any:
    """Construct Qwen3 from the locked snapshot; never re-read process env."""
    try:
        return build_reranker(
            "qwen3",
            model=str(settings["model"]),
            base_url=str(settings.get("_base_url") or ""),
            instruct=str(settings["instruct"]),
            timeout_seconds=float(settings["timeout_seconds"]),
            max_attempts=int(settings["max_attempts"]),
            backoff_seconds=float(settings["backoff_seconds"]),
            max_inflight=int(settings["max_inflight"]),
            max_document_chars=int(settings["max_document_chars"]),
        )
    except Exception:
        raise AblationError("failed to construct locked qwen3 reranker") from None


def sanitize_error_type(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    if _SAFE_ERROR_TYPE_RE.fullmatch(text):
        return text
    return "provider_error"


def credential_source_records(repo_root: Path) -> list[dict[str, str]]:
    return [
        {"key": item.key, "source": item.source}
        for item in describe_credential_sources(root=Path(repo_root), keys=("DASHSCOPE_API_KEY",))
    ]


def call_accounting(
    *,
    embed_total: int,
    embed_this_invocation: int,
    embed_expected: int,
    qwen3_total: int,
    qwen3_this_invocation: int,
    qwen3_expected: int,
) -> dict[str, Any]:
    return {
        "query_embedding_calls": int(embed_total),
        "query_embedding_calls_total": int(embed_total),
        "query_embedding_calls_this_invocation": int(embed_this_invocation),
        "query_embedding_calls_expected": int(embed_expected),
        "qwen3_calls": int(qwen3_total),
        "qwen3_calls_total": int(qwen3_total),
        "qwen3_calls_this_invocation": int(qwen3_this_invocation),
        "qwen3_calls_expected": int(qwen3_expected),
        "chunk_reembed_calls": 0,
        "billing_semantics": BILLING_SEMANTICS,
        "exactly_once": False,
        "call_count_basis": "persisted_complete_cases",
        "crash_before_atomic_persist_may_repeat": True,
        "unobserved_inflight_remote_calls_possible": True,
        "call_accounting_note": (
            "Totals count persisted complete cases only: one query embedding "
            "and three Qwen3 reranks per complete case. A crash after a remote "
            "call and before persist can create extra provider calls that this "
            "program cannot observe. These totals are not a provider invoice "
            "and are not exactly-once billed calls."
        ),
    }


def case_ids_sha256(case_ids: list[str] | tuple[str, ...] | set[str]) -> str:
    ordered = sorted({str(item) for item in case_ids if str(item)})
    return hashlib.sha256(
        json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def empty_focus_identity() -> dict[str, Any]:
    return {
        "source_status": "recomputed",
        "path_exists": False,
        "per_case_sha256": "",
        "focus_case_count": 0,
        "focus_case_ids": [],
        "focus_case_ids_sha256": case_ids_sha256([]),
        "source_row_count": 0,
        "expected_test_case_ids_sha256": "",
    }


def focus_identity_fields(provenance: Mapping[str, Any]) -> dict[str, Any]:
    empty = empty_focus_identity()
    return {
        "source_status": str(provenance.get("source_status") or empty["source_status"]),
        "path_exists": bool(provenance.get("path_exists")),
        "per_case_sha256": str(provenance.get("per_case_sha256") or ""),
        "focus_case_count": int(provenance.get("focus_case_count") or 0),
        "focus_case_ids_sha256": str(provenance.get("focus_case_ids_sha256") or empty["focus_case_ids_sha256"]),
        "source_row_count": int(provenance.get("source_row_count") or 0),
        "expected_test_case_ids_sha256": str(provenance.get("expected_test_case_ids_sha256") or ""),
    }


def expected_call_budget(*, n_cases: int = SPLIT_TEST_SIZE) -> dict[str, int]:
    cases = int(n_cases)
    return {
        "query_embedding_calls_expected": cases,
        "qwen3_rerank_calls_expected": cases * len(ARM_ORDER),
        "chunk_reembed_calls_expected": 0,
        "rerank_calls_per_case": len(ARM_ORDER),
        "channel_fetch_k": LOCKED_CHANNEL_FETCH_K,
        "rerank_document_slots": cases * sum(spec.rerank_k for spec in ARM_SPECS.values()),
    }


def ablation_config_payload(
    *,
    rerank: dict[str, Any] | None = None,
    source_schema_sha256: str = "",
    source_collection_manifest_sha256: str = "",
    dataset_hash: str = "",
    focus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = public_rerank_settings(rerank) if rerank is not None else resolved_rerank_settings()
    return {
        "schema_version": ABLATION_SCHEMA,
        "experiment_role": EXPERIMENT_ROLE,
        "split": LOCKED_SPLIT,
        "index_scope": LOCKED_INDEX_SCOPE,
        "session_id": SOURCE_INDEX_SESSION_ID,
        "embedding_provider": LOCKED_EMBEDDING_PROVIDER,
        "embedding_model": LOCKED_EMBEDDING_MODEL_NAME,
        "embedding_dimension": LOCKED_EMBEDDING_DIMENSION,
        "dense_rrf_weight": LOCKED_DENSE_RRF_WEIGHT,
        "bm25_rrf_weight": LOCKED_BM25_RRF_WEIGHT,
        "channel_fetch_k": LOCKED_CHANNEL_FETCH_K,
        "final_k": LOCKED_FINAL_K,
        "arms": {name: asdict(spec) for name, spec in ARM_SPECS.items()},
        "rerank": {
            "provider": "qwen3",
            "model": settings["model"],
            "instruct": settings["instruct"],
            "timeout_seconds": settings["timeout_seconds"],
            "max_attempts": settings["max_attempts"],
            "backoff_seconds": settings["backoff_seconds"],
            "max_inflight": settings["max_inflight"],
            "max_document_chars": settings["max_document_chars"],
            "base_url_sha256": settings.get("base_url_sha256") or rerank_base_url_sha256(""),
        },
        "source_schema_sha256": source_schema_sha256,
        "source_collection_manifest_sha256": source_collection_manifest_sha256,
        "dataset_hash": dataset_hash,
        "focus": dict(focus or empty_focus_identity()),
    }


def config_hash_for(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_exact_output_path(path: Path, *, repo_root: Path, dirname: str) -> bool:
    resolved = Path(path).expanduser().resolve()
    target = (Path(repo_root) / "outputs" / dirname).resolve()
    return resolved == target


def _is_named_output_path(path: Path, *, repo_root: Path, dirname: str) -> bool:
    resolved = Path(path).expanduser().resolve()
    target = (Path(repo_root) / "outputs" / dirname).resolve()
    return resolved == target or target in resolved.parents


def is_locked_ablation_output_path(path: Path, *, repo_root: Path) -> bool:
    return _is_exact_output_path(path, repo_root=repo_root, dirname=ABLATION_OUTPUT_DIRNAME)


def is_locked_preflight_output_path(path: Path, *, repo_root: Path) -> bool:
    return _is_exact_output_path(path, repo_root=repo_root, dirname=ABLATION_PREFLIGHT_DIRNAME)


def is_candidate_depth_output_path(path: Path, *, repo_root: Path) -> bool:
    return any(
        _is_named_output_path(path, repo_root=repo_root, dirname=name)
        for name in (
            FIRST_ATTEMPT_OUTPUT_DIRNAME,
            DEPTH_SCORING_DIRNAME,
            FAILED_PREFLIGHT_OUTPUT_DIRNAME,
            DEPTH_PREFLIGHT2_DIRNAME,
        )
    ) or is_first_attempt_output_path(path, repo_root=repo_root) or is_scoring_output_path(
        path, repo_root=repo_root
    ) or is_failed_preflight_output_path(path, repo_root=repo_root)


def validate_ablation_request(
    *,
    split: str,
    confirm_exposed_diagnostic: bool,
    allow_remote: bool,
    embedding_provider: str,
    output_dir: str | Path,
    repo_root: str | Path,
    session_id: str = SOURCE_INDEX_SESSION_ID,
    preflight_only: bool = False,
    resume: bool = False,
    enforce_locked_output_dir: bool = False,
) -> None:
    raw = str(split or "").strip().lower()
    if raw in {"dev", "confirmation"}:
        raise SplitError(
            "candidate-pool ablation refuses confirmation/dev; confirmation-50 is consumed"
        )
    if raw == "all":
        raise SplitError("candidate-pool ablation refuses --split all")
    if raw != LOCKED_SPLIT:
        raise SplitError("candidate-pool ablation only allows --split test")
    if not confirm_exposed_diagnostic:
        raise AblationError(
            "refusing exposed test-100 ablation without --confirm-exposed-diagnostic"
        )
    provider = embedding_provider.strip().lower()
    if provider not in {LOCKED_EMBEDDING_PROVIDER, "deterministic"}:
        raise AblationError(
            f"unsupported embedding_provider {embedding_provider!r}; real runs use dashscope"
        )
    requested_session = str(session_id or "").strip()
    if requested_session == FORBIDDEN_QUERY_SESSION_ID:
        raise AblationError(
            "refusing query session_id 'financebench-candidate-depth'; "
            f"historical index was written with {SOURCE_INDEX_SESSION_ID}"
        )
    if requested_session and requested_session != SOURCE_INDEX_SESSION_ID:
        raise AblationError(
            f"candidate-pool ablation is locked to session_id={SOURCE_INDEX_SESSION_ID}"
        )
    out = Path(output_dir)
    root = Path(repo_root)
    if is_historical_output_path(out, repo_root=root):
        raise AblationError(f"refusing to write into historical FinanceBench directory {output_dir}")
    if is_candidate_depth_output_path(out, repo_root=root):
        raise AblationError(
            f"refusing to write into candidate-depth directory {output_dir}; "
            f"use outputs/{ABLATION_OUTPUT_DIRNAME}/"
        )
    if preflight_only and is_locked_ablation_output_path(out, repo_root=root):
        raise AblationError(
            f"refusing to write preflight into the scoring directory {output_dir}; "
            f"use outputs/{ABLATION_PREFLIGHT_DIRNAME}/"
        )
    if not preflight_only and is_locked_preflight_output_path(out, repo_root=root):
        raise AblationError(
            f"refusing to write scoring into the preflight directory {output_dir}; "
            f"use outputs/{ABLATION_OUTPUT_DIRNAME}/"
        )
    if enforce_locked_output_dir:
        if preflight_only:
            if not is_locked_preflight_output_path(out, repo_root=root):
                raise AblationError(
                    "real CLI preflight only allows "
                    f"outputs/{ABLATION_PREFLIGHT_DIRNAME}/"
                )
        elif not is_locked_ablation_output_path(out, repo_root=root):
            raise AblationError(
                "real CLI scoring only allows "
                f"outputs/{ABLATION_OUTPUT_DIRNAME}/"
            )
    if not preflight_only and provider != "deterministic":
        require_allow_remote(
            mode="hybrid-qwen3",
            embedding_provider=embedding_provider,
            allow_remote=allow_remote,
        )
    if resume and preflight_only:
        raise AblationError("refusing --resume with --preflight-only")


def truncate_hits(hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return [dict(hit) for hit in list(hits or [])[: int(k)]]


def fuse_rrf(
    *,
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    bm25_rrf_weight: float = LOCKED_BM25_RRF_WEIGHT,
    k: int,
) -> list[dict[str, Any]]:
    if dense_hits and bm25_hits:
        fused = reciprocal_rank_fusion(
            [dense_hits, bm25_hits],
            retrieval_method="hybrid_dense_bm25_rrf",
            weights=[LOCKED_DENSE_RRF_WEIGHT, float(bm25_rrf_weight)],
        )
        return fused[:k]
    if bm25_hits:
        return bm25_hits[:k]
    return dense_hits[:k]


def construct_arm_pool(
    *,
    bm25_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    spec: ArmSpec,
) -> list[dict[str, Any]]:
    bm25 = truncate_hits(bm25_hits, spec.channel_k)
    dense = truncate_hits(dense_hits, spec.channel_k)
    pool = fuse_rrf(dense_hits=dense, bm25_hits=bm25, k=spec.rrf_k)
    if len(pool) > spec.rerank_k:
        pool = pool[: spec.rerank_k]
    return pool


def _safe_candidates(hits: list[dict[str, Any]], *, k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits[:k], start=1):
        rows.append(
            {
                "rank": rank,
                "chunk_id": hit.get("chunk_id"),
                "doc_name": normalize_doc_name(
                    str(hit.get("document_id") or hit.get("filename") or hit.get("doc_name") or "")
                ),
                "page": hit.get("page"),
            }
        )
    return rows


def _pages(hits: list[dict[str, Any]]) -> list[tuple[str, int]]:
    return retrieved_page_keys(hits)


def _gold_in_pages(pages: list[tuple[str, int]], gold: set[tuple[str, int]]) -> bool:
    return any(page in gold for page in pages)


def _candidate_chars(query: str, hits: list[dict[str, Any]]) -> int:
    return len(query or "") + sum(len(str(hit.get("text") or "")) for hit in hits)


def _score_pages(pages: list[tuple[str, int]], gold: set[tuple[str, int]]) -> dict[str, Any]:
    rank = first_gold_rank(pages, gold)
    return {
        "first_gold_rank": rank,
        "hit_at": {
            "1": int(hit_at_k(pages, gold, k=1)),
            "5": int(hit_at_k(pages, gold, k=5)),
            "10": int(hit_at_k(pages, gold, k=10)),
        },
        "recall_at": {
            "5": round(recall_at_k(pages, gold, k=5), 4),
            "10": round(recall_at_k(pages, gold, k=10), 4),
        },
        "mrr": round(mean_reciprocal_rank(pages[:LOCKED_FINAL_K], gold), 4),
        "ndcg_at": {
            "5": round(ndcg_at_k(pages, gold, k=5), 4),
            "10": round(ndcg_at_k(pages, gold, k=10), 4),
        },
    }


def classify_arm_failure(
    *,
    gold_in_pool: bool,
    gold_in_final: bool,
    ingestion_failure: bool,
    fallback: bool,
    error_type: str,
) -> str:
    if ingestion_failure:
        return "ingestion_failure"
    if gold_in_final:
        return "hit_at_10"
    if gold_in_pool:
        return "gold_in_pool_not_in_final_top10"
    if fallback or error_type:
        return "rerank_fallback_or_error_miss"
    return "gold_not_in_rerank_pool"


def both_channels_empty(bm25_hits: list[dict[str, Any]], dense_hits: list[dict[str, Any]]) -> bool:
    return not list(bm25_hits or []) and not list(dense_hits or [])


def retrieve_channel_lists(
    *,
    store: Any,
    query: str,
    company: str,
    session_id: str,
    candidate_k: int = LOCKED_CHANNEL_FETCH_K,
    index_scope: str = LOCKED_INDEX_SCOPE,
) -> dict[str, list[dict[str, Any]]]:
    companies = None if index_scope == "corpus" else [company]
    bm25_hits = list(
        store.bm25_search(
            query,
            session_id=session_id,
            companies=companies,
            top_k=candidate_k,
        )
        or []
    )
    dense_hits = list(
        store.vector_search(
            query,
            session_id=session_id,
            companies=companies,
            top_k=candidate_k,
        )
        or []
    )
    return {"bm25": bm25_hits, "dense": dense_hits}


def rerank_arm(
    *,
    reranker: Any,
    query: str,
    pool: list[dict[str, Any]],
    spec: ArmSpec,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chars = _candidate_chars(query, pool)
    ranked, meta = reranker.rerank(query, pool, top_k=spec.final_k)
    final = list(ranked or [])[: spec.final_k]
    fallback = bool((meta or {}).get("rerank_fallback"))
    error_type = sanitize_error_type(str((meta or {}).get("rerank_error_type") or ""))
    qwen3_ok = not fallback and not error_type
    record = {
        "model": str((meta or {}).get("rerank_model") or settings["model"]),
        "instruct": settings["instruct"],
        "candidate_count": len(pool),
        "rerank_latency_ms": float((meta or {}).get("rerank_latency_ms") or 0.0),
        "usage": int((meta or {}).get("rerank_tokens") or 0),
        "chars": chars,
        "fallback": fallback,
        "error_type": error_type,
        "degraded": fallback or bool(error_type),
        "qwen3_ok": qwen3_ok,
        "provider": str((meta or {}).get("rerank_provider") or ""),
    }
    return final, record


def _paired_row(case_id: str, arm: dict[str, Any]) -> dict[str, Any]:
    scores = arm.get("scores") or {}
    rerank = arm.get("rerank") or {}
    return {
        "case_id": case_id,
        "page": {
            "hit_at": scores.get("hit_at") or {},
            "mrr": scores.get("mrr") or 0.0,
            "ndcg_at": scores.get("ndcg_at") or {},
            "first_relevant_rank": scores.get("first_gold_rank") or 0,
        },
        "failure_class": arm.get("failure_class") or "",
        "latency_ms": rerank.get("rerank_latency_ms") or rerank.get("latency_ms") or 0.0,
        "rerank_fallback": bool(rerank.get("fallback")),
        "error_type": rerank.get("error_type") or "",
    }


def _hit10_movement(base_rows: list[dict[str, Any]], cand_rows: list[dict[str, Any]]) -> dict[str, int]:
    improved = degraded = tied = 0
    for base, cand in zip(base_rows, cand_rows, strict=True):
        base_hit = float(((base.get("page") or {}).get("hit_at") or {}).get("10") or 0) >= 0.5
        cand_hit = float(((cand.get("page") or {}).get("hit_at") or {}).get("10") or 0) >= 0.5
        if cand_hit and not base_hit:
            improved += 1
        elif base_hit and not cand_hit:
            degraded += 1
        else:
            tied += 1
    return {"improved": improved, "degraded": degraded, "tied": tied}


def _arm_summary(rows: list[dict[str, Any]], arm_name: str) -> dict[str, Any]:
    arms = [((row.get("arms") or {}).get(arm_name) or {}) for row in rows]
    latencies = [float((arm.get("rerank") or {}).get("rerank_latency_ms") or 0.0) for arm in arms]
    usages = [float((arm.get("rerank") or {}).get("usage") or 0.0) for arm in arms]
    chars = [float((arm.get("rerank") or {}).get("chars") or 0.0) for arm in arms]
    cand_counts = [float((arm.get("rerank") or {}).get("candidate_count") or 0.0) for arm in arms]
    fallbacks = sum(1 for arm in arms if (arm.get("rerank") or {}).get("fallback"))
    errors = sum(1 for arm in arms if (arm.get("rerank") or {}).get("error_type"))
    qwen3_ok_rows = [arm for arm in arms if (arm.get("rerank") or {}).get("qwen3_ok")]
    def _mean_hit(items: list[dict[str, Any]], k: str) -> float:
        return round(mean(float(((item.get("scores") or {}).get("hit_at") or {}).get(k) or 0) for item in items), 4)

    def _mean_metric(items: list[dict[str, Any]], field: str, k: str | None = None) -> float:
        values: list[float] = []
        for item in items:
            scores = item.get("scores") or {}
            if k is None:
                values.append(float(scores.get(field) or 0.0))
            else:
                values.append(float((scores.get(field) or {}).get(k) or 0.0))
        return round(mean(values), 4)

    p50 = round(percentile_sorted(latencies, 0.50), 2) if latencies else 0.0
    p95 = round(percentile_sorted(latencies, 0.95), 2) if latencies else 0.0
    return {
        "cases": len(arms),
        "page_hit_at_1": _mean_hit(arms, "1"),
        "page_hit_at_5": _mean_hit(arms, "5"),
        "page_hit_at_10": _mean_hit(arms, "10"),
        "page_recall_at_5": _mean_metric(arms, "recall_at", "5"),
        "page_recall_at_10": _mean_metric(arms, "recall_at", "10"),
        "mrr": _mean_metric(arms, "mrr"),
        "ndcg_at_5": _mean_metric(arms, "ndcg_at", "5"),
        "ndcg_at_10": _mean_metric(arms, "ndcg_at", "10"),
        "page_hit_at_10_qwen3_ok": _mean_hit(qwen3_ok_rows, "10") if qwen3_ok_rows else None,
        "qwen3_ok_cases": len(qwen3_ok_rows),
        "fallback_cases": fallbacks,
        "provider_error_cases": errors,
        "mean_rerank_candidates": round(mean(cand_counts), 4) if cand_counts else 0.0,
        "mean_rerank_usage": round(mean(usages), 4) if usages else 0.0,
        "mean_rerank_chars": round(mean(chars), 4) if chars else 0.0,
        "p50_rerank_latency_ms": p50,
        "p95_rerank_latency_ms": p95,
        "latency_scope": "qwen3_rerank_only_not_e2e",
        "spec": asdict(ARM_SPECS[arm_name]),
    }


def load_depth_focus_provenance(
    repo_root: Path,
    *,
    expected_case_ids: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    path = Path(repo_root) / CANDIDATE_DEPTH_V2_PER_CASE
    identity = empty_focus_identity()
    expected = [str(item) for item in (expected_case_ids or []) if str(item)]
    if expected:
        identity["expected_test_case_ids_sha256"] = case_ids_sha256(expected)
    if not path.is_file():
        return identity
    try:
        digest = sha256_file(path)
        rows = read_jsonl(path)
    except Exception as exc:
        raise AblationError(
            "candidate-depth v2 per_case.jsonl exists but is unreadable; refusing silent recompute"
        ) from exc
    if not rows:
        raise AblationError(
            "candidate-depth v2 per_case.jsonl exists but is empty; refusing silent recompute"
        )
    seen: set[str] = set()
    ordered_ids: list[str] = []
    focus_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise AblationError(
                f"candidate-depth v2 per_case.jsonl line {index} is not an object; refusing silent recompute"
            )
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise AblationError(
                f"candidate-depth v2 per_case.jsonl line {index} is missing case_id; refusing silent recompute"
            )
        if case_id in seen:
            raise AblationError(
                "candidate-depth v2 per_case.jsonl has duplicate case_id; refusing silent recompute"
            )
        seen.add(case_id)
        ordered_ids.append(case_id)
        modes = row.get("modes")
        if not isinstance(modes, dict):
            raise AblationError(
                f"candidate-depth v2 per_case.jsonl line {index} is missing modes; refusing silent recompute"
            )
        failure = str(((modes.get("hybrid_rrf") or {}).get("failure_class") or ""))
        if failure in FOCUS_FAILURE_CLASSES:
            focus_ids.append(case_id)
    if expected:
        if seen != set(expected):
            raise AblationError(
                "candidate-depth v2 per_case.jsonl case IDs do not match the current test split; "
                "refusing silent recompute"
            )
        if len(expected) == SPLIT_TEST_SIZE and len(focus_ids) != EXPECTED_FOCUS_CASES:
            raise AblationError(
                "candidate-depth v2 per_case.jsonl focus count "
                f"{len(focus_ids)} != {EXPECTED_FOCUS_CASES}; refusing silent recompute"
            )
    if not focus_ids:
        raise AblationError(
            "candidate-depth v2 per_case.jsonl has no rank 11-30 focus cases; refusing silent recompute"
        )
    focus_ids_sorted = sorted(focus_ids)
    return {
        "source_status": "candidate_depth_v2",
        "path_exists": True,
        "per_case_sha256": digest,
        "focus_case_count": len(focus_ids_sorted),
        "focus_case_ids": focus_ids_sorted,
        "focus_case_ids_sha256": case_ids_sha256(focus_ids_sorted),
        "source_row_count": len(ordered_ids),
        "expected_test_case_ids_sha256": case_ids_sha256(expected) if expected else case_ids_sha256(ordered_ids),
    }


def _load_depth_focus_ids(repo_root: Path) -> list[str] | None:
    provenance = load_depth_focus_provenance(repo_root)
    if provenance["source_status"] != "candidate_depth_v2":
        return None
    return list(provenance["focus_case_ids"])


def _focus_analysis(rows: list[dict[str, Any]], *, depth_ids: list[str] | None) -> dict[str, Any]:
    return focus_analysis(rows, provenance={"source_status": "candidate_depth_v2" if depth_ids else "recomputed", "focus_case_ids": depth_ids or []})


def focus_analysis(rows: list[dict[str, Any]], *, provenance: Mapping[str, Any]) -> dict[str, Any]:
    computed = [
        row for row in rows if 11 <= int(row.get("hybrid_rrf50_first_gold_rank") or 0) <= 30
    ]
    source_status = str(provenance.get("source_status") or "recomputed")
    if source_status == "candidate_depth_v2":
        wanted = {str(item) for item in (provenance.get("focus_case_ids") or []) if item}
        focus = [row for row in rows if str(row.get("case_id") or "") in wanted]
        source = "candidate_depth_v2"
    else:
        focus = computed
        source = "recomputed"
    a_miss_b_hit = 0
    a_miss_c_hit = 0
    c_rescued = 0
    c_still_miss = 0
    gold_in_pool_not_final = 0
    gold_not_in_pool = 0
    reasons: dict[str, int] = {}
    for row in focus:
        arms = row.get("arms") or {}
        a_hit = int((((arms.get("A") or {}).get("scores") or {}).get("hit_at") or {}).get("10") or 0)
        b_hit = int((((arms.get("B") or {}).get("scores") or {}).get("hit_at") or {}).get("10") or 0)
        c_hit = int((((arms.get("C") or {}).get("scores") or {}).get("hit_at") or {}).get("10") or 0)
        if not a_hit and b_hit:
            a_miss_b_hit += 1
        if not a_hit and c_hit:
            a_miss_c_hit += 1
            c_rescued += 1
        if not c_hit:
            c_still_miss += 1
            reason = str((arms.get("C") or {}).get("failure_class") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason == "gold_in_pool_not_in_final_top10":
                gold_in_pool_not_final += 1
            if reason == "gold_not_in_rerank_pool":
                gold_not_in_pool += 1
    return {
        "source": source,
        "source_status": source_status,
        "path_exists": bool(provenance.get("path_exists")),
        "per_case_sha256": str(provenance.get("per_case_sha256") or ""),
        "focus_case_count": int(provenance.get("focus_case_count") or len(focus)),
        "focus_case_ids_sha256": str(provenance.get("focus_case_ids_sha256") or ""),
        "source_row_count": int(provenance.get("source_row_count") or 0),
        "focus_cases": len(focus),
        "computed_rank_11_30": len(computed),
        "a_miss_b_hit": a_miss_b_hit,
        "a_miss_c_hit": a_miss_c_hit,
        "c_rescued_from_a_miss": c_rescued,
        "c_still_miss": c_still_miss,
        "c_miss_gold_in_rerank_pool_not_final_top10": gold_in_pool_not_final,
        "c_miss_gold_not_in_rerank_pool": gold_not_in_pool,
        "c_miss_reasons": reasons,
        "case_ids": [str(row.get("case_id") or "") for row in focus],
    }


def _pair_report(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    base_rows = [_paired_row(str(row["case_id"]), (row.get("arms") or {}).get(baseline) or {}) for row in rows]
    cand_rows = [_paired_row(str(row["case_id"]), (row.get("arms") or {}).get(candidate) or {}) for row in rows]
    compared = compare_paired_systems(
        base_rows,
        cand_rows,
        baseline_name=baseline,
        candidate_name=candidate,
        n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
        seed=DEFAULT_BOOTSTRAP_SEED,
    )
    compared["hit_at_10_movement"] = _hit10_movement(base_rows, cand_rows)
    compared["delta_hit_at_5"] = compared["paired_bootstrap"]["delta_hit_at_5"]["mean_delta"]
    compared["delta_hit_at_10"] = compared["paired_bootstrap"]["delta_hit_at_10"]["mean_delta"]
    compared["delta_mrr"] = compared["paired_bootstrap"]["delta_mrr"]["mean_delta"]
    compared["delta_ndcg_at_10"] = compared["paired_bootstrap"]["delta_ndcg_at_10"]["mean_delta"]
    return compared


def _arm_qwen3_ok(row: dict[str, Any], arm_name: str) -> bool:
    return bool((((row.get("arms") or {}).get(arm_name) or {}).get("rerank") or {}).get("qwen3_ok"))


def _any_arm_degraded(rows: list[dict[str, Any]]) -> bool:
    return any(
        bool((arm.get("rerank") or {}).get("fallback")) or bool((arm.get("rerank") or {}).get("error_type"))
        for row in rows
        for arm in (row.get("arms") or {}).values()
    )


def _pair_qwen3_ok_rows(rows: list[dict[str, Any]], baseline: str, candidate: str) -> list[dict[str, Any]]:
    return [row for row in rows if _arm_qwen3_ok(row, baseline) and _arm_qwen3_ok(row, candidate)]


def _pair_block(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    candidate: str,
    primary_comparison_valid: bool,
) -> dict[str, Any]:
    descriptive = _pair_report(rows, baseline=baseline, candidate=candidate) if rows else {
        "n": 0,
        "delta_hit_at_5": None,
        "delta_hit_at_10": None,
        "delta_mrr": None,
        "delta_ndcg_at_10": None,
    }
    ok_rows = _pair_qwen3_ok_rows(rows, baseline, candidate)
    qwen3_ok_paired = _pair_report(ok_rows, baseline=baseline, candidate=candidate) if ok_rows else None
    status = "qwen3_primary" if primary_comparison_valid else "degraded_descriptive"
    return {
        **descriptive,
        "primary_comparison_valid": bool(primary_comparison_valid),
        "comparison_status": status,
        "n_descriptive": len(rows),
        "qwen3_ok_complete_cases": len(ok_rows),
        "qwen3_ok_paired": qwen3_ok_paired,
        "descriptive_only": not primary_comparison_valid,
    }


def build_paired_report(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, int]:
    all_ok = [row for row in rows if all(_arm_qwen3_ok(row, name) for name in ARM_ORDER)]
    primary_valid = (not _any_arm_degraded(rows)) and bool(rows)
    paired = {
        key: _pair_block(
            rows,
            baseline=baseline,
            candidate=candidate,
            primary_comparison_valid=primary_valid,
        )
        for key, baseline, candidate in _PAIR_KEYS
    }
    return paired, primary_valid, len(all_ok)


def render_results_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    arms = summary.get("arms") or {}
    pairs = report.get("paired") or {}
    focus = summary.get("rank_11_30") or {}
    lines = [
        "# FinanceBench exposed test-100 candidate-pool / Qwen3 ablation",
        "",
        "This is a **post-hoc ablation** on the already-exposed test-100.",
        "It is **not** held-out, **not** product accuracy, and **not** a new benchmark score.",
        f"experiment_role: `{EXPERIMENT_ROLE}`",
        "",
        "## Arms",
        "",
        "| Arm | Channels | RRF | Qwen3 pool | Final |",
        "|---|---:|---:|---:|---:|",
        "| A baseline | 20+20 | 20 | 20 | 10 |",
        "| B deeper retrieval | 50+50 | 20 | 20 | 10 |",
        "| C proposed | 50+50 | 30 | 30 | 10 |",
        "",
        "## Page metrics",
        "",
        "| Arm | Hit@1 | Hit@5 | Hit@10 | MRR | nDCG@10 | fallback | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARM_ORDER:
        arm = arms.get(name) or {}
        lines.append(
            f"| {name} | {arm.get('page_hit_at_1')} | {arm.get('page_hit_at_5')} | "
            f"{arm.get('page_hit_at_10')} | {arm.get('mrr')} | {arm.get('ndcg_at_10')} | "
            f"{arm.get('fallback_cases')} | {arm.get('provider_error_cases')} |"
        )
    lines.extend(["", "## Paired deltas", ""])
    primary_valid = bool(report.get("primary_comparison_valid"))
    if primary_valid:
        lines.append("Primary Qwen3 comparison is valid and covers every completed case with zero fallback/error.")
    else:
        lines.extend(
            [
                "Primary comparison status: **degraded/descriptive**.",
                "`primary_comparison_valid=false` because at least one arm has Qwen3 fallback or provider error.",
                "All-case paired deltas below are descriptive only and are **not** a valid Qwen3 ablation conclusion.",
                "Failed cases are retained in the descriptive tables; they are not silently dropped.",
                "Qwen3-only paired sensitivity uses cases where **both** compared arms have `qwen3_ok=true`.",
            ]
        )
    lines.append(f"all_arms_qwen3_ok_cases: {report.get('all_arms_qwen3_ok_cases')}")
    for key in ("B_vs_A", "C_vs_A", "C_vs_B"):
        pair = pairs.get(key) or {}
        lines.append(
            f"- {key} [{pair.get('comparison_status')}]: descriptive n={pair.get('n_descriptive')} "
            f"ΔHit@5={pair.get('delta_hit_at_5')} ΔHit@10={pair.get('delta_hit_at_10')} "
            f"ΔMRR={pair.get('delta_mrr')} ΔnDCG@10={pair.get('delta_ndcg_at_10')}; "
            f"qwen3_ok_complete_cases={pair.get('qwen3_ok_complete_cases')}"
        )
    lines.extend(
        [
            "",
            "## Rank 11–30 focus",
            "",
            f"- source_status: {focus.get('source_status') or focus.get('source')}",
            f"- per_case_sha256: {focus.get('per_case_sha256') or '(recomputed)'}",
            f"- focus_case_count: {focus.get('focus_case_count')}",
            f"- focus_case_ids_sha256: {focus.get('focus_case_ids_sha256')}",
            f"- focus cases in this run: {focus.get('focus_cases')}",
            f"- A miss, B hit: {focus.get('a_miss_b_hit')}",
            f"- A miss, C hit: {focus.get('a_miss_c_hit')}",
            f"- C still miss: {focus.get('c_still_miss')}",
            "",
            "## Latency scope",
            "",
            "- `rerank_latency_ms` measures Qwen3 rerank only.",
            "- It is **not** end-to-end query latency.",
            "- It cannot be used to claim how much production total latency arm C adds.",
            "- Shared `channel_retrieval_latency_ms` is BM25 Top-50 + Dense Top-50, not production Top-20 retrieval.",
            "",
            "## Resume / billing",
            "",
            f"- billing_semantics: `{report.get('billing_semantics') or BILLING_SEMANTICS}`",
            "- Completed cases are not re-called.",
            "- A crash after a remote call but before atomic persist may repeat that case (at-least-once, not exactly-once).",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_preflight_report(
    out: Path,
    *,
    selected: dict[str, Any],
    copied_uri: str,
    query_session_id: str,
    tenant_id: str,
    scope_check: dict[str, Any],
    code_snapshot: dict[str, Any],
    config: dict[str, Any],
    credential_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    budget = expected_call_budget(n_cases=SPLIT_TEST_SIZE)
    max_chars = int(((config.get("rerank") or {}).get("max_document_chars") or 4000))
    accounting = call_accounting(
        embed_total=0,
        embed_this_invocation=0,
        embed_expected=budget["query_embedding_calls_expected"],
        qwen3_total=0,
        qwen3_this_invocation=0,
        qwen3_expected=budget["qwen3_rerank_calls_expected"],
    )
    report = {
        "schema_version": ABLATION_SCHEMA,
        "status": "PREFLIGHT_OK",
        "experiment_role": EXPERIMENT_ROLE,
        "held_out": False,
        "product_accuracy_claim": False,
        **accounting,
        **budget,
        "rerank_document_char_ceiling": budget["rerank_document_slots"] * max_chars,
        "query_session_id": query_session_id,
        "index_uri_original": selected.get("uri"),
        "index_uri_copy": copied_uri,
        "source_index": {
            "commit": selected.get("source_index_commit") or SOURCE_INDEX_COMMIT,
            "worktree_dirty": selected.get("source_index_worktree_dirty", SOURCE_INDEX_WORKTREE_DIRTY),
            "chunker": selected.get("source_index_chunker") or SOURCE_INDEX_CHUNKER,
            "session_id": query_session_id,
            "tenant_id": tenant_id,
        },
        "scope_check": scope_check,
        "config_hash": config_hash_for(config),
        "credential_sources": credential_sources,
        "diagnostic_code": {
            "commit": code_snapshot.get("lumenfin_commit"),
            "worktree_dirty": bool(code_snapshot.get("worktree_dirty")),
        },
        "latency_scope": {
            "rerank_latency_ms": "Qwen3 rerank only; not end-to-end query latency",
            "channel_retrieval_latency_ms": "shared BM25 Top-50 + Dense Top-50; not production Top-20",
            "not_production_e2e": True,
        },
        "billing_semantics": BILLING_SEMANTICS,
        "exactly_once": False,
        "disclaimer": (
            "Preflight only. Operates on the copied index only. Not a FinanceBench score. "
            "Does not re-embed 52,518 chunks."
        ),
    }
    write_json(out / "preflight.json", report)
    return report


def _completed_ids(path: Path) -> set[str]:
    return {str(row["case_id"]) for row in _read_complete_per_case_rows(path)}


def _read_complete_per_case_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw_lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if index == len(raw_lines) - 1:
                continue
            raise AblationError("per_case.jsonl is corrupt; refusing resume") from None
        if not isinstance(payload, dict):
            raise AblationError("per_case.jsonl is corrupt; refusing resume")
        arms = payload.get("arms") or {}
        case_id = str(payload.get("case_id") or "").strip()
        if case_id and all(name in arms for name in ARM_ORDER):
            rows.append(payload)
    return rows


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AblationError(f"refusing --resume without {label}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AblationError(f"refusing --resume: {label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise AblationError(f"refusing --resume: {label} is not an object")
    return payload


def _identity_from(payload: Mapping[str, Any]) -> dict[str, str]:
    focus = payload.get("focus") or (payload.get("config") or {}).get("focus") or {}
    return {
        "config_hash": str(payload.get("config_hash") or ""),
        "diagnostic_commit": str(
            payload.get("diagnostic_commit")
            or payload.get("diagnostic_code_commit")
            or payload.get("commit")
            or ""
        ),
        "source_schema_sha256": str(payload.get("source_schema_sha256") or ""),
        "source_collection_manifest_sha256": str(payload.get("source_collection_manifest_sha256") or ""),
        "focus_per_case_sha256": str(
            payload.get("focus_per_case_sha256") or focus.get("per_case_sha256") or ""
        ),
        "focus_case_ids_sha256": str(
            payload.get("focus_case_ids_sha256") or focus.get("focus_case_ids_sha256") or ""
        ),
    }


def _require_pairwise_nested_completed_ids(
    *,
    manifest_ids: set[str],
    checkpoint_ids: set[str],
    per_case_ids: set[str],
) -> None:
    pairs = (
        (manifest_ids, checkpoint_ids),
        (manifest_ids, per_case_ids),
        (checkpoint_ids, per_case_ids),
    )
    if any(not (left <= right or right <= left) for left, right in pairs):
        raise AblationError(
            "refusing --resume: completed case IDs diverge across manifest/checkpoint/per_case"
        )


def _resume_ids_from(payload: Mapping[str, Any], *, label: str) -> set[str]:
    raw = payload.get("complete_case_ids")
    if raw is None:
        raw = payload.get("completed_case_ids")
    if raw is None:
        raise AblationError(f"refusing --resume: {label} is missing completed case IDs")
    if not isinstance(raw, list) or any(not str(item).strip() for item in raw):
        raise AblationError(f"refusing --resume: {label} completed case IDs are invalid")
    ids = {str(item) for item in raw}
    count = payload.get("completed_cases")
    if count is None:
        raise AblationError(f"refusing --resume: {label} is missing completed_cases")
    try:
        completed_cases = int(count)
    except (TypeError, ValueError) as exc:
        raise AblationError(f"refusing --resume: {label} completed_cases is invalid") from exc
    if completed_cases != len(ids):
        raise AblationError(f"refusing --resume: {label} completed_cases does not match completed case IDs")
    return ids


def _validate_resume(
    out: Path,
    *,
    config_hash: str,
    commit: str,
    source_schema_sha256: str,
    source_collection_manifest_sha256: str,
    focus: Mapping[str, Any],
    expected_n: int,
    config: dict[str, Any],
) -> set[str]:
    manifest = _read_json_object(out / "manifest.json", label="manifest.json")
    checkpoint = _read_json_object(out / "checkpoint.json", label="checkpoint.json")
    per_case_path = out / "per_case.jsonl"
    if not per_case_path.is_file():
        raise AblationError("refusing --resume without per_case.jsonl")
    rows = _read_complete_per_case_rows(per_case_path)
    per_case_ids = {str(row["case_id"]) for row in rows}
    focus_fields = focus_identity_fields(focus)
    expected = {
        "config_hash": config_hash,
        "diagnostic_commit": commit,
        "source_schema_sha256": source_schema_sha256,
        "source_collection_manifest_sha256": source_collection_manifest_sha256,
        "focus_per_case_sha256": str(focus_fields["per_case_sha256"] or ""),
        "focus_case_ids_sha256": str(focus_fields["focus_case_ids_sha256"] or ""),
    }
    for label, payload in (("manifest.json", manifest), ("checkpoint.json", checkpoint)):
        found = _identity_from(payload)
        for name, wanted in expected.items():
            if found[name] != wanted:
                raise AblationError(f"refusing --resume: {label} {name} mismatch")
    manifest_ids = _resume_ids_from(manifest, label="manifest.json")
    checkpoint_ids = _resume_ids_from(checkpoint, label="checkpoint.json")
    _require_pairwise_nested_completed_ids(
        manifest_ids=manifest_ids,
        checkpoint_ids=checkpoint_ids,
        per_case_ids=per_case_ids,
    )
    # per_case complete rows are the durable source of truth after a crash
    # between the three writes.
    _write_progress(
        out,
        digest=config_hash,
        commit=commit,
        source_schema_sha256=source_schema_sha256,
        source_collection_manifest_sha256=source_collection_manifest_sha256,
        completed=per_case_ids,
        embed_total=len(per_case_ids),
        qwen3_total=len(per_case_ids) * len(ARM_ORDER),
        expected_n=expected_n,
        status="running",
        config=config,
        focus=focus_fields,
        rows=rows,
    )
    return set(per_case_ids)


def _progress_payload(
    *,
    digest: str,
    commit: str,
    source_schema_sha256: str,
    source_collection_manifest_sha256: str,
    completed: set[str],
    embed_total: int,
    qwen3_total: int,
    expected_n: int,
    status: str,
    config: dict[str, Any] | None = None,
    focus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    focus_fields = focus_identity_fields(focus or empty_focus_identity())
    payload: dict[str, Any] = {
        "schema_version": ABLATION_SCHEMA,
        "status": status,
        "experiment_role": EXPERIMENT_ROLE,
        "config_hash": digest,
        "diagnostic_commit": commit,
        "diagnostic_code_commit": commit,
        "source_schema_sha256": source_schema_sha256,
        "source_collection_manifest_sha256": source_collection_manifest_sha256,
        "complete_case_ids": sorted(completed),
        "completed_case_ids": sorted(completed),
        "completed_cases": len(completed),
        "cases": len(completed),
        "query_embedding_calls": embed_total,
        "query_embedding_calls_total": embed_total,
        "qwen3_calls": qwen3_total,
        "qwen3_calls_total": qwen3_total,
        "chunk_reembed_calls": 0,
        "billing_semantics": BILLING_SEMANTICS,
        "exactly_once": False,
        "call_count_basis": "persisted_complete_cases",
        "unobserved_inflight_remote_calls_possible": True,
        "expected_calls": expected_call_budget(n_cases=expected_n),
        "focus": focus_fields,
        "focus_per_case_sha256": focus_fields["per_case_sha256"],
        "focus_case_ids_sha256": focus_fields["focus_case_ids_sha256"],
        "index_copy_relpath": f"{INDEX_WORK_DIRNAME}/{INDEX_COPY_DBNAME}",
    }
    if config is not None:
        payload["config"] = config
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    cleaned = redact_mapping(dict(payload))
    assert_no_secrets(cleaned)
    _atomic_write_text(path, json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n")


def _per_case_text(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        cleaned = redact_mapping(dict(row))
        assert_no_secrets(cleaned)
        lines.append(json.dumps(cleaned, ensure_ascii=False))
    return ("\n".join(lines) + "\n") if lines else ""


def _write_progress(
    out: Path,
    *,
    digest: str,
    commit: str,
    source_schema_sha256: str,
    source_collection_manifest_sha256: str,
    completed: set[str],
    embed_total: int,
    qwen3_total: int,
    expected_n: int,
    status: str,
    config: dict[str, Any],
    focus: Mapping[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> None:
    payload = _progress_payload(
        digest=digest,
        commit=commit,
        source_schema_sha256=source_schema_sha256,
        source_collection_manifest_sha256=source_collection_manifest_sha256,
        completed=completed,
        embed_total=embed_total,
        qwen3_total=qwen3_total,
        expected_n=expected_n,
        status=status,
        config=config,
        focus=focus,
    )
    out.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        _atomic_write_text(out / "per_case.jsonl", _per_case_text(rows))
    _atomic_write_json(out / "checkpoint.json", payload)
    _atomic_write_json(out / "manifest.json", payload)


def resolve_copied_index_uri(out: Path, source_uri: str, *, resume: bool) -> str:
    dest_parent = Path(out) / INDEX_WORK_DIRNAME
    dest = dest_parent / INDEX_COPY_DBNAME
    if resume:
        if not dest.exists():
            raise AblationError(
                f"refusing --resume without existing {INDEX_WORK_DIRNAME}/{INDEX_COPY_DBNAME}"
            )
        return str(dest)
    return str(copy_index_for_query(source_uri, dest_parent))


def assert_ablation_row_redacted(row: dict[str, Any]) -> None:
    assert_per_case_redacted(row)

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                if lowered in {"rerank_error", "error", "api_key", "authorization", "base_url", "_base_url"}:
                    raise ValueError(f"per-case ablation leaked forbidden key {key}")
                if "length" == lowered and str((value.get("key") or "")).upper().endswith("_KEY"):
                    raise ValueError("credential_sources must not record secret length")
                _walk(item)
            return
        if isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(row)
    serialized = json.dumps(row, ensure_ascii=False)
    if _LEAK_RE.search(serialized):
        raise ValueError("per-case ablation leaked credential, endpoint, or provider body")


def score_case_arms(
    question: FinanceBenchQuestion,
    *,
    bm25_hits: list[dict[str, Any]],
    dense_hits: list[dict[str, Any]],
    reranker: Any,
    settings: dict[str, Any],
    documents: dict[str, DocumentInfo],
    zero_chunk_names: set[str],
    session_id: str,
) -> dict[str, Any]:
    scoring_gold, affected_by_zero_chunk, ingestion_failure = gold_pages_for_scoring(
        question, zero_chunk_names
    )
    hybrid50 = fuse_rrf(dense_hits=dense_hits, bm25_hits=bm25_hits, k=LOCKED_CHANNEL_FETCH_K)
    hybrid_rank = first_gold_rank(_pages(hybrid50), scoring_gold)
    arms: dict[str, Any] = {}
    for name in ARM_ORDER:
        spec = ARM_SPECS[name]
        pool = construct_arm_pool(bm25_hits=bm25_hits, dense_hits=dense_hits, spec=spec)
        if len(pool) != spec.rerank_k and len(pool) > spec.rerank_k:
            pool = pool[: spec.rerank_k]
        pool_pages = _pages(pool)
        gold_in_pool = _gold_in_pages(pool_pages, scoring_gold)
        final, rerank_meta = rerank_arm(
            reranker=reranker,
            query=question.question,
            pool=pool,
            spec=spec,
            settings=settings,
        )
        final_pages = _pages(final)
        gold_in_final = _gold_in_pages(final_pages, scoring_gold)
        scores = _score_pages(final_pages, scoring_gold)
        failure = classify_arm_failure(
            gold_in_pool=gold_in_pool,
            gold_in_final=gold_in_final,
            ingestion_failure=ingestion_failure,
            fallback=bool(rerank_meta.get("fallback")),
            error_type=str(rerank_meta.get("error_type") or ""),
        )
        arms[name] = {
            "spec": asdict(spec),
            "channel_k": spec.channel_k,
            "rrf_k": spec.rrf_k,
            "rerank_k": spec.rerank_k,
            "final_k": spec.final_k,
            "pre_rerank": _safe_candidates(pool, k=spec.rerank_k),
            "final": _safe_candidates(final, k=spec.final_k),
            "first_gold_rank_pre": first_gold_rank(pool_pages, scoring_gold),
            "scores": scores,
            "rerank": rerank_meta,
            "gold_in_rerank_pool": gold_in_pool,
            "gold_in_final_top10": gold_in_final,
            "failure_class": failure,
            "ingestion_failure": ingestion_failure,
        }
    row = {
        "schema_version": ABLATION_SCHEMA,
        "experiment_role": EXPERIMENT_ROLE,
        "case_id": question.case_id,
        "financebench_id": question.financebench_id,
        "company": question.company,
        "gold_pages": [{"doc_name": page.doc_name, "page": page.page_one} for page in gold_pages_for(question)],
        "affected_by_zero_chunk": affected_by_zero_chunk,
        "session_id": session_id,
        "hybrid_rrf50_first_gold_rank": hybrid_rank,
        "in_rank_11_30_focus": 11 <= hybrid_rank <= 30,
        "query_embedding_calls": 1,
        "qwen3_calls": len(ARM_ORDER),
        "chunk_reembed_calls": 0,
        "arms": arms,
    }
    assert_ablation_row_redacted(row)
    serialized = json.dumps(row, ensure_ascii=False)
    for token in ("SECRET_QUESTION", "SECRET_ANSWER", "SECRET_EVIDENCE", "LEAKED_CHUNK"):
        if token in serialized:
            raise ValueError("per-case ablation leaked forbidden content")
    return row


def _source_index_fields(selected: dict[str, Any], *, session_id: str, tenant_id: str) -> dict[str, Any]:
    return {
        "commit": selected.get("source_index_commit") or SOURCE_INDEX_COMMIT,
        "worktree_dirty": selected.get("source_index_worktree_dirty", SOURCE_INDEX_WORKTREE_DIRTY),
        "chunker": selected.get("source_index_chunker") or SOURCE_INDEX_CHUNKER,
        "session_id": session_id,
        "tenant_id": tenant_id,
        "schema_sha256": selected.get("source_schema_sha256") or "",
        "collection_manifest_sha256": selected.get("source_collection_manifest_sha256") or "",
        "not_current_chunker": True,
    }


def _zero_chunk_names(index_report: dict[str, Any]) -> set[str]:
    return {normalize_doc_name(str(name)) for name in index_report.get("zero_chunk_documents") or []}


def run_candidate_pool_ablation(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    split: str = LOCKED_SPLIT,
    confirm_exposed_diagnostic: bool = False,
    allow_remote: bool = False,
    embedding_provider: str = LOCKED_EMBEDDING_PROVIDER,
    session_id: str = SOURCE_INDEX_SESSION_ID,
    store: Any | None = None,
    reranker: Any | None = None,
    index_inspection: dict[str, Any] | None = None,
    skip_index_copy: bool = False,
    expected_questions: int | None = EXPECTED_OPEN_SOURCE_QUESTIONS,
    require_clean_worktree: bool = True,
    worktree_dirty: bool | None = None,
    preflight_only: bool = False,
    resume: bool = False,
    index_query_client: Any | None = None,
    canary_companies: tuple[str, ...] = DEFAULT_CANARY_COMPANIES,
    enforce_locked_output_dir: bool = False,
    stop_after: int | None = None,
) -> dict[str, Any]:
    validate_ablation_request(
        split=split,
        confirm_exposed_diagnostic=confirm_exposed_diagnostic,
        allow_remote=allow_remote,
        embedding_provider=embedding_provider,
        output_dir=output_dir,
        repo_root=repo_root,
        session_id=session_id,
        preflight_only=preflight_only,
        resume=resume,
        enforce_locked_output_dir=enforce_locked_output_dir,
    )
    if not resume:
        try:
            require_fresh_output_dir(output_dir)
        except CandidateDepthError as exc:
            raise AblationError(str(exc)) from exc
    code_snapshot = require_clean_diagnostic_worktree(
        repo_root,
        require_clean_worktree=require_clean_worktree,
        worktree_dirty=worktree_dirty,
    )
    inspection = index_inspection or inspect_financebench_indexes(repo_root)
    selected = require_compatible_index(inspection)
    source_session_id, source_tenant_id = resolve_source_scope(selected)
    query_session_id = resolve_query_session_id(session_id, source_session_id)
    rerank_settings = snapshot_rerank_settings()
    commit = str(code_snapshot.get("lumenfin_commit") or "")
    source_schema_sha256 = str(selected.get("source_schema_sha256") or "")
    source_collection_manifest_sha256 = str(selected.get("source_collection_manifest_sha256") or "")
    out = Path(output_dir)
    documents: dict[str, DocumentInfo] = {}
    selected_questions: list[FinanceBenchQuestion] = []
    expected_n = SPLIT_TEST_SIZE
    dataset_hash = str(selected.get("dataset_hash") or "")
    focus_provenance: dict[str, Any]
    if preflight_only:
        focus_provenance = load_depth_focus_provenance(Path(repo_root))
    else:
        questions, documents, paths = load_financebench_dataset(
            dataset_dir,
            expected_questions=expected_questions,
            require_pdfs=False,
        )
        dataset_hash = sha256_file(paths.questions_path)
        if dataset_hash != str(selected.get("dataset_hash") or ""):
            raise IndexIncompatibleError("dataset hash does not match the compatible index sidecar")
        assignment = assign_splits(questions)
        selected_questions = questions_for_split(questions, assignment, split)
        if expected_questions == EXPECTED_OPEN_SOURCE_QUESTIONS and len(selected_questions) != SPLIT_TEST_SIZE:
            raise AblationError(f"expected {SPLIT_TEST_SIZE} test questions, found {len(selected_questions)}")
        expected_n = len(selected_questions)
        focus_provenance = load_depth_focus_provenance(
            Path(repo_root),
            expected_case_ids=[question.case_id for question in selected_questions],
        )
    config = ablation_config_payload(
        rerank=rerank_settings,
        source_schema_sha256=source_schema_sha256,
        source_collection_manifest_sha256=source_collection_manifest_sha256,
        dataset_hash=dataset_hash,
        focus=focus_identity_fields(focus_provenance),
    )
    digest = config_hash_for(config)
    completed: set[str] = set()
    if resume:
        completed = _validate_resume(
            out,
            config_hash=digest,
            commit=commit,
            source_schema_sha256=source_schema_sha256,
            source_collection_manifest_sha256=source_collection_manifest_sha256,
            focus=focus_provenance,
            expected_n=expected_n,
            config=config,
        )
    copied_uri = ""
    if store is None:
        if skip_index_copy:
            raise AblationError("refusing to open the original Milvus Lite index without a copy")
        out.mkdir(parents=True, exist_ok=True)
        copied_uri = resolve_copied_index_uri(out, str(selected["uri"]), resume=resume)
    verify_uri = copied_uri or str(selected.get("uri") or "")
    if store is None or index_query_client is not None:
        try:
            scope_check = verify_copied_index_scope(
                uri=verify_uri,
                collection_name=str(selected.get("collection_name") or "financebench_eval"),
                expected_session_id=query_session_id,
                expected_tenant_id=source_tenant_id,
                expected_row_count=int(selected.get("chunks") or EXPECTED_CHUNKS),
                canary_companies=canary_companies,
                client=index_query_client,
            )
        except IndexSessionError as exc:
            raise AblationError(str(exc)) from exc
    else:
        scope_check = {
            "skipped": True,
            "session_id": query_session_id,
            "tenant_id": source_tenant_id,
            "query_embedding_calls": 0,
        }
    credential_sources = credential_source_records(Path(repo_root))
    if preflight_only:
        return _write_preflight_report(
            out,
            selected=selected,
            copied_uri=copied_uri,
            query_session_id=query_session_id,
            tenant_id=source_tenant_id,
            scope_check=scope_check,
            code_snapshot=code_snapshot,
            config=config,
            credential_sources=credential_sources,
        )

    remaining = [question for question in selected_questions if question.case_id not in completed]
    out.mkdir(parents=True, exist_ok=True)
    embed_total = len(completed)
    qwen3_total = len(completed) * len(ARM_ORDER)
    embed_this = 0
    qwen3_this = 0
    rows: list[dict[str, Any]] = _read_complete_per_case_rows(out / "per_case.jsonl") if resume else []
    _write_progress(
        out,
        digest=digest,
        commit=commit,
        source_schema_sha256=source_schema_sha256,
        source_collection_manifest_sha256=source_collection_manifest_sha256,
        completed=completed,
        embed_total=embed_total,
        qwen3_total=qwen3_total,
        expected_n=expected_n,
        status="running",
        config=config,
        focus=focus_provenance,
        rows=None if resume else [],
    )
    work_store = store
    opened_remote_store = False
    work_reranker = reranker
    if remaining:
        if work_store is None:
            from .retrieval import build_eval_store

            work_store = build_eval_store(
                uri=copied_uri,
                embedding_provider=embedding_provider,
                embedding_dimension=LOCKED_EMBEDDING_DIMENSION,
                collection_name=str(selected.get("collection_name") or "financebench_eval"),
                allow_remote=allow_remote,
                mode="hybrid-qwen3",
                embedding_model=LOCKED_EMBEDDING_MODEL_NAME,
            )
            opened_remote_store = embedding_provider.strip().lower() in REMOTE_EMBEDDING_PROVIDERS
        if work_reranker is None:
            work_reranker = build_locked_qwen3_reranker(rerank_settings)
    zero_chunk = _zero_chunk_names(selected)
    prefix_all_empty = True
    try:
        for index, question in enumerate(selected_questions, start=1):
            if question.case_id in completed:
                continue
            started = time.perf_counter()
            lists = retrieve_channel_lists(
                store=work_store,
                query=question.question,
                company=question.company,
                session_id=query_session_id,
            )
            channel_ms = round((time.perf_counter() - started) * 1000.0, 2)
            if opened_remote_store or store is not None:
                embed_this += 1
                embed_total += 1
            if prefix_all_empty and index <= EMPTY_RETRIEVAL_FAIL_FAST:
                if both_channels_empty(lists["bm25"], lists["dense"]):
                    if index >= EMPTY_RETRIEVAL_FAIL_FAST:
                        raise InvalidEmptyRetrievalError(
                            "first "
                            f"{EMPTY_RETRIEVAL_FAIL_FAST} questions returned empty BM25 and Dense "
                            "candidates; refusing to call Qwen3 or record scores",
                            query_embedding_calls=embed_this,
                        )
                else:
                    prefix_all_empty = False
            row = score_case_arms(
                question,
                bm25_hits=lists["bm25"],
                dense_hits=lists["dense"],
                reranker=work_reranker,
                settings=rerank_settings,
                documents=documents,
                zero_chunk_names=zero_chunk,
                session_id=query_session_id,
            )
            qwen3_this += int(row.get("qwen3_calls") or 0)
            qwen3_total += int(row.get("qwen3_calls") or 0)
            row["config_hash"] = digest
            row["channel_retrieval_latency_ms"] = channel_ms
            row["latency_scope"] = {
                "rerank_latency_ms": "Qwen3 rerank only; not end-to-end query latency",
                "channel_retrieval_latency_ms": (
                    "shared BM25 Top-50 + Dense Top-50; not production Top-20 retrieval"
                ),
                "not_production_e2e": True,
            }
            assert_ablation_row_redacted(row)
            rows.append(row)
            completed.add(question.case_id)
            _write_progress(
                out,
                digest=digest,
                commit=commit,
                source_schema_sha256=source_schema_sha256,
                source_collection_manifest_sha256=source_collection_manifest_sha256,
                completed=completed,
                embed_total=embed_total,
                qwen3_total=qwen3_total,
                expected_n=expected_n,
                status="running",
                config=config,
                focus=focus_provenance,
                rows=rows,
            )
            if stop_after is not None and len(completed) >= int(stop_after):
                break
    finally:
        closer = getattr(work_store, "close", None)
        if store is None and callable(closer):
            closer()

    accounting = call_accounting(
        embed_total=embed_total,
        embed_this_invocation=embed_this,
        embed_expected=expected_n,
        qwen3_total=qwen3_total,
        qwen3_this_invocation=qwen3_this,
        qwen3_expected=expected_n * len(ARM_ORDER),
    )
    if len(rows) < expected_n:
        _write_progress(
            out,
            digest=digest,
            commit=commit,
            source_schema_sha256=source_schema_sha256,
            source_collection_manifest_sha256=source_collection_manifest_sha256,
            completed=completed,
            embed_total=embed_total,
            qwen3_total=qwen3_total,
            expected_n=expected_n,
            status="interrupted",
            config=config,
            focus=focus_provenance,
            rows=rows,
        )
        return {
            "schema_version": ABLATION_SCHEMA,
            "status": "interrupted",
            "experiment_role": EXPERIMENT_ROLE,
            "cases": len(rows),
            "config_hash": digest,
            **accounting,
        }

    paired, primary_valid, all_ok_n = build_paired_report(rows)
    primary_valid = bool(
        primary_valid and all_ok_n == len(rows) == expected_n and not _any_arm_degraded(rows)
    )
    for pair in paired.values():
        pair["primary_comparison_valid"] = primary_valid
        pair["comparison_status"] = "qwen3_primary" if primary_valid else "degraded_descriptive"
        pair["descriptive_only"] = not primary_valid
    summary = {
        "cases": len(rows),
        "arms": {name: _arm_summary(rows, name) for name in ARM_ORDER},
        "rank_11_30": focus_analysis(rows, provenance=focus_provenance),
        **accounting,
        "primary_comparison_valid": primary_valid,
        "all_arms_qwen3_ok_cases": all_ok_n,
        "comparison_status": "qwen3_primary" if primary_valid else "degraded_descriptive",
        "latency_scope": {
            "rerank_latency_ms": "Qwen3 rerank only; not end-to-end query latency",
            "channel_retrieval_latency_ms": "shared BM25 Top-50 + Dense Top-50; not production Top-20",
            "not_production_e2e": True,
        },
    }
    source_fields = _source_index_fields(
        selected, session_id=query_session_id, tenant_id=source_tenant_id
    )
    env = environment_payload(
        repo_root=Path(repo_root),
        dataset_hash=dataset_hash,
        split_manifest_hash="",
        embedding_provider=embedding_provider,
        embedding_model=LOCKED_EMBEDDING_MODEL_NAME,
        rerank_provider="qwen3",
        rerank_model=str(rerank_settings["model"]),
        chunk_size=900,
        chunk_overlap=120,
        collection_name=str(selected.get("collection_name") or ""),
        bm25_rrf_weight=LOCKED_BM25_RRF_WEIGHT,
        top_k=LOCKED_FINAL_K,
        mode="candidate_pool_ablation",
        split=split,
        remote_calls_enabled=allow_remote,
        extra={
            **experiment_governance(split, LOCKED_INDEX_SCOPE),
            "diagnostic_schema": ABLATION_SCHEMA,
            "experiment_role": EXPERIMENT_ROLE,
            "product_accuracy_claim": False,
            "held_out_status": "exposed_test",
            "config_hash": digest,
            "query_session_id": query_session_id,
            "source_index_session_id": source_fields["session_id"],
            "source_index_tenant_id": source_fields["tenant_id"],
            "source_index_commit": source_fields["commit"],
            "source_index_chunker": source_fields["chunker"],
            "scope_check": scope_check,
            **accounting,
            "rerank_settings": public_rerank_settings(rerank_settings),
            "credential_sources": credential_sources,
        },
    )
    failures = [
        row
        for row in rows
        if any(
            (arm.get("rerank") or {}).get("fallback")
            or (arm.get("rerank") or {}).get("error_type")
            or (arm.get("failure_class") not in {"hit_at_10"})
            for arm in (row.get("arms") or {}).values()
        )
    ]
    report = {
        "schema_version": ABLATION_SCHEMA,
        "status": "recorded",
        "experiment_role": EXPERIMENT_ROLE,
        "held_out": False,
        "product_accuracy_claim": False,
        "split": split,
        "cases": len(rows),
        "config_hash": digest,
        "source_index": source_fields,
        "scope_check": scope_check,
        "summary": summary,
        "environment": env,
        "primary_comparison_valid": primary_valid,
        "all_arms_qwen3_ok_cases": all_ok_n,
        "comparison_status": "qwen3_primary" if primary_valid else "degraded_descriptive",
        **accounting,
        "disclaimer": (
            "Exposed test-100 post-hoc candidate-pool / Qwen3 ablation. "
            "Not held-out, not product accuracy, and not a confirmation-50 result. "
            "The reused Milvus index was built by the pre-overlap-fix chunker. "
            "rerank_latency_ms is Qwen3-only and is not production end-to-end latency."
        ),
    }
    write_json(out / "environment.json", env)
    write_json(out / "summary.json", report)
    write_json(out / "paired.json", paired)
    write_jsonl(out / "failures.jsonl", failures)
    (out / "results.md").write_text(render_results_markdown({**report, "paired": paired}), encoding="utf-8")
    _write_progress(
        out,
        digest=digest,
        commit=commit,
        source_schema_sha256=source_schema_sha256,
        source_collection_manifest_sha256=source_collection_manifest_sha256,
        completed=completed,
        embed_total=embed_total,
        qwen3_total=qwen3_total,
        expected_n=expected_n,
        status="recorded",
        config=config,
        focus=focus_provenance,
        rows=rows,
    )
    report["paired"] = paired
    return report
