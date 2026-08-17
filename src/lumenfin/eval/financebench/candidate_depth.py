"""Exposed test-100 candidate-depth diagnostic (eval-only, not a held-out).

Retrieves BM25@50, Dense@50, RRF(1.1)@50, and BM25∪Dense oracle coverage.
Does not call Qwen3. Does not change production retriever defaults.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from ...rag.hybrid_retriever import reciprocal_rank_fusion
from .constants import (
    DEFAULT_BM25_RRF_WEIGHT,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_DASHSCOPE_EMBEDDING_DIM,
    DEFAULT_TOP_K,
    REMOTE_EMBEDDING_PROVIDERS,
    SCHEMA_VERSION,
)
from .index_inspect import (
    EXPECTED_CHUNKS,
    SECTION_METADATA,
    SOURCE_INDEX_CHUNKER,
    SOURCE_INDEX_COMMIT,
    SOURCE_INDEX_WORKTREE_DIRTY,
    IndexIncompatibleError,
    inspect_financebench_indexes,
    is_historical_output_path,
    require_compatible_index,
)
from .loader import load_financebench_dataset, normalize_doc_name
from .metrics import hit_at_k, mean, mean_reciprocal_rank
from .qrels import gold_pages_for, retrieved_page_keys
from .reporting import environment_payload, git_snapshot, sha256_file, write_json, write_jsonl
from .retrieval import require_allow_remote
from .schema import DocumentInfo, FinanceBenchQuestion
from .split import SplitError, assign_splits, experiment_governance, questions_for_split

DIAGNOSTIC_SCHEMA = "financebench_candidate_depth.v1"
DIAGNOSTIC_CANDIDATE_K = 50
LOCKED_SPLIT = "test"
LOCKED_INDEX_SCOPE = "company"
LOCKED_CANDIDATE_K = DIAGNOSTIC_CANDIDATE_K
LOCKED_EMBEDDING_PROVIDER = "dashscope"
LOCKED_EMBEDDING_MODEL = "text-embedding-v4"
LOCKED_EMBEDDING_DIMENSION = DEFAULT_DASHSCOPE_EMBEDDING_DIM
LOCKED_DENSE_RRF_WEIGHT = 1.0
LOCKED_BM25_RRF_WEIGHT = DEFAULT_BM25_RRF_WEIGHT
RANKED_MODES = ("bm25", "dense", "hybrid_rrf")
DIAGNOSTIC_MODES = RANKED_MODES + ("oracle_union",)
DEPTH_BUCKETS = (
    "gold_rank_11_20",
    "gold_rank_21_30",
    "gold_rank_31_50",
    "gold_not_in_top50",
    "wrong_document",
    "wrong_period",
    "ingestion_failure",
)
FORBIDDEN_PER_CASE_KEYS = frozenset(
    {
        "question",
        "answer",
        "justification",
        "evidence_text",
        "evidence_text_full_page",
        "text",
        "chunk_text",
        "question_reasoning",
        "question_type",
    }
)
_PERIOD_RE = re.compile(r"(20\d{2}(?:Q[1-4])?)", re.IGNORECASE)


class CandidateDepthError(ValueError):
    """Raised when the diagnostic CLI/request is invalid."""


def _directory_has_entries(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def require_fresh_output_dir(output_dir: str | Path) -> None:
    target = Path(output_dir)
    if target.is_file():
        raise CandidateDepthError(f"refusing to reuse output path that is a file: {target}")
    if _directory_has_entries(target):
        raise CandidateDepthError(
            f"refusing to reuse non-empty output directory {target}; "
            "choose a new path instead of deleting or overwriting"
        )


def require_clean_diagnostic_worktree(
    repo_root: str | Path,
    *,
    require_clean_worktree: bool,
    worktree_dirty: bool | None = None,
) -> dict[str, Any]:
    snapshot = git_snapshot(Path(repo_root))
    dirty = bool(snapshot.get("worktree_dirty")) if worktree_dirty is None else bool(worktree_dirty)
    if require_clean_worktree and dirty:
        raise CandidateDepthError(
            "candidate-depth diagnostic requires a clean worktree before copying the index or calling remote APIs"
        )
    return snapshot


def validate_candidate_depth_request(
    *,
    split: str,
    confirm_exposed_diagnostic: bool,
    allow_remote: bool,
    embedding_provider: str,
    output_dir: str | Path,
    repo_root: str | Path,
    embedding_dimension: int = LOCKED_EMBEDDING_DIMENSION,
    candidate_k: int = LOCKED_CANDIDATE_K,
    index_scope: str = LOCKED_INDEX_SCOPE,
    bm25_rrf_weight: float = LOCKED_BM25_RRF_WEIGHT,
    dense_rrf_weight: float = LOCKED_DENSE_RRF_WEIGHT,
    embedding_model: str = LOCKED_EMBEDDING_MODEL,
) -> None:
    raw = str(split or "").strip().lower()
    if raw in {"dev", "confirmation"}:
        raise SplitError(
            "candidate-depth diagnostic refuses confirmation/dev; confirmation-50 is consumed"
        )
    if raw == "all":
        raise SplitError("candidate-depth diagnostic refuses --split all")
    if raw != LOCKED_SPLIT:
        raise SplitError("candidate-depth diagnostic only allows --split test")
    if not confirm_exposed_diagnostic:
        raise CandidateDepthError(
            "refusing exposed test-100 diagnostic without --confirm-exposed-diagnostic"
        )
    if str(index_scope or "").strip().lower() != LOCKED_INDEX_SCOPE:
        raise CandidateDepthError("candidate-depth diagnostic is locked to company scope")
    if int(candidate_k) != LOCKED_CANDIDATE_K:
        raise CandidateDepthError("candidate-depth diagnostic is locked to candidate_k=50")
    if int(embedding_dimension) != LOCKED_EMBEDDING_DIMENSION:
        raise CandidateDepthError("candidate-depth diagnostic is locked to embedding dimension 1024")
    if str(embedding_model or "").strip() != LOCKED_EMBEDDING_MODEL:
        raise CandidateDepthError("candidate-depth diagnostic is locked to text-embedding-v4")
    if abs(float(bm25_rrf_weight) - LOCKED_BM25_RRF_WEIGHT) > 1e-9:
        raise CandidateDepthError("candidate-depth diagnostic is locked to BM25 RRF weight 1.1")
    if abs(float(dense_rrf_weight) - LOCKED_DENSE_RRF_WEIGHT) > 1e-9:
        raise CandidateDepthError("candidate-depth diagnostic is locked to dense RRF weight 1.0")
    provider = embedding_provider.strip().lower()
    if provider not in {LOCKED_EMBEDDING_PROVIDER, "deterministic"}:
        raise CandidateDepthError(
            f"unsupported embedding_provider {embedding_provider!r}; real runs use dashscope"
        )
    require_allow_remote(
        mode="dense",
        embedding_provider=embedding_provider,
        allow_remote=allow_remote,
    )
    if is_historical_output_path(Path(output_dir), repo_root=Path(repo_root)):
        raise CandidateDepthError(
            f"refusing to write into historical FinanceBench directory {output_dir}"
        )


def page_key(doc_name: object, page: object) -> tuple[str, int] | None:
    name = normalize_doc_name(str(doc_name or ""))
    try:
        page_one = int(page)
    except (TypeError, ValueError):
        return None
    if not name:
        return None
    return (name, page_one)


def first_gold_rank(retrieved_pages: list[tuple[str, int]], gold_pages: set[tuple[str, int]]) -> int:
    for rank, item in enumerate(retrieved_pages, start=1):
        if item in gold_pages:
            return rank
    return 0


def hit_at_depths(
    retrieved_pages: list[tuple[str, int]],
    gold_pages: set[tuple[str, int]],
    *,
    ks: tuple[int, ...] = (10, 20, 30, 50),
) -> dict[str, float]:
    return {str(k): hit_at_k(retrieved_pages, gold_pages, k=k) for k in ks}


def mrr_at_k(
    retrieved_pages: list[tuple[str, int]],
    gold_pages: set[tuple[str, int]],
    *,
    k: int = DIAGNOSTIC_CANDIDATE_K,
) -> float:
    return mean_reciprocal_rank(retrieved_pages[:k], gold_pages)


def unique_page_count(hits: list[dict[str, Any]], *, k: int) -> int:
    return len(retrieved_page_keys(hits[:k]))


def duplicate_page_occupancy(hits: list[dict[str, Any]], *, k: int) -> float:
    window = hits[:k]
    if not window:
        return 0.0
    unique = unique_page_count(window, k=len(window))
    return round(1.0 - (unique / len(window)), 4)


def channel_recall_label(
    *,
    bm25_pages: list[tuple[str, int]],
    dense_pages: list[tuple[str, int]],
    gold_pages: set[tuple[str, int]],
) -> str:
    bm25_hit = any(page in gold_pages for page in bm25_pages)
    dense_hit = any(page in gold_pages for page in dense_pages)
    if bm25_hit and dense_hit:
        return "both"
    if bm25_hit:
        return "bm25_only"
    if dense_hit:
        return "dense_only"
    return "neither"


def _period_token(doc_name: str, documents: dict[str, DocumentInfo] | None = None) -> str:
    info = (documents or {}).get(normalize_doc_name(doc_name))
    if info and info.period:
        return str(info.period).strip().upper()
    match = _PERIOD_RE.search(doc_name)
    return match.group(1).upper() if match else ""


def _doc_company(doc_name: str, documents: dict[str, DocumentInfo] | None = None) -> str:
    info = (documents or {}).get(normalize_doc_name(doc_name))
    return str(info.company).strip() if info and info.company else ""


def classify_depth_failure(
    *,
    retrieved_pages: list[tuple[str, int]],
    gold_pages: set[tuple[str, int]],
    ingestion_failure: bool = False,
    documents: dict[str, DocumentInfo] | None = None,
    gold_company: str = "",
    top_k: int = DEFAULT_TOP_K,
) -> str:
    if ingestion_failure:
        return "ingestion_failure"
    rank = first_gold_rank(retrieved_pages, gold_pages)
    if 1 <= rank <= top_k:
        return "hit_at_10"
    if 11 <= rank <= 20:
        return "gold_rank_11_20"
    if 21 <= rank <= 30:
        return "gold_rank_21_30"
    if 31 <= rank <= 50:
        return "gold_rank_31_50"
    retrieved_docs = {doc for doc, _page in retrieved_pages}
    gold_docs = {doc for doc, _page in gold_pages}
    gold_periods = {_period_token(doc, documents) for doc in gold_docs}
    retrieved_periods = {_period_token(doc, documents) for doc in retrieved_docs}
    retrieved_companies = {_doc_company(doc, documents) for doc in retrieved_docs}
    company = (gold_company or "").strip()
    same_company = bool(company) and any(
        item.lower() == company.lower() for item in retrieved_companies if item
    )
    period_mismatch = bool(gold_periods and retrieved_periods and gold_periods.isdisjoint(retrieved_periods))
    if same_company and period_mismatch:
        return "wrong_period"
    if retrieved_docs and gold_docs and retrieved_docs.isdisjoint(gold_docs):
        return "wrong_document"
    return "gold_not_in_top50"


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


def assert_per_case_redacted(row: dict[str, Any]) -> None:
    serialized = str(row)

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                lowered = str(key).lower()
                if lowered in FORBIDDEN_PER_CASE_KEYS or lowered.endswith("_text"):
                    raise ValueError(f"per-case diagnostic leaked forbidden key {key}")
                _walk(item)
            return
        if isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(row)
    for token in ("evidence_text", "chunk body"):
        if token in serialized:
            raise ValueError("per-case diagnostic leaked evidence or chunk body")


def best_channel_rank(bm25_rank: int, dense_rank: int) -> int:
    found = [rank for rank in (int(bm25_rank), int(dense_rank)) if rank > 0]
    return min(found) if found else 0


def oracle_union_metrics(
    *,
    bm25_pages: list[tuple[str, int]],
    dense_pages: list[tuple[str, int]],
    gold_pages: set[tuple[str, int]],
    ks: tuple[int, ...] = (10, 20, 30, 50),
) -> dict[str, Any]:
    hit_at: dict[str, int] = {}
    coverage_at: dict[str, float] = {}
    for k in ks:
        pool = set(bm25_pages[:k]) | set(dense_pages[:k])
        hit_at[str(k)] = int(any(page in gold_pages for page in pool))
        coverage_at[str(k)] = (
            round(len({page for page in gold_pages if page in pool}) / len(gold_pages), 4)
            if gold_pages
            else 0.0
        )
    return {
        "hit_at": hit_at,
        "coverage_at": coverage_at,
        "unique_sections_top10": SECTION_METADATA,
    }


def score_mode(
    hits: list[dict[str, Any]],
    gold_pages: set[tuple[str, int]],
    *,
    k: int = DIAGNOSTIC_CANDIDATE_K,
) -> dict[str, Any]:
    pages = retrieved_page_keys(hits[:k])
    rank = first_gold_rank(pages, gold_pages)
    hits_at = hit_at_depths(pages, gold_pages)
    return {
        "first_gold_rank": rank,
        "hit_at": {key: int(value) for key, value in hits_at.items()},
        "mrr_at_50": round(mrr_at_k(pages, gold_pages, k=k), 4),
        "unique_pages_top10": unique_page_count(hits, k=10),
        "unique_pages_top20": unique_page_count(hits, k=20),
        "unique_pages_top50": unique_page_count(hits, k=k),
        "unique_sections_top10": SECTION_METADATA,
        "duplicate_page_occupancy_top10": duplicate_page_occupancy(hits, k=10),
        "duplicate_page_occupancy_top20": duplicate_page_occupancy(hits, k=20),
        "duplicate_page_occupancy_top50": duplicate_page_occupancy(hits, k=k),
        "candidates": _safe_candidates(hits, k=k),
    }


def union_hits(bm25_hits: list[dict[str, Any]], dense_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in list(bm25_hits) + list(dense_hits):
        chunk_id = str(hit.get("chunk_id") or "")
        key = chunk_id or f"{hit.get('document_id')}:{hit.get('page')}:{len(merged)}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


def fuse_rrf(
    *,
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    bm25_rrf_weight: float = DEFAULT_BM25_RRF_WEIGHT,
    k: int = DIAGNOSTIC_CANDIDATE_K,
) -> list[dict[str, Any]]:
    if dense_hits and bm25_hits:
        return reciprocal_rank_fusion(
            [dense_hits, bm25_hits],
            retrieval_method="hybrid_dense_bm25_rrf",
            weights=[LOCKED_DENSE_RRF_WEIGHT, float(bm25_rrf_weight)],
        )[:k]
    if bm25_hits:
        return bm25_hits[:k]
    return dense_hits[:k]


def retrieve_candidate_lists(
    *,
    store: Any,
    query: str,
    company: str,
    session_id: str,
    candidate_k: int,
    index_scope: str,
    bm25_rrf_weight: float,
) -> dict[str, list[dict[str, Any]]]:
    companies = None if index_scope == "corpus" else [company]
    bm25_hits = store.bm25_search(
        query,
        session_id=session_id,
        companies=companies,
        top_k=candidate_k,
    )
    dense_hits = store.vector_search(
        query,
        session_id=session_id,
        companies=companies,
        top_k=candidate_k,
    )
    return {
        "bm25": list(bm25_hits or []),
        "dense": list(dense_hits or []),
        "hybrid_rrf": fuse_rrf(
            dense_hits=list(dense_hits or []),
            bm25_hits=list(bm25_hits or []),
            bm25_rrf_weight=bm25_rrf_weight,
            k=candidate_k,
        ),
        "oracle_union": union_hits(list(bm25_hits or []), list(dense_hits or [])),
    }


def gold_pages_for_scoring(
    question: FinanceBenchQuestion,
    zero_chunk_names: set[str],
) -> tuple[set[tuple[str, int]], bool, bool]:
    pages = gold_pages_for(question)
    zero = {normalize_doc_name(name) for name in zero_chunk_names}
    retrievable = {page.key for page in pages if normalize_doc_name(page.doc_name) not in zero}
    affected = any(normalize_doc_name(page.doc_name) in zero for page in pages)
    all_blocked = bool(pages) and not retrievable
    scoring = retrievable if retrievable else {page.key for page in pages}
    return scoring, affected, all_blocked


def score_case(
    question: FinanceBenchQuestion,
    lists: dict[str, list[dict[str, Any]]],
    *,
    documents: dict[str, DocumentInfo],
    zero_chunk_names: set[str],
    candidate_k: int = DIAGNOSTIC_CANDIDATE_K,
) -> dict[str, Any]:
    gold_public = [
        {"doc_name": page.doc_name, "page": page.page_one} for page in gold_pages_for(question)
    ]
    scoring_gold, affected_by_zero_chunk, ingestion_failure = gold_pages_for_scoring(
        question, zero_chunk_names
    )
    mode_scores: dict[str, Any] = {}
    ranked_pages: dict[str, list[tuple[str, int]]] = {}
    for mode in RANKED_MODES:
        hits = lists.get(mode) or []
        scored = score_mode(hits, scoring_gold, k=candidate_k)
        ranked_pages[mode] = retrieved_page_keys(hits[:candidate_k])
        scored["failure_class"] = classify_depth_failure(
            retrieved_pages=ranked_pages[mode],
            gold_pages=scoring_gold,
            ingestion_failure=ingestion_failure,
            documents=documents,
            gold_company=question.company,
        )
        mode_scores[mode] = scored
    bm25_pages = ranked_pages.get("bm25") or []
    dense_pages = ranked_pages.get("dense") or []
    oracle = oracle_union_metrics(
        bm25_pages=bm25_pages,
        dense_pages=dense_pages,
        gold_pages=scoring_gold,
    )
    mode_scores["oracle_union"] = oracle
    channel = channel_recall_label(
        bm25_pages=bm25_pages,
        dense_pages=dense_pages,
        gold_pages=scoring_gold,
    )
    row = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "case_id": question.case_id,
        "financebench_id": question.financebench_id,
        "company": question.company,
        "gold_pages": gold_public,
        "affected_by_zero_chunk": affected_by_zero_chunk,
        "ingestion_failure": ingestion_failure,
        "best_channel_rank": best_channel_rank(
            int((mode_scores["bm25"].get("first_gold_rank") or 0)),
            int((mode_scores["dense"].get("first_gold_rank") or 0)),
        ),
        "channel_recall": channel,
        "modes": mode_scores,
    }
    assert_per_case_redacted(row)
    return row


def _rank_movement(baseline: int, candidate: int) -> str:
    if candidate and (not baseline or candidate < baseline):
        return "improved"
    if baseline and (not candidate or candidate > baseline):
        return "degraded"
    return "tied"


def _mean_hit(rows: list[dict[str, Any]], mode: str, k: str) -> float:
    values = [float(((row.get("modes") or {}).get(mode) or {}).get("hit_at", {}).get(k) or 0) for row in rows]
    return round(mean(values), 4)


def _mean_mrr(rows: list[dict[str, Any]], mode: str) -> float:
    values = [float(((row.get("modes") or {}).get(mode) or {}).get("mrr_at_50") or 0.0) for row in rows]
    return round(mean(values), 4)


def _mean_rank_or_none(rows: list[dict[str, Any]], mode: str) -> float | None:
    ranks = [
        int(((row.get("modes") or {}).get(mode) or {}).get("first_gold_rank") or 0)
        for row in rows
        if int(((row.get("modes") or {}).get(mode) or {}).get("first_gold_rank") or 0) > 0
    ]
    if not ranks:
        return None
    return round(mean(float(item) for item in ranks), 4)


def aggregate_depth(rows: list[dict[str, Any]]) -> dict[str, Any]:
    channel_counts = {"bm25_only": 0, "dense_only": 0, "both": 0, "neither": 0}
    depth_counts = {key: 0 for key in DEPTH_BUCKETS}
    rrf_vs_dense = {"improved": 0, "degraded": 0, "tied": 0}
    rrf_vs_bm25 = {"improved": 0, "degraded": 0, "tied": 0}
    rrf_worse_than_best_channel = 0
    wrong_period_in_candidate_50 = 0
    for row in rows:
        channel_counts[str(row.get("channel_recall") or "neither")] = (
            channel_counts.get(str(row.get("channel_recall") or "neither"), 0) + 1
        )
        hybrid = (row.get("modes") or {}).get("hybrid_rrf") or {}
        failure = str(hybrid.get("failure_class") or "")
        if failure in depth_counts:
            depth_counts[failure] += 1
        if failure == "wrong_period":
            wrong_period_in_candidate_50 += 1
        bm25_rank = int(((row.get("modes") or {}).get("bm25") or {}).get("first_gold_rank") or 0)
        dense_rank = int(((row.get("modes") or {}).get("dense") or {}).get("first_gold_rank") or 0)
        rrf_rank = int(hybrid.get("first_gold_rank") or 0)
        rrf_vs_dense[_rank_movement(dense_rank, rrf_rank)] += 1
        rrf_vs_bm25[_rank_movement(bm25_rank, rrf_rank)] += 1
        best = min((item for item in (bm25_rank, dense_rank) if item > 0), default=0)
        if best and (not rrf_rank or rrf_rank > best):
            rrf_worse_than_best_channel += 1
    modes = {}
    for mode in RANKED_MODES:
        modes[mode] = {
            "page_hit_at_10": _mean_hit(rows, mode, "10"),
            "page_hit_at_20": _mean_hit(rows, mode, "20"),
            "page_hit_at_30": _mean_hit(rows, mode, "30"),
            "page_hit_at_50": _mean_hit(rows, mode, "50"),
            "mrr_at_50": _mean_mrr(rows, mode),
            "mean_first_gold_rank_when_found": _mean_rank_or_none(rows, mode),
            "mean_unique_pages_top10": round(
                mean(
                    float(((row.get("modes") or {}).get(mode) or {}).get("unique_pages_top10") or 0)
                    for row in rows
                ),
                4,
            ),
            "mean_duplicate_page_occupancy_top10": round(
                mean(
                    float(
                        ((row.get("modes") or {}).get(mode) or {}).get(
                            "duplicate_page_occupancy_top10"
                        )
                        or 0.0
                    )
                    for row in rows
                ),
                4,
            ),
            "unique_sections_top10": SECTION_METADATA,
        }
    oracle_rows = [(row.get("modes") or {}).get("oracle_union") or {} for row in rows]
    modes["oracle_union"] = {
        "page_hit_at_10": round(mean(float((item.get("hit_at") or {}).get("10") or 0) for item in oracle_rows), 4),
        "page_hit_at_20": round(mean(float((item.get("hit_at") or {}).get("20") or 0) for item in oracle_rows), 4),
        "page_hit_at_30": round(mean(float((item.get("hit_at") or {}).get("30") or 0) for item in oracle_rows), 4),
        "page_hit_at_50": round(mean(float((item.get("hit_at") or {}).get("50") or 0) for item in oracle_rows), 4),
        "coverage_at_10": round(mean(float((item.get("coverage_at") or {}).get("10") or 0) for item in oracle_rows), 4),
        "coverage_at_20": round(mean(float((item.get("coverage_at") or {}).get("20") or 0) for item in oracle_rows), 4),
        "coverage_at_30": round(mean(float((item.get("coverage_at") or {}).get("30") or 0) for item in oracle_rows), 4),
        "coverage_at_50": round(mean(float((item.get("coverage_at") or {}).get("50") or 0) for item in oracle_rows), 4),
        "unique_sections_top10": SECTION_METADATA,
    }
    misses = sum(depth_counts[key] for key in DEPTH_BUCKETS)
    recoverable_11_30 = depth_counts["gold_rank_11_20"] + depth_counts["gold_rank_21_30"]
    return {
        "cases": len(rows),
        "modes": modes,
        "top10_miss_depth": depth_counts,
        "top10_misses": misses,
        "channel_recall": channel_counts,
        "rrf_vs_dense": rrf_vs_dense,
        "rrf_vs_bm25": rrf_vs_bm25,
        "rrf_worse_than_best_channel": rrf_worse_than_best_channel,
        "wrong_period_in_candidate_50": wrong_period_in_candidate_50,
        "mean_best_channel_rank_when_found": round(
            mean(
                float(row["best_channel_rank"])
                for row in rows
                if int(row.get("best_channel_rank") or 0) > 0
            ),
            4,
        )
        if any(int(row.get("best_channel_rank") or 0) > 0 for row in rows)
        else None,
        "recoverable_11_30": recoverable_11_30,
    }


def recommend_from_aggregate(summary: dict[str, Any]) -> list[dict[str, Any]]:
    depth = summary.get("top10_miss_depth") or {}
    channel = summary.get("channel_recall") or {}
    misses = max(int(summary.get("top10_misses") or 0), 1)
    rec_11_30 = int(summary.get("recoverable_11_30") or 0)
    rank_31_50 = int(depth.get("gold_rank_31_50") or 0)
    not_in_50 = int(depth.get("gold_not_in_top50") or 0)
    exclusive = int(channel.get("bm25_only") or 0) + int(channel.get("dense_only") or 0)
    rrf_worse = int(summary.get("rrf_worse_than_best_channel") or 0)
    templates = [
        {
            "id": "expand_candidate_pool_30",
            "triggered": rec_11_30 >= max(8, int(0.3 * misses)) and rec_11_30 >= rank_31_50 and rec_11_30 >= not_in_50,
            "advice": "若大量 gold rank 在 11～30：建议候选池扩到 30",
        },
        {
            "id": "evaluate_candidate_pool_50",
            "triggered": rank_31_50 >= max(8, int(0.3 * misses)) and rank_31_50 >= rec_11_30,
            "advice": "若大量 gold rank 在 31～50：评估候选池 50 及成本",
        },
        {
            "id": "section_aware_or_query_decomposition",
            "triggered": not_in_50 >= max(8, int(0.5 * misses)),
            "advice": "若大部分 gold 不在 Top-50：优先 section-aware chunking / query decomposition",
        },
        {
            "id": "keep_hybrid",
            "triggered": exclusive >= max(5, int(0.1 * int(summary.get("cases") or 0))),
            "advice": "若 BM25 和 Dense 各自有明显独占召回：保留 Hybrid",
        },
        {
            "id": "inspect_fusion_or_rerank_pool",
            "triggered": rrf_worse >= max(8, int(0.2 * int(summary.get("cases") or 0))),
            "advice": "若候选中有 gold 但 RRF 排序明显下降：检查融合权重或增加 rerank 池",
        },
    ]
    return [
        {
            **item,
            "status": "exposed_diagnostic_not_final",
            "claim": "exposed_test_100_post_hoc_diagnostic",
            "held_out": False,
            "product_accuracy": False,
        }
        for item in templates
    ]


def render_results_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    modes = summary.get("modes") or {}
    depth = summary.get("top10_miss_depth") or {}
    channel = summary.get("channel_recall") or {}
    lines = [
        "# FinanceBench exposed test-100 candidate-depth diagnostic",
        "",
        "This is a **post-hoc diagnostic** on the already-exposed test-100.",
        "It is **not** held-out, **not** product accuracy, and **not** a new benchmark score.",
        "Qwen3 was not called. Chunk corpus was not re-embedded.",
        "",
        f"- schema: `{report.get('schema_version')}`",
        f"- split: `{report.get('split')}`",
        f"- experiment_role: `{report.get('experiment_role')}`",
        f"- compatible_index: `{((report.get('index') or {}).get('uri'))}`",
        f"- section metadata: `{SECTION_METADATA}`",
        "",
        "## Page metrics by mode",
        "",
        "| Mode | Hit@10 | Hit@20 | Hit@30 | Hit@50 | MRR@50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in RANKED_MODES:
        row = modes.get(mode) or {}
        lines.append(
            f"| {mode} | {row.get('page_hit_at_10', 'NOT_RUN')} | "
            f"{row.get('page_hit_at_20', 'NOT_RUN')} | {row.get('page_hit_at_30', 'NOT_RUN')} | "
            f"{row.get('page_hit_at_50', 'NOT_RUN')} | {row.get('mrr_at_50', 'NOT_RUN')} |"
        )
    oracle = modes.get("oracle_union") or {}
    lines.extend(
        [
            "",
            "Oracle union is the set coverage of BM25 Top-K pages ∪ Dense Top-K pages. "
            "It has no MRR and no concatenation rank. Use `best_channel_rank` for rank.",
            "",
            f"- oracle Hit@10/20/30/50: {oracle.get('page_hit_at_10')} / "
            f"{oracle.get('page_hit_at_20')} / {oracle.get('page_hit_at_30')} / "
            f"{oracle.get('page_hit_at_50')}",
            f"- oracle Coverage@10/20/30/50: {oracle.get('coverage_at_10')} / "
            f"{oracle.get('coverage_at_20')} / {oracle.get('coverage_at_30')} / "
            f"{oracle.get('coverage_at_50')}",
            f"- mean best_channel_rank (when found): {summary.get('mean_best_channel_rank_when_found')}",
        ]
    )
    lines.extend(
        [
            "",
            "## Top-10 miss depth (hybrid RRF)",
            "",
            "| Bucket | Count |",
            "|---|---:|",
        ]
    )
    for key in DEPTH_BUCKETS:
        lines.append(f"| {key} | {depth.get(key, 0)} |")
    lines.extend(
        [
            "",
            "## Channel recall @50",
            "",
            f"- BM25-only: {channel.get('bm25_only', 0)}",
            f"- Dense-only: {channel.get('dense_only', 0)}",
            f"- both: {channel.get('both', 0)}",
            f"- neither: {channel.get('neither', 0)}",
            "",
            "## Recommendations (exposed diagnostic, not final)",
            "",
        ]
    )
    for item in report.get("recommendations") or []:
        flag = "TRIGGERED" if item.get("triggered") else "not triggered"
        lines.append(f"- `{item.get('id')}` [{flag}]: {item.get('advice')}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _ignore_lock(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "LOCK"}


def copy_index_for_query(source_uri: str | Path, dest_dir: str | Path) -> Path:
    source = Path(source_uri)
    dest_parent = Path(dest_dir)
    dest = dest_parent / "eval.db"
    if dest.exists() or _directory_has_entries(dest_parent):
        raise CandidateDepthError(
            f"refusing to overwrite existing index work directory {dest_parent}"
        )
    dest_parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest, ignore=_ignore_lock)
    return dest


def _zero_chunk_names(index_report: dict[str, Any]) -> set[str]:
    return {normalize_doc_name(str(name)) for name in index_report.get("zero_chunk_documents") or []}


def run_candidate_depth_diagnostic(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    split: str = LOCKED_SPLIT,
    confirm_exposed_diagnostic: bool = False,
    allow_remote: bool = False,
    embedding_provider: str = LOCKED_EMBEDDING_PROVIDER,
    embedding_dimension: int = LOCKED_EMBEDDING_DIMENSION,
    candidate_k: int = LOCKED_CANDIDATE_K,
    bm25_rrf_weight: float = LOCKED_BM25_RRF_WEIGHT,
    dense_rrf_weight: float = LOCKED_DENSE_RRF_WEIGHT,
    index_scope: str = LOCKED_INDEX_SCOPE,
    embedding_model: str = LOCKED_EMBEDDING_MODEL,
    store: Any | None = None,
    index_inspection: dict[str, Any] | None = None,
    skip_index_copy: bool = False,
    expected_questions: int | None = 150,
    session_id: str = "financebench-candidate-depth",
    parse_pdfs: bool = False,
    require_clean_worktree: bool = True,
    worktree_dirty: bool | None = None,
) -> dict[str, Any]:
    if parse_pdfs:
        raise CandidateDepthError("candidate-depth diagnostic refuses PDF parsing")
    validate_candidate_depth_request(
        split=split,
        confirm_exposed_diagnostic=confirm_exposed_diagnostic,
        allow_remote=allow_remote,
        embedding_provider=embedding_provider,
        embedding_dimension=embedding_dimension,
        candidate_k=candidate_k,
        index_scope=index_scope,
        bm25_rrf_weight=bm25_rrf_weight,
        dense_rrf_weight=dense_rrf_weight,
        embedding_model=embedding_model,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    require_fresh_output_dir(output_dir)
    code_snapshot = require_clean_diagnostic_worktree(
        repo_root,
        require_clean_worktree=require_clean_worktree,
        worktree_dirty=worktree_dirty,
    )
    inspection = index_inspection or inspect_financebench_indexes(repo_root)
    selected = require_compatible_index(inspection)
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
    out = Path(output_dir)
    work_store = store
    copied_uri = ""
    if work_store is None:
        if skip_index_copy:
            raise CandidateDepthError("refusing to open the original Milvus Lite index without a copy")
        from .retrieval import build_eval_store

        out.mkdir(parents=True, exist_ok=True)
        copied_uri = str(copy_index_for_query(selected["uri"], out / "_index_work"))
        work_store = build_eval_store(
            uri=copied_uri,
            embedding_provider=embedding_provider,
            embedding_dimension=embedding_dimension,
            collection_name=str(selected.get("collection_name") or "financebench_eval"),
            allow_remote=allow_remote,
            mode="dense",
            embedding_model=str(selected.get("embedding_model") or embedding_model),
        )
    zero_chunk = _zero_chunk_names(selected)
    rows: list[dict[str, Any]] = []
    query_embed_calls = 0
    try:
        for question in selected_questions:
            lists = retrieve_candidate_lists(
                store=work_store,
                query=question.question,
                company=question.company,
                session_id=session_id,
                candidate_k=candidate_k,
                index_scope=index_scope,
                bm25_rrf_weight=bm25_rrf_weight,
            )
            if embedding_provider.strip().lower() in REMOTE_EMBEDDING_PROVIDERS:
                query_embed_calls += 1
            rows.append(
                score_case(
                    question,
                    lists,
                    documents=documents,
                    zero_chunk_names=zero_chunk,
                    candidate_k=candidate_k,
                )
            )
    finally:
        closer = getattr(work_store, "close", None)
        if store is None and callable(closer):
            closer()

    summary = aggregate_depth(rows)
    recommendations = recommend_from_aggregate(summary)
    governance = experiment_governance(split, index_scope)
    out.mkdir(parents=True, exist_ok=True)
    env = environment_payload(
        repo_root=Path(repo_root),
        dataset_hash=dataset_hash,
        split_manifest_hash="",
        embedding_provider=embedding_provider,
        embedding_model=str(selected.get("embedding_model") or embedding_model),
        rerank_provider="none",
        rerank_model="",
        chunk_size=DEFAULT_CHUNK_CHARS,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        collection_name=str(selected.get("collection_name") or ""),
        bm25_rrf_weight=bm25_rrf_weight,
        top_k=candidate_k,
        mode="candidate_depth",
        split=split,
        remote_calls_enabled=allow_remote,
        extra={
            **governance,
            "diagnostic_schema": DIAGNOSTIC_SCHEMA,
            "candidate_k": candidate_k,
            "dense_rrf_weight": LOCKED_DENSE_RRF_WEIGHT,
            "rerank_called": False,
            "qwen3_calls": 0,
            "chunk_reembed_calls": 0,
            "query_embedding_calls": query_embed_calls,
            "query_embedding_calls_expected": len(selected_questions),
            "section_metadata": SECTION_METADATA,
            "index_uri_original": selected.get("uri"),
            "index_uri_copy": copied_uri,
            "index_copy_required": True,
            "product_accuracy_claim": False,
            "held_out_status": "exposed_test",
            "experiment_role": "post_hoc_candidate_depth_diagnostic",
            "source_index_commit": selected.get("source_index_commit") or SOURCE_INDEX_COMMIT,
            "source_index_worktree_dirty": selected.get("source_index_worktree_dirty", SOURCE_INDEX_WORKTREE_DIRTY),
            "source_index_chunker": selected.get("source_index_chunker") or SOURCE_INDEX_CHUNKER,
            "source_schema_sha256": selected.get("source_schema_sha256") or "",
            "source_collection_manifest_sha256": selected.get("source_collection_manifest_sha256") or "",
            "diagnostic_code_commit": code_snapshot.get("lumenfin_commit"),
            "diagnostic_code_worktree_dirty": bool(code_snapshot.get("worktree_dirty")),
            "index_not_current_chunker": True,
        },
    )
    report = {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "parent_schema_version": SCHEMA_VERSION,
        "status": "recorded",
        "split": split,
        "split_status": "exposed_test",
        "experiment_role": "post_hoc_candidate_depth_diagnostic",
        "held_out": False,
        "product_accuracy_claim": False,
        "qwen3_calls": 0,
        "chunk_reembed_calls": 0,
        "query_embedding_calls": query_embed_calls,
        "query_embedding_calls_expected": len(selected_questions),
        "reembed_all_chunks": False,
        "expected_chunk_reembeds_if_forced": EXPECTED_CHUNKS,
        "section_metadata": SECTION_METADATA,
        "index": selected,
        "index_inspection": {
            "compatible": inspection.get("compatible"),
            "opened_milvus_client": False,
            "modified_original_index": False,
        },
        "source_index": {
            "commit": selected.get("source_index_commit") or SOURCE_INDEX_COMMIT,
            "worktree_dirty": selected.get("source_index_worktree_dirty", SOURCE_INDEX_WORKTREE_DIRTY),
            "chunker": selected.get("source_index_chunker") or SOURCE_INDEX_CHUNKER,
            "schema_sha256": selected.get("source_schema_sha256") or "",
            "collection_manifest_sha256": selected.get("source_collection_manifest_sha256") or "",
            "not_current_chunker": True,
        },
        "diagnostic_code": {
            "commit": code_snapshot.get("lumenfin_commit"),
            "worktree_dirty": bool(code_snapshot.get("worktree_dirty")),
        },
        "summary": summary,
        "recommendations": recommendations,
        "environment": env,
        "disclaimer": (
            "Exposed test-100 post-hoc candidate-depth diagnostic. Not held-out, "
            "not product accuracy, not a new FinanceBench score, and not a "
            "confirmation-50 result. The reused Milvus index was built by the "
            "pre-overlap-fix chunker and is not an index of the current chunker."
        ),
    }
    write_json(out / "environment.json", env)
    write_json(out / "summary.json", report)
    write_jsonl(out / "per_case.jsonl", rows)
    (out / "results.md").write_text(render_results_markdown(report), encoding="utf-8")
    return report
