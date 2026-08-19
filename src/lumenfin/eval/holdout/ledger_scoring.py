"""Eval-only LEDGER public-dev scoring with shared candidates across arms."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from .governance import HoldoutError
from .ledger_corpus import LedgerPublicDevDataset
from .page_collapse import (
    duplicate_page_occupancy,
    page_identity_coverage_top_k,
    unique_pages_top_k,
)
from .ranking import ARM_SPECS, prepare_rerank_pool, summarize_ranking_cases

CandidateRetriever = Callable[
    [str, str, int],
    tuple[list[dict[str, Any]], Mapping[str, Any]],
]
RerankCallable = Callable[
    [str, list[dict[str, Any]], int, str],
    tuple[list[dict[str, Any]], Mapping[str, Any]],
]


def _ledger_hit_doc_id(hit: Mapping[str, Any]) -> str:
    value = hit.get("document_id")
    if not isinstance(value, str) or not value.strip():
        raise HoldoutError("LEDGER retrieval hit needs a document_id")
    return value.strip()


def _validate_candidates(
    hits: list[dict[str, Any]],
    *,
    company_key: str,
    corpus_doc_ids: set[str],
    expected_max: int,
) -> None:
    if not hits:
        raise HoldoutError("LEDGER public_dev retrieval returned no candidates")
    if len(hits) > expected_max:
        raise HoldoutError("LEDGER retrieval returned more than the shared source window")
    chunk_ids: set[str] = set()
    for hit in hits:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in chunk_ids:
            raise HoldoutError("LEDGER retrieval returned missing or duplicate chunk_id")
        chunk_ids.add(chunk_id)
        if _ledger_hit_doc_id(hit) not in corpus_doc_ids:
            raise HoldoutError("LEDGER retrieval returned a page outside the dev corpus")
        companies = hit.get("companies")
        if not isinstance(companies, list) or company_key not in companies:
            raise HoldoutError("LEDGER retrieval returned a cross-company candidate")


def _validate_reranked(
    final_hits: list[dict[str, Any]],
    *,
    pool: list[dict[str, Any]],
    final_k: int,
) -> None:
    if pool and not final_hits:
        raise HoldoutError("LEDGER reranker returned no hits for a non-empty pool")
    if len(final_hits) > final_k:
        raise HoldoutError("LEDGER reranker returned more than final_k hits")
    pool_ids = {str(hit["chunk_id"]) for hit in pool}
    final_ids = [str(hit.get("chunk_id") or "") for hit in final_hits]
    if any(not chunk_id or chunk_id not in pool_ids for chunk_id in final_ids):
        raise HoldoutError("LEDGER reranker returned a hit outside its input pool")
    if len(set(final_ids)) != len(final_ids):
        raise HoldoutError("LEDGER reranker returned duplicate chunk_id")


def _graded_ndcg(
    ranked_doc_ids: list[str],
    qrels: Mapping[str, int],
    *,
    k: int,
) -> float:
    def _dcg(relevances: list[int]) -> float:
        return sum(
            (2**relevance - 1) / math.log2(rank + 1)
            for rank, relevance in enumerate(relevances, start=1)
        )

    seen: set[str] = set()
    actual: list[int] = []
    for doc_id in ranked_doc_ids[:k]:
        if doc_id in seen:
            actual.append(0)
            continue
        seen.add(doc_id)
        actual.append(int(qrels.get(doc_id, 0)))
    ideal = sorted((int(value) for value in qrels.values()), reverse=True)[:k]
    denominator = _dcg(ideal)
    return 0.0 if denominator <= 0.0 else _dcg(actual) / denominator


def _nonnegative_meta_int(
    meta: Mapping[str, Any],
    field: str,
    *,
    source: str,
) -> int:
    value = meta.get(field)
    if isinstance(value, bool):
        raise HoldoutError(f"LEDGER {source} metadata field {field} is invalid")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise HoldoutError(
            f"LEDGER {source} metadata needs integer {field}"
        ) from exc
    if value != normalized or normalized < 0:
        raise HoldoutError(f"LEDGER {source} metadata field {field} is invalid")
    return normalized


def _score_arm(
    *,
    query_id: str,
    qrels: Mapping[str, int],
    pool: list[dict[str, Any]],
    final_hits: list[dict[str, Any]],
) -> dict[str, Any]:
    positives = {doc_id for doc_id, relevance in qrels.items() if relevance > 0}
    pool_doc_ids = [_ledger_hit_doc_id(hit) for hit in pool]
    final_doc_ids = [_ledger_hit_doc_id(hit) for hit in final_hits[:10]]
    pool_hit = any(doc_id in positives for doc_id in pool_doc_ids)
    first_positive_rank = next(
        (rank for rank, doc_id in enumerate(final_doc_ids, start=1) if doc_id in positives),
        0,
    )
    hit_at_10 = first_positive_rank > 0
    if hit_at_10:
        failure_class = "hit_at_10"
    elif pool_hit:
        failure_class = "gold_in_pool_not_in_final_top10"
    else:
        failure_class = "gold_not_in_rerank_pool"
    return {
        "case_id": query_id,
        "pool_size": len(pool),
        "final_size": len(final_doc_ids),
        "gold_page_count": len(positives),
        "pool_hit": pool_hit,
        "hit_at_5": float(
            any(doc_id in positives for doc_id in final_doc_ids[:5])
        ),
        "hit_at_10": float(hit_at_10),
        "mrr": round(1.0 / first_positive_rank if first_positive_rank else 0.0, 4),
        "ndcg_at_10": round(_graded_ndcg(final_doc_ids, qrels, k=10), 4),
        "unique_pages_top10": unique_pages_top_k(final_hits, k=10),
        "page_identity_coverage_top10": page_identity_coverage_top_k(
            final_hits, k=10
        ),
        "duplicate_page_occupancy_top10": duplicate_page_occupancy(
            final_hits, k=10
        ),
        "failure_class": failure_class,
    }


def score_ledger_public_dev(
    dataset: LedgerPublicDevDataset,
    *,
    retrieve_candidates: CandidateRetriever,
    retrieval_requires_remote: bool,
    rerank: RerankCallable | None = None,
    allow_remote: bool = False,
    max_cases: int | None = None,
) -> dict[str, Any]:
    """Score A_prod/R_page from one shared Top-20 retrieval result per query."""
    if rerank is not None and not allow_remote:
        raise HoldoutError("LEDGER reranking requires explicit allow_remote=True")
    if retrieval_requires_remote and not allow_remote:
        raise HoldoutError(
            "LEDGER remote retrieval requires explicit allow_remote=True"
        )
    if max_cases is not None and max_cases <= 0:
        raise ValueError("max_cases must be > 0")
    queries = list(dataset.queries[:max_cases])
    if not queries:
        raise HoldoutError("LEDGER public_dev scoring selection is empty")

    corpus_doc_ids = {
        str(document["ledger_doc_id"]) for document in dataset.page_documents
    }
    arm_rows: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in ARM_SPECS
    }
    retrieval_calls = 0
    retrieval_remote_calls = 0
    rerank_calls = 0
    rerank_attempts = 0
    rerank_fallbacks = 0
    arm_rerank_attempts = {arm: 0 for arm in ARM_SPECS}

    for query in queries:
        query_text = str(query["query_text"])
        company_key = str(query["company_key"])
        hits, retrieval_meta = retrieve_candidates(
            query_text,
            company_key,
            ARM_SPECS["A_prod"].source_k,
        )
        retrieval_calls += 1
        query_retrieval_remote_calls = _nonnegative_meta_int(
            retrieval_meta,
            "remote_calls",
            source="retrieval",
        )
        if query_retrieval_remote_calls and not retrieval_requires_remote:
            raise HoldoutError(
                "LEDGER retrieval performed undeclared remote calls"
            )
        retrieval_remote_calls += query_retrieval_remote_calls
        _validate_candidates(
            hits,
            company_key=company_key,
            corpus_doc_ids=corpus_doc_ids,
            expected_max=ARM_SPECS["A_prod"].source_k,
        )
        qrels = {
            str(item["doc_id"]): int(item["relevance"])
            for item in query["qrels"]
        }
        for arm_name, spec in ARM_SPECS.items():
            pool = prepare_rerank_pool(hits, arm=spec)
            if rerank is None:
                final_hits = list(pool[: spec.final_k])
            else:
                final_hits, rerank_meta = rerank(
                    query_text,
                    pool,
                    spec.final_k,
                    arm_name,
                )
                rerank_calls += 1
                attempts = _nonnegative_meta_int(
                    rerank_meta,
                    "rerank_attempts",
                    source="reranker",
                )
                rerank_attempts += attempts
                arm_rerank_attempts[arm_name] += attempts
                rerank_fallbacks += int(
                    bool(rerank_meta.get("rerank_fallback"))
                )
                _validate_reranked(
                    final_hits,
                    pool=pool,
                    final_k=spec.final_k,
                )
            arm_rows[arm_name].append(
                _score_arm(
                    query_id=str(query["query_id"]),
                    qrels=qrels,
                    pool=pool,
                    final_hits=final_hits,
                )
            )

    summaries = {
        arm: summarize_ranking_cases(rows, arm=arm)
        for arm, rows in arm_rows.items()
    }
    for arm_name, summary in summaries.items():
        summary["scoring_status"] = (
            "public_dev_remote_rerank" if rerank is not None else "public_dev_offline_prerank"
        )
        summary["remote_calls"] = arm_rerank_attempts[arm_name]
    return {
        "schema_version": "lumenfin_ledger_public_dev_scoring.v1",
        "split": "public_dev",
        "cases": len(queries),
        "arms": summaries,
        "call_accounting": {
            "retrieval_calls": retrieval_calls,
            "retrieval_remote_calls": retrieval_remote_calls,
            "rerank_calls": rerank_calls,
            "rerank_attempts": rerank_attempts,
            "rerank_fallbacks": rerank_fallbacks,
            "remote_calls": retrieval_remote_calls + rerank_attempts,
        },
        "primary_comparison_valid": rerank is not None and rerank_fallbacks == 0,
        "per_case": arm_rows,
    }
