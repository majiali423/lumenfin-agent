"""Pure offline ranking arms and page-level metrics for synthetic validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from lumenfin.eval.financebench.metrics import (
    hit_at_k,
    mean,
    mean_reciprocal_rank,
    ndcg_at_k,
)

from .page_collapse import (
    collapse_to_unique_pages,
    duplicate_page_occupancy,
    normalize_document_identity,
    page_identity_coverage_top_k,
    page_key,
    unique_pages_top_k,
)


@dataclass(frozen=True)
class RankingArm:
    name: str
    pool_strategy: str
    rerank_k: int
    final_k: int


ARM_SPECS = {
    "A_prod": RankingArm(
        name="A_prod",
        pool_strategy="ranked_chunks",
        rerank_k=20,
        final_k=10,
    ),
    "R_page": RankingArm(
        name="R_page",
        pool_strategy="unique_pages_backfilled",
        rerank_k=20,
        final_k=10,
    ),
}


def prepare_rerank_pool(
    ranked_hits: list[dict[str, Any]],
    *,
    arm: str | RankingArm,
) -> list[dict[str, Any]]:
    spec = ARM_SPECS[arm] if isinstance(arm, str) else arm
    if spec.pool_strategy == "ranked_chunks":
        return list(ranked_hits[: spec.rerank_k])
    if spec.pool_strategy == "unique_pages_backfilled":
        return collapse_to_unique_pages(ranked_hits, k=spec.rerank_k)
    raise ValueError(f"unsupported holdout pool strategy: {spec.pool_strategy}")


def gold_page_keys(question: dict[str, Any]) -> set[tuple[str, int]]:
    gold: set[tuple[str, int]] = set()
    for item in question.get("evidence") or []:
        doc_name = normalize_document_identity(item.get("evidence_doc_name"))
        page = item.get("evidence_page_num_one")
        if not doc_name or isinstance(page, bool):
            continue
        try:
            page_one = int(page)
        except (TypeError, ValueError):
            continue
        if page_one > 0:
            gold.add((doc_name, page_one))
    return gold


def ranked_page_keys(hits: Iterable[dict[str, Any]]) -> list[object]:
    """Preserve every rank slot; unidentified pages remain non-relevant ``None``."""
    return [page_key(hit) for hit in hits]


def evaluate_ranking_case(
    question: dict[str, Any],
    *,
    rerank_pool: list[dict[str, Any]],
    final_hits: list[dict[str, Any]],
    final_k: int = 10,
) -> dict[str, Any]:
    if final_k <= 0:
        raise ValueError("final_k must be > 0")
    gold = gold_page_keys(question)
    if not gold:
        raise ValueError("question must contain at least one valid gold page")
    pool_pages = ranked_page_keys(rerank_pool)
    final_pages = ranked_page_keys(final_hits[:final_k])
    pool_hit = bool(hit_at_k(pool_pages, gold, k=len(pool_pages)))
    hit_10 = bool(hit_at_k(final_pages, gold, k=min(10, final_k)))
    if hit_10:
        failure_class = "hit_at_10"
    elif pool_hit:
        failure_class = "gold_in_pool_not_in_final_top10"
    else:
        failure_class = "gold_not_in_rerank_pool"
    return {
        "case_id": str(question.get("case_id") or ""),
        "pool_size": len(rerank_pool),
        "final_size": min(len(final_hits), final_k),
        "gold_page_count": len(gold),
        "pool_hit": pool_hit,
        "hit_at_5": hit_at_k(final_pages, gold, k=min(5, final_k)),
        "hit_at_10": float(hit_10),
        "mrr": round(mean_reciprocal_rank(final_pages, gold), 4),
        "ndcg_at_10": round(ndcg_at_k(final_pages, gold, k=min(10, final_k)), 4),
        "unique_pages_top10": unique_pages_top_k(final_hits, k=min(10, final_k)),
        "page_identity_coverage_top10": page_identity_coverage_top_k(
            final_hits, k=min(10, final_k)
        ),
        "duplicate_page_occupancy_top10": duplicate_page_occupancy(
            final_hits, k=min(10, final_k)
        ),
        "failure_class": failure_class,
    }


def summarize_ranking_cases(
    rows: list[dict[str, Any]],
    *,
    arm: str | RankingArm,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty holdout ranking result")
    spec = ARM_SPECS[arm] if isinstance(arm, str) else arm
    counts = {
        name: sum(str(row.get("failure_class") or "") == name for row in rows)
        for name in (
            "hit_at_10",
            "gold_in_pool_not_in_final_top10",
            "gold_not_in_rerank_pool",
        )
    }
    return {
        "arm": asdict(spec),
        "cases": len(rows),
        "pool_hit_rate": round(mean(float(bool(row.get("pool_hit"))) for row in rows), 4),
        "page_hit_at_5": round(mean(float(row.get("hit_at_5") or 0) for row in rows), 4),
        "page_hit_at_10": round(mean(float(row.get("hit_at_10") or 0) for row in rows), 4),
        "mrr": round(mean(float(row.get("mrr") or 0) for row in rows), 4),
        "ndcg_at_10": round(mean(float(row.get("ndcg_at_10") or 0) for row in rows), 4),
        "mean_unique_pages_top10": round(
            mean(float(row.get("unique_pages_top10") or 0) for row in rows), 4
        ),
        "mean_page_identity_coverage_top10": round(
            mean(
                float(row.get("page_identity_coverage_top10") or 0)
                for row in rows
            ),
            4,
        ),
        "mean_duplicate_page_occupancy_top10": round(
            mean(
                float(row.get("duplicate_page_occupancy_top10") or 0)
                for row in rows
            ),
            4,
        ),
        "failure_class_counts": counts,
        "scoring_status": "synthetic_offline_only",
        "remote_calls": 0,
    }
