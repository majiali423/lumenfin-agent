from __future__ import annotations

import math
import random
from typing import Any, Sequence

from .constants import DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_BOOTSTRAP_SEED
from .metrics import percentile_sorted


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, float | int | str]:
    """Bootstrap the paired delta (candidate - baseline) by resampling query indices."""
    if len(baseline) != len(candidate):
        raise ValueError("paired bootstrap requires aligned baseline and candidate scores")
    count = len(baseline)
    if count == 0:
        return {
            "n": 0,
            "baseline_mean": 0.0,
            "candidate_mean": 0.0,
            "mean_delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "status": "EMPTY",
        }
    base = [float(value) for value in baseline]
    cand = [float(value) for value in candidate]
    observed_delta = _mean(cand) - _mean(base)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(count) for _ in range(count)]
        sample_base = _mean([base[index] for index in indices])
        sample_cand = _mean([cand[index] for index in indices])
        deltas.append(sample_cand - sample_base)
    return {
        "n": count,
        "baseline_mean": round(_mean(base), 4),
        "candidate_mean": round(_mean(cand), 4),
        "mean_delta": round(observed_delta, 4),
        "ci95_low": round(percentile_sorted(deltas, alpha / 2), 4),
        "ci95_high": round(percentile_sorted(deltas, 1.0 - (alpha / 2)), 4),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "status": "computed",
    }


def binomial_coefficient(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def mcnemar_exact(
    baseline_only: int,
    candidate_only: int,
) -> dict[str, float | int | str]:
    """Two-sided McNemar exact test on discordant paired binary outcomes."""
    b = int(baseline_only)
    c = int(candidate_only)
    if b < 0 or c < 0:
        raise ValueError("McNemar counts must be non-negative")
    n = b + c
    if n == 0:
        p_value = 1.0
    else:
        k = min(b, c)
        tail = sum(binomial_coefficient(n, i) for i in range(k + 1))
        p_value = min(1.0, (2 * tail) / (1 << n))
    return {
        "test_name": "mcnemar_exact",
        "baseline_only": b,
        "candidate_only": c,
        "discordant": n,
        "p_value": round(p_value, 6),
        "sample_size_discordant": n,
        "status": "computed",
    }


def mcnemar_table(
    baseline_hits: Sequence[float],
    candidate_hits: Sequence[float],
) -> dict[str, float | int | str]:
    if len(baseline_hits) != len(candidate_hits):
        raise ValueError("McNemar table requires aligned paired outcomes")
    both_hit = 0
    baseline_only = 0
    candidate_only = 0
    neither_hit = 0
    for base, cand in zip(baseline_hits, candidate_hits, strict=True):
        base_hit = float(base) >= 0.5
        cand_hit = float(cand) >= 0.5
        if base_hit and cand_hit:
            both_hit += 1
        elif base_hit:
            baseline_only += 1
        elif cand_hit:
            candidate_only += 1
        else:
            neither_hit += 1
    exact = mcnemar_exact(baseline_only, candidate_only)
    return {
        "both_hit": both_hit,
        "baseline_only": baseline_only,
        "candidate_only": candidate_only,
        "neither_hit": neither_hit,
        "n": len(baseline_hits),
        **exact,
    }


def _page_scores(rows: list[dict[str, Any]], field: str, *, k: str | None = None) -> list[float]:
    values: list[float] = []
    for row in rows:
        page = row.get("page") or {}
        if field == "hit_at" and k is not None:
            values.append(float((page.get("hit_at") or {}).get(k) or 0.0))
        else:
            values.append(float(page.get(field) or 0.0))
    return values


def _rank_outcome(row: dict[str, Any]) -> int:
    return int((row.get("page") or {}).get("first_relevant_rank") or 0)


def compare_paired_systems(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    baseline_name: str,
    candidate_name: str,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    by_base = {str(row["case_id"]): row for row in baseline_rows}
    by_cand = {str(row["case_id"]): row for row in candidate_rows}
    case_ids = sorted(set(by_base) & set(by_cand))
    aligned_base = [by_base[case_id] for case_id in case_ids]
    aligned_cand = [by_cand[case_id] for case_id in case_ids]
    hit5_base = _page_scores(aligned_base, "hit_at", k="5")
    hit5_cand = _page_scores(aligned_cand, "hit_at", k="5")
    hit10_base = _page_scores(aligned_base, "hit_at", k="10")
    hit10_cand = _page_scores(aligned_cand, "hit_at", k="10")
    improved = 0
    degraded = 0
    tied = 0
    never_retrieved = 0
    for base, cand in zip(aligned_base, aligned_cand, strict=True):
        base_rank = _rank_outcome(base)
        cand_rank = _rank_outcome(cand)
        if base_rank <= 0 and cand_rank <= 0:
            never_retrieved += 1
        if cand_rank and (not base_rank or cand_rank < base_rank):
            improved += 1
        elif base_rank and (not cand_rank or cand_rank > base_rank):
            degraded += 1
        else:
            tied += 1

    def _fail_count(rows: list[dict[str, Any]], name: str) -> int:
        return sum(1 for row in rows if str(row.get("failure_class") or "") == name)

    def _ndcg10(rows: list[dict[str, Any]]) -> list[float]:
        return [
            float((row.get("page") or {}).get("ndcg_at", {}).get("10") or 0.0) for row in rows
        ]

    latencies = [float(row.get("latency_ms") or 0.0) for row in aligned_cand]
    ordered = sorted(latencies)

    def _pct(q: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return round(ordered[index], 2)

    ndcg_base = _ndcg10(aligned_base)
    ndcg_cand = _ndcg10(aligned_cand)
    mrr_base = _page_scores(aligned_base, "mrr")
    mrr_cand = _page_scores(aligned_cand, "mrr")
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "n": len(case_ids),
        "absolute": {
            "baseline_hit_at_5": round(_mean(hit5_base), 4),
            "candidate_hit_at_5": round(_mean(hit5_cand), 4),
            "baseline_hit_at_10": round(_mean(hit10_base), 4),
            "candidate_hit_at_10": round(_mean(hit10_cand), 4),
            "baseline_mrr": round(_mean(mrr_base), 4),
            "candidate_mrr": round(_mean(mrr_cand), 4),
            "baseline_ndcg_at_10": round(_mean(ndcg_base), 4),
            "candidate_ndcg_at_10": round(_mean(ndcg_cand), 4),
        },
        "paired_bootstrap": {
            "delta_hit_at_5": paired_bootstrap(hit5_base, hit5_cand, n_bootstrap=n_bootstrap, seed=seed),
            "delta_hit_at_10": paired_bootstrap(hit10_base, hit10_cand, n_bootstrap=n_bootstrap, seed=seed),
            "delta_mrr": paired_bootstrap(mrr_base, mrr_cand, n_bootstrap=n_bootstrap, seed=seed),
            "delta_ndcg_at_10": paired_bootstrap(ndcg_base, ndcg_cand, n_bootstrap=n_bootstrap, seed=seed),
        },
        "mcnemar": {
            "hit_at_5": mcnemar_table(hit5_base, hit5_cand),
            "hit_at_10": mcnemar_table(hit10_base, hit10_cand),
        },
        "rank_movement": {
            "improved": improved,
            "degraded": degraded,
            "tied": tied,
            "never_retrieved": never_retrieved,
        },
        "failure_counts": {
            "baseline_wrong_document": _fail_count(aligned_base, "wrong_document"),
            "candidate_wrong_document": _fail_count(aligned_cand, "wrong_document"),
            "baseline_miss_all": _fail_count(aligned_base, "miss_all"),
            "candidate_miss_all": _fail_count(aligned_cand, "miss_all"),
            "baseline_ingestion_failure": _fail_count(aligned_base, "ingestion_failure"),
            "candidate_ingestion_failure": _fail_count(aligned_cand, "ingestion_failure"),
        },
        "system": {
            "candidate_p50_ms": _pct(0.50),
            "candidate_p95_ms": _pct(0.95),
            "rerank_fallback": sum(1 for row in aligned_cand if row.get("rerank_fallback")),
            "provider_errors": sum(1 for row in aligned_cand if row.get("error_type")),
        },
        "significance_note": (
            "Independent CIs describe each system's uncertainty. Paired delta CI and "
            "McNemar decide whether the same queries changed. A significant p-value "
            "does not authorize a held-out claim on an exposed or dirty run."
        ),
    }
