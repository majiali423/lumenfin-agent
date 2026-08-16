from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

from .constants import DEFAULT_BOOTSTRAP_SAMPLES, DEFAULT_BOOTSTRAP_SEED


def hit_at_k(retrieved: Sequence[object], relevant: set[object], *, k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in retrieved[:k]) else 0.0


def recall_at_k(retrieved: Sequence[object], relevant: set[object], *, k: int) -> float:
    if not relevant:
        return 0.0
    found = {item for item in retrieved[:k] if item in relevant}
    return len(found) / len(relevant)


def mean_reciprocal_rank(retrieved: Sequence[object], relevant: set[object]) -> float:
    if not relevant:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: Sequence[int]) -> float:
    return sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg_at_k(retrieved: Sequence[object], relevant: set[object], *, k: int) -> float:
    if not relevant:
        return 0.0
    grades = [1 if item in relevant else 0 for item in retrieved[:k]]
    ideal = [1] * min(k, len(relevant))
    ideal_dcg = _dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return _dcg(grades) / ideal_dcg


def single_gold_recall_equals_hit(relevant_count: int) -> bool:
    return relevant_count == 1


def percentile_sorted(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires a non-empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("percentile q must be in [0, 1]")
    ordered = sorted(values)
    if q <= 0:
        return float(ordered[0])
    if q >= 1:
        return float(ordered[-1])
    index = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered)) - 1)))
    return float(ordered[index])


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, float | int | str]:
    observed = [float(value) for value in values]
    if not observed:
        return {
            "n": 0,
            "mean": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "status": "EMPTY",
        }
    rng = random.Random(seed)
    means: list[float] = []
    count = len(observed)
    for _ in range(n_bootstrap):
        sample = [observed[rng.randrange(count)] for _ in range(count)]
        means.append(sum(sample) / count)
    low = percentile_sorted(means, alpha / 2)
    high = percentile_sorted(means, 1.0 - (alpha / 2))
    return {
        "n": count,
        "mean": round(sum(observed) / count, 4),
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "status": "computed",
    }


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)
