from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_MAX_WORKERS = int(os.getenv("MAS_COMPANY_PARALLELISM", "4"))


def map_in_parallel(
    fn: Callable[[T], R],
    items: list[T],
    *,
    max_workers: int | None = None,
) -> list[R]:
    """Fan-out/fan-in helper that preserves input order."""
    if not items:
        return []
    if len(items) == 1:
        return [fn(items[0])]

    workers = max_workers or DEFAULT_MAX_WORKERS
    workers = max(1, min(workers, len(items)))
    if workers == 1:
        return [fn(item) for item in items]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))
