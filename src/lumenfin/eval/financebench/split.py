from __future__ import annotations

import hashlib
from typing import Iterable

from .constants import (
    FROZEN_SPLITS,
    SPLIT_DEV_SIZE,
    SPLIT_SALT,
    SPLIT_TEST_SIZE,
    SPLIT_VERSION,
    TUNABLE_SPLITS,
)
from .schema import FinanceBenchQuestion, SplitName


class SplitError(ValueError):
    """Raised when a deterministic split cannot be assigned."""


def _sort_key(financebench_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{SPLIT_SALT}|{financebench_id}".encode("utf-8")).hexdigest()
    return digest, financebench_id


def assign_splits(
    questions: Iterable[FinanceBenchQuestion],
    *,
    n_dev: int = SPLIT_DEV_SIZE,
    n_test: int = SPLIT_TEST_SIZE,
) -> dict[str, SplitName]:
    unique_ids = sorted({question.financebench_id for question in questions}, key=_sort_key)
    total = len(unique_ids)
    canonical_total = SPLIT_DEV_SIZE + SPLIT_TEST_SIZE
    if n_dev == SPLIT_DEV_SIZE and n_test == SPLIT_TEST_SIZE and total != canonical_total:
        n_dev = min(SPLIT_DEV_SIZE, (total * SPLIT_DEV_SIZE) // canonical_total)
        n_test = total - n_dev
    expected = n_dev + n_test
    if total != expected:
        raise SplitError(
            f"split requires exactly {expected} unique financebench_id values, found {total}"
        )
    mapping: dict[str, SplitName] = {}
    for index, financebench_id in enumerate(unique_ids):
        mapping[financebench_id] = "dev" if index < n_dev else "test"
    return mapping


def split_manifest(
    questions: Iterable[FinanceBenchQuestion],
    assignment: dict[str, SplitName],
) -> dict[str, object]:
    ordered = sorted(questions, key=lambda item: item.financebench_id)
    n_dev = sum(1 for split in assignment.values() if split == "dev")
    n_test = sum(1 for split in assignment.values() if split == "test")
    return {
        "version": SPLIT_VERSION,
        "salt": SPLIT_SALT,
        "n_dev": n_dev,
        "n_test": n_test,
        "rule": "sha256(salt|financebench_id) lexicographic; first n_dev -> dev, rest -> test",
        "order_independent": True,
        "cases": [
            {
                "financebench_id": question.financebench_id,
                "case_id": question.case_id,
                "split": assignment[question.financebench_id],
                "company": question.company,
                "question_type": question.question_type,
            }
            for question in ordered
        ],
    }


def questions_for_split(
    questions: Iterable[FinanceBenchQuestion],
    assignment: dict[str, SplitName],
    split: str,
) -> list[FinanceBenchQuestion]:
    if split == "all":
        return list(questions)
    if split not in {"dev", "test"}:
        raise SplitError(f"unknown split {split!r}; expected dev, test, or all")
    return [item for item in questions if assignment[item.financebench_id] == split]


def forbid_test_split_tuning(split: str, *, tuning: bool) -> None:
    normalized = str(split or "").strip().lower()
    if not tuning:
        return
    if normalized in FROZEN_SPLITS or normalized == "all":
        raise SplitError(
            "test split is frozen and must not be used for threshold tuning, "
            "prompt fitting, or case deletion"
        )
    if normalized not in TUNABLE_SPLITS:
        raise SplitError(f"unknown split {split!r} for tuning")
