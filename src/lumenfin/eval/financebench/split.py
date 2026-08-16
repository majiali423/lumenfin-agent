from __future__ import annotations

import hashlib
from typing import Iterable

from .constants import (
    FROZEN_SPLITS,
    SPLIT_ALIASES,
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
    """Assign development/test splits independently of JSONL file order.

    All ids are sorted by ``sha256(salt|financebench_id)``, then the first
    ``n_dev`` become ``dev`` and the remainder become ``test``. The mapping is
    therefore stable for a given id set.
    """
    unique_ids = sorted({question.financebench_id for question in questions}, key=_sort_key)
    total = len(unique_ids)
    canonical_total = SPLIT_DEV_SIZE + SPLIT_TEST_SIZE
    if n_dev == SPLIT_DEV_SIZE and n_test == SPLIT_TEST_SIZE and total != canonical_total:
        n_dev = min(SPLIT_DEV_SIZE, (total * SPLIT_DEV_SIZE) // canonical_total)
        n_test = total - n_dev
    expected = n_dev + n_test
    if total != expected:
        raise SplitError(
            f"split requires exactly {expected} unique financebench_id values, "
            f"found {total}"
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
        "split_status": {
            "dev": "confirmation",
            "test": "exposed_test",
        },
        "held_out_claim": "test-100 is exposed; confirmation-50 is the remaining unseen split",
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


def canonicalize_split(split: str) -> str:
    normalized = str(split or "").strip().lower()
    return SPLIT_ALIASES.get(normalized, normalized)


def experiment_governance(split: str, index_scope: str) -> dict[str, str | bool]:
    """Name the experimental role so exposed test-100 is never called held-out."""
    canonical = canonicalize_split(split)
    if canonical == "test":
        split_status = "exposed_test"
        if index_scope == "company":
            role = "post_hoc_paired_diagnostic"
        else:
            role = "exploratory_baseline"
        held_out = False
    elif canonical == "dev":
        split_status = "confirmation"
        role = "confirmation"
        held_out = True
    elif canonical == "all":
        split_status = "all"
        role = (
            "post_hoc_paired_diagnostic"
            if index_scope == "company"
            else "exploratory_baseline"
        )
        held_out = False
    else:
        split_status = canonical or "unknown"
        role = "unknown"
        held_out = False
    return {
        "canonical_split": canonical,
        "split_status": split_status,
        "experiment_role": role,
        "held_out": held_out,
        "held_out_claim": "forbidden" if not held_out else "confirmation_only",
    }


def questions_for_split(
    questions: Iterable[FinanceBenchQuestion],
    assignment: dict[str, SplitName],
    split: str,
) -> list[FinanceBenchQuestion]:
    canonical = canonicalize_split(split)
    if canonical == "all":
        return list(questions)
    if canonical not in {"dev", "test"}:
        raise SplitError(f"unknown split {split!r}; expected dev, test, confirmation, or all")
    return [item for item in questions if assignment[item.financebench_id] == canonical]


def forbid_test_split_tuning(split: str, *, tuning: bool) -> None:
    """No FinanceBench split may be used to fit thresholds after test-100 exposure."""
    if not tuning:
        return
    canonical = canonicalize_split(split)
    if canonical in FROZEN_SPLITS or canonical not in TUNABLE_SPLITS:
        raise SplitError(
            "FinanceBench splits are frozen. test-100 is an exposed exploratory "
            "baseline; confirmation-50 may be scored once after a frozen config "
            "and must not be used for threshold tuning, prompt fitting, or case deletion"
        )
