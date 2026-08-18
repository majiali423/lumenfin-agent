"""Governance guards for the unseen holdout ranking phase."""

from __future__ import annotations

from pathlib import Path

from lumenfin.eval.financebench.split import canonicalize_split

HOLDOUT_SPLIT = "holdout"
CONSUMED_SPLITS = frozenset({"test", "dev", "confirmation", "all"})


class HoldoutError(ValueError):
    """Raised when a holdout request violates the frozen evaluation protocol."""


def validate_holdout_split(split: str) -> str:
    raw = str(split or "").strip().lower()
    canonical = canonicalize_split(raw)
    if raw in CONSUMED_SPLITS or canonical in CONSUMED_SPLITS:
        raise HoldoutError(
            "holdout ranking refuses consumed FinanceBench splits "
            f"(requested {split!r})"
        )
    if raw != HOLDOUT_SPLIT:
        raise HoldoutError(
            f"holdout ranking only allows --split {HOLDOUT_SPLIT}, got {split!r}"
        )
    return HOLDOUT_SPLIT


def _contains_financebench_component(path: Path) -> bool:
    return any("financebench" in part.casefold() for part in path.parts)


def validate_holdout_dataset_dir(
    dataset_dir: str | Path,
    *,
    repo_root: str | Path,
) -> Path:
    target = Path(dataset_dir).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    if _contains_financebench_component(target):
        raise HoldoutError("holdout ranking refuses FinanceBench dataset paths")
    if not target.is_dir():
        raise HoldoutError(f"holdout dataset directory not found: {target}")

    # Resolve first so a symlink inside the repository cannot point into a
    # FinanceBench checkout while appearing to be a holdout path.
    try:
        relative = target.relative_to(root)
    except ValueError:
        relative = None
    if relative is not None and _contains_financebench_component(relative):
        raise HoldoutError("holdout ranking refuses FinanceBench dataset paths")
    return target


def resolve_holdout_questions_path(
    dataset_dir: str | Path,
    questions_path: str | Path | None = None,
) -> Path:
    root = Path(dataset_dir).expanduser().resolve()
    candidate = (
        Path(questions_path).expanduser().resolve()
        if questions_path is not None
        else (root / "schema_example.jsonl").resolve()
    )
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HoldoutError(
            "holdout questions file must be inside the validated holdout dataset directory"
        ) from exc
    if _contains_financebench_component(candidate):
        raise HoldoutError("holdout ranking refuses FinanceBench question paths")
    return candidate


def validate_holdout_request(
    *,
    split: str,
    dataset_dir: str | Path,
    repo_root: str | Path,
    allow_remote: bool = False,
) -> Path:
    if allow_remote:
        raise HoldoutError(
            "holdout scaffold is validate-only; remote scoring is not enabled"
        )
    validate_holdout_split(split)
    return validate_holdout_dataset_dir(dataset_dir, repo_root=repo_root)
