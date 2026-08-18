"""Schema validation for a new, non-FinanceBench holdout dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .governance import HoldoutError

REQUIRED_QUESTION_FIELDS = ("case_id", "company", "doc_name", "question", "evidence")
FORBIDDEN_QUESTION_FIELDS = ("financebench_id",)


def holdout_file_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_holdout_questions(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise HoldoutError(f"missing holdout questions file: {target}")
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HoldoutError(f"cannot read holdout questions file: {target}") from exc

    rows: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise HoldoutError(
                f"corrupt JSON on line {line_no} of {target.name}"
            ) from exc
        if not isinstance(payload, dict):
            raise HoldoutError(f"holdout line {line_no} must be an object")
        _validate_question(payload, line_no=line_no)
        case_id = str(payload["case_id"]).strip()
        if case_id in seen_case_ids:
            raise HoldoutError(f"duplicate holdout case_id on line {line_no}")
        seen_case_ids.add(case_id)
        rows.append(payload)

    if not rows:
        raise HoldoutError(f"holdout questions file is empty: {target}")
    return rows


def _required_text(payload: dict[str, Any], field: str, *, line_no: int) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise HoldoutError(f"holdout line {line_no} needs non-empty {field}")
    return value.strip()


def _validate_question(payload: dict[str, Any], *, line_no: int) -> None:
    for field in FORBIDDEN_QUESTION_FIELDS:
        if field in payload:
            raise HoldoutError(
                f"holdout line {line_no} must not include {field}; "
                "do not reuse FinanceBench rows"
            )
    missing = [field for field in REQUIRED_QUESTION_FIELDS if field not in payload]
    if missing:
        raise HoldoutError(f"holdout line {line_no} missing required fields: {missing}")

    case_id = _required_text(payload, "case_id", line_no=line_no)
    if "financebench" in case_id.casefold():
        raise HoldoutError(
            f"holdout line {line_no} uses a FinanceBench-derived case_id"
        )
    _required_text(payload, "company", line_no=line_no)
    _required_text(payload, "doc_name", line_no=line_no)
    _required_text(payload, "question", line_no=line_no)

    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise HoldoutError(f"holdout line {line_no} needs non-empty evidence")
    for item in evidence:
        if not isinstance(item, dict):
            raise HoldoutError(f"holdout line {line_no} evidence must be objects")
        _required_text(item, "evidence_doc_name", line_no=line_no)
        page = item.get("evidence_page_num_one")
        if isinstance(page, bool):
            raise HoldoutError(
                f"holdout line {line_no} evidence_page_num_one must be an integer >= 1"
            )
        try:
            page_one = int(page)
        except (TypeError, ValueError) as exc:
            raise HoldoutError(
                f"holdout line {line_no} evidence_page_num_one must be an integer >= 1"
            ) from exc
        if page_one <= 0 or str(page_one) != str(page).strip():
            raise HoldoutError(
                f"holdout line {line_no} evidence_page_num_one must be an integer >= 1"
            )
