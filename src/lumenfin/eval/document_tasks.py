"""Load independent document-task gold. Never reads Agent outputs as labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = ROOT / "tests" / "fixtures" / "document_tasks" / "lumenfin_document_tasks_v1.json"
FAMILIES = (
    "financial_fact",
    "ratio_calc",
    "multipage_period",
    "comparison_scope",
    "evidenced_risk",
    "refuse_or_clarify",
)
SPLITS = ("dev", "test", "pilot")
ALLOWED_REVIEW = {
    "source_quote_checked",
    "needs_human_review",
    "human_reviewed",
    "frozen",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    target = path or CATALOG_PATH
    payload = json.loads(target.read_text(encoding="utf-8"))
    _validate_catalog(payload)
    return payload


def _validate_catalog(payload: dict[str, Any]) -> None:
    required = (
        "dataset_id",
        "version",
        "inclusion_rules",
        "exclusion_rules",
        "source_limitations",
        "groups",
        "tasks",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"catalog missing {missing}")
    seen: set[str] = set()
    for task in payload["tasks"]:
        task_id = str(task.get("id") or "")
        if not task_id or task_id in seen:
            raise ValueError(f"duplicate or empty task id: {task_id!r}")
        seen.add(task_id)
        if task.get("family") not in FAMILIES:
            raise ValueError(f"{task_id} unknown family")
        if task.get("split") not in {"dev", "test"}:
            raise ValueError(f"{task_id} split must be dev or test")
        if task.get("expected_action") not in {"answer", "refuse", "clarify", "answer_or_clarify"}:
            raise ValueError(f"{task_id} expected_action invalid")
        for doc in task.get("allowed_documents") or []:
            rel = ROOT / str(doc["path"])
            digest = sha256_file(rel)
            if digest != doc["sha256"]:
                raise ValueError(f"{task_id} document hash mismatch for {doc['path']}")
        if task.get("gold_origin") == "product_output":
            raise ValueError(f"{task_id} gold must not come from product output")
        status = str(task.get("review_status") or "")
        if status not in ALLOWED_REVIEW:
            raise ValueError(f"{task_id} review_status {status!r} is not allowed")
        if status in {"human_reviewed", "frozen"} and not task.get("human_reviewer"):
            raise ValueError(f"{task_id} human review status requires human_reviewer")


def tasks_for(*, split: str | None = None, pilot_only: bool = False) -> list[dict[str, Any]]:
    catalog = load_catalog()
    rows = list(catalog["tasks"])
    if split:
        rows = [row for row in rows if row["split"] == split]
    if pilot_only:
        rows = [row for row in rows if row.get("pilot")]
    return rows


def group_split_map(catalog: dict[str, Any] | None = None) -> dict[str, str]:
    payload = catalog or load_catalog()
    return {str(item["id"]): str(item["split"]) for item in payload["groups"]}


def freeze_readiness(catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    """Exit-code-2 inventory. None of these blockers are program exceptions."""
    payload = catalog or load_catalog()
    test_tasks = [row for row in payload["tasks"] if row.get("split") == "test"]
    frozen_test = [row for row in test_tasks if row.get("review_status") == "frozen"]
    blocked_groups = [
        item["id"] for item in payload["groups"] if item.get("status") == "blocked_needs_source_extract"
    ]
    needs_human = [row["id"] for row in payload["tasks"] if row.get("review_status") == "needs_human_review"]
    blockers = [
        {
            "id": "no_frozen_test_gold",
            "reason": "test split has no frozen gold tasks",
            "observed": {"test_tasks": len(test_tasks), "frozen_test_tasks": len(frozen_test)},
            "unlock": "Add 60 test tasks with second-person or documented single-annotator freeze after human review; set review_status=frozen.",
        },
        {
            "id": "issuer_groups_missing_extracts",
            "reason": "planned test issuer groups lack committed source extracts",
            "observed": {"groups_blocked": blocked_groups},
            "unlock": "Commit hash-locked extracts for amd/intc/avgo/meta/googl/tsm (or replacements) and attach tasks. Do not reuse FinanceBench/LEDGER holdout as unseen.",
        },
        {
            "id": "no_independent_human_review",
            "reason": "catalog is not independently human-reviewed; fixture extract checks are not a freeze",
            "observed": {
                "catalog_review_status": payload.get("review_status"),
                "pilot_needs_human_review": needs_human,
            },
            "unlock": "A human records human_reviewer on every frozen test task. Model-assisted labels stay candidate gold.",
        },
        {
            "id": "pilot_not_a_test_freeze",
            "reason": "24-task dev pilot is not the 120-task freeze; remaining test volume is not built",
            "observed": {
                "pilot_tasks": sum(1 for row in payload["tasks"] if row.get("pilot")),
                "version": payload.get("version"),
            },
            "unlock": "Do not freeze until the test split exists and gold is reviewed. Do not pad to 120 without sources.",
        },
    ]
    return {
        "frozen": False,
        "program_exception": False,
        "exit_code": 2,
        "blockers": blockers,
        "test_tasks_in_file": len(test_tasks),
        "groups_blocked": blocked_groups,
        "needs_human_review": needs_human,
    }


def _page_text(path: Path, page: int | None) -> str:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        import fitz

        doc = fitz.open(path)
        try:
            if page is None:
                return "\n".join(doc.load_page(i).get_text() or "" for i in range(doc.page_count))
            index = int(page) - 1
            if index < 0 or index >= doc.page_count:
                return ""
            return doc.load_page(index).get_text() or ""
        finally:
            doc.close()
    return path.read_text(encoding="utf-8")


def evidence_quote_issues(task: dict[str, Any]) -> list[str]:
    """Return quote/page mismatches. Empty means quotes were found in allowed files."""
    docs = {str(item["document_id"]): ROOT / item["path"] for item in task.get("allowed_documents") or []}
    issues: list[str] = []
    evidence_rows: list[dict[str, Any]] = []
    for fact in task.get("required_facts") or []:
        for group in fact.get("accepted_evidence_sets") or []:
            evidence_rows.extend(group)
    for point in task.get("required_risk_points") or []:
        for group in point.get("accepted_evidence_sets") or []:
            evidence_rows.extend(group)
    for item in evidence_rows:
        doc_id = str(item.get("document_id") or "")
        quote = str(item.get("quote") or "").strip()
        page = item.get("page")
        path = docs.get(doc_id)
        if not quote:
            continue
        if path is None or not path.is_file():
            issues.append(f"{task.get('id')}: missing document {doc_id}")
            continue
        blob = _page_text(path, int(page) if page is not None else None)
        if quote not in blob:
            issues.append(f"{task.get('id')}: quote not on page {page} of {doc_id}")
    return issues
