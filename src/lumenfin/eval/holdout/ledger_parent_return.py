"""Eval-only retrieve-child / return-parent-page. Does not change production RAG."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .governance import HoldoutError
from .ledger_parent_probe import page_text, unique_document_ids
from .section_schema import SECTION_METADATA_UNAVAILABLE, attach_section_metadata

LOCKED_STRATEGY = "retrieved_page_full"
LOCKED_RADIUS = 0
FROZEN_QUERIES_PER_COMPANY = 50
PREFIX_CASES_PER_COMPANY = 10
HOLDOUT_CASES_PER_COMPANY = 40
SCHEMA_VERSION = "lumenfin_ledger_parent_page_return.v1"


def select_frozen_slice(
    candidate_rows: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    *,
    start: int,
    count: int,
) -> list[dict[str, Any]]:
    if start < 0 or count <= 0:
        raise HoldoutError("frozen slice bounds are invalid")
    by_id = {str(row["query_id"]): dict(row) for row in candidate_rows}
    selected: list[dict[str, Any]] = []
    for plan in plans:
        query_ids = [str(query_id) for query_id in plan["query_ids"]]
        if len(query_ids) != FROZEN_QUERIES_PER_COMPANY:
            raise HoldoutError("frozen company query list is not 50")
        if start + count > len(query_ids):
            raise HoldoutError("frozen company slice exceeds the 50-query list")
        slice_ids = query_ids[start : start + count]
        seen: set[str] = set()
        for query_id in slice_ids:
            if query_id in seen:
                raise HoldoutError("frozen slice contains duplicate query ids")
            seen.add(query_id)
            row = by_id.get(query_id)
            if row is None:
                raise HoldoutError("frozen slice query is missing from the candidate cache")
            selected.append(row)
    expected = count * len(plans)
    if len(selected) != expected:
        raise HoldoutError("frozen slice coverage mismatch")
    return selected


def assert_disjoint_from_prefix(
    selected_ids: Sequence[str],
    prefix_ids: Sequence[str],
) -> None:
    overlap = set(selected_ids) & set(prefix_ids)
    if overlap:
        raise HoldoutError("parent-page holdout overlaps the e2e canary prefix")
    if len(set(selected_ids)) != len(list(selected_ids)):
        raise HoldoutError("parent-page holdout contains duplicate query ids")


def build_parent_page_hits(
    final_identity: Sequence[Mapping[str, Any]],
    page_by_id: Mapping[str, str],
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for document_id in unique_document_ids(final_identity):
        text = page_text(page_by_id, document_id)
        hit = {
            "chunk_id": document_id,
            "document_id": document_id,
            "text": text,
            "parent_chunk_id": document_id,
            "section_id": SECTION_METADATA_UNAVAILABLE,
            "section_title": SECTION_METADATA_UNAVAILABLE,
            "retrieval_method": "eval_parent_page_return",
        }
        hits.append(attach_section_metadata(hit))
        hits[-1]["parent_chunk_id"] = document_id
    if not hits:
        raise HoldoutError("parent page return produced no hits")
    return hits


def parent_prompt_char_cap(hits: Sequence[Mapping[str, Any]]) -> int:
    longest = max((len(str(hit.get("text") or "")) for hit in hits), default=1)
    return max(1, longest)
