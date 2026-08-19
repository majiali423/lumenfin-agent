"""Eval-only LEDGER page-parent packing. Does not change production chunking."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .governance import HoldoutError
from .ledger_e2e_taxonomy import gold_number_in_text
from .section_schema import SECTION_METADATA_UNAVAILABLE, section_metadata_for

SCHEMA_VERSION = "lumenfin_ledger_parent_pack_probe.v1"
PAGE_ID_RE = re.compile(r"^(?P<report>.+)/page_(?P<page>\d{4})$")
PACK_STRATEGIES = (
    "chunk_final",
    "retrieved_page_full",
    "retrieved_page_window_1",
    "gold_page_full",
)


def parse_ledger_page_id(document_id: str) -> tuple[str, int]:
    match = PAGE_ID_RE.fullmatch(str(document_id or "").strip())
    if match is None:
        raise HoldoutError("taxonomy page document_id is not a LEDGER page id")
    return match.group("report"), int(match.group("page"))


def format_ledger_page_id(report_id: str, page_zero: int) -> str:
    if page_zero < 0:
        raise HoldoutError("LEDGER page index must be >= 0")
    return f"{report_id}/page_{page_zero:04d}"


def page_text(page_by_id: Mapping[str, str], document_id: str) -> str:
    text = page_by_id.get(document_id)
    if text is None:
        raise HoldoutError("LEDGER page text is missing from the public-dev corpus")
    return str(text)


def neighbor_page_ids(document_id: str, *, radius: int) -> list[str]:
    if radius < 0:
        raise HoldoutError("page-window radius must be >= 0")
    report_id, page_zero = parse_ledger_page_id(document_id)
    return [
        format_ledger_page_id(report_id, index)
        for index in range(page_zero - radius, page_zero + radius + 1)
        if index >= 0
    ]


def unique_document_ids(identity: Sequence[Mapping[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in identity:
        document_id = str(item.get("document_id") or "").strip()
        if not document_id:
            raise HoldoutError("parent probe identity is missing document_id")
        if document_id in seen:
            continue
        seen.add(document_id)
        ordered.append(document_id)
    return ordered


def pack_pages(
    document_ids: Sequence[str],
    page_by_id: Mapping[str, str],
    *,
    radius: int = 0,
) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for document_id in document_ids:
        for neighbor in neighbor_page_ids(document_id, radius=radius):
            if neighbor in seen:
                continue
            text = page_by_id.get(neighbor)
            if text is None:
                continue
            seen.add(neighbor)
            parts.append(str(text))
    return "\n".join(parts)


def attach_page_parent(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """Mark the LEDGER page as parent. Do not infer section titles from text."""
    document_id = str(chunk.get("document_id") or "").strip()
    if not document_id:
        raise HoldoutError("page parent requires document_id")
    parse_ledger_page_id(document_id)
    attached = dict(chunk)
    attached.update(section_metadata_for(chunk))
    attached["parent_chunk_id"] = document_id
    if attached.get("section_title") == "":
        attached["section_title"] = SECTION_METADATA_UNAVAILABLE
    return attached


def recoverability(
    *,
    gold_value: float,
    chunk_final_text: str,
    retrieved_page_ids: Sequence[str],
    gold_page_ids: Sequence[str],
    page_by_id: Mapping[str, str],
) -> dict[str, Any]:
    if not gold_page_ids:
        raise HoldoutError("parent probe case has no gold pages")
    missing_gold = [page_id for page_id in gold_page_ids if page_id not in page_by_id]
    if missing_gold:
        raise HoldoutError("LEDGER gold page text is missing from the public-dev corpus")
    unique_retrieved = unique_document_ids(
        [{"document_id": document_id} for document_id in retrieved_page_ids]
    )
    packed = {
        "chunk_final": gold_number_in_text(gold_value, chunk_final_text),
        "retrieved_page_full": gold_number_in_text(
            gold_value,
            pack_pages(unique_retrieved, page_by_id, radius=0),
        ),
        "retrieved_page_window_1": gold_number_in_text(
            gold_value,
            pack_pages(unique_retrieved, page_by_id, radius=1),
        ),
        "gold_page_full": gold_number_in_text(
            gold_value,
            pack_pages(list(gold_page_ids), page_by_id, radius=0),
        ),
    }
    if set(packed) != set(PACK_STRATEGIES):
        raise HoldoutError("parent probe strategy coverage mismatch")
    return {
        "recovered": packed,
        "retrieved_unique_pages": len(unique_retrieved),
        "gold_pages": len(gold_page_ids),
        "gold_pages_in_corpus": len(gold_page_ids),
    }
