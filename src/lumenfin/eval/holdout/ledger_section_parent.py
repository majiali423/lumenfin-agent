"""Eval-only page-parent retrieval units. Does not change production chunking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .governance import HoldoutError
from .ledger_parent_probe import parse_ledger_page_id
from .section_schema import SECTION_METADATA_UNAVAILABLE, attach_section_metadata

SCHEMA_VERSION = "lumenfin_ledger_section_parent.v1"
LOCKED_INDEX_UNIT = "page_parent_full"
COLLECTION_NAME = "lumenfin_ledger_section_parent_bm25"


def parent_page_index_unit(document: Mapping[str, Any]) -> dict[str, Any]:
    """Index one LEDGER page as the retrieval unit. Never infer section titles."""
    pages = list(document.get("pages") or [])
    if len(pages) != 1:
        raise HoldoutError("section-parent page document must contain exactly one page")
    text = str(pages[0] or "").strip()
    if not text:
        raise HoldoutError("section-parent page text is empty")
    document_id = str(document.get("document_id") or "").strip()
    parse_ledger_page_id(document_id)
    companies = [
        str(item)
        for item in (
            document.get("issuer_companies")
            or document.get("detected_companies")
            or []
        )
        if str(item).strip()
    ]
    if not companies:
        raise HoldoutError("section-parent page is missing company tags")
    filename = str(document.get("filename") or "").strip()
    if not filename:
        raise HoldoutError("section-parent page is missing filename")
    page_zero = document.get("ledger_page_zero")
    try:
        page_one = int(page_zero) + 1
    except (TypeError, ValueError) as exc:
        raise HoldoutError("section-parent page index is invalid") from exc
    if page_one <= 0:
        raise HoldoutError("section-parent page index is invalid")
    chunk = {
        "chunk_id": document_id,
        "document_id": document_id,
        "filename": filename,
        "page": page_one,
        "text": text,
        "companies": companies,
        "chunk_type": "eval_parent_page",
        "char_count": len(text),
        "parent_chunk_id": document_id,
        "section_id": SECTION_METADATA_UNAVAILABLE,
        "section_title": SECTION_METADATA_UNAVAILABLE,
        "retrieval_method": "eval_section_parent_page",
    }
    attached = attach_section_metadata(chunk)
    attached["parent_chunk_id"] = document_id
    attached["section_title"] = SECTION_METADATA_UNAVAILABLE
    attached["section_id"] = SECTION_METADATA_UNAVAILABLE
    return attached


def select_company_pages(
    page_documents: Sequence[Mapping[str, Any]],
    company_keys: Sequence[str],
) -> list[dict[str, Any]]:
    wanted = {str(key) for key in company_keys}
    if not wanted:
        raise HoldoutError("section-parent company selection is empty")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document in page_documents:
        companies = {
            str(item)
            for item in (
                document.get("issuer_companies")
                or document.get("detected_companies")
                or []
            )
        }
        if not (companies & wanted):
            continue
        document_id = str(document.get("document_id") or "").strip()
        if not document_id or document_id in seen:
            raise HoldoutError("section-parent corpus has missing or duplicate pages")
        seen.add(document_id)
        selected.append(dict(document))
    if not selected:
        raise HoldoutError("section-parent corpus selected no pages")
    return selected


def pool_hit(hits: Sequence[Mapping[str, Any]], qrels: Sequence[Mapping[str, Any]]) -> bool:
    positives = {
        str(item["doc_id"])
        for item in qrels
        if int(item["relevance"]) > 0
    }
    if not positives:
        raise HoldoutError("section-parent query has no positive qrels")
    return any(str(hit.get("document_id") or "") in positives for hit in hits)


def recommend_next(*, hybrid_pool_hits: int, parent_pool_hits: int, cases: int) -> str:
    if cases <= 0:
        raise HoldoutError("section-parent recommendation has no cases")
    if parent_pool_hits <= hybrid_pool_hits:
        return "do_not_embed_page_parent_index"
    return "hybrid_page_parent_index"
