from __future__ import annotations

import re
from typing import Any, Iterable

from .loader import normalize_doc_name
from .schema import CaseQrels, ChunkQrel, FinanceBenchQuestion, GoldPage, MatchReason

_WS_RE = re.compile(r"\s+")
MIN_SPAN_CHARS = 40
MIN_SHORT_SPAN_CHARS = 12


def normalize_span(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "").strip().lower())


def gold_pages_for(question: FinanceBenchQuestion) -> tuple[GoldPage, ...]:
    unique: dict[tuple[str, int], GoldPage] = {}
    for span in question.evidence:
        page = GoldPage(
            doc_name=span.evidence_doc_name,
            page_one=span.evidence_page_num_one,
            page_zero=span.evidence_page_num_zero,
        )
        unique[page.key] = page
    return tuple(unique.values())


def page_qrel_records(questions: Iterable[FinanceBenchQuestion]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for question in questions:
        for span in question.evidence:
            records.append(
                {
                    "query_id": question.case_id,
                    "financebench_id": question.financebench_id,
                    "evidence_doc_name": span.evidence_doc_name,
                    "evidence_page_num_zero": span.evidence_page_num_zero,
                    "evidence_page_num_one": span.evidence_page_num_one,
                    "evidence_text": span.evidence_text,
                }
            )
    return records


def _same_document(chunk: dict[str, Any], doc_name: str) -> bool:
    candidates = [
        chunk.get("document_id"),
        chunk.get("source_document_id"),
        chunk.get("filename"),
        chunk.get("doc_name"),
    ]
    normalized_gold = normalize_doc_name(doc_name)
    return any(normalize_doc_name(str(value or "")) == normalized_gold for value in candidates)


def _chunk_page(chunk: dict[str, Any]) -> int | None:
    page = chunk.get("page")
    if page is None or page == "":
        return None
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def auditable_span_overlap(chunk_text: str, evidence_text: str) -> bool:
    chunk_norm = normalize_span(chunk_text)
    evidence_norm = normalize_span(evidence_text)
    if not chunk_norm or not evidence_norm:
        return False
    if len(evidence_norm) < MIN_SHORT_SPAN_CHARS:
        return False
    if evidence_norm in chunk_norm:
        return True
    if len(chunk_norm) >= MIN_SPAN_CHARS and chunk_norm in evidence_norm:
        return True
    return False


def map_chunks_to_qrels(
    question: FinanceBenchQuestion,
    chunks: Iterable[dict[str, Any]],
) -> CaseQrels:
    notes: list[str] = []
    pages = gold_pages_for(question)
    page_provenance_ok = True
    qrels: list[ChunkQrel] = []
    seen: set[tuple[str, str]] = set()
    chunk_list = list(chunks)
    if not chunk_list:
        notes.append("no_chunks_for_mapping")
    for chunk in chunk_list:
        chunk_id = str(chunk.get("chunk_id") or "")
        if not chunk_id:
            notes.append("chunk_missing_id")
            continue
        page = _chunk_page(chunk)
        if page is None:
            page_provenance_ok = False
            notes.append(f"missing_page:{chunk_id}")
            continue
        for span in question.evidence:
            if not _same_document(chunk, span.evidence_doc_name):
                continue
            reason: MatchReason | None = None
            if page == span.evidence_page_num_one:
                reason = "page_cover"
            elif auditable_span_overlap(str(chunk.get("text") or ""), span.evidence_text):
                reason = "span_overlap"
            if reason is None:
                continue
            key = (chunk_id, reason)
            if key in seen:
                continue
            seen.add(key)
            qrels.append(
                ChunkQrel(
                    chunk_id=chunk_id,
                    doc_name=span.evidence_doc_name,
                    page_one=page,
                    match_reason=reason,
                )
            )
    if not page_provenance_ok:
        notes.append("page_provenance_gap")
    if not pages:
        notes.append("no_gold_pages")
    if pages and not qrels:
        notes.append("no_relevant_chunks")
    gold_chunk_ids = tuple(dict.fromkeys(item.chunk_id for item in qrels))
    return CaseQrels(
        case_id=question.case_id,
        financebench_id=question.financebench_id,
        gold_pages=pages,
        gold_chunk_ids=gold_chunk_ids,
        chunk_qrels=tuple(qrels),
        page_provenance_ok=page_provenance_ok,
        notes=tuple(dict.fromkeys(notes)),
    )


def retrieved_page_keys(hits: list[dict[str, Any]]) -> list[tuple[str, int]]:
    ordered: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for hit in hits:
        doc_name = normalize_doc_name(
            str(hit.get("document_id") or hit.get("filename") or hit.get("doc_name") or "")
        )
        page = _chunk_page(hit)
        if not doc_name or page is None:
            continue
        key = (doc_name, page)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered
