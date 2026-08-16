from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .constants import FALLBACK_MANIFEST_NAME
from .loader import normalize_doc_name
from .schema import FinanceBenchQuestion


def load_fallback_documents(dataset_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_dir) / FALLBACK_MANIFEST_NAME
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
        return [item for item in payload["documents"] if isinstance(item, dict)]
    return []


def fallback_doc_names(records: Iterable[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in records:
        raw = item.get("doc_name") or item.get("document_id") or ""
        if raw:
            names.add(normalize_doc_name(str(raw)))
    return names


def gold_doc_names(question: FinanceBenchQuestion) -> set[str]:
    names = {normalize_doc_name(question.doc_name)}
    for span in question.evidence:
        names.add(normalize_doc_name(span.evidence_doc_name))
    return {name for name in names if name}


def question_uses_fallback(question: FinanceBenchQuestion, fallback_names: set[str]) -> bool:
    normalized = {normalize_doc_name(name) for name in fallback_names}
    return bool(gold_doc_names(question) & normalized)


def question_uses_zero_chunk(question: FinanceBenchQuestion, zero_chunk_names: set[str]) -> bool:
    normalized = {normalize_doc_name(name) for name in zero_chunk_names}
    return bool(gold_doc_names(question) & normalized)


def build_ingestion_report(
    *,
    questions: list[FinanceBenchQuestion],
    parsed_documents: dict[str, dict[str, Any]],
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    fallback_records: list[dict[str, Any]],
    missing_pdfs: list[str] | None = None,
) -> dict[str, Any]:
    fallback_names = fallback_doc_names(fallback_records)
    parsed_names = {normalize_doc_name(name) for name in parsed_documents}
    real_pdf_names = parsed_names - fallback_names
    zero_chunk: list[str] = []
    pages_parsed = 0
    chunks_created = 0
    for doc_name, parsed in parsed_documents.items():
        pages = parsed.get("pages") or []
        pages_parsed += len(pages) if isinstance(pages, list) else 0
        chunk_count = len(chunks_by_doc.get(doc_name) or [])
        chunks_created += chunk_count
        if chunk_count == 0:
            zero_chunk.append(doc_name)
    zero_chunk_names = {normalize_doc_name(name) for name in zero_chunk}
    docs_with_gold = sorted(
        {
            name
            for question in questions
            for name in gold_doc_names(question)
            if name in parsed_names
        }
    )
    fallback_questions = [
        question.case_id for question in questions if question_uses_fallback(question, fallback_names)
    ]
    zero_chunk_questions = [
        question.case_id
        for question in questions
        if question_uses_zero_chunk(question, zero_chunk_names)
    ]
    return {
        "document_count": len(parsed_documents),
        "real_pdf_count": len(real_pdf_names),
        "fallback_pdf_count": len(fallback_names & parsed_names) if parsed_names else len(fallback_names),
        "zero_chunk_documents": sorted(zero_chunk),
        "zero_chunk_status": {
            name: "ingestion_failure" for name in sorted(zero_chunk)
        },
        "pages_parsed": pages_parsed,
        "chunks_created": chunks_created,
        "documents_with_gold_questions": docs_with_gold,
        "questions_affected_by_fallback": fallback_questions,
        "questions_affected_by_zero_chunk": zero_chunk_questions,
        "missing_pdfs": list(missing_pdfs or []),
        "fallback_doc_names": sorted(fallback_names),
    }


def cohort_case_ids(
    questions: list[FinanceBenchQuestion],
    *,
    fallback_names: set[str],
    cohort: str,
) -> set[str]:
    if cohort == "all":
        return {question.case_id for question in questions}
    if cohort == "fallback":
        return {
            question.case_id
            for question in questions
            if question_uses_fallback(question, fallback_names)
        }
    if cohort == "real_pdf":
        return {
            question.case_id
            for question in questions
            if not question_uses_fallback(question, fallback_names)
        }
    raise ValueError(f"unknown cohort {cohort!r}")
