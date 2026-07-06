from __future__ import annotations

import re
from typing import Any


FINANCIAL_KEYWORDS = (
    "revenue",
    "ebitda",
    "margin",
    "profit",
    "r&d",
    "research",
    "supply chain",
    "risk",
    "cash flow",
    "operating income",
    "收入",
    "营收",
    "研发",
    "供应链",
    "风险",
    "利润",
)


def _classify_chunk(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("supply chain", "供应链")) and any(
        word in lowered for word in ("risk", "风险", "constraint", "concentration")
    ):
        return "risk_signal"
    hits = sum(1 for keyword in FINANCIAL_KEYWORDS if keyword in lowered)
    if hits >= 2:
        return "financial_metric"
    if any(word in lowered for word in ("risk", "supply chain", "风险", "供应链")):
        return "risk_signal"
    return "narrative"


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if paragraphs:
        return paragraphs
    return [line.strip() for line in text.splitlines() if line.strip()]


def _merge_small_chunks(parts: list[str], max_chars: int) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}\n\n{part}".strip() if buffer else part
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            merged.append(buffer)
        if len(part) <= max_chars:
            buffer = part
        else:
            for start in range(0, len(part), max_chars):
                merged.append(part[start : start + max_chars])
            buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


def chunk_document(
    document: dict[str, Any],
    *,
    max_chunk_chars: int = 900,
    overlap_chars: int = 120,
) -> list[dict[str, Any]]:
    """Page-aware chunking with financial-signal tagging for hybrid retrieval."""
    pages: list[str] = document.get("pages") or []
    if not pages and document.get("text"):
        pages = _split_paragraphs(document["text"])

    chunks: list[dict[str, Any]] = []
    companies = document.get("detected_companies", [])
    document_id = document.get("document_id", "unknown")
    filename = document.get("filename", "unknown")

    for page_number, page_text in enumerate(pages, start=1):
        paragraphs = _merge_small_chunks(_split_paragraphs(page_text), max_chunk_chars)
        for chunk_index, paragraph in enumerate(paragraphs):
            if overlap_chars and chunk_index > 0 and len(paragraph) > overlap_chars:
                paragraph = paragraph[max(0, overlap_chars // 2) :]
            chunk_id = f"{document_id}:p{page_number}:c{chunk_index}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number,
                    "text": paragraph,
                    "companies": companies,
                    "chunk_type": _classify_chunk(paragraph),
                    "char_count": len(paragraph),
                }
            )
    return chunks
