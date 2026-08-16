from __future__ import annotations

import re
from typing import Any

from .schema import FinanceBenchQuestion

_NUMERIC_RE = re.compile(
    r"\b(how much|what is the|amount|ratio|margin|capex|revenue|eps|million|billion|percent|%)\b",
    re.IGNORECASE,
)
_CALC_RE = re.compile(
    r"\b(calculate|computed?|change|delta|difference|growth|cagr|exclude|if we|driven|ratio)\b",
    re.IGNORECASE,
)
_PERIOD_RE = re.compile(
    r"\b(fy\s*20\d{2}|fiscal|year[- ]end|as of|q[1-4]|compared to|versus|vs\.?)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(not|never|except|excluding|no longer|without|did not|isn't|is not)\b",
    re.IGNORECASE,
)


def classify_case(question: FinanceBenchQuestion) -> dict[str, Any]:
    gold_docs = {span.evidence_doc_name for span in question.evidence}
    gold_pages = {(span.evidence_doc_name, span.evidence_page_num_one) for span in question.evidence}
    text = question.question
    reasoning = question.question_reasoning.lower()
    numeric_extraction = bool(_NUMERIC_RE.search(text)) or "information extraction" in reasoning
    multi_step = (
        "numerical reasoning" in reasoning
        or "logical reasoning" in reasoning
        or bool(_CALC_RE.search(text))
    )
    return {
        "question_type": question.question_type or "unknown",
        "reasoning_type": question.question_reasoning or "unknown",
        "company": question.company,
        "document_type": (question.document.doc_type if question.document else "")
        or _doc_type_from_name(question.doc_name),
        "evidence_pages": "multi_page" if len(gold_pages) > 1 else "single_page",
        "numeric_extraction": numeric_extraction,
        "multi_step_calculation": multi_step,
        "period_disambiguation": bool(_PERIOD_RE.search(text)),
        "negation": bool(_NEGATION_RE.search(text)),
        "cross_document": len(gold_docs) > 1,
        "gold_page_count": len(gold_pages),
        "gold_doc_count": len(gold_docs),
    }


def _doc_type_from_name(doc_name: str) -> str:
    lowered = doc_name.lower()
    for token in ("10k", "10-k", "10q", "10-q", "8k", "8-k", "earnings", "transcript"):
        if token in lowered:
            return token.replace("-", "").upper() if token[0].isdigit() else token
    return "unknown"


def classify_failure(
    *,
    retrieved_pages: list[tuple[str, int]],
    gold_pages: set[tuple[str, int]],
    top_k: int,
    empty: bool,
    provider_error: str = "",
    degraded: bool = False,
) -> str:
    if provider_error:
        return "provider_error"
    if empty:
        return "empty_retrieval"
    if not gold_pages:
        return "invalid_gold"
    if not any(page in gold_pages for page in retrieved_pages[:top_k]):
        retrieved_docs = {doc for doc, _page in retrieved_pages[:top_k]}
        gold_docs = {doc for doc, _page in gold_pages}
        if retrieved_docs and retrieved_docs.isdisjoint(gold_docs):
            return "wrong_document"
        return "miss_all"
    if degraded:
        return "degraded_hit"
    first_rank = next(
        (rank for rank, page in enumerate(retrieved_pages, start=1) if page in gold_pages),
        None,
    )
    if first_rank and first_rank > 1:
        return "rank_gt_1"
    return "hit"
