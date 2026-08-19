"""Local-only LEDGER e2e failure taxonomy. Does not change production RAG."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .governance import HoldoutError
from .ledger_e2e import RELATIVE_TOLERANCE, numeric_match

SCHEMA_VERSION = "lumenfin_ledger_e2e_failure_taxonomy.v1"
MAX_DOCUMENT_CHARS_DEFAULT = 4000
NUMBER_RE = re.compile(
    r"(?<![\d.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\d.])"
)

LEAK_CLASSES = (
    "numeric_match",
    "unsupported_match",
    "retrieval_pool_miss",
    "ranking_top10_miss",
    "evidence_gap_unselected_chunk",
    "evidence_gap_number_absent",
    "generation_abstain",
    "generation_miss",
)

NEXT_WORKSTREAM = {
    "numeric_match": "none",
    "unsupported_match": "do_not_credit_generation",
    "retrieval_pool_miss": "section_parent_retrieval",
    "ranking_top10_miss": "ranking",
    "evidence_gap_unselected_chunk": "same_page_parent_or_table_pack",
    "evidence_gap_number_absent": "section_parent_retrieval",
    "generation_abstain": "generation_prompt_unseen_queries",
    "generation_miss": "generation_prompt_unseen_queries",
}


def extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    normalized = str(text or "").replace("\u00a0", " ")
    for match in NUMBER_RE.finditer(normalized):
        raw = match.group(0).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            numbers.append(value)
    return numbers


def gold_number_in_text(
    gold_value: float,
    text: str,
    *,
    relative_tolerance: float = RELATIVE_TOLERANCE,
) -> bool:
    for predicted in extract_numbers(text):
        if numeric_match(
            predicted,
            gold_value,
            relative_tolerance=relative_tolerance,
        )["matched"]:
            return True
    return False


def _positive_doc_ids(qrels: Sequence[Mapping[str, Any]] | Mapping[str, int]) -> set[str]:
    if isinstance(qrels, Mapping):
        items = (
            {"doc_id": doc_id, "relevance": relevance}
            for doc_id, relevance in qrels.items()
        )
    else:
        items = qrels
    positives: set[str] = set()
    for item in items:
        doc_id = str(item.get("doc_id") or "").strip()
        try:
            relevance = int(item.get("relevance"))
        except (TypeError, ValueError) as exc:
            raise HoldoutError("taxonomy qrel relevance is invalid") from exc
        if doc_id and relevance > 0:
            positives.add(doc_id)
    if not positives:
        raise HoldoutError("taxonomy case has no positive qrels")
    return positives


def _hits_by_chunk(hits: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_chunk: dict[str, Mapping[str, Any]] = {}
    for hit in hits:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in by_chunk:
            raise HoldoutError("taxonomy candidate hits have missing or duplicate chunk_id")
        by_chunk[chunk_id] = hit
    return by_chunk


def _joined_text(
    hits: Sequence[Mapping[str, Any]],
    *,
    max_document_chars: int,
) -> str:
    parts = []
    limit = max(1, int(max_document_chars))
    for hit in hits:
        parts.append(str(hit.get("text") or "")[:limit])
    return "\n".join(parts)


def classify_e2e_case(
    *,
    pool_hits: Sequence[Mapping[str, Any]],
    final_identity: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]] | Mapping[str, int],
    gold_value: float,
    numeric_matched: bool,
    abstain: bool,
    max_document_chars: int = MAX_DOCUMENT_CHARS_DEFAULT,
) -> dict[str, Any]:
    positives = _positive_doc_ids(qrels)
    by_chunk = _hits_by_chunk(pool_hits)
    pool_doc_ids = {str(hit.get("document_id") or "") for hit in pool_hits}
    pool_hit = bool(positives & pool_doc_ids)
    final_chunk_ids = [str(item.get("chunk_id") or "").strip() for item in final_identity]
    if any(not chunk_id or chunk_id not in by_chunk for chunk_id in final_chunk_ids):
        raise HoldoutError("taxonomy final identity is outside the frozen pool")
    if len(set(final_chunk_ids)) != len(final_chunk_ids):
        raise HoldoutError("taxonomy final identity contains duplicate chunk_id")
    final_hits = [by_chunk[chunk_id] for chunk_id in final_chunk_ids]
    hit_at_10 = any(str(hit.get("document_id") or "") in positives for hit in final_hits)
    gold_page_pool_hits = [
        hit
        for hit in pool_hits
        if str(hit.get("document_id") or "") in positives
    ]
    number_in_final = gold_number_in_text(
        gold_value,
        _joined_text(final_hits, max_document_chars=max_document_chars),
    )
    number_on_gold_page_pool = gold_number_in_text(
        gold_value,
        _joined_text(gold_page_pool_hits, max_document_chars=max_document_chars),
    )
    if not pool_hit:
        ranking_class = "gold_not_in_rerank_pool"
        leak = "unsupported_match" if numeric_matched else "retrieval_pool_miss"
    elif not hit_at_10:
        ranking_class = "gold_in_pool_not_in_final_top10"
        leak = "unsupported_match" if numeric_matched else "ranking_top10_miss"
    else:
        ranking_class = "hit_at_10"
        if numeric_matched:
            leak = "numeric_match"
        elif number_in_final:
            leak = "generation_abstain" if abstain else "generation_miss"
        elif number_on_gold_page_pool:
            leak = "evidence_gap_unselected_chunk"
        else:
            leak = "evidence_gap_number_absent"
    if leak not in LEAK_CLASSES:
        raise HoldoutError("taxonomy produced an unknown leak class")
    return {
        "ranking_class": ranking_class,
        "leak_class": leak,
        "next_workstream": NEXT_WORKSTREAM[leak],
        "pool_hit": pool_hit,
        "hit_at_10": hit_at_10,
        "number_in_final_context": number_in_final,
        "number_on_gold_page_pool_chunks": number_on_gold_page_pool,
        "final_size": len(final_hits),
        "gold_page_pool_chunks": len(gold_page_pool_hits),
    }


def classify_parent_page_generate_case(
    *,
    pool_hits: Sequence[Mapping[str, Any]],
    final_identity: Sequence[Mapping[str, Any]],
    parent_hits: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]] | Mapping[str, int],
    gold_value: float,
    numeric_matched: bool,
    abstain: bool,
    chunk_max_document_chars: int = MAX_DOCUMENT_CHARS_DEFAULT,
    parent_max_document_chars: int,
) -> dict[str, Any]:
    """Rank from frozen chunks; score parent-page prompt text separately."""
    ranking = classify_e2e_case(
        pool_hits=pool_hits,
        final_identity=final_identity,
        qrels=qrels,
        gold_value=gold_value,
        numeric_matched=numeric_matched,
        abstain=abstain,
        max_document_chars=chunk_max_document_chars,
    )
    number_in_parent = gold_number_in_text(
        gold_value,
        _joined_text(
            parent_hits,
            max_document_chars=parent_max_document_chars,
        ),
    )
    ranking_class = ranking["ranking_class"]
    if ranking_class == "gold_not_in_rerank_pool":
        leak = "unsupported_match" if numeric_matched else "retrieval_pool_miss"
    elif ranking_class == "gold_in_pool_not_in_final_top10":
        leak = "unsupported_match" if numeric_matched else "ranking_top10_miss"
    elif numeric_matched:
        leak = "numeric_match"
    elif number_in_parent:
        leak = "generation_abstain" if abstain else "generation_miss"
    else:
        leak = "evidence_gap_number_absent"
    if leak not in LEAK_CLASSES:
        raise HoldoutError("parent-page taxonomy produced an unknown leak class")
    return {
        "ranking_class": ranking_class,
        "leak_class": leak,
        "next_workstream": NEXT_WORKSTREAM[leak],
        "pool_hit": ranking["pool_hit"],
        "hit_at_10": ranking["hit_at_10"],
        "number_in_final_context": number_in_parent,
        "number_on_gold_page_pool_chunks": ranking[
            "number_on_gold_page_pool_chunks"
        ],
        "final_size": ranking["final_size"],
        "gold_page_pool_chunks": ranking["gold_page_pool_chunks"],
        "parent_pages": len(parent_hits),
    }


def recommend_next_workstream(counts: Mapping[str, int]) -> str:
    ranking_left = int(counts.get("ranking_top10_miss") or 0)
    retrieval_left = int(counts.get("retrieval_pool_miss") or 0) + int(
        counts.get("evidence_gap_number_absent") or 0
    )
    packing_left = int(counts.get("evidence_gap_unselected_chunk") or 0)
    generation_left = int(counts.get("generation_abstain") or 0) + int(
        counts.get("generation_miss") or 0
    )
    if ranking_left >= 5:
        return "ranking"
    if retrieval_left + packing_left >= generation_left:
        return "section_parent_retrieval"
    if generation_left:
        return "generation_prompt_unseen_queries"
    return "none"
