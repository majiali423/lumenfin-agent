"""Eval-only LEDGER numeric answer scoring. Does not change production RAG."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .governance import HoldoutError
from .ledger import iter_ledger_parquet_rows
from ...structured_answer import (
    AllowedEvidence,
    CITATION_SOURCE_LEGACY_TEXT,
    CITATION_SOURCE_STRUCTURED,
    CITATION_SOURCE_UNAVAILABLE,
    STRUCTURED_ANSWER_SCHEMA_VERSION,
    StructuredAnswerError,
    validate_structured_answer,
)

SCHEMA_VERSION = "lumenfin_ledger_e2e_scoring.v1"
RELATIVE_TOLERANCE = 0.01
SCALE_FACTORS = (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
SYSTEM_PROMPT = (
    "Extract one KPI number from the numbered passages. "
    "Reply with JSON only: "
    '{"value": <number-or-null>, "cited_chunk_ids": [<id>], "abstain": <bool>}. '
    "Use null and abstain=true when the passages do not contain the answer. "
    "Do not invent numbers."
)


def load_ledger_gold_values(
    parquet_path: str | Any,
    *,
    query_ids: Iterable[str],
) -> dict[str, float]:
    wanted = {str(query_id) for query_id in query_ids}
    if not wanted:
        raise HoldoutError("LEDGER e2e gold-value selection is empty")
    found: dict[str, float] = {}
    for row in iter_ledger_parquet_rows(parquet_path):
        query_id = str(row.get("query_id") or "")
        if query_id not in wanted or query_id in found:
            continue
        found[query_id] = _require_finite_float(row.get("value"), field="gold value")
        if len(found) == len(wanted):
            break
    missing = wanted - set(found)
    if missing:
        raise HoldoutError("LEDGER e2e gold values are incomplete")
    return found


def _require_finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise HoldoutError(f"LEDGER e2e {field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise HoldoutError(f"LEDGER e2e {field} must be a finite number") from exc
    if not math.isfinite(number):
        raise HoldoutError(f"LEDGER e2e {field} must be a finite number")
    return number


def parse_answer_payload(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise HoldoutError("LEDGER e2e generator returned empty text")
    match = JSON_OBJECT_RE.search(text)
    if match is None:
        raise HoldoutError("LEDGER e2e generator did not return JSON")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise HoldoutError("LEDGER e2e generator JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise HoldoutError("LEDGER e2e generator JSON must be an object")
    abstain = bool(payload.get("abstain"))
    raw_value = payload.get("value")
    value: float | None
    if raw_value is None:
        value = None
    else:
        value = _require_finite_float(raw_value, field="predicted value")
    cited = payload.get("citations")
    source = CITATION_SOURCE_STRUCTURED
    if cited is None:
        cited = payload.get("cited_chunk_ids") or []
        source = CITATION_SOURCE_LEGACY_TEXT if cited else CITATION_SOURCE_UNAVAILABLE
    if not isinstance(cited, list) or any(not str(item).strip() for item in cited):
        raise HoldoutError("LEDGER e2e cited_chunk_ids must be non-empty strings")
    cited_ids = [str(item).strip() for item in cited]
    if len(set(cited_ids)) != len(cited_ids):
        raise HoldoutError("LEDGER e2e cited_chunk_ids contain duplicates")
    schema_version = str(
        payload.get("structured_answer_schema_version") or payload.get("schema_version") or ""
    )
    if schema_version and schema_version != STRUCTURED_ANSWER_SCHEMA_VERSION:
        raise HoldoutError("LEDGER e2e structured_answer_schema_version is unsupported")
    if payload.get("citations") is not None:
        source = CITATION_SOURCE_STRUCTURED
        schema_version = schema_version or STRUCTURED_ANSWER_SCHEMA_VERSION
    if abstain and value is not None:
        raise HoldoutError("LEDGER e2e abstain cannot include a numeric value")
    return {
        "value": value,
        "cited_chunk_ids": cited_ids,
        "citations": cited_ids,
        "abstain": abstain,
        "citation_source": source if cited_ids else CITATION_SOURCE_UNAVAILABLE,
        "structured_answer_schema_version": schema_version or None,
        "answer": str(payload.get("answer") or ""),
    }


def numeric_match(
    predicted: float | None,
    gold: float,
    *,
    relative_tolerance: float = RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    if predicted is None or not math.isfinite(predicted):
        return {
            "matched": False,
            "scale_factor": None,
            "relative_error": None,
        }
    best: tuple[float, float] | None = None
    for scale in SCALE_FACTORS:
        scaled = predicted * scale
        denominator = abs(gold) if gold != 0.0 else 1.0
        error = abs(scaled - gold) / denominator
        if best is None or error < best[0]:
            best = (error, scale)
    assert best is not None
    return {
        "matched": best[0] <= relative_tolerance,
        "scale_factor": best[1],
        "relative_error": round(best[0], 6),
    }


def citation_supported(
    cited_chunk_ids: list[str],
    hits: list[Mapping[str, Any]],
    qrels: Mapping[str, int],
) -> bool:
    by_chunk = {
        str(hit.get("chunk_id") or ""): str(hit.get("document_id") or "")
        for hit in hits
    }
    positives = {doc_id for doc_id, relevance in qrels.items() if int(relevance) > 0}
    for chunk_id in cited_chunk_ids:
        document_id = by_chunk.get(chunk_id)
        if document_id and document_id in positives:
            return True
    return False


def build_generation_prompt(
    *,
    query_text: str,
    hits: list[Mapping[str, Any]],
    max_document_chars: int,
) -> str:
    lines = [f"Question: {query_text}", "", "Passages:"]
    for index, hit in enumerate(hits, start=1):
        chunk_id = str(hit.get("chunk_id") or "")
        text = str(hit.get("text") or "")[: max(1, int(max_document_chars))]
        lines.append(f"[{index}] chunk_id={chunk_id}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def score_generated_answer(
    *,
    gold_value: float,
    parsed: Mapping[str, Any],
    hits: list[Mapping[str, Any]],
    qrels: Mapping[str, int],
) -> dict[str, Any]:
    match = numeric_match(parsed.get("value"), gold_value)
    cited = list(parsed.get("citations") or parsed.get("cited_chunk_ids") or [])
    supported = citation_supported(cited, hits, qrels)
    accounting = account_ledger_citations(
        cited_chunk_ids=cited,
        hits=hits,
        qrels=qrels,
        citation_source=str(parsed.get("citation_source") or CITATION_SOURCE_UNAVAILABLE),
        tenant_id=str(parsed.get("tenant_id") or ""),
        session_id=str(parsed.get("session_id") or ""),
    )
    abstain = bool(parsed.get("abstain"))
    if abstain:
        outcome = "abstain"
    elif match["matched"]:
        outcome = "numeric_match"
    else:
        outcome = "numeric_miss"
    return {
        "numeric_match": bool(match["matched"]),
        "citation_supported": supported,
        "abstain": abstain,
        "scale_factor": match["scale_factor"],
        "relative_error": match["relative_error"],
        "outcome": outcome,
        "citation_accounting": accounting,
        "citation_source": accounting["citation_source"],
    }


def account_ledger_citations(
    *,
    cited_chunk_ids: list[str],
    hits: list[Mapping[str, Any]],
    qrels: Mapping[str, int],
    citation_source: str,
    tenant_id: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    allowed = [
        AllowedEvidence(
            chunk_id=str(hit.get("chunk_id") or ""),
            tenant_id=str(hit.get("tenant_id") or tenant_id),
            session_id=str(hit.get("session_id") or session_id),
            verified=not bool(hit.get("unverified")),
            stale=bool(hit.get("stale_repair_attempt")),
        )
        for hit in hits
        if str(hit.get("chunk_id") or "").strip()
    ]
    unknown = 0
    unverified = 0
    cross_scope = 0
    valid = 0
    for chunk_id in cited_chunk_ids:
        try:
            validate_structured_answer(
                {
                    "answer": "ok",
                    "citations": [chunk_id],
                    "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                    "citation_source": CITATION_SOURCE_STRUCTURED,
                    "workflow_status": "completed",
                    "has_factual_conclusion": False,
                },
                allowed=allowed,
                expected_tenant_id=tenant_id,
                expected_session_id=session_id,
                require_citation_for_factual=False,
            )
            valid += 1
        except StructuredAnswerError as exc:
            message = str(exc)
            if "unknown" in message:
                unknown += 1
            elif "unverified" in message or "stale" in message:
                unverified += 1
            elif "tenant" in message or "session" in message:
                cross_scope += 1
            else:
                unknown += 1
    supported = citation_supported(cited_chunk_ids, hits, qrels)
    return {
        "structured_citation_present": citation_source == CITATION_SOURCE_STRUCTURED and bool(cited_chunk_ids),
        "legacy_fallback": citation_source == CITATION_SOURCE_LEGACY_TEXT,
        "no_citation": not cited_chunk_ids,
        "valid_citation": valid,
        "unknown_citation": unknown,
        "unverified_citation": unverified,
        "cross_run_or_tenant_citation": cross_scope,
        "supported_claim": bool(supported),
        "unsupported_claim": bool(cited_chunk_ids) and not supported,
        "citation_source": citation_source if cited_chunk_ids else CITATION_SOURCE_UNAVAILABLE,
    }
