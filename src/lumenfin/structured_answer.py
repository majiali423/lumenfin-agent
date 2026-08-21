"""Versioned structured answer + citation protocol.

Citations are stable retrieval chunk IDs from the current run's verified
evidence. This module does not invent chunk IDs from prose and does not
change retrieval defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

STRUCTURED_ANSWER_SCHEMA_VERSION = "1.0"
CITATION_SOURCE_STRUCTURED = "structured"
CITATION_SOURCE_LEGACY_TEXT = "legacy_text"
CITATION_SOURCE_UNAVAILABLE = "unavailable"
_SUPPORTED_SCHEMA_VERSIONS = frozenset({STRUCTURED_ANSWER_SCHEMA_VERSION})
_DISPLAY_MARKER_RE = re.compile(r"\[(\d+)\]")
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|DASHSCOPE_API_KEY|DEEPSEEK_API_KEY)",
    re.IGNORECASE,
)


class StructuredAnswerError(ValueError):
    """Fail-closed structured-answer validation error (no secrets, no full docs)."""


@dataclass(frozen=True)
class AllowedEvidence:
    chunk_id: str
    tenant_id: str = ""
    session_id: str = ""
    verified: bool = False
    stale: bool = False


@dataclass(frozen=True)
class StructuredAnswer:
    answer: str
    citations: tuple[str, ...] = ()
    schema_version: str = STRUCTURED_ANSWER_SCHEMA_VERSION
    citation_source: str = CITATION_SOURCE_STRUCTURED
    workflow_status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": list(self.citations),
            "structured_answer_schema_version": self.schema_version,
            "citation_source": self.citation_source,
            "workflow_status": self.workflow_status,
        }


def normalize_answer(answer: object) -> str:
    if not isinstance(answer, str):
        raise StructuredAnswerError("structured answer must be a string")
    return answer.replace("\r\n", "\n").strip()


def normalize_citations(raw: object) -> list[str]:
    if not isinstance(raw, list):
        raise StructuredAnswerError("citations must be a string array")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if item is None or isinstance(item, bool) or isinstance(item, (int, float)):
            raise StructuredAnswerError("citation ids must be non-empty strings")
        if not isinstance(item, str):
            raise StructuredAnswerError("citation ids must be non-empty strings")
        chunk_id = item.strip()
        if not chunk_id:
            raise StructuredAnswerError("citation ids must be non-empty strings")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        ordered.append(chunk_id)
    return ordered


def map_display_markers_to_chunk_ids(
    markers: Iterable[object],
    index_to_chunk_id: Mapping[int, str],
) -> list[str]:
    """Map explicit [n] display markers through a program-owned evidence table.

    Missing or unmapped markers fail closed. This never guesses IDs from prose.
    """
    mapped: list[str] = []
    seen: set[str] = set()
    for marker in markers:
        try:
            index = int(marker)
        except (TypeError, ValueError) as exc:
            raise StructuredAnswerError("display citation marker is not a passage index") from exc
        chunk_id = str(index_to_chunk_id.get(index) or "").strip()
        if not chunk_id:
            raise StructuredAnswerError("display citation marker is not in the evidence map")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        mapped.append(chunk_id)
    return mapped


def extract_display_markers(text: str) -> list[int]:
    return [int(match.group(1)) for match in _DISPLAY_MARKER_RE.finditer(text or "")]


def validate_structured_answer(
    payload: Mapping[str, Any],
    *,
    allowed: Iterable[AllowedEvidence],
    expected_tenant_id: str = "",
    expected_session_id: str = "",
    require_citation_for_factual: bool = True,
) -> StructuredAnswer:
    schema_version = str(
        payload.get("structured_answer_schema_version")
        or payload.get("schema_version")
        or STRUCTURED_ANSWER_SCHEMA_VERSION
    )
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise StructuredAnswerError("unsupported structured_answer_schema_version")
    answer = normalize_answer(payload.get("answer", ""))
    citations = normalize_citations(payload.get("citations") or [])
    workflow_status = str(payload.get("workflow_status") or "completed")
    citation_source = str(payload.get("citation_source") or CITATION_SOURCE_STRUCTURED)
    if citation_source not in {
        CITATION_SOURCE_STRUCTURED,
        CITATION_SOURCE_LEGACY_TEXT,
        CITATION_SOURCE_UNAVAILABLE,
    }:
        raise StructuredAnswerError("unsupported citation_source")

    by_id = {item.chunk_id: item for item in allowed if item.chunk_id}
    for chunk_id in citations:
        record = by_id.get(chunk_id)
        if record is None:
            raise StructuredAnswerError("citation refers to an unknown chunk")
        if record.stale:
            raise StructuredAnswerError("citation refers to a stale repair attempt")
        if not record.verified:
            raise StructuredAnswerError("citation refers to unverified evidence")
        if expected_tenant_id and record.tenant_id and record.tenant_id != expected_tenant_id:
            raise StructuredAnswerError("citation tenant does not match the current run")
        if expected_session_id and record.session_id and record.session_id != expected_session_id:
            raise StructuredAnswerError("citation session does not match the current run")

    incomplete = workflow_status == "incomplete_data" or citation_source == CITATION_SOURCE_UNAVAILABLE
    factual = (not incomplete) and _looks_factual(answer, payload)
    allowed_verified = [item for item in by_id.values() if item.verified and not item.stale]
    if require_citation_for_factual and factual and allowed_verified and not citations:
        raise StructuredAnswerError("factual verified answer requires at least one citation")
    if incomplete and _has_unsupported_ratio_claim(answer) and not citations:
        raise StructuredAnswerError("incomplete_data cannot carry unsupported financial ratios")

    return StructuredAnswer(
        answer=answer,
        citations=tuple(citations),
        schema_version=schema_version,
        citation_source=citation_source if citations or incomplete else CITATION_SOURCE_UNAVAILABLE,
        workflow_status=workflow_status,
    )


def allowed_evidence_from_state(state: Mapping[str, Any]) -> list[AllowedEvidence]:
    tenant_id = str(state.get("rag_tenant_id") or state.get("tenant_id") or "")
    session_id = str(state.get("thread_id") or state.get("run_id") or "")
    verified_ids = _verified_chunk_ids(state)
    allowed: list[AllowedEvidence] = []
    seen: set[str] = set()
    for hits in (state.get("rag_evidence") or {}).values():
        if not isinstance(hits, list):
            continue
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            chunk_id = str(hit.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            hit_tenant = str(hit.get("tenant_id") or tenant_id)
            hit_session = str(hit.get("session_id") or session_id)
            allowed.append(
                AllowedEvidence(
                    chunk_id=chunk_id,
                    tenant_id=hit_tenant,
                    session_id=hit_session,
                    verified=chunk_id in verified_ids,
                    stale=bool(hit.get("stale_repair_attempt")),
                )
            )
    for chunk_id in verified_ids:
        if chunk_id in seen:
            continue
        allowed.append(
            AllowedEvidence(
                chunk_id=chunk_id,
                tenant_id=tenant_id,
                session_id=session_id,
                verified=True,
                stale=False,
            )
        )
    return allowed


def build_structured_answer_from_state(state: Mapping[str, Any]) -> StructuredAnswer:
    """Deterministically collect verified chunk IDs. Never guesses from prose."""
    workflow_status = str(state.get("workflow_status") or "completed")
    answer = normalize_answer(
        str(state.get("final_report") or state.get("executive_summary") or "")
    )
    citations = list(_verified_chunk_ids(state))
    incomplete = workflow_status == "incomplete_data" or bool(state.get("fatal_data_gap"))
    allowed = allowed_evidence_from_state(state)
    has_verified_rag = any(item.verified and not item.stale for item in allowed)
    if incomplete:
        citation_source = CITATION_SOURCE_UNAVAILABLE
        citations = []
    elif citations:
        citation_source = CITATION_SOURCE_STRUCTURED
    elif has_verified_rag:
        citation_source = CITATION_SOURCE_STRUCTURED
    else:
        citation_source = CITATION_SOURCE_UNAVAILABLE
    payload = {
        "answer": answer,
        "citations": citations,
        "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
        "citation_source": citation_source,
        "workflow_status": "incomplete_data" if incomplete else workflow_status,
    }
    return validate_structured_answer(
        payload,
        allowed=allowed,
        expected_tenant_id=str(state.get("rag_tenant_id") or state.get("tenant_id") or ""),
        expected_session_id=str(state.get("thread_id") or state.get("run_id") or ""),
        require_citation_for_factual=has_verified_rag and not incomplete,
    )


def _verified_chunk_ids(state: Mapping[str, Any]) -> tuple[str, ...]:
    from .claims.models import claims_from_state, filter_verified

    ordered: list[str] = []
    seen: set[str] = set()
    for claim in filter_verified(claims_from_state(dict(state))):
        for ref in claim.evidence_refs:
            chunk_id = str(getattr(ref, "chunk_id", None) or "").strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return tuple(ordered)


def _looks_factual(answer: str, payload: Mapping[str, Any]) -> bool:
    if payload.get("has_factual_conclusion") is False:
        return False
    if payload.get("has_factual_conclusion") is True:
        return True
    lowered = answer.lower()
    return any(
        token in lowered
        for token in ("revenue", "ebitda", "margin", "ratio", "billion", "million", "%")
    )


def _has_unsupported_ratio_claim(answer: str) -> bool:
    lowered = answer.lower()
    return any(token in lowered for token in ("ebitda margin", "operating margin", "p/e", "pe ratio"))


def redact_structured_error(message: str) -> str:
    text = _SECRET_RE.sub("[redacted]", str(message or ""))
    if len(text) > 240:
        return text[:240] + "…"
    return text
