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
CITATION_SOURCE_LEGACY_STRUCTURED = "legacy_structured"
CITATION_SOURCE_UNAVAILABLE = "unavailable"
# Read-only alias for unpublished local drafts; writers emit legacy_structured.
CITATION_SOURCE_LEGACY_TEXT = "legacy_text"
CITATION_VALIDATION_PASSED = "passed"
CITATION_VALIDATION_FAILED = "failed"
CITATION_PATH_VERIFIED = "verified_evidence.chunk_id"
CITATION_PATH_UNAVAILABLE = "unavailable"
CITATION_PATH_VALIDATION_FAILED = "validation_failed"
_SUPPORTED_SCHEMA_VERSIONS = frozenset({STRUCTURED_ANSWER_SCHEMA_VERSION})
_WRITE_CITATION_SOURCES = frozenset(
    {
        CITATION_SOURCE_STRUCTURED,
        CITATION_SOURCE_LEGACY_STRUCTURED,
        CITATION_SOURCE_UNAVAILABLE,
    }
)
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
    citation_validation: str = CITATION_VALIDATION_PASSED
    citation_path: str = CITATION_PATH_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": list(self.citations),
            "structured_answer_schema_version": self.schema_version,
            "citation_source": self.citation_source,
            "workflow_status": self.workflow_status,
            "citation_validation": self.citation_validation,
            "citation_path": self.citation_path,
        }


def canonicalize_citation_source(raw: object) -> str:
    value = str(raw or "").strip()
    if value == CITATION_SOURCE_LEGACY_TEXT:
        return CITATION_SOURCE_LEGACY_STRUCTURED
    if value in _WRITE_CITATION_SOURCES:
        return value
    raise StructuredAnswerError("unsupported citation_source")


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
    schema_version = str(payload.get("structured_answer_schema_version") or "").strip()
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise StructuredAnswerError("unsupported structured_answer_schema_version")
    answer = normalize_answer(payload.get("answer", ""))
    citations = normalize_citations(payload.get("citations") or [])
    workflow_status = str(payload.get("workflow_status") or "completed")
    citation_source = canonicalize_citation_source(
        payload.get("citation_source") or CITATION_SOURCE_STRUCTURED
    )

    by_id = _index_allowed(allowed)
    for chunk_id in citations:
        record = by_id.get(chunk_id)
        if record is None:
            raise StructuredAnswerError("citation refers to an unknown chunk")
        if record.stale:
            raise StructuredAnswerError("citation refers to a stale repair attempt")
        if not record.verified:
            raise StructuredAnswerError("citation refers to unverified evidence")
        if expected_tenant_id and record.tenant_id != expected_tenant_id:
            raise StructuredAnswerError("citation tenant does not match the current run")
        if expected_session_id and record.session_id != expected_session_id:
            raise StructuredAnswerError("citation session does not match the current run")

    incomplete = workflow_status == "incomplete_data"
    allowed_verified = [item for item in by_id.values() if item.verified and not item.stale]
    if require_citation_for_factual and not incomplete and allowed_verified and not citations:
        raise StructuredAnswerError("factual verified answer requires at least one citation")
    if incomplete and payload.get("has_verified_numeric_claims"):
        raise StructuredAnswerError("incomplete_data cannot carry verified numeric claims")

    if not citations:
        citation_source = CITATION_SOURCE_UNAVAILABLE
    citation_path = CITATION_PATH_VERIFIED if citations else CITATION_PATH_UNAVAILABLE
    return StructuredAnswer(
        answer=answer,
        citations=tuple(citations),
        schema_version=schema_version,
        citation_source=citation_source,
        workflow_status=workflow_status,
        citation_validation=CITATION_VALIDATION_PASSED,
        citation_path=citation_path,
    )


def degraded_structured_answer(
    *,
    answer: str,
    workflow_status: str,
    error: StructuredAnswerError | str,
) -> dict[str, Any]:
    return {
        "answer": str(answer or ""),
        "citations": [],
        "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
        "citation_source": CITATION_SOURCE_UNAVAILABLE,
        "workflow_status": workflow_status or "completed",
        "citation_validation": CITATION_VALIDATION_FAILED,
        "citation_path": CITATION_PATH_VALIDATION_FAILED,
        "validation_error": redact_structured_error(str(error)),
    }


def allowed_evidence_from_state(state: Mapping[str, Any]) -> list[AllowedEvidence]:
    tenant_id = str(state.get("rag_tenant_id") or state.get("tenant_id") or "")
    session_id = str(state.get("thread_id") or state.get("run_id") or "")
    verified_ids = set(_verified_chunk_ids(state))
    allowed: list[AllowedEvidence] = []
    seen: dict[str, AllowedEvidence] = {}
    for hits in (state.get("rag_evidence") or {}).values():
        if not isinstance(hits, list):
            continue
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            chunk_id = str(hit.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            record = AllowedEvidence(
                chunk_id=chunk_id,
                tenant_id=str(hit.get("tenant_id") or tenant_id),
                session_id=str(hit.get("session_id") or session_id),
                verified=chunk_id in verified_ids,
                stale=bool(hit.get("stale_repair_attempt")),
            )
            previous = seen.get(chunk_id)
            if previous is not None and _identity(previous) != _identity(record):
                raise StructuredAnswerError("citation allowlist has conflicting metadata")
            if previous is None:
                seen[chunk_id] = record
                allowed.append(record)
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
    allowed_ids = {item.chunk_id for item in allowed if item.verified and not item.stale}
    citations = [chunk_id for chunk_id in citations if chunk_id in allowed_ids]
    has_verified_rag = bool(allowed_ids)
    if incomplete:
        citation_source = CITATION_SOURCE_UNAVAILABLE
        citations = []
    elif citations:
        citation_source = CITATION_SOURCE_STRUCTURED
    else:
        citation_source = CITATION_SOURCE_UNAVAILABLE
    payload = {
        "answer": answer,
        "citations": citations,
        "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
        "citation_source": citation_source,
        "workflow_status": "incomplete_data" if incomplete else workflow_status,
        "has_verified_numeric_claims": _has_verified_numeric_claims(state),
    }
    return validate_structured_answer(
        payload,
        allowed=allowed,
        expected_tenant_id=str(state.get("rag_tenant_id") or state.get("tenant_id") or ""),
        expected_session_id=str(state.get("thread_id") or state.get("run_id") or ""),
        require_citation_for_factual=has_verified_rag and not incomplete,
    )


def public_structured_answer_fields(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the API triple only when it is complete and valid."""
    structured = result.get("structured_answer")
    if not isinstance(structured, dict):
        return None
    if (
        structured.get("validation_error")
        or structured.get("citation_validation") == CITATION_VALIDATION_FAILED
    ):
        return None
    version = str(structured.get("structured_answer_schema_version") or "").strip()
    if version != STRUCTURED_ANSWER_SCHEMA_VERSION:
        return None
    citations_raw = structured.get("citations")
    answer_raw = structured.get("answer")
    if not isinstance(answer_raw, str) or not isinstance(citations_raw, list):
        return None
    try:
        citations = normalize_citations(citations_raw)
    except StructuredAnswerError:
        return None
    report = str(result.get("final_report") or "")
    return {
        "answer": report,
        "citations": citations,
        "structured_answer_schema_version": version,
    }


def _index_allowed(allowed: Iterable[AllowedEvidence]) -> dict[str, AllowedEvidence]:
    by_id: dict[str, AllowedEvidence] = {}
    for item in allowed:
        if not item.chunk_id:
            continue
        previous = by_id.get(item.chunk_id)
        if previous is not None and _identity(previous) != _identity(item):
            raise StructuredAnswerError("citation allowlist has conflicting metadata")
        by_id[item.chunk_id] = item
    return by_id


def _identity(item: AllowedEvidence) -> tuple[str, str, bool, bool]:
    return (item.tenant_id, item.session_id, item.verified, item.stale)


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


def _has_verified_numeric_claims(state: Mapping[str, Any]) -> bool:
    from .claims.models import claims_from_state, filter_verified

    for claim in filter_verified(claims_from_state(dict(state))):
        if claim.claim_type in {"numeric", "growth"} and claim.value is not None:
            return True
    return False


def redact_structured_error(message: str) -> str:
    text = _SECRET_RE.sub("[redacted]", str(message or ""))
    if len(text) > 240:
        return text[:240] + "…"
    return text
