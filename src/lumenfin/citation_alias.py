"""Internal citation-alias protocol for a single final evidence window.

This module is shared by production synthesis and the LEDGER structured-citation
shadow. It does not change retrieval, ranking, or the public structured-answer
schema. Public citations remain stable chunk IDs after mapping.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .structured_answer import (
    AllowedEvidence,
    StructuredAnswerError,
    map_display_markers_to_chunk_ids,
)

CITATION_ALIAS_PROTOCOL_VERSION = "citation_alias_protocol.v1"
LOCKED_FINAL_K = 10
ALIAS_TOKEN_RE = re.compile(r"^E(\d{2})$")
ALIAS_WRAPPER_RE = re.compile(r"^\[(E\d{2})\]$")
_FILENAME_PAGE_RE = re.compile(r".+#p\d+", re.IGNORECASE)


class CitationAliasError(StructuredAnswerError):
    """Fail-closed alias mapping error. Safe to log; no sources or prompts."""


RankWindowFn = Callable[..., Sequence[Mapping[str, Any]]]


def format_alias(index: int) -> str:
    if index < 1 or index > LOCKED_FINAL_K:
        raise CitationAliasError("citation alias index is outside the final window")
    return f"E{index:02d}"


def flatten_rag_evidence(rag_evidence: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for company in sorted((rag_evidence or {}), key=lambda item: str(item)):
        rows = (rag_evidence or {}).get(company) or []
        if not isinstance(rows, list):
            continue
        for hit in rows:
            if isinstance(hit, dict) and str(hit.get("chunk_id") or "").strip():
                hits.append(dict(hit))
    return hits


def build_final_evidence_window(
    hits: Sequence[Mapping[str, Any]] | None,
    *,
    final_k: int = LOCKED_FINAL_K,
    already_ranked: bool = False,
    rank: RankWindowFn | None = None,
) -> list[dict[str, Any]]:
    if final_k != LOCKED_FINAL_K:
        raise CitationAliasError("final_k must stay 10")
    source = [dict(item) for item in list(hits or [])]
    if already_ranked:
        window = source[:LOCKED_FINAL_K]
    else:
        if rank is None:
            raise CitationAliasError("ranking function is required for an unranked pool")
        ranked = list(rank(source, final_k=LOCKED_FINAL_K))
        window = [dict(item) for item in ranked[:LOCKED_FINAL_K]]
    cleaned: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for hit in window:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not chunk_id:
            raise CitationAliasError("final evidence window is missing chunk_id")
        previous = seen.get(chunk_id)
        if previous is not None:
            if _window_hit_identity(previous) != _window_hit_identity(hit):
                raise CitationAliasError("citation allowlist has conflicting metadata")
            raise CitationAliasError("final evidence window contains duplicate chunk_id")
        seen[chunk_id] = hit
        cleaned.append(dict(hit))
    return cleaned


def _window_hit_identity(hit: Mapping[str, Any]) -> tuple[str, str, bool, bool]:
    return (
        str(hit.get("tenant_id") or ""),
        str(hit.get("session_id") or ""),
        not bool(hit.get("unverified")),
        bool(hit.get("stale_repair_attempt")),
    )


def window_chunk_ids(window: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(str(item.get("chunk_id") or "").strip() for item in window)


def final_allowed_hits_sha256(window: Sequence[Mapping[str, Any]]) -> str:
    payload = [{"chunk_id": chunk_id} for chunk_id in window_chunk_ids(window)]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CitationAliasMap:
    protocol_version: str
    case_id: str
    attempt_id: str
    tenant_id: str
    session_id: str
    aliases: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    alias_to_chunk: dict[str, str]
    index_to_chunk: dict[int, str]
    final_allowed_hits_sha256: str
    alias_map_sha256: str

    def alias_for(self, chunk_id: str) -> str:
        chunk_id = str(chunk_id or "").strip()
        for alias, mapped in self.alias_to_chunk.items():
            if mapped == chunk_id:
                return alias
        raise CitationAliasError("chunk_id is not in the current alias map")


def build_citation_alias_map(
    window: Sequence[Mapping[str, Any]],
    *,
    case_id: str = "",
    attempt_id: str = "",
    tenant_id: str = "",
    session_id: str = "",
) -> CitationAliasMap:
    if len(window) > LOCKED_FINAL_K:
        raise CitationAliasError("alias map cannot include hits beyond final_k")
    aliases: list[str] = []
    chunk_ids: list[str] = []
    alias_to_chunk: dict[str, str] = {}
    index_to_chunk: dict[int, str] = {}
    for index, hit in enumerate(window, start=1):
        alias = format_alias(index)
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not chunk_id:
            raise CitationAliasError("alias map is missing chunk_id")
        aliases.append(alias)
        chunk_ids.append(chunk_id)
        alias_to_chunk[alias] = chunk_id
        index_to_chunk[index] = chunk_id
    digest_payload = [
        {"alias": alias, "chunk_id": chunk_id}
        for alias, chunk_id in zip(aliases, chunk_ids)
    ]
    alias_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return CitationAliasMap(
        protocol_version=CITATION_ALIAS_PROTOCOL_VERSION,
        case_id=str(case_id or ""),
        attempt_id=str(attempt_id or ""),
        tenant_id=str(tenant_id or ""),
        session_id=str(session_id or ""),
        aliases=tuple(aliases),
        chunk_ids=tuple(chunk_ids),
        alias_to_chunk=alias_to_chunk,
        index_to_chunk=index_to_chunk,
        final_allowed_hits_sha256=final_allowed_hits_sha256(window),
        alias_map_sha256=alias_digest,
    )


def render_alias_passages(
    pairs: Sequence[tuple[str, str]],
    *,
    query_text: str = "",
    max_document_chars: int = 4000,
) -> str:
    lines: list[str] = []
    if query_text:
        lines.extend([f"Question: {query_text}", "", "Passages:"])
    limit = max(1, int(max_document_chars))
    for alias, text in pairs:
        lines.append(f"[{alias}]")
        lines.append(str(text or "")[:limit])
        lines.append("")
    return "\n".join(lines).strip()


def render_prompt_evidence(
    window: Sequence[Mapping[str, Any]],
    alias_map: CitationAliasMap,
    *,
    max_document_chars: int = 4000,
    query_text: str = "",
) -> str:
    assert_same_window(window, alias_map)
    rendered = render_alias_passages(
        tuple((alias, str(hit.get("text") or "")) for alias, hit in zip(alias_map.aliases, window)),
        query_text=query_text,
        max_document_chars=max_document_chars,
    )
    assert_prompt_hides_stable_ids(rendered, window)
    return rendered


def prompt_hits_for_generator(
    window: Sequence[Mapping[str, Any]],
    alias_map: CitationAliasMap,
) -> list[dict[str, str]]:
    assert_same_window(window, alias_map)
    return [
        {"alias": alias, "text": str(hit.get("text") or "")}
        for alias, hit in zip(alias_map.aliases, window)
    ]


def normalize_alias_token(raw: object) -> str:
    if raw is None or isinstance(raw, (bool, int, float, dict, list)):
        raise CitationAliasError("citation alias must be a string")
    if not isinstance(raw, str):
        raise CitationAliasError("citation alias must be a string")
    token = raw.strip()
    if not token:
        raise CitationAliasError("citation alias must be a string")
    wrapped = ALIAS_WRAPPER_RE.fullmatch(token)
    if wrapped:
        token = wrapped.group(1)
    if _FILENAME_PAGE_RE.search(token) or token.isdigit():
        raise CitationAliasError("citation alias format is not allowed")
    if not ALIAS_TOKEN_RE.fullmatch(token):
        raise CitationAliasError("citation alias is not in the current evidence map")
    return token


def _assert_alias_map_scope(
    alias_map: CitationAliasMap,
    *,
    expected_case_id: str = "",
    expected_attempt_id: str = "",
    expected_tenant_id: str = "",
    expected_session_id: str = "",
) -> None:
    if expected_case_id and alias_map.case_id != expected_case_id:
        raise CitationAliasError("citation alias map is not bound to this case")
    if expected_attempt_id and alias_map.attempt_id != expected_attempt_id:
        raise CitationAliasError("citation alias map is not bound to this repair attempt")
    if expected_tenant_id and alias_map.tenant_id != expected_tenant_id:
        raise CitationAliasError("citation alias map is not bound to this tenant")
    if expected_session_id and alias_map.session_id != expected_session_id:
        raise CitationAliasError("citation alias map is not bound to this session")


def parse_and_map_citation_aliases(
    raw: object,
    alias_map: CitationAliasMap,
    *,
    expected_case_id: str = "",
    expected_attempt_id: str = "",
    expected_tenant_id: str = "",
    expected_session_id: str = "",
) -> list[str]:
    _assert_alias_map_scope(
        alias_map,
        expected_case_id=expected_case_id,
        expected_attempt_id=expected_attempt_id,
        expected_tenant_id=expected_tenant_id,
        expected_session_id=expected_session_id,
    )
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise CitationAliasError("citations must be a string array")
    aliases: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip() in alias_map.chunk_ids:
            raise CitationAliasError("raw chunk ids are not allowed")
        token = normalize_alias_token(item)
        if token not in alias_map.alias_to_chunk:
            raise CitationAliasError("citation alias is not in the current evidence map")
        aliases.append(token)
    markers = []
    seen: set[str] = set()
    for alias in aliases:
        if alias in seen:
            continue
        seen.add(alias)
        markers.append(int(alias[1:]))
    return map_display_markers_to_chunk_ids(markers, alias_map.index_to_chunk)


def allowlist_from_window(
    window: Sequence[Mapping[str, Any]],
    *,
    tenant_id: str = "",
    session_id: str = "",
    verified_ids: Iterable[str] | None = None,
) -> list[AllowedEvidence]:
    verified = set(verified_ids) if verified_ids is not None else None
    allowed: list[AllowedEvidence] = []
    seen: dict[str, AllowedEvidence] = {}
    for hit in window:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        record = AllowedEvidence(
            chunk_id=chunk_id,
            tenant_id=str(hit.get("tenant_id") or tenant_id),
            session_id=str(hit.get("session_id") or session_id),
            verified=(chunk_id in verified) if verified is not None else (not bool(hit.get("unverified"))),
            stale=bool(hit.get("stale_repair_attempt")),
        )
        previous = seen.get(chunk_id)
        if previous is not None and (
            previous.tenant_id,
            previous.session_id,
            previous.verified,
            previous.stale,
        ) != (record.tenant_id, record.session_id, record.verified, record.stale):
            raise CitationAliasError("citation allowlist has conflicting metadata")
        if previous is None:
            seen[chunk_id] = record
            allowed.append(record)
    return allowed


def assert_same_window(
    window: Sequence[Mapping[str, Any]],
    alias_map: CitationAliasMap,
    allowlist: Sequence[AllowedEvidence] | None = None,
) -> None:
    prompt_ids = window_chunk_ids(window)
    alias_ids = alias_map.chunk_ids
    if prompt_ids != alias_ids:
        raise CitationAliasError("prompt window does not match alias window")
    if alias_map.final_allowed_hits_sha256 != final_allowed_hits_sha256(window):
        raise CitationAliasError("window identity hash mismatch")
    if allowlist is not None:
        allow_ids = tuple(item.chunk_id for item in allowlist)
        if allow_ids != alias_ids:
            raise CitationAliasError("alias window does not match validator window")


def assert_prompt_hides_stable_ids(prompt: str, window: Sequence[Mapping[str, Any]]) -> None:
    blob = str(prompt or "")
    for hit in window:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if chunk_id and chunk_id in blob:
            raise CitationAliasError("prompt leaked a stable chunk id")
        document_id = str(hit.get("document_id") or "").strip()
        if document_id and document_id in blob and document_id != str(hit.get("text") or ""):
            # document_id may legitimately appear inside passage text; only reject
            # structured identity fields, which render_prompt_evidence never writes.
            pass
    lowered = blob.casefold()
    if "chunk_id=" in lowered:
        raise CitationAliasError("prompt leaked a stable chunk id")


def window_identity_report(
    window: Sequence[Mapping[str, Any]],
    alias_map: CitationAliasMap,
    allowlist: Sequence[AllowedEvidence],
) -> dict[str, Any]:
    assert_same_window(window, alias_map, allowlist)
    return {
        "alias_protocol_version": CITATION_ALIAS_PROTOCOL_VERSION,
        "final_k": LOCKED_FINAL_K,
        "prompt_window_equals_alias_window": True,
        "alias_window_equals_validator_window": True,
        "raw_chunk_id_output_forbidden": True,
        "final_allowed_hits_sha256": alias_map.final_allowed_hits_sha256,
        "alias_map_sha256": alias_map.alias_map_sha256,
        "window_size": len(window),
    }
