"""Optional section/parent fields; unavailable metadata is never inferred."""

from __future__ import annotations

from typing import Any, Mapping

SECTION_METADATA_UNAVAILABLE = "NOT_AVAILABLE"
OPTIONAL_SECTION_FIELDS = ("section_id", "section_title", "parent_chunk_id")


def _read_optional(raw: object) -> str:
    if raw is None:
        return SECTION_METADATA_UNAVAILABLE
    text = str(raw).strip()
    if not text:
        return SECTION_METADATA_UNAVAILABLE
    unavailable = text.upper().replace(" ", "_")
    if unavailable in {"NOT_AVAILABLE", "NA", "N/A", "NONE", "NULL"}:
        return SECTION_METADATA_UNAVAILABLE
    return text


def section_metadata_for(chunk: Mapping[str, Any]) -> dict[str, str]:
    """Read explicit metadata only; never derive section identity from text."""
    return {
        field: _read_optional(chunk.get(field))
        for field in OPTIONAL_SECTION_FIELDS
    }


def attach_section_metadata(chunk: Mapping[str, Any]) -> dict[str, Any]:
    attached = dict(chunk)
    attached.update(section_metadata_for(chunk))
    return attached
