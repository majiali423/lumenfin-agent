"""Period identity parsing and matching helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

@dataclass(frozen=True)
class PeriodMatch:
    status: str
    matched: bool
    source: str | None = None


@dataclass(frozen=True)
class PeriodIdentity:
    kind: str
    year: int | None = None
    quarter: int | None = None
    date_value: str | None = None
    raw: str | None = None


_PERIOD_TYPE_TOKENS = frozenset({"annual", "quarter", "ttm", "latest", "model"})
_ACCEPTABLE_PERIOD_STATUS = frozenset({"exact", "text_match", "not_required"})
_FACTUAL_PERIOD_SOURCES = frozenset(
    {
        "sec_companyfacts",
        "provider_record",
        "document_text",
        "table_header",
        "structured_table",
        "filing_fact",
    }
)
_ASSUMED_PERIOD_SOURCES = frozenset(
    {
        "query",
        "query_assumption",
        "upload_filename",
        "issuer_convention",
        "model",
        "latest",
        "annual",
        "quarter",
        "unknown",
    }
)
_ASSUMED_PERIOD_ALIGNMENTS = frozenset(
    {"assumed_from_query", "fallback_latest", "upload_labeled", "unspecified", "unknown"}
)


def is_factual_period_provenance(
    *,
    period: str | None,
    period_source: str | None,
    period_alignment: str | None,
) -> bool:
    """Return whether upstream explicitly bound a concrete period to a factual source."""
    identity = parse_period_identity(period)
    source = str(period_source or "").strip().casefold()
    alignment = str(period_alignment or "").strip().casefold()
    if identity.kind == "unknown" or not source:
        return False
    if source in _ASSUMED_PERIOD_SOURCES or source not in _FACTUAL_PERIOD_SOURCES:
        return False
    if alignment != "exact":
        return False
    return True


def parse_period_identity(value: str | None) -> PeriodIdentity:
    """Parse a concrete fiscal/calendar period label into a comparable identity."""
    raw = str(value or "").strip()
    if not raw:
        return PeriodIdentity("unknown", raw=None)
    if raw.lower() in _PERIOD_TYPE_TOKENS:
        return PeriodIdentity("unknown", raw=raw)

    upper = raw.upper()
    date_match = re.fullmatch(r"(20\d{2})-(\d{2})-(\d{2})", upper)
    if date_match:
        year = int(date_match.group(1))
        month = int(date_match.group(2))
        quarter = (month - 1) // 3 + 1
        return PeriodIdentity(
            "date", year=year, quarter=quarter, date_value=date_match.group(0), raw=raw
        )

    q_match = re.search(
        r"(?:^|\b)(?:Q\s*([1-4])\s*(20\d{2})|(20\d{2})\s*Q\s*([1-4]))(?:\b|$)",
        upper,
    )
    if q_match:
        if q_match.group(1) and q_match.group(2):
            return PeriodIdentity(
                "quarter", year=int(q_match.group(2)), quarter=int(q_match.group(1)), raw=raw
            )
        return PeriodIdentity(
            "quarter", year=int(q_match.group(3)), quarter=int(q_match.group(4)), raw=raw
        )

    fy_match = re.search(
        r"(?:^|\b)(?:FY\s*(20\d{2})|FISCAL\s+YEAR\s*(20\d{2}))(?:\b|$)",
        upper,
    )
    if fy_match:
        year = int(fy_match.group(1) or fy_match.group(2))
        return PeriodIdentity("fiscal_year", year=year, raw=raw)

    year_match = re.fullmatch(r"(20\d{2})", upper)
    if year_match:
        return PeriodIdentity("calendar_year", year=int(year_match.group(1)), raw=raw)

    # Fallback: extract year if present but keep unknown kind so we do not over-match.
    years = re.findall(r"20\d{2}", upper)
    if len(years) == 1 and re.search(r"\bQ[1-4]\b", upper):
        q = int(re.search(r"\bQ([1-4])\b", upper).group(1))
        return PeriodIdentity("quarter", year=int(years[0]), quarter=q, raw=raw)
    if len(years) == 1 and re.search(r"\bFY\b|FISCAL", upper):
        return PeriodIdentity("fiscal_year", year=int(years[0]), raw=raw)
    return PeriodIdentity("unknown", year=int(years[0]) if years else None, raw=raw)


def _period_identities_compatible(left: PeriodIdentity, right: PeriodIdentity) -> bool:
    # Both unspecified: compatible when the claim itself does not require a concrete period.
    if left.kind == "unknown" and right.kind == "unknown":
        return True
    if left.kind == "unknown" or right.kind == "unknown":
        return False
    if left.year is None or right.year is None or left.year != right.year:
        return False
    if left.kind != right.kind:
        return False
    if left.kind in {"fiscal_year", "calendar_year"}:
        return True
    if left.kind == "quarter" and right.kind == "quarter":
        return left.quarter == right.quarter
    if left.kind == "date" and right.kind == "date":
        return bool(left.date_value and left.date_value == right.date_value)
    return False


def _extract_period_identities_from_text(text: str) -> list[PeriodIdentity]:
    found: list[PeriodIdentity] = []
    upper = (text or "").upper()
    for match in re.finditer(r"\bQ\s*([1-4])\s*(20\d{2})\b|\b(20\d{2})\s*Q\s*([1-4])\b", upper):
        if match.group(1) and match.group(2):
            found.append(
                PeriodIdentity(
                    "quarter",
                    year=int(match.group(2)),
                    quarter=int(match.group(1)),
                    raw=match.group(0),
                )
            )
        else:
            found.append(
                PeriodIdentity(
                    "quarter",
                    year=int(match.group(3)),
                    quarter=int(match.group(4)),
                    raw=match.group(0),
                )
            )
    for match in re.finditer(r"\bFY\s*(20\d{2})\b|\bFISCAL\s+YEAR\s*(20\d{2})\b", upper):
        year = int(match.group(1) or match.group(2))
        found.append(PeriodIdentity("fiscal_year", year=year, raw=match.group(0)))
    for match in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", upper):
        year = int(match.group(1))
        month = int(match.group(2))
        found.append(
            PeriodIdentity(
                "date",
                year=year,
                quarter=(month - 1) // 3 + 1,
                date_value=match.group(0),
                raw=match.group(0),
            )
        )
    return found


def _period_identity_spans(text: str) -> list[tuple[PeriodIdentity, int, int]]:
    """Return explicit fiscal/date period identities with their text spans."""
    found: list[tuple[PeriodIdentity, int, int]] = []
    upper = text.upper()
    patterns = (
        r"\bQ\s*([1-4])\s*(20\d{2})\b|\b(20\d{2})\s*Q\s*([1-4])\b",
        r"\bFY\s*(20\d{2})\b|\bFISCAL\s+YEAR\s*(20\d{2})\b",
        r"\b(20\d{2})-(\d{2})-(\d{2})\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, upper):
            found.append((parse_period_identity(match.group(0)), match.start(), match.end()))
    found.sort(key=lambda item: item[1])
    return found


def _local_context_bounds(text: str, number_start: int, number_end: int) -> tuple[int, int]:
    """Find sentence/table-row boundaries around one numeric span."""
    boundaries = [0]
    for match in re.finditer(r"[\r\n;|]+|(?<=[.!?])\s+(?=[A-Z\u4e00-\u9fff])", text or ""):
        boundaries.extend((match.start(), match.end()))
    boundaries.append(len(text or ""))
    left = max((point for point in boundaries if point <= number_start), default=0)
    right = min((point for point in boundaries if point >= number_end), default=len(text or ""))
    return left, right


def _local_context_for_number(text: str, number_start: int, number_end: int) -> str:
    """Return the sentence or table row that owns a target numeric span."""
    left, right = _local_context_bounds(text or "", number_start, number_end)
    return (text or "")[left:right].strip()


def _match_period_near_number(
    claim_period: str | None,
    text: str,
    number_start: int,
    number_end: int,
    fallback_period: str | None,
) -> PeriodMatch:
    left, right = _local_context_bounds(text, number_start, number_end)
    local = text[left:right]
    spans = _period_identity_spans(local)
    claim_id = parse_period_identity(claim_period)
    fallback_id = parse_period_identity(fallback_period)
    if spans:
        center = ((number_start - left) + (number_end - left)) / 2.0
        ranked = sorted(
            ((abs(((start + end) / 2.0) - center), identity) for identity, start, end in spans),
            key=lambda item: item[0],
        )
        nearest_distance = ranked[0][0]
        nearest = [identity for distance, identity in ranked if distance == nearest_distance]
        if len({(item.kind, item.year, item.quarter, getattr(item, "date_value", None)) for item in nearest}) > 1:
            return PeriodMatch("ambiguous", False, "text")
        selected = nearest[0]
        if fallback_id.kind != "unknown" and not _period_identities_compatible(selected, fallback_id):
            return PeriodMatch("metadata_conflict", False, "text+evidence.period")
        if _period_identities_compatible(claim_id, selected):
            return PeriodMatch("text_match", True, "text")
        return PeriodMatch("mismatch", False, "text")
    return match_period(claim_period, fallback_period, "")


def _nearest_period_identity(
    text: str, number_start: int, number_end: int
) -> PeriodIdentity | None:
    left, right = _local_context_bounds(text, number_start, number_end)
    spans = _period_identity_spans(text[left:right])
    if not spans:
        return None
    center = ((number_start - left) + (number_end - left)) / 2.0
    ranked = sorted(
        ((abs(((start + end) / 2.0) - center), identity) for identity, start, end in spans),
        key=lambda item: item[0],
    )
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        first, second = ranked[0][1], ranked[1][1]
        if not _period_identities_compatible(first, second):
            return None
    return ranked[0][1]


def _periods_compatible(claim_period: str | None, evidence_period: str | None, text: str) -> bool:
    return match_period(claim_period, evidence_period, text).matched


def _period_status(claim_period: str | None, evidence_period: str | None, text: str) -> str:
    """Compatibility wrapper: exact/text_match/not_required → ok; else status name."""
    result = match_period(claim_period, evidence_period, text)
    if result.status in _ACCEPTABLE_PERIOD_STATUS:
        return "ok"
    return result.status


def match_period(
    claim_period: str | None,
    evidence_period: str | None,
    text: str = "",
    *,
    period_type: str | None = None,
    prefer_local_text: bool = False,
) -> PeriodMatch:
    """Compare claim period identity against evidence period / local text."""
    del period_type  # period_type never proves a concrete FY/Q identity.
    if not claim_period:
        return PeriodMatch("not_required", True, None)

    claim_id = parse_period_identity(claim_period)
    if claim_id.kind == "unknown" and claim_id.year is None:
        return PeriodMatch("not_required", True, None)

    local_ids = _extract_period_identities_from_text(text or "")
    if local_ids:
        compatible = [item for item in local_ids if _period_identities_compatible(claim_id, item)]
        conflicting = [item for item in local_ids if not _period_identities_compatible(claim_id, item)]
        # Any explicit conflicting label in the inspected window wins over metadata.
        if conflicting:
            return PeriodMatch("mismatch", False, "text")
        if compatible:
            return PeriodMatch("text_match", True, "text")

    if prefer_local_text and local_ids:
        # Local labels already handled above; empty compatible set means unknown locally.
        return PeriodMatch("unknown", False, "text")

    specific = str(evidence_period or "").strip()
    if specific.lower() in _PERIOD_TYPE_TOKENS:
        specific = ""
    if specific:
        evidence_id = parse_period_identity(specific)
        if evidence_id.kind == "unknown" and evidence_id.year is None:
            return PeriodMatch("unknown", False, None)
        if _period_identities_compatible(claim_id, evidence_id):
            return PeriodMatch("exact", True, "evidence.period")
        return PeriodMatch("mismatch", False, "evidence.period")

    return PeriodMatch("unknown", False, None)


def match_local_period(
    claim_period: str | None,
    local_text: str,
    fallback_period: str | None,
    period_type: str | None = None,
) -> PeriodMatch:
    """Prefer explicit local FY/Q labels; fall back to evidence.period only when absent."""
    return match_period(
        claim_period,
        fallback_period,
        local_text,
        period_type=period_type,
        prefer_local_text=True,
    )
