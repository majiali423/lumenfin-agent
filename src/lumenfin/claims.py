"""Claim → Evidence Binding layer.

Builds internal claim objects from structured state, verifies them against
evidence (RAG page anchors and/or fundamentals text), and exposes only
verified claims for the report synthesizer.

No prompt-forced citations: citations come from evidence already present
on the claim after structural verification.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .metrics_schema import get_fundamental, period_label_from_meta
from .tools import AST_RATIO_KEYS, has_computable_fundamentals

ClaimType = Literal["numeric", "growth", "risk_conclusion", "investment_conclusion"]
Verification = Literal["verified", "unverified", "rejected"]

FORMULA_INPUTS: dict[str, dict[str, str]] = {
    "ebitda_margin": {"ebitda": "ebitda", "revenue": "revenue"},
    "operating_margin": {"operating_income": "operating_income", "revenue": "revenue"},
    "r_and_d_intensity": {"r_and_d": "r_and_d", "revenue": "revenue"},
}

METRIC_LABELS = {
    "ebitda_margin": "EBITDA margin",
    "operating_margin": "Operating margin",
    "r_and_d_intensity": "R&D intensity",
    "estimated_net_margin": "Estimated net margin",
    "estimated_fcf_margin": "Estimated FCF margin",
    "pe_ratio": "P/E (TTM)",
    "monthly_return": "Monthly return",
    "range_position": "52-week range position",
}

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue", "revenues", "total revenue", "net sales", "收入", "营收"),
    "r_and_d": (
        "r&d",
        "r & d",
        "research and development",
        "research & development",
        "rd expense",
        "研发",
    ),
    "operating_income": ("operating income", "operating profit", "营业利润", "经营利润"),
    "ebitda": ("ebitda",),
    "operating_margin": ("operating margin", "营业利润率"),
    "ebitda_margin": ("ebitda margin",),
    "r_and_d_intensity": ("r&d intensity", "r&d as a percentage", "研发强度"),
}

_STRUCTURED_SOURCES = frozenset(
    {
        "sec_companyfacts",
        "yahoo_fundamentals",
        "document_extracted",
        "sample_db",
        "fundamentals",
        "sample_financial_data",
    }
)


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    entity: str
    citation: str
    source_type: str
    text: str
    page: int | None = None
    period: str | None = None
    period_type: str | None = None
    metric_name: str | None = None
    value: float | None = None
    unit: str | None = None
    confidence: str | None = None
    period_source: str | None = None
    period_alignment: str | None = None
    source_record_id: str | None = None
    citation_trusted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_structured_fields(self) -> bool:
        return self.metric_name is not None and self.value is not None


@dataclass
class Claim:
    claim_id: str
    entity: str
    claim_type: ClaimType
    statement: str
    value: float | str | None = None
    unit: str | None = None
    period: str | None = None
    metric_name: str | None = None
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    verification: Verification = "unverified"
    verify_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @property
    def primary_citation(self) -> str:
        if not self.evidence_refs:
            return ""
        return self.evidence_refs[0].citation

    def render_with_citation(self, *, humanize: bool = False) -> str:
        cite = self.primary_citation
        if cite and humanize:
            from .reporting import humanize_citation

            cite = humanize_citation(cite)
        if cite:
            return f"{self.statement} [{cite}]"
        return self.statement


def claim_to_dict(claim: Claim) -> dict[str, Any]:
    return claim.to_dict()


def claims_from_state(state: dict[str, Any]) -> list[Claim]:
    raw = state.get("claims") or []
    out: list[Claim] = []
    for item in raw:
        if isinstance(item, Claim):
            out.append(item)
            continue
        if not isinstance(item, dict):
            continue
        refs = []
        for ref in (item.get("evidence_refs") or []):
            if isinstance(ref, EvidenceRef):
                refs.append(ref)
            elif isinstance(ref, dict):
                allowed = {
                    key: ref[key]
                    for key in (
                        "evidence_id",
                        "entity",
                        "citation",
                        "source_type",
                        "text",
                        "page",
                        "period",
                        "period_type",
                        "metric_name",
                        "value",
                        "unit",
                        "confidence",
                        "period_source",
                        "period_alignment",
                        "source_record_id",
                        "citation_trusted",
                    )
                    if key in ref
                }
                refs.append(EvidenceRef(**allowed))
            else:
                refs.append(ref)
        out.append(
            Claim(
                claim_id=str(item.get("claim_id") or ""),
                entity=str(item.get("entity") or ""),
                claim_type=item.get("claim_type") or "numeric",  # type: ignore[arg-type]
                statement=str(item.get("statement") or ""),
                value=item.get("value"),
                unit=item.get("unit"),
                period=item.get("period"),
                metric_name=item.get("metric_name"),
                evidence_refs=refs,
                verification=item.get("verification") or "unverified",  # type: ignore[arg-type]
                verify_reason=str(item.get("verify_reason") or ""),
            )
        )
    return out


def filter_verified(claims: list[Claim]) -> list[Claim]:
    return [c for c in claims if c.verification == "verified" and c.evidence_refs]


def verified_by_entity(
    claims: list[Claim],
    entity: str,
    *,
    claim_type: ClaimType | None = None,
    metric_name: str | None = None,
) -> list[Claim]:
    out = []
    for claim in filter_verified(claims):
        if claim.entity != entity:
            continue
        if claim_type and claim.claim_type != claim_type:
            continue
        if metric_name and claim.metric_name != metric_name:
            continue
        out.append(claim)
    return out


def _fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def _fmt_num(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4g}"


def _is_usable_number_token(token: str, value: float) -> bool:
    """Drop ambiguous tokens that create false substring hits (e.g. '0' inside '2024')."""
    t = (token or "").replace(",", "").strip()
    if not t or t in {".", "-", "+", "0", "1", "-1"}:
        return False
    # Bare 1–2 digit tokens are too collision-prone as substrings.
    digits = re.sub(r"[^\d]", "", t)
    if len(digits) <= 1:
        return False
    if len(digits) <= 2 and abs(float(value)) < 2.0:
        return False
    return True


def _number_variants(value: float) -> list[str]:
    variants = {
        _fmt_num(value),
        f"{value:.1f}",
        f"{value:.2f}",
        f"{value:.0f}",
        f"{value:,.0f}",
        f"{value:,.1f}",
    }
    # Absolute amounts (internal billion_usd): also match filing million / raw USD forms.
    if abs(value) > 1.5:
        variants.add(f"{value * 1000:.0f}")  # billion → million
        variants.add(f"{value * 1_000_000_000:.0f}")  # billion → raw USD
    # Compact without commas for PDF text matches.
    variants |= {v.replace(",", "") for v in list(variants)}
    return [v for v in variants if _is_usable_number_token(v, value)]


def _token_in_text(text: str, token: str) -> bool:
    """Substring match; short tokens require digit-boundary isolation."""
    t = token.replace(",", "")
    if not t:
        return False
    if len(re.sub(r"[^\d]", "", t)) <= 2:
        return bool(re.search(rf"(?<![\d.]){re.escape(t)}(?![\d.])", text))
    return t in text


def _text_contains_number(text: str, value: float | None, *, tol: float = 0.02) -> bool:
    if value is None or not text:
        return False
    lowered = text.lower().replace(",", "")
    target = float(value)
    for token in _number_variants(target):
        if _token_in_text(lowered, token):
            return True
    # Relative match for percentages already formatted in text like 34.8%
    pct = target * 100.0 if abs(target) <= 1.5 else target
    for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", text):
        try:
            found = float(match.group(1))
        except ValueError:
            continue
        if abs(found - pct) <= max(0.15, abs(pct) * tol):
            return True
    # Scale-tolerant absolute match: internal billion vs filing millions (e.g. 29.5 ↔ 29510).
    if abs(target) > 1.5:
        million_target = target * 1000.0
        for match in re.finditer(r"-?\d{3,}(?:\.\d+)?", lowered):
            try:
                found = float(match.group(0))
            except ValueError:
                continue
            if abs(found - million_target) <= max(50.0, abs(million_target) * tol):
                return True
            raw_target = target * 1_000_000_000.0
            if abs(found - raw_target) <= max(1_000_000.0, abs(raw_target) * tol):
                return True
    return False


@dataclass(frozen=True)
class EvidenceMatch:
    matched: bool
    matched_value: float | None
    matched_metric: bool
    matched_period: bool
    matched_unit: bool
    unit_conversion: str | None
    match_span: str | None
    confidence: str
    reason: str


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
    if alignment in _ASSUMED_PERIOD_ALIGNMENTS:
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
        return PeriodIdentity("date", year=year, quarter=quarter, raw=raw)

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
    annual = {"fiscal_year", "calendar_year"}
    if left.kind in annual and right.kind in annual:
        return True
    if left.kind == "quarter" and right.kind == "quarter":
        return left.quarter == right.quarter
    if left.kind == "date" and right.kind == "date":
        return left.quarter == right.quarter
    # date may align to quarter of same year/quarter number
    if {left.kind, right.kind} <= {"date", "quarter"}:
        return left.quarter == right.quarter
    # annual vs quarter/date is never the same identity
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


def _values_close(left: float, right: float, *, tol: float = 0.02) -> bool:
    return _text_values_close(left, right, tol=tol)


def _structured_values_close(left: float, right: float) -> bool:
    a = float(left)
    b = float(right)
    return abs(a - b) <= max(1e-6, abs(b) * 1e-4)


def _text_values_close(left: float, right: float, *, tol: float = 0.02) -> bool:
    a = float(left)
    b = float(right)
    return abs(a - b) <= max(1e-9, abs(b) * tol, 0.05 if abs(b) >= 1 else 1e-6)


def _unit_declared_near(text: str, idx: int, *, window: int = 60) -> bool:
    snippet = (text or "")[max(0, idx - window) : idx + window].lower()
    return bool(
        re.search(
            r"\b(?:billion|million|thousand|usd|u\.?s\.?\s*dollars?)\b|"
            r"\$|in\s+millions|in\s+billions|in\s+thousands",
            snippet,
        )
    )


def _metric_alias_near(text: str, metric_name: str, center: int, *, window: int = 80) -> bool:
    return _metric_alias_distance(text, metric_name, center, window=window) is not None


def _metric_alias_distance(
    text: str, metric_name: str, center: int, *, window: int = 100
) -> int | None:
    aliases = METRIC_ALIASES.get(metric_name) or (metric_name.replace("_", " "),)
    start = max(0, center - window)
    end = min(len(text), center + window)
    snippet = text[start:end].casefold()
    best: int | None = None
    for alias in aliases:
        token = alias.casefold()
        if not token:
            continue
        pos = snippet.find(token)
        while pos >= 0:
            abs_pos = start + pos
            dist = abs(abs_pos - center)
            if best is None or dist < best:
                best = dist
            pos = snippet.find(token, pos + 1)
    return best


def _find_number_span(
    text: str,
    value: float,
    *,
    unit: str | None,
    metric_name: str | None = None,
    tol: float = 0.02,
) -> tuple[int, float, str | None, str] | None:
    """Return (index, found_value, unit_conversion, span) for the best numeric hit."""
    lowered = text.lower().replace(",", "")
    target = float(value)
    candidates: list[tuple[int, float, str | None, str, int]] = []

    def _score(idx: int) -> int:
        if not metric_name:
            return 0
        dist = _metric_alias_distance(text, metric_name, idx)
        if dist is None:
            return 10_000
        return dist

    for token in _number_variants(target):
        clean = token.replace(",", "")
        start_at = 0
        while True:
            idx = lowered.find(clean, start_at)
            if idx < 0:
                break
            if _token_in_text(lowered, clean):
                conversion = None
                found_val = target
                try:
                    as_float = float(clean)
                except ValueError:
                    as_float = target
                if abs(target) > 1.5 and abs(as_float - target * 1000.0) <= max(
                    50.0, abs(target * 1000.0) * tol
                ):
                    nearby = lowered[max(0, idx - 80) : idx + len(clean) + 24]
                    if "million" not in nearby and "in millions" not in lowered:
                        start_at = idx + max(len(clean), 1)
                        continue
                    conversion = "million_to_billion"
                    found_val = as_float / 1000.0
                elif abs(target) > 1.5 and abs(
                    as_float - target * 1_000_000_000.0
                ) <= max(1_000_000.0, abs(target * 1_000_000_000.0) * tol):
                    conversion = "raw_usd_to_billion"
                    found_val = as_float / 1_000_000_000.0
                span_start = max(0, idx - 48)
                span_end = min(len(text), idx + max(len(clean), 1) + 8)
                candidates.append(
                    (
                        idx,
                        found_val,
                        conversion,
                        text[span_start:span_end].strip(),
                        _score(idx),
                    )
                )
            start_at = idx + max(len(clean), 1)

    if unit == "ratio" or abs(target) <= 1.5:
        pct = target * 100.0 if abs(target) <= 1.5 else target
        for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", text):
            try:
                found = float(match.group(1))
            except ValueError:
                continue
            if abs(found - pct) <= max(0.15, abs(pct) * tol):
                candidates.append(
                    (
                        match.start(),
                        found / 100.0 if abs(target) <= 1.5 else found,
                        None,
                        match.group(0),
                        _score(match.start()),
                    )
                )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[4], item[0]))
        best = candidates[0]
        return best[0], best[1], best[2], best[3]

    million_target = target * 1000.0
    for match in re.finditer(r"-?\d{3,}(?:\.\d+)?", lowered):
        try:
            found = float(match.group(0))
        except ValueError:
            continue
        after = lowered[match.end() : match.end() + 24]
        before = lowered[max(0, match.start() - 80) : match.start()]
        if abs(found - million_target) <= max(50.0, abs(million_target) * tol):
            if "million" in after or "million" in before or "in millions" in lowered:
                candidates.append(
                    (
                        match.start(),
                        found / 1000.0,
                        "million_to_billion",
                        match.group(0),
                        _score(match.start()),
                    )
                )
            continue
        raw_target = target * 1_000_000_000.0
        if abs(found - raw_target) <= max(1_000_000.0, abs(raw_target) * tol):
            candidates.append(
                (
                    match.start(),
                    found / 1_000_000_000.0,
                    "raw_usd_to_billion",
                    match.group(0),
                    _score(match.start()),
                )
            )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[4], item[0]))
    best = candidates[0]
    return best[0], best[1], best[2], best[3]


def _number_token_end(text: str, start: int) -> int:
    match = re.match(r"[-+]?\s*[\d,.]+", (text or "")[start:])
    return start + (len(match.group(0)) if match else 1)



def match_numeric_evidence(
    evidence: EvidenceRef,
    *,
    entity: str,
    metric_name: str,
    value: float,
    unit: str | None,
    period: str | None,
    formula_inputs: dict[str, float] | None = None,
) -> EvidenceMatch:
    """Bind a numeric claim to evidence with metric/period/unit/entity constraints."""
    if evidence.entity and entity and evidence.entity.casefold() != entity.casefold():
        return EvidenceMatch(
            False, None, False, False, False, None, None, "none", "entity_mismatch"
        )

    text = evidence.text or ""
    period_type = getattr(evidence, "period_type", None)
    hit: tuple[int, float, str | None, str] | None = None
    if formula_inputs:
        # Formula periods are owned by individual input spans, never top-level metadata alone.
        period_result = PeriodMatch("not_required", True, None)
    elif evidence.has_structured_fields:
        period_result = match_period(period, evidence.period, "", period_type=period_type)
    else:
        hit = _find_number_span(text, float(value), unit=unit, metric_name=metric_name)
        if hit is None:
            return EvidenceMatch(
                False, None, False, False, False, None, None, "none", "number_not_found"
            )
        number_end = _number_token_end(text, hit[0])
        period_result = _match_period_near_number(
            period, text, hit[0], number_end, evidence.period
        )
    if period_result.status in {"mismatch", "metadata_conflict", "ambiguous"}:
        reason = {
            "mismatch": "period_mismatch",
            "metadata_conflict": "period_metadata_conflict",
            "ambiguous": "period_ambiguous",
        }[period_result.status]
        return EvidenceMatch(
            False, None, False, False, False, None, None, "none", reason
        )
    if period_result.status == "unknown":
        return EvidenceMatch(
            False, None, False, False, False, None, None, "none", "period_unknown"
        )
    period_ok = period_result.matched
    period_state = "ok" if period_ok else period_result.status

    # Structured field path: display text is not the source of truth.
    if evidence.has_structured_fields and not formula_inputs:
        if str(evidence.metric_name) != str(metric_name):
            return EvidenceMatch(
                False,
                float(evidence.value) if evidence.value is not None else None,
                False,
                period_ok,
                False,
                None,
                None,
                "none",
                "metric_name_mismatch",
            )
        if not _structured_values_close(float(evidence.value), float(value)):
            return EvidenceMatch(
                False,
                float(evidence.value),
                True,
                period_ok,
                True,
                None,
                None,
                "none",
                "metric_value_mismatch",
            )
        if not (evidence.unit or "").strip():
            return EvidenceMatch(
                False,
                float(evidence.value),
                True,
                period_ok,
                False,
                None,
                None,
                "none",
                "unit_missing",
            )
        if unit and str(evidence.unit) != str(unit):
            return EvidenceMatch(
                False,
                float(evidence.value),
                True,
                period_ok,
                False,
                None,
                None,
                "none",
                "unit_mismatch",
            )
        if evidence.confidence is None:
            return EvidenceMatch(
                False,
                float(evidence.value),
                True,
                period_ok,
                True,
                None,
                None,
                "none",
                "confidence_missing",
            )
        if evidence.confidence != "high":
            return EvidenceMatch(
                False,
                float(evidence.value),
                True,
                period_ok,
                True,
                None,
                None,
                "low",
                "low_confidence_evidence",
            )
        if not (evidence.citation or "").strip():
            return EvidenceMatch(
                False,
                float(evidence.value),
                True,
                period_ok,
                True,
                None,
                None,
                "none",
                "citation_missing",
            )
        if not evidence.citation_trusted and not (evidence.source_record_id or "").strip():
            return EvidenceMatch(
                False,
                float(evidence.value),
                True,
                period_ok,
                True,
                None,
                None,
                "none",
                "source_record_missing",
            )
        return EvidenceMatch(
            True,
            float(evidence.value),
            True,
            True,
            True,
            None,
            None,
            "high",
            "structured_field_bound",
        )

    structured_source = (
        evidence.source_type in _STRUCTURED_SOURCES
        or any(key in (evidence.citation or "") for key in _STRUCTURED_SOURCES)
        or evidence.evidence_id.startswith("ev_fund_")
    )
    # Never treat source_type=structured as a free pass when structured fields are absent
    # and the claim has metric context — fall through to text rules without skipping checks.
    structured = structured_source and evidence.has_structured_fields

    if formula_inputs:
        missing = [name for name, amount in formula_inputs.items() if amount is None]
        if missing:
            return EvidenceMatch(
                False,
                None,
                False,
                period_ok,
                False,
                None,
                None,
                "none",
                "formula_input_incomplete",
            )
        input_periods: list[PeriodIdentity] = []
        for input_name, input_value in formula_inputs.items():
            if evidence.has_structured_fields:
                # Single structured metric record cannot alone bind a multi-input formula.
                if str(evidence.metric_name) != str(input_name) or not _structured_values_close(
                    float(evidence.value), float(input_value)
                ):
                    return EvidenceMatch(
                        False,
                        None,
                        False,
                        period_ok,
                        False,
                        None,
                        None,
                        "none",
                        "formula_input_incomplete",
                    )
                local_period = match_local_period(
                    period, evidence.text or "", evidence.period, period_type
                )
                if local_period.status == "unknown":
                    return EvidenceMatch(
                        False, None, False, False, False, None, None, "none",
                        "formula_input_period_unknown",
                    )
                if local_period.status == "mismatch":
                    return EvidenceMatch(
                        False, None, False, False, False, None, None, "none",
                        "formula_input_period_mismatch",
                    )
                input_periods.append(
                    parse_period_identity(evidence.period)
                    if evidence.period
                    else parse_period_identity(period)
                )
                continue
            hit = _find_number_span(
                text, float(input_value), unit="billion_usd", metric_name=input_name
            )
            if hit is None:
                return EvidenceMatch(
                    False,
                    None,
                    False,
                    period_ok,
                    False,
                    None,
                    None,
                    "none",
                    "formula_input_incomplete",
                )
            idx, _, conversion, span = hit
            number_end = _number_token_end(text, idx)
            local_period = _match_period_near_number(
                period, text, idx, number_end, evidence.period
            )
            if local_period.status == "unknown":
                return EvidenceMatch(
                    False,
                    None,
                    False,
                    False,
                    False,
                    None,
                    span,
                    "none",
                    "formula_input_period_unknown",
                )
            if local_period.status == "mismatch":
                return EvidenceMatch(
                    False,
                    None,
                    False,
                    False,
                    False,
                    None,
                    span,
                    "none",
                    "formula_input_period_mismatch",
                )
            if local_period.status == "ambiguous":
                return EvidenceMatch(
                    False, None, False, False, False, None, span, "none",
                    "formula_input_period_ambiguous",
                )
            if local_period.status == "metadata_conflict":
                return EvidenceMatch(
                    False, None, False, False, False, None, span, "none",
                    "period_metadata_conflict",
                )
            local_identity = _nearest_period_identity(text, idx, number_end)
            if local_identity is not None:
                input_periods.append(local_identity)
            elif evidence.period:
                input_periods.append(parse_period_identity(evidence.period))
            else:
                input_periods.append(parse_period_identity(period))
            if not _metric_alias_near(text, input_name, idx):
                return EvidenceMatch(
                    False,
                    None,
                    False,
                    period_ok,
                    bool(conversion),
                    conversion,
                    span,
                    "none",
                    "metric_label_mismatch",
                )
            if not conversion and not _unit_declared_near(text, idx):
                return EvidenceMatch(
                    False,
                    None,
                    False,
                    period_ok,
                    False,
                    None,
                    span,
                    "low",
                    "unit_ambiguous",
                )
        claim_id = parse_period_identity(period) if period else PeriodIdentity("unknown")
        if claim_id.kind != "unknown":
            for item in input_periods:
                if item.kind == "unknown" or not _period_identities_compatible(claim_id, item):
                    return EvidenceMatch(
                        False, None, False, False, False, None, None, "none",
                        "formula_input_period_mismatch",
                    )
            for left, right in zip(input_periods, input_periods[1:]):
                if not _period_identities_compatible(left, right):
                    return EvidenceMatch(
                        False, None, False, False, False, None, None, "none",
                        "formula_input_period_mismatch",
                    )
        conf = "high" if structured or structured_source else "medium"
        return EvidenceMatch(
            True,
            float(value),
            True,
            True,
            True,
            None,
            None,
            conf,
            "formula_inputs_bound",
        )

    hit = hit or _find_number_span(text, float(value), unit=unit, metric_name=metric_name)
    if hit is None:
        return EvidenceMatch(
            False, None, False, period_ok, False, None, None, "none", "number_not_found"
        )
    idx, found, conversion, span = hit
    metric_ok = _metric_alias_near(text, metric_name, idx)
    if not metric_ok:
        return EvidenceMatch(
            False,
            found,
            False,
            period_ok,
            bool(conversion) or unit in {None, "billion_usd", "ratio"},
            conversion,
            span,
            "none",
            "metric_label_mismatch",
        )

    unit_ok = True
    conf = "medium"
    if conversion is None and unit == "billion_usd" and abs(float(value)) > 1.5:
        after = text.lower()[idx : idx + 40]
        before = text.lower()[max(0, idx - 80) : idx]
        if (
            "billion" not in after
            and "million" not in after
            and "million" not in before
            and "in millions" not in before
            and "billion" not in before
            and not _unit_declared_near(text, idx)
        ):
            if re.search(r"\b\d{4,}\b", span) or not _unit_declared_near(text, idx):
                return EvidenceMatch(
                    False,
                    found,
                    True,
                    period_ok,
                    False,
                    None,
                    span,
                    "low",
                    "unit_ambiguous",
                )
    if conversion:
        unit_ok = True
        conf = "high"
    elif _unit_declared_near(text, idx):
        conf = "high"

    return EvidenceMatch(
        True,
        found,
        True,
        period_ok,
        unit_ok,
        conversion,
        span,
        conf,
        "bound",
    )


def _page_from_citation(citation: str) -> int | None:
    match = re.search(r"#p(\d+)\b", citation or "", flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _collect_evidence_pool(state: dict[str, Any], company: str) -> list[EvidenceRef]:
    pool: list[EvidenceRef] = []
    seen: set[str] = set()

    def add(
        *,
        evidence_id: str,
        citation: str,
        source_type: str,
        text: str,
        period: str | None = None,
        period_type: str | None = None,
        metric_name: str | None = None,
        value: float | None = None,
        unit: str | None = None,
        confidence: str | None = None,
        period_source: str | None = None,
        period_alignment: str | None = None,
        source_record_id: str | None = None,
        citation_trusted: bool = False,
    ) -> None:
        if not citation or evidence_id in seen:
            return
        seen.add(evidence_id)
        specific = period
        ptype = period_type
        if specific and str(specific).lower() in _PERIOD_TYPE_TOKENS:
            ptype = ptype or str(specific).lower()
            specific = None
        pool.append(
            EvidenceRef(
                evidence_id=evidence_id,
                entity=company,
                citation=citation,
                source_type=source_type,
                text=text or "",
                page=_page_from_citation(citation),
                period=specific,
                period_type=ptype,
                metric_name=metric_name,
                value=value,
                unit=unit,
                confidence=confidence,
                period_source=period_source,
                period_alignment=period_alignment,
                source_record_id=source_record_id,
                citation_trusted=citation_trusted,
            )
        )

    for index, hit in enumerate((state.get("rag_evidence") or {}).get(company) or []):
        supplied_citation = hit.get("citation") or hit.get("source")
        citation = str(supplied_citation or f"rag:{company}:{index}")
        hit_period = str(hit.get("period") or "") or None
        hit_period_source = str(hit.get("period_source") or "") or None
        hit_period_alignment = str(hit.get("period_alignment") or "") or None
        hit_record = str(hit.get("source_record_id") or hit.get("provider_record_id") or "") or None
        trusted_citation = bool(supplied_citation)
        if hit_period and not is_factual_period_provenance(
            period=hit_period,
            period_source=hit_period_source,
            period_alignment=hit_period_alignment,
        ):
            hit_period = None
        add(
            evidence_id=f"ev_rag_{company}_{index}",
            citation=citation,
            source_type=str(hit.get("source_type") or "rag"),
            text=str(hit.get("text") or hit.get("snippet") or ""),
            period=hit_period,
            period_source=hit_period_source,
            period_alignment=hit_period_alignment,
            source_record_id=hit_record,
            citation_trusted=trusted_citation,
        )

    payload = (state.get("retrieved_docs") or {}).get(company) or {}
    period = period_label_from_meta(payload.get("fundamentals_meta"))
    structured = str(payload.get("structured_source") or "none")
    market_data = payload.get("market_data") or {}
    provenance = payload.get("fundamental_provenance") or {}
    if market_data:
        labels = {
            "revenue": "revenue",
            "ebitda": "EBITDA",
            "r_and_d": "R&D",
            "operating_income": "operating income",
        }
        for key, label in labels.items():
            raw = get_fundamental(market_data, key)
            if raw is None:
                continue
            field_prov = provenance.get(key) if isinstance(provenance, dict) else None
            if not isinstance(field_prov, dict):
                field_prov = {}
            field_period = str(field_prov.get("period") or "") or None
            field_period_type = str(field_prov.get("period_type") or "") or None
            field_period_source = str(field_prov.get("period_source") or "") or None
            field_period_alignment = str(field_prov.get("period_alignment") or "") or None
            source_record_id = str(
                field_prov.get("source_record_id") or field_prov.get("provider_record_id") or ""
            ) or None
            supplied_citation = str(field_prov.get("citation") or "") or None
            if field_period and field_period.lower() in _PERIOD_TYPE_TOKENS:
                field_period_type = field_period_type or field_period.lower()
                field_period = None
            record_proof = bool(supplied_citation or source_record_id)
            if not record_proof or not is_factual_period_provenance(
                period=field_period,
                period_source=field_period_source,
                period_alignment=field_period_alignment,
            ):
                field_period = None
            field_source = str(field_prov.get("source") or "fundamentals")
            field_conf = (
                str(field_prov["confidence"])
                if field_prov.get("confidence") is not None
                else None
            )
            source_type = field_source if field_source != "none" else "fundamentals"
            display = f"{company} {label} was {float(raw)} billion USD."
            display_citation = (
                supplied_citation
                or f"lumenfin:{source_type}:{company}:{field_period or 'unknown'}:{key}"
            )
            add(
                evidence_id=f"ev_fund_{company}_{key}_{field_period or 'unknown'}",
                citation=display_citation,
                source_type=source_type,
                text=display,
                period=field_period,
                period_type=field_period_type,
                metric_name=key,
                value=float(raw),
                unit="billion_usd",
                confidence=field_conf,
                period_source=field_period_source,
                period_alignment=field_period_alignment,
                source_record_id=source_record_id,
                citation_trusted=bool(supplied_citation),
            )

    supply = payload.get("supply_chain") or {}
    if supply:
        signals = [str(s) for s in (supply.get("signals") or [])]
        text = (
            f"{company} supply chain risk level is {supply.get('risk_level', 'unknown')}. "
            f"Signals: {'; '.join(signals)}"
        )
        add(
            evidence_id=f"ev_supply_{company}",
            citation=f"lumenfin:supply_chain:{company}:{period}",
            source_type="supply_chain",
            text=text,
            period=period,
        )

    for index, doc in enumerate(payload.get("source_documents") or []):
        citation = str(doc.get("citation") or doc.get("filename") or f"source:{company}:{index}")
        add(
            evidence_id=f"ev_doc_{company}_{index}",
            citation=citation,
            source_type=str(doc.get("source_type") or "document"),
            text=str(doc.get("excerpt") or doc.get("text") or ""),
            period=str(doc.get("period") or "") or None,
            period_source=str(doc.get("period_source") or "") or None,
            period_alignment=str(doc.get("period_alignment") or "") or None,
            source_record_id=str(doc.get("source_record_id") or "") or None,
            citation_trusted=bool(doc.get("citation") or doc.get("filename")),
        )

    scores = (state.get("risk_scores") or {}).get(company) or {}
    if isinstance(scores, dict) and scores:
        parts = [f"{k}={v}" for k, v in scores.items() if isinstance(v, (int, float))]
        if parts:
            add(
                evidence_id=f"ev_risk_{company}",
                citation=f"lumenfin:risk_model:{company}:model",
                source_type="risk_model",
                text=(
                    f"{company} model-derived risk scores are screening indicators: "
                    + ", ".join(parts)
                    + "."
                ),
                period="model",
            )

    snapshot = (state.get("market_snapshots") or {}).get(company) or {}
    if snapshot.get("current_price") is not None:
        details = []
        for key in ("current_price", "trailing_pe", "monthly_return", "fifty_two_week_high", "fifty_two_week_low"):
            if snapshot.get(key) is not None:
                details.append(f"{key}={snapshot.get(key)}")
        add(
            evidence_id=f"ev_mkt_{company}",
            citation=f"lumenfin:market_snapshot:{company}:{snapshot.get('fetched_at') or 'latest'}",
            source_type="market_data",
            text=(
                f"{company} live market snapshot from {snapshot.get('provider') or 'unknown'}: "
                + ", ".join(details)
                + "."
            ),
            period="latest",
        )
    return pool


def _fund_refs_for_values(
    pool: list[EvidenceRef],
    values: list[float | None],
    *,
    entity: str | None = None,
    metric_name: str | None = None,
    unit: str | None = None,
    period: str | None = None,
    formula_inputs: dict[str, float] | None = None,
) -> list[EvidenceRef]:
    """Return fundamentals evidence that structurally matches the claim."""
    needed = [v for v in values if v is not None]
    fund_refs = [r for r in pool if r.evidence_id.startswith("ev_fund_")]

    if formula_inputs and entity:
        bound: list[EvidenceRef] = []
        for input_name, input_value in formula_inputs.items():
            hit = None
            for ref in fund_refs:
                result = match_numeric_evidence(
                    ref,
                    entity=entity,
                    metric_name=str(input_name),
                    value=float(input_value),
                    unit="billion_usd",
                    period=period,
                )
                if result.matched and result.confidence == "high":
                    hit = ref
                    break
            if hit is None:
                for ref in pool:
                    if ref in bound:
                        continue
                    result = match_numeric_evidence(
                        ref,
                        entity=entity,
                        metric_name=str(input_name),
                        value=float(input_value),
                        unit="billion_usd",
                        period=period,
                    )
                    if result.matched and result.confidence in {"high", "medium"}:
                        hit = ref
                        break
            if hit is None:
                return []
            bound.append(hit)
        return bound

    if fund_refs and metric_name and entity and needed:
        matched = [
            r
            for r in fund_refs
            if match_numeric_evidence(
                r,
                entity=entity,
                metric_name=metric_name,
                value=float(needed[0]),
                unit=unit,
                period=period,
            ).matched
        ]
        if matched:
            return matched[:1]
        return []
    if metric_name:
        # Structured metric context present: do not fall back to whole-text number match.
        return []
    if fund_refs and (
        not needed or any(_text_contains_number(r.text, v) for r in fund_refs for v in needed)
    ):
        usable = [
            r
            for r in fund_refs
            if not needed or any(_text_contains_number(r.text, v) for v in needed)
        ]
        if usable:
            return usable[:1]
    for r in pool:
        cite = r.citation or ""
        if not any(
            key in cite
            for key in (
                "sample_financial_data",
                "sec_companyfacts",
                "yahoo_fundamentals",
                "document_extracted",
            )
        ) and r.source_type not in {
            "sec_companyfacts",
            "yahoo_fundamentals",
            "document_extracted",
            "fundamentals",
            "sample_db",
        }:
            continue
        if metric_name and entity and needed:
            if match_numeric_evidence(
                r,
                entity=entity,
                metric_name=metric_name,
                value=float(needed[0]),
                unit=unit,
                period=period,
            ).matched:
                return [r]
            continue
        if needed and not any(_text_contains_number(r.text, v) for v in needed):
            continue
        return [r]
    return []


def _prefer_refs_for_values(
    pool: list[EvidenceRef],
    values: list[float | None],
    *,
    require_values: list[float | None] | None = None,
    entity: str | None = None,
    metric_name: str | None = None,
    unit: str | None = None,
    period: str | None = None,
    formula_inputs: dict[str, float] | None = None,
) -> list[EvidenceRef]:
    """Prefer page-anchored RAG that structurally matches; else fundamentals text."""
    gate = [v for v in (require_values if require_values is not None else values) if v is not None]
    scan = [v for v in values if v is not None]
    matched: list[EvidenceRef] = []
    for ref in pool:
        if ref.source_type not in {"rag", "document"} and "#p" not in (ref.citation or "").lower():
            continue
        if metric_name and entity:
            probe_value = float((gate or scan)[0]) if (gate or scan) else float(values[0] or 0)
            result = match_numeric_evidence(
                ref,
                entity=entity,
                metric_name=metric_name,
                value=probe_value if not formula_inputs else float(values[-1] or probe_value),
                unit=unit,
                period=period,
                formula_inputs=formula_inputs,
            )
            if result.matched:
                matched.append(ref)
            continue
        probe = gate or scan
        if probe and any(_text_contains_number(ref.text, v) for v in probe):
            matched.append(ref)
    if matched:
        matched.sort(key=lambda r: (0 if r.page is not None else 1, r.evidence_id))
        return matched[:2]

    return _fund_refs_for_values(
        pool,
        gate or scan,
        entity=entity,
        metric_name=metric_name,
        unit=unit,
        period=period,
        formula_inputs=formula_inputs,
    )


def _verify_numeric(
    claim: Claim,
    refs: list[EvidenceRef],
    input_values: list[float | None],
    *,
    pool: list[EvidenceRef] | None = None,
    formula_inputs: dict[str, float] | None = None,
) -> Claim:
    needed = [v for v in input_values if v is not None]

    def _bind_formula(candidates: list[EvidenceRef]) -> tuple[list[EvidenceRef], str]:
        assert formula_inputs is not None
        bound: list[EvidenceRef] = []
        last_reason = "formula_input_incomplete"
        input_ids: dict[str, str] = {}
        input_periods: list[PeriodIdentity] = []
        search_space = list(candidates)
        if pool:
            for ref in pool:
                if ref not in search_space:
                    search_space.append(ref)
        for input_name, input_value in formula_inputs.items():
            hit = None
            for ref in search_space:
                result = match_numeric_evidence(
                    ref,
                    entity=claim.entity,
                    metric_name=str(input_name),
                    value=float(input_value),
                    unit="billion_usd",
                    period=claim.period,
                )
                if result.matched and result.matched_period and (
                    result.confidence == "high"
                    or (
                        not ref.has_structured_fields
                        and result.confidence == "medium"
                    )
                ):
                    hit = ref
                    last_reason = result.reason
                    break
                last_reason = result.reason
            if hit is None:
                return [], last_reason
            bound.append(hit)
            input_ids[str(input_name)] = hit.evidence_id
            if hit.period:
                input_periods.append(parse_period_identity(hit.period))
            else:
                local_ids = _extract_period_identities_from_text(hit.text or "")
                input_periods.append(local_ids[0] if local_ids else PeriodIdentity("unknown"))
        claim_id = parse_period_identity(claim.period) if claim.period else PeriodIdentity("unknown")
        if claim_id.kind != "unknown":
            for item in input_periods:
                if item.kind == "unknown" or not _period_identities_compatible(claim_id, item):
                    return [], "formula_input_period_mismatch"
            for left, right in zip(input_periods, input_periods[1:]):
                if not _period_identities_compatible(left, right):
                    return [], "formula_input_period_mismatch"
        # Keep formula input map in verify_reason for auditability.
        claim.verify_reason = f"formula_inputs={input_ids}"
        return bound, "formula_inputs_bound"

    def _usable_from(candidates: list[EvidenceRef]) -> tuple[list[EvidenceRef], str]:
        if formula_inputs and claim.metric_name:
            return _bind_formula(candidates)
        usable: list[EvidenceRef] = []
        last_reason = "number_not_found"
        for ref in candidates:
            if claim.metric_name and isinstance(claim.value, (int, float)):
                result = match_numeric_evidence(
                    ref,
                    entity=claim.entity,
                    metric_name=str(claim.metric_name),
                    value=float(claim.value),
                    unit=claim.unit,
                    period=claim.period,
                )
                if (
                    result.matched
                    and result.matched_period
                    and result.confidence in {"high", "medium"}
                ):
                    # Verified numeric facts require high confidence when structured fields exist.
                    if ref.has_structured_fields and result.confidence != "high":
                        last_reason = result.reason
                        continue
                    usable.append(ref)
                    last_reason = result.reason
                else:
                    last_reason = result.reason
            elif not claim.metric_name and needed and any(
                _text_contains_number(ref.text, v) for v in needed
            ):
                usable.append(ref)
                last_reason = "legacy_number_match"
        return usable, last_reason

    usable, reason = _usable_from(refs)
    if not usable and pool is not None:
        usable, reason = _usable_from(
            _fund_refs_for_values(
                pool,
                input_values,
                entity=claim.entity,
                metric_name=claim.metric_name,
                unit=claim.unit,
                period=claim.period,
                formula_inputs=formula_inputs,
            )
        )
    if not refs and not usable:
        if pool and claim.period:
            relevant = [
                ref
                for ref in pool
                if ref.evidence_id.startswith("ev_fund_")
                and (
                    (formula_inputs and ref.metric_name in formula_inputs)
                    or (not formula_inputs and ref.metric_name == claim.metric_name)
                )
            ]
            if relevant and any(ref.period is None for ref in relevant):
                reason = "formula_input_period_unknown" if formula_inputs else "period_unknown"
        claim.verification = "rejected"
        claim.verify_reason = f"No evidence text contains the metric inputs ({reason})."
        claim.evidence_refs = []
        return claim
    if not usable:
        claim.verification = "rejected"
        claim.verify_reason = f"Candidate evidence rejected: {reason}."
        claim.evidence_refs = []
        return claim
    claim.evidence_refs = usable
    claim.verification = "verified"
    detail = claim.verify_reason if claim.verify_reason.startswith("formula_inputs=") else ""
    claim.verify_reason = (
        f"Metric/period/unit-bound evidence ({reason})"
        + (f"; {detail}" if detail else "")
        + (
            f"; formula_inputs={sorted(formula_inputs)}"
            if formula_inputs and not detail
            else ""
        )
    )
    return claim


def build_claims(state: dict[str, Any]) -> list[Claim]:
    """Build and verify claims from structured state (no LLM)."""
    claims: list[Claim] = []
    companies = list(state.get("companies") or [])
    metrics_by_co = state.get("financial_metrics") or {}
    retrieved = state.get("retrieved_docs") or {}
    fatal_gap = bool(state.get("fatal_data_gap"))

    for company in companies:
        pool = _collect_evidence_pool(state, company)
        payload = retrieved.get(company) or {}
        market_data = payload.get("market_data") or {}
        period = period_label_from_meta(payload.get("fundamentals_meta"))
        metrics = metrics_by_co.get(company) or {}
        structured = str(payload.get("structured_source") or "none")
        # Fail-closed: do not mint verified numeric "facts" when fundamentals are absent.
        block_numeric = fatal_gap or structured == "none" or not has_computable_fundamentals(payload)

        # --- Numeric ratio claims (AST) ---
        if not block_numeric:
            for metric_name in AST_RATIO_KEYS:
                value = metrics.get(metric_name)
                if not isinstance(value, (int, float)):
                    continue
                input_map = FORMULA_INPUTS.get(metric_name) or {}
                input_vals = [get_fundamental(market_data, src) for src in input_map.values()]
                formula_inputs = {
                    src: float(get_fundamental(market_data, src))
                    for src in input_map.values()
                    if get_fundamental(market_data, src) is not None
                }
                label = METRIC_LABELS.get(metric_name, metric_name)
                statement = f"{company} {label} is {_fmt_pct(float(value))} for {period}."
                claim = Claim(
                    claim_id=f"cl_num_{company}_{metric_name}",
                    entity=company,
                    claim_type="numeric",
                    statement=statement,
                    value=float(value),
                    unit="ratio",
                    period=period,
                    metric_name=metric_name,
                )
                refs = _prefer_refs_for_values(
                    pool,
                    input_vals + [float(value)],
                    require_values=input_vals,
                    entity=company,
                    metric_name=metric_name,
                    unit="ratio",
                    period=period,
                    formula_inputs=formula_inputs or None,
                )
                claims.append(
                    _verify_numeric(
                        claim,
                        refs,
                        input_vals,
                        pool=pool,
                        formula_inputs=formula_inputs or None,
                    )
                )

            # Absolute fundamentals as numeric claims (for ledger + consistency)
            for key, label in (
                ("revenue", "Revenue"),
                ("ebitda", "EBITDA"),
                ("operating_income", "Operating income"),
                ("r_and_d", "R&D expense"),
            ):
                raw = get_fundamental(market_data, key)
                if raw is None:
                    continue
                statement = f"{company} {label} is {_fmt_num(float(raw))} billion USD for {period}."
                claim = Claim(
                    claim_id=f"cl_abs_{company}_{key}",
                    entity=company,
                    claim_type="numeric",
                    statement=statement,
                    value=float(raw),
                    unit="billion_usd",
                    period=period,
                    metric_name=key,
                )
                refs = _prefer_refs_for_values(
                    pool,
                    [float(raw)],
                    require_values=[float(raw)],
                    entity=company,
                    metric_name=key,
                    unit="billion_usd",
                    period=period,
                )
                claims.append(_verify_numeric(claim, refs, [float(raw)], pool=pool))

            # Market snapshot numerics only when structured fundamentals exist.
            snapshot = (state.get("market_snapshots") or {}).get(company) or {}
            pe = snapshot.get("trailing_pe")
            if isinstance(pe, (int, float)):
                claim = Claim(
                    claim_id=f"cl_num_{company}_pe_ratio",
                    entity=company,
                    claim_type="numeric",
                    statement=f"{company} trailing P/E is {float(pe):.2f}x (live market snapshot).",
                    value=float(pe),
                    unit="multiple",
                    period="latest",
                    metric_name="pe_ratio",
                )
                mkt_refs = [r for r in pool if r.evidence_id.startswith("ev_mkt_")]
                if mkt_refs and _text_contains_number(mkt_refs[0].text, float(pe)):
                    claim.evidence_refs = mkt_refs[:1]
                    claim.verification = "verified"
                    claim.verify_reason = "Bound to live market snapshot evidence."
                else:
                    claim.verification = "rejected"
                    claim.verify_reason = "No market snapshot evidence for P/E."
                claims.append(claim)
        else:
            claim = Claim(
                claim_id=f"cl_num_{company}_blocked",
                entity=company,
                claim_type="numeric",
                statement=(
                    f"{company}: numeric claims withheld — structured fundamentals unavailable "
                    f"(fatal_data_gap={fatal_gap}, structured_source={structured})."
                ),
                metric_name="numeric_blocked",
                verification="rejected",
                verify_reason="Fail-closed: refusing verified numeric claims without computable fundamentals.",
            )
            claims.append(claim)

        # --- Growth claims: only with multi-period fundamentals ---
        if not block_numeric:
            growth_claim = _build_growth_claim(company, market_data, period, pool)
            if growth_claim is not None:
                claims.append(growth_claim)
        else:
            claims.append(
                Claim(
                    claim_id=f"cl_growth_{company}_revenue",
                    entity=company,
                    claim_type="growth",
                    statement=f"{company}: revenue growth claim withheld under fail-closed data gap.",
                    metric_name="revenue_growth",
                    period=period,
                    verification="rejected",
                    verify_reason="Fail-closed: no computable fundamentals for growth.",
                )
            )

        # --- Risk conclusions ---
        supply = payload.get("supply_chain") or {}
        risk_level = str(supply.get("risk_level") or "unknown")
        risk_scores = (state.get("risk_scores") or {}).get(company) or {}
        if supply or risk_scores or block_numeric:
            if block_numeric:
                statement = (
                    f"{company} data-limitation risk is elevated: no AST-computable fundamentals "
                    f"(structured_source={structured})."
                )
                metric_name = "data_limitation_risk"
            else:
                statement = (
                    f"{company} supply-chain risk signal is '{risk_level}'"
                    + (
                        f" with model supply_chain_risk={risk_scores.get('supply_chain_risk')}"
                        if isinstance(risk_scores.get("supply_chain_risk"), (int, float))
                        else ""
                    )
                    + "."
                )
                metric_name = "supply_chain_risk"
            claim = Claim(
                claim_id=f"cl_risk_{company}_supply",
                entity=company,
                claim_type="risk_conclusion",
                statement=statement,
                value=risk_level if not block_numeric else "elevated",
                period=period,
                metric_name=metric_name,
            )
            refs = [r for r in pool if r.evidence_id.startswith("ev_supply_")]
            rag_risk = [
                r
                for r in pool
                if (r.page is not None or "#p" in r.citation.lower())
                and re.search(r"risk|supply|supplier|concentration", r.text, re.I)
            ]
            if rag_risk:
                refs = rag_risk[:1] + refs[:1]
            if block_numeric:
                # Bind to explicit fail-closed provenance rather than market snapshots.
                claim.evidence_refs = [
                    EvidenceRef(
                        evidence_id=f"ev_gap_{company}",
                        entity=company,
                        citation=f"lumenfin:data_gap:{company}:{structured}",
                        source_type="data_gap",
                        text=statement,
                        period=period,
                    )
                ]
                claim.verification = "verified"
                claim.verify_reason = "Fail-closed data-limitation risk bound to structured_source=none provenance."
            elif risk_level.lower() in {"unknown", "n/a", "none"} and not block_numeric:
                # Do not promote an "unknown" supply-chain placeholder into material risk/thesis.
                claim.verification = "rejected"
                claim.verify_reason = (
                    "Supply-chain signal is unknown; withheld from material risk conclusions."
                )
                claim.evidence_refs = []
            elif refs:
                claim.evidence_refs = refs[:2]
                claim.verification = "verified"
                claim.verify_reason = "Bound to supply-chain / risk evidence."
            else:
                risk_model = [r for r in pool if r.evidence_id.startswith("ev_risk_")]
                if risk_model:
                    claim.evidence_refs = risk_model[:1]
                    claim.verification = "verified"
                    claim.verify_reason = "Bound to risk-model screening evidence (not a filing fact)."
                else:
                    claim.verification = "rejected"
                    claim.verify_reason = "No risk evidence available."
            claims.append(claim)

        if not block_numeric:
            for dim in ("financial_risk", "market_risk", "operational_risk"):
                score = risk_scores.get(dim)
                if not isinstance(score, (int, float)):
                    continue
                level = "Low" if score < 3.5 else ("Moderate" if score < 6.5 else "Elevated")
                claim = Claim(
                    claim_id=f"cl_risk_{company}_{dim}",
                    entity=company,
                    claim_type="risk_conclusion",
                    statement=f"{company} {dim.replace('_', ' ')} screening score is {score:.1f}/10 ({level}).",
                    value=float(score),
                    unit="score_1_10",
                    period="model",
                    metric_name=dim,
                )
                risk_refs = [r for r in pool if r.evidence_id.startswith("ev_risk_")]
                mkt_refs = [r for r in pool if r.evidence_id.startswith("ev_mkt_")] if dim == "market_risk" else []
                refs = (mkt_refs or risk_refs)[:1]
                if refs:
                    claim.evidence_refs = refs
                    claim.verification = "verified"
                    claim.verify_reason = "Bound to risk-model / market evidence as screening conclusion."
                else:
                    claim.verification = "rejected"
                    claim.verify_reason = "Missing risk-model evidence."
                claims.append(claim)

    # --- Investment conclusions: only compose from verified numeric+risk ---
    verified_so_far = filter_verified(claims)
    for company in companies:
        payload = retrieved.get(company) or {}
        structured = str(payload.get("structured_source") or "none")
        block_numeric = fatal_gap or structured == "none" or not has_computable_fundamentals(payload)
        num = verified_by_entity(verified_so_far, company, claim_type="numeric")
        risk = verified_by_entity(verified_so_far, company, claim_type="risk_conclusion")
        # Prefer EBITDA margin; fall back to operating margin for PDF extracts lacking EBITDA.
        profit_claims = [c for c in num if c.metric_name == "ebitda_margin"] or [
            c for c in num if c.metric_name == "operating_margin"
        ]
        claim = Claim(
            claim_id=f"cl_inv_{company}_thesis",
            entity=company,
            claim_type="investment_conclusion",
            statement="",
            metric_name="research_thesis",
        )
        if block_numeric or not profit_claims or not risk:
            claim.verification = "rejected"
            claim.verify_reason = (
                "Investment conclusion omitted: requires verified numeric profitability "
                "and verified risk conclusion."
            )
            claim.statement = (
                f"{company}: no evidence-backed investment conclusion — missing verified "
                "profitability and/or risk claims."
            )
            claims.append(claim)
            continue
        profit = profit_claims[0]
        # Prefer scored screening dimensions; never anchor thesis on unknown supply-chain.
        usable_risk = [
            c
            for c in risk
            if not (
                c.metric_name == "supply_chain_risk"
                and str(c.value).lower() in {"unknown", "n/a", "none"}
            )
        ]
        scored_risk = [
            c
            for c in usable_risk
            if c.metric_name in {"financial_risk", "operational_risk", "market_risk", "data_limitation_risk"}
        ]
        risk_c = scored_risk[0] if scored_risk else (usable_risk[0] if usable_risk else None)
        if risk_c is None:
            claim.verification = "rejected"
            claim.verify_reason = (
                "Investment conclusion omitted: no material verified risk conclusion "
                "(unknown supply-chain alone is insufficient)."
            )
            claim.statement = (
                f"{company}: no evidence-backed investment conclusion — missing material "
                "verified risk claims."
            )
            claims.append(claim)
            continue
        profit_v = float(profit.value) if isinstance(profit.value, (int, float)) else 0.0
        # Operating-margin thresholds are slightly lower than EBITDA-margin screens.
        strong, adequate = (0.25, 0.15) if profit.metric_name == "ebitda_margin" else (0.20, 0.12)
        if profit_v >= strong:
            stance = "quality-screening research thesis (not a recommendation)"
        elif profit_v >= adequate:
            stance = "neutral quality-compounder research screen (not a recommendation)"
        else:
            stance = "defensive research posture pending operational evidence (not a recommendation)"
        risk_label = (
            f"{risk_c.metric_name}={risk_c.value}"
            if isinstance(risk_c.value, (int, float))
            else str(risk_c.value)
        )
        claim.statement = (
            f"{company} supports a {stance} based on verified {profit.metric_name} "
            f"({_fmt_pct(profit_v)}) and verified risk screen ({risk_label})."
        )
        claim.evidence_refs = list(profit.evidence_refs[:1]) + list(risk_c.evidence_refs[:1])
        claim.verification = "verified"
        claim.verify_reason = "Composed only from verified numeric + risk claims."
        claim.value = stance
        claims.append(claim)

    return claims


def _build_growth_claim(
    company: str,
    market_data: dict[str, Any],
    period: str,
    pool: list[EvidenceRef],
) -> Claim | None:
    """Emit growth claim only when two periods of revenue exist structurally."""
    current = get_fundamental(market_data, "revenue")
    prior = None
    prior_key = None
    for key, value in (market_data or {}).items():
        match = re.match(r"revenue_(20\d{2})$", str(key))
        if not match:
            continue
        year = int(match.group(1))
        # Prefer an older year than current period tag if present.
        try:
            prior = float(value)
            prior_key = str(key)
        except (TypeError, ValueError):
            continue
        _ = year
    # Also accept explicit prior_revenue
    if prior is None and market_data.get("prior_revenue") is not None:
        try:
            prior = float(market_data["prior_revenue"])
            prior_key = "prior_revenue"
        except (TypeError, ValueError):
            prior = None

    claim = Claim(
        claim_id=f"cl_growth_{company}_revenue",
        entity=company,
        claim_type="growth",
        statement="",
        metric_name="revenue_growth",
        period=period,
    )
    if current is None or prior is None or prior == 0:
        # Explicitly reject heuristic/scenario growth — do not invent.
        claim.verification = "rejected"
        claim.verify_reason = (
            "No multi-period revenue pair in structured fundamentals; "
            "refusing heuristic growth claims."
        )
        claim.statement = (
            f"{company}: revenue growth claim withheld — multi-period fundamentals unavailable."
        )
        return claim

    growth = (float(current) - float(prior)) / abs(float(prior))
    claim.value = growth
    claim.unit = "ratio"
    claim.statement = (
        f"{company} revenue growth is {_fmt_pct(growth)} "
        f"(from {prior_key}={_fmt_num(float(prior))} to revenue={_fmt_num(float(current))} billion USD)."
    )
    refs = _prefer_refs_for_values(pool, [float(current), float(prior), growth])
    return _verify_numeric(claim, refs, [float(current), float(prior)])


def binding_summary(claims: list[Claim]) -> dict[str, Any]:
    verified = filter_verified(claims)
    by_type: dict[str, dict[str, int]] = {}
    for claim in claims:
        bucket = by_type.setdefault(claim.claim_type, {"total": 0, "verified": 0, "rejected": 0})
        bucket["total"] += 1
        if claim.verification == "verified":
            bucket["verified"] += 1
        elif claim.verification == "rejected":
            bucket["rejected"] += 1
    page_bound = sum(1 for c in verified if any(r.page is not None or "#p" in r.citation.lower() for r in c.evidence_refs))
    return {
        "total_claims": len(claims),
        "verified_claims": len(verified),
        "rejected_claims": sum(1 for c in claims if c.verification == "rejected"),
        "page_anchored_verified": page_bound,
        "by_type": by_type,
        "bind_rate": round(len(verified) / len(claims), 4) if claims else 0.0,
    }


def format_verified_claims_ledger(claims: list[Claim]) -> list[str]:
    from .reporting import humanize_citation

    verified = filter_verified(claims)
    lines = [
        "## Appendix A. Verified Claims Ledger",
        "",
        "*Audit appendix: only structurally verified claims may back material assertions. "
        "Internal claim IDs are retained here for traceability.*",
        "",
    ]
    if not verified:
        lines.append("- (none — report will withhold evidence-backed financial assertions)")
        lines.append("")
        return lines
    lines.append("| Entity | Type | Statement | Source |")
    lines.append("|--------|------|-----------|--------|")
    for claim in verified:
        cite = humanize_citation(claim.primary_citation).replace("|", "/")
        stmt = claim.statement.replace("|", "/")
        lines.append(
            f"| {claim.entity} | {claim.claim_type} | {stmt} | {cite} |"
        )
    lines.append("")
    return lines
