"""Numeric evidence matching helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import METRIC_ALIASES, EvidenceRef, _STRUCTURED_SOURCES
from .period import (
    PeriodIdentity,
    PeriodMatch,
    _match_period_near_number,
    _nearest_period_identity,
    _period_identities_compatible,
    is_factual_period_provenance,
    match_local_period,
    match_period,
    parse_period_identity,
)

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
        claim_period_id = parse_period_identity(period)
        if claim_period_id.kind != "unknown" and not is_factual_period_provenance(
            period=evidence.period,
            period_source=evidence.period_source,
            period_alignment=evidence.period_alignment,
        ):
            return EvidenceMatch(
                False,
                float(evidence.value) if evidence.value is not None else None,
                bool(evidence.metric_name == metric_name),
                False,
                bool(evidence.unit),
                None,
                None,
                "none",
                "period_provenance_invalid",
            )
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

