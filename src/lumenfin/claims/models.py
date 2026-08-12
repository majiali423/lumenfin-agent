"""Claim models and shared constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

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
            from ..reporting import humanize_citation

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

