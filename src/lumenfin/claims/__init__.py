"""Claim → Evidence Binding layer.

Builds internal claim objects from structured state, verifies them against
evidence (RAG page anchors and/or fundamentals text), and exposes only
verified claims for the report synthesizer.

No prompt-forced citations: citations come from evidence already present
on the claim after structural verification.
"""

from __future__ import annotations

from .binding import (
    _collect_evidence_pool,
    _fund_refs_for_values,
    _prefer_refs_for_values,
    _verify_numeric,
)
from .build import (
    _build_growth_claim,
    binding_summary,
    build_claims,
    format_verified_claims_ledger,
)
from .models import (
    FORMULA_INPUTS,
    METRIC_ALIASES,
    METRIC_LABELS,
    Claim,
    ClaimType,
    EvidenceRef,
    Verification,
    claim_to_dict,
    claims_from_state,
    filter_verified,
    verified_by_entity,
)
from .numeric import (
    EvidenceMatch,
    _fmt_num,
    _fmt_pct,
    _number_variants,
    _text_contains_number,
    match_numeric_evidence,
)
from .period import (
    PeriodIdentity,
    PeriodMatch,
    _period_identities_compatible,
    is_factual_period_provenance,
    match_local_period,
    match_period,
    parse_period_identity,
)

__all__ = [
    "FORMULA_INPUTS",
    "METRIC_ALIASES",
    "METRIC_LABELS",
    "Claim",
    "ClaimType",
    "EvidenceMatch",
    "EvidenceRef",
    "PeriodIdentity",
    "PeriodMatch",
    "Verification",
    "_build_growth_claim",
    "_collect_evidence_pool",
    "_fmt_num",
    "_fmt_pct",
    "_fund_refs_for_values",
    "_number_variants",
    "_period_identities_compatible",
    "_prefer_refs_for_values",
    "_text_contains_number",
    "_verify_numeric",
    "binding_summary",
    "build_claims",
    "claim_to_dict",
    "claims_from_state",
    "filter_verified",
    "format_verified_claims_ledger",
    "is_factual_period_provenance",
    "match_local_period",
    "match_numeric_evidence",
    "match_period",
    "parse_period_identity",
    "verified_by_entity",
]
