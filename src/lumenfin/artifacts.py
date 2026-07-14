from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


StructuredSource = Literal[
    "sample_db",
    "document_extracted",
    "sec_companyfacts",
    "yahoo_fundamentals",
    "none",
]
ViolationSeverity = Literal["critical", "high", "medium", "low"]


@dataclass(frozen=True)
class RetrievalProvenance:
    structured_source: StructuredSource
    market_provider: str
    market_status: str
    rag_enabled: bool
    rag_hit_count: int
    document_count: int
    data_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalConfidence:
    overall: float
    market_data: float
    live_market: float
    rag_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalArtifact:
    company: str
    market_data: dict[str, Any]
    supply_chain: dict[str, Any]
    earnings_call_quotes: list[str]
    source_documents: list[dict[str, Any]]
    market_snapshot: dict[str, Any]
    profile: str
    rag_hits: list[dict[str, Any]]
    provenance: RetrievalProvenance
    confidence: RetrievalConfidence
    structured_source: StructuredSource = "none"
    appendix: dict[str, Any] = field(default_factory=dict)
    fundamentals_meta: dict[str, Any] = field(default_factory=dict)

    def to_legacy_payload(self) -> dict[str, Any]:
        """Backward-compatible dict stored under state['retrieved_docs'][company]."""
        payload: dict[str, Any] = {
            "market_data": dict(self.market_data),
            "supply_chain": dict(self.supply_chain),
            "earnings_call_quotes": list(self.earnings_call_quotes),
            "source_documents": list(self.source_documents),
            "live_market": dict(self.market_snapshot),
            "structured_source": self.structured_source,
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence.to_dict(),
        }
        if self.appendix:
            payload["appendix"] = dict(self.appendix)
        if self.fundamentals_meta:
            payload["fundamentals_meta"] = dict(self.fundamentals_meta)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "market_data": self.market_data,
            "supply_chain": self.supply_chain,
            "earnings_call_quotes": self.earnings_call_quotes,
            "source_documents": self.source_documents,
            "market_snapshot": self.market_snapshot,
            "profile": self.profile,
            "rag_hits": self.rag_hits,
            "provenance": self.provenance.to_dict(),
            "confidence": self.confidence.to_dict(),
            "structured_source": self.structured_source,
            "appendix": self.appendix,
            "fundamentals_meta": self.fundamentals_meta,
        }


@dataclass(frozen=True)
class Violation:
    code: str
    severity: ViolationSeverity
    message: str
    company: str = ""
    repair_target: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.company:
            payload.pop("company", None)
        if not self.repair_target:
            payload.pop("repair_target", None)
        return payload


@dataclass(frozen=True)
class RepairPolicy:
    code: str
    target: str
    priority: int

    def matches(self, violation: Violation) -> bool:
        return violation.code == self.code


def score_retrieval_confidence(
    *,
    market_data: dict[str, Any],
    live_market: dict[str, Any],
    rag_hits: list[dict[str, Any]],
) -> RetrievalConfidence:
    core_keys = ("revenue_2025", "ebitda_2025", "operating_income_2025")
    present = sum(1 for key in core_keys if market_data.get(key) not in (None, "", 0))
    market_data_score = round(present / len(core_keys), 3)

    status = str(live_market.get("status") or "")
    if live_market.get("current_price") is not None:
        live_market_score = 1.0
    elif status == "failed":
        live_market_score = 0.0
    else:
        live_market_score = 0.35

    rag_coverage = round(min(1.0, len(rag_hits) / 3), 3) if rag_hits else 0.0
    overall = round(market_data_score * 0.5 + live_market_score * 0.25 + rag_coverage * 0.25, 3)
    return RetrievalConfidence(
        overall=overall,
        market_data=market_data_score,
        live_market=live_market_score,
        rag_coverage=rag_coverage,
    )
