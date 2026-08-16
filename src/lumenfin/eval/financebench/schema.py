from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SplitName = Literal["dev", "test"]
MatchReason = Literal["page_cover", "span_overlap"]


@dataclass(frozen=True)
class EvidenceSpan:
    evidence_doc_name: str
    evidence_page_num_zero: int
    evidence_page_num_one: int
    evidence_text: str
    evidence_text_full_page: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Full page extracts are gold provenance, not eval output.
        payload.pop("evidence_text_full_page", None)
        return payload


@dataclass(frozen=True)
class DocumentInfo:
    doc_name: str
    company: str = ""
    doc_type: str = ""
    period: str = ""
    gics_sector: str = ""
    ticker: str = ""
    pdf_filename: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "company": self.company,
            "doc_type": self.doc_type,
            "period": self.period,
            "gics_sector": self.gics_sector,
            "ticker": self.ticker,
            "pdf_filename": self.pdf_filename,
        }


@dataclass(frozen=True)
class FinanceBenchQuestion:
    financebench_id: str
    case_id: str
    question: str
    answer: str
    justification: str
    question_type: str
    question_reasoning: str
    domain_question_num: str
    company: str
    doc_name: str
    dataset_subset_label: str
    evidence: tuple[EvidenceSpan, ...]
    document: DocumentInfo | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "financebench_id": self.financebench_id,
            "case_id": self.case_id,
            "question": self.question,
            "answer": self.answer,
            "justification": self.justification,
            "question_type": self.question_type,
            "question_reasoning": self.question_reasoning,
            "domain_question_num": self.domain_question_num,
            "company": self.company,
            "doc_name": self.doc_name,
            "dataset_subset_label": self.dataset_subset_label,
            "evidence": [item.to_dict() for item in self.evidence],
            "document": self.document.to_public_dict() if self.document else None,
        }


@dataclass(frozen=True)
class GoldPage:
    doc_name: str
    page_one: int
    page_zero: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.doc_name, self.page_one)


@dataclass(frozen=True)
class ChunkQrel:
    chunk_id: str
    doc_name: str
    page_one: int
    match_reason: MatchReason


@dataclass(frozen=True)
class CaseQrels:
    case_id: str
    financebench_id: str
    gold_pages: tuple[GoldPage, ...]
    gold_chunk_ids: tuple[str, ...]
    chunk_qrels: tuple[ChunkQrel, ...]
    page_provenance_ok: bool
    notes: tuple[str, ...] = ()
    page_chunk_ids: tuple[str, ...] = ()
    span_chunk_ids: tuple[str, ...] = ()
    span_mapped_count: int = 0
    span_unmapped_count: int = 0

    @property
    def single_gold_page(self) -> bool:
        return len({item.key for item in self.gold_pages}) == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "financebench_id": self.financebench_id,
            "gold_pages": [
                {
                    "doc_name": page.doc_name,
                    "page_one": page.page_one,
                    "page_zero": page.page_zero,
                }
                for page in self.gold_pages
            ],
            "gold_chunk_ids": list(self.gold_chunk_ids),
            "gold_chunk_ids_deprecated": True,
            "gold_chunk_ids_semantics": (
                "union of page-derived and evidence-span chunk ids; prefer "
                "page_chunk_ids / span_chunk_ids"
            ),
            "page_chunk_ids": list(self.page_chunk_ids),
            "span_chunk_ids": list(self.span_chunk_ids),
            "span_mapped_count": self.span_mapped_count,
            "span_unmapped_count": self.span_unmapped_count,
            "chunk_qrels": [
                {
                    "chunk_id": item.chunk_id,
                    "doc_name": item.doc_name,
                    "page_one": item.page_one,
                    "match_reason": item.match_reason,
                }
                for item in self.chunk_qrels
            ],
            "page_provenance_ok": self.page_provenance_ok,
            "single_gold_page": self.single_gold_page,
            "notes": list(self.notes),
        }
