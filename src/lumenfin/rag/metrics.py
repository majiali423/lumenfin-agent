from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .chunking import chunk_document

CITATION_PATTERN = re.compile(r"^.+#p\d+$", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]+")


def _normalize_terms(terms: list[str]) -> list[str]:
    return [term.strip().lower() for term in terms if term.strip()]


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def find_relevant_chunks(
    document: dict[str, Any],
    relevant_terms: list[str],
) -> list[dict[str, Any]]:
    """Ground-truth chunks whose text contains any of the relevant terms."""
    terms = _normalize_terms(relevant_terms)
    if not terms:
        return []
    relevant: list[dict[str, Any]] = []
    for chunk in chunk_document(document):
        text = chunk["text"].lower()
        if any(term in text for term in terms):
            relevant.append(chunk)
    return relevant


def recall_at_k(
    retrieved: list[dict[str, Any]],
    relevant_ids: set[str],
    *,
    k: int,
) -> float:
    """Fraction of ground-truth relevant chunks found in the top-k results."""
    if not relevant_ids:
        return 1.0
    top_k = retrieved[:k]
    found = {item.get("chunk_id") for item in top_k if item.get("chunk_id") in relevant_ids}
    return len(found) / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved: list[dict[str, Any]],
    relevant_ids: set[str],
) -> float:
    """Reciprocal rank of the first relevant chunk (0 if none)."""
    if not relevant_ids:
        return 1.0
    for rank, item in enumerate(retrieved, start=1):
        if item.get("chunk_id") in relevant_ids:
            return 1.0 / rank
    return 0.0


def is_valid_citation(citation: str | None) -> bool:
    return bool(citation and CITATION_PATTERN.match(citation.strip()))


def citation_coverage(hits: list[dict[str, Any]]) -> float:
    """Share of retrieved hits that carry a well-formed filename#page citation."""
    if not hits:
        return 0.0
    valid = sum(1 for hit in hits if is_valid_citation(hit.get("citation")))
    return valid / len(hits)


def citation_recall_at_k(
    retrieved: list[dict[str, Any]],
    relevant_citations: set[str],
    *,
    k: int,
) -> float:
    """Fraction of relevant citations present in top-k retrieved hits."""
    if not relevant_citations:
        return 1.0
    top_k = retrieved[:k]
    found = {
        hit.get("citation")
        for hit in top_k
        if hit.get("citation") in relevant_citations and is_valid_citation(hit.get("citation"))
    }
    return len(found) / len(relevant_citations)


def groundedness_score(
    hits: list[dict[str, Any]],
    *,
    query: str,
    relevant_terms: list[str],
) -> float:
    """
    Heuristic faithfulness / groundedness without an LLM judge.

    Rank-weighted token overlap between retrieval hits and query + ground-truth terms.
    """
    anchor_tokens = _tokenize(query)
    anchor_tokens.update(_normalize_terms(relevant_terms))
    if not anchor_tokens or not hits:
        return 0.0

    weighted = 0.0
    weight_sum = 0.0
    for rank, hit in enumerate(hits, start=1):
        hit_tokens = _tokenize(hit.get("text", ""))
        if not hit_tokens:
            continue
        overlap = len(anchor_tokens & hit_tokens) / len(anchor_tokens)
        weight = 1.0 / rank
        weighted += overlap * weight
        weight_sum += weight
    return round(weighted / weight_sum, 4) if weight_sum else 0.0


def citations_in_report(report: str, citations: list[str]) -> list[str]:
    """Citations from retrieval that appear verbatim in a generated report."""
    if not report:
        return []
    return [citation for citation in citations if citation and citation in report]


def report_citation_coverage(report: str, citations: list[str]) -> float:
    """Share of retrieved citations echoed in the final report text."""
    if not citations:
        return 1.0
    matched = citations_in_report(report, citations)
    return len(matched) / len(citations)


@dataclass
class RagEvalResult:
    case_id: str
    company: str
    query: str
    passed: bool
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    citation_coverage: float = 0.0
    citation_recall_at_k: dict[int, float] = field(default_factory=dict)
    groundedness: float = 0.0
    relevant_chunk_count: int = 0
    retrieved_count: int = 0
    top_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "company": self.company,
            "query": self.query,
            "passed": self.passed,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "citation_coverage": self.citation_coverage,
            "citation_recall_at_k": self.citation_recall_at_k,
            "groundedness": self.groundedness,
            "relevant_chunk_count": self.relevant_chunk_count,
            "retrieved_count": self.retrieved_count,
            "top_citations": self.top_citations,
        }


def evaluate_retrieval_case(
    *,
    case_id: str,
    company: str,
    query: str,
    document: dict[str, Any],
    retrieved: list[dict[str, Any]],
    relevant_terms: list[str],
    k_values: list[int] | None = None,
    min_recall_at_k: int = 3,
    min_groundedness: float = 0.2,
) -> RagEvalResult:
    """Score one retrieval eval case with Recall@K, MRR, citation, and groundedness metrics."""
    k_values = sorted(set(k_values or [1, 3, 5]))
    relevant_chunks = find_relevant_chunks(document, relevant_terms)
    relevant_ids = {chunk["chunk_id"] for chunk in relevant_chunks}
    relevant_citations = {
        f"{chunk['filename']}#p{chunk['page']}"
        for chunk in relevant_chunks
    }

    recall_scores = {k: recall_at_k(retrieved, relevant_ids, k=k) for k in k_values}
    citation_recall_scores = {
        k: citation_recall_at_k(retrieved, relevant_citations, k=k) for k in k_values
    }
    mrr = mean_reciprocal_rank(retrieved, relevant_ids)
    cov = citation_coverage(retrieved)
    grounded = groundedness_score(retrieved, query=query, relevant_terms=relevant_terms)

    recall_ok = recall_scores.get(min_recall_at_k, 0.0) >= 1.0 if relevant_ids else True
    mrr_ok = mrr > 0.0 if relevant_ids else True
    citation_ok = cov >= 1.0 if retrieved else True
    grounded_ok = grounded >= min_groundedness if retrieved else True
    passed = recall_ok and mrr_ok and citation_ok and grounded_ok

    return RagEvalResult(
        case_id=case_id,
        company=company,
        query=query,
        passed=passed,
        recall_at_k={k: round(v, 4) for k, v in recall_scores.items()},
        mrr=round(mrr, 4),
        citation_coverage=round(cov, 4),
        citation_recall_at_k={k: round(v, 4) for k, v in citation_recall_scores.items()},
        groundedness=grounded,
        relevant_chunk_count=len(relevant_chunks),
        retrieved_count=len(retrieved),
        top_citations=[hit.get("citation", "") for hit in retrieved[:5] if hit.get("citation")],
    )


def summarize_eval_results(results: list[RagEvalResult]) -> dict[str, Any]:
    if not results:
        return {"cases": 0, "passed": 0, "pass_rate": 0.0}

    k_keys = sorted({k for result in results for k in result.recall_at_k})
    aggregate: dict[str, Any] = {
        "cases": len(results),
        "passed": sum(1 for result in results if result.passed),
        "pass_rate": round(sum(1 for result in results if result.passed) / len(results), 4),
        "mean_mrr": round(sum(result.mrr for result in results) / len(results), 4),
        "mean_citation_coverage": round(
            sum(result.citation_coverage for result in results) / len(results), 4
        ),
        "mean_groundedness": round(sum(result.groundedness for result in results) / len(results), 4),
    }
    for k in k_keys:
        aggregate[f"mean_recall_at_{k}"] = round(
            sum(result.recall_at_k.get(k, 0.0) for result in results) / len(results), 4
        )
        aggregate[f"mean_citation_recall_at_{k}"] = round(
            sum(result.citation_recall_at_k.get(k, 0.0) for result in results) / len(results), 4
        )
    return aggregate
