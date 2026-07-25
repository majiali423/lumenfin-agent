"""Cross-company RAG hit dedupe for peer-table / shared-chunk retrieval."""

from __future__ import annotations

from typing import Any


def _fingerprint(hit: dict[str, Any]) -> str:
    chunk_id = str(hit.get("chunk_id") or "").strip()
    if chunk_id:
        return f"id:{chunk_id}"
    citation = str(hit.get("citation") or "").strip()
    text = str(hit.get("text") or "").strip().lower()[:180]
    return f"cite:{citation}|{text}"


def _score(hit: dict[str, Any]) -> float:
    for key in ("rerank_score", "fusion_score", "score"):
        value = hit.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _company_mentioned(company: str, text: str) -> bool:
    return company.lower() in (text or "").lower()


def dedupe_cross_company_rag_hits(
    rag_evidence: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Assign shared citations/texts to the best-matching company only.

    Company-exclusive chunks (``companies == [this_company]``) are always kept.
    Shared fingerprints prefer the company named in the hit text, then highest score.
    """
    evidence = rag_evidence or {}
    if len(evidence) <= 1:
        return {company: list(hits or []) for company, hits in evidence.items()}

    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for company, hits in evidence.items():
        for hit in hits or []:
            if not isinstance(hit, dict):
                continue
            groups.setdefault(_fingerprint(hit), []).append((company, hit))

    winners: dict[str, set[str]] = {}
    for fingerprint, items in groups.items():
        companies = {company for company, _ in items}
        if len(companies) == 1:
            winners[fingerprint] = companies
            continue

        # Exclusive tagging wins for that company even if fingerprint collides.
        exclusive: set[str] = set()
        for company, hit in items:
            tagged = [str(c) for c in (hit.get("companies") or [])]
            if tagged == [company]:
                exclusive.add(company)
        if exclusive:
            winners[fingerprint] = exclusive
            continue

        text = str(items[0][1].get("text") or "")
        mentioned = {company for company, _ in items if _company_mentioned(company, text)}
        if len(mentioned) == 1:
            winners[fingerprint] = mentioned
            continue
        pool = mentioned or companies
        best_company = max(
            ((company, hit) for company, hit in items if company in pool),
            key=lambda item: _score(item[1]),
        )[0]
        winners[fingerprint] = {best_company}

    result: dict[str, list[dict[str, Any]]] = {company: [] for company in evidence}
    for company, hits in evidence.items():
        seen: set[str] = set()
        for hit in hits or []:
            if not isinstance(hit, dict):
                continue
            fingerprint = _fingerprint(hit)
            if fingerprint in seen:
                continue
            allowed = winners.get(fingerprint) or {company}
            tagged = [str(c) for c in (hit.get("companies") or [])]
            if tagged == [company] or company in allowed:
                result[company].append(hit)
                seen.add(fingerprint)
    return result
