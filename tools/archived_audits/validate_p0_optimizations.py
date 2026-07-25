"""ARCHIVED AUDIT SCRIPT.

Historical purpose: quick Apple/NVIDIA entity and fact-ranking check.
Replacement: document-primary-entity and SEC HTML fact tests.
Last compatible schema: historical e2e_real fixture layout.
Not part of the supported release interface; do not run on production fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumenfin.documents import parse_pdf_document
from lumenfin.rag.chunking import chunk_document
from lumenfin.rag.hybrid_retriever import _hits_from_scored_chunks


def main() -> int:
    nvda = parse_pdf_document(ROOT / "fixtures" / "e2e_real" / "nvda_fy2025_10k_sec.pdf")
    aapl = parse_pdf_document(ROOT / "fixtures" / "e2e_real" / "aapl_fy2024_10k_sec.pdf")
    print("P0-1 NVDA detected=", nvda["detected_companies"], "primary=", nvda.get("primary_company"))
    print("P0-1 AAPL detected=", aapl["detected_companies"], "primary=", aapl.get("primary_company"))
    assert nvda["detected_companies"] == ["NVIDIA"], nvda["detected_companies"]
    assert aapl["detected_companies"] == ["Apple"], aapl["detected_companies"]

    chunks = chunk_document(aapl)
    facts = [
        c
        for c in chunks
        if c.get("financial_fact") and c["financial_fact"].get("metric") == "revenue"
    ]
    print("P0-2 apple revenue facts=", len(facts))
    if facts:
        print(" sample=", facts[0]["text"])
    hits = _hits_from_scored_chunks(
        chunks,
        company="Apple",
        query="What was Apple FY2024 revenue / net sales?",
        top_k=5,
    )
    print("P0-2 top hit=", (hits[0]["text"][:140] if hits else None))
    print(" fact=", (hits[0].get("financial_fact") if hits else None))
    assert hits, "expected retrieval hits"
    fact = hits[0].get("financial_fact") or {}
    assert fact or "391" in hits[0]["text"] or "net sales" in hits[0]["text"].lower()
    if fact:
        # Prefer consolidated / total when both segment and total exist in fixture text.
        print(" scope=", fact.get("scope"), "value=", fact.get("value"))
    print("P0 validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
