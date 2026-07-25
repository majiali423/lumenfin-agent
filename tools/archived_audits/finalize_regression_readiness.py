#!/usr/bin/env python3
"""ARCHIVED AUDIT SCRIPT.

Historical purpose: write an ad-hoc P0/P1 readiness score.
Replacement: Release_Checklist.md and deterministic release gates.
Last compatible schema: pre-release artifact layout.
Not part of the supported release interface; do not run on production fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "e2e_production_audit" / "raw"


def main() -> int:
    agents = json.loads((RAW / "module3_agents.json").read_text(encoding="utf-8")) if (RAW / "module3_agents.json").exists() else []
    rag = json.loads((RAW / "module2_rag.json").read_text(encoding="utf-8")) if (RAW / "module2_rag.json").exists() else {}
    deep = json.loads((RAW / "p0_deep_validation.json").read_text(encoding="utf-8")) if (RAW / "p0_deep_validation.json").exists() else {}

    ag02 = next((a for a in agents if a.get("id") == "ag02_nvda_pdf_live"), {})
    companies = ag02.get("companies") or []
    issuer_ok = isinstance(companies, list) and len(companies) <= 2 and "NVIDIA" in companies
    cite_avg = 0.0
    ok_agents = [a for a in agents if a.get("ok") and a.get("report_scores")]
    if ok_agents:
        cite_avg = sum(a["report_scores"].get("citation", 0) for a in ok_agents) / len(ok_agents)
    rel = [float(q.get("relevance_0_5") or 0) for q in (rag.get("queries") or [])]
    rag_mean = sum(rel) / len(rel) if rel else 0.0
    fact_rate = float(deep.get("financial_fact_hit_rate") or 0)
    compare_ok = bool((deep.get("compare_intent") or {}).get("amd_allowed"))
    html_ok = bool((deep.get("apple_html_revenue_top") or {}).get("prefers_consolidated") or (deep.get("entity") or {}).get("aapl_html"))

    # Honest 0-10 scorecard (not inflated)
    score = 5.0
    dims = {
        "可信金融数据 grounding": f"{'improved' if fact_rate >= 0.5 else 'weak'}; fact_hit_rate={fact_rate:.0%}, rag_mean={rag_mean:.2f}/5",
        "稳定 entity routing": f"{'pass' if issuer_ok and compare_ok else 'partial'}; ag02_companies={companies}",
        "可解释 citation": f"report citation mean={cite_avg:.1f}/10",
        "真实 analyst workflow": "HITL + fail-closed present; table-native fidelity still limited on PDF path",
    }
    if issuer_ok:
        score += 1.5
    if compare_ok:
        score += 0.5
    if fact_rate >= 0.55:
        score += 1.0
    elif fact_rate >= 0.35:
        score += 0.5
    if rag_mean >= 4.0:
        score += 0.5
    if cite_avg >= 6.5:
        score += 0.5
    if html_ok:
        score += 0.5
    score = min(9.0, round(score, 1))  # cap <10 until native statement gold QA

    gaps = [
        "Gold-labeled FY line-item QA (consolidated vs segment) across issuers, not heuristic hit rates.",
        "Native issuer PDF / iXBRL (not only HTML text + DOM tables) for page-faithful citations.",
        "Hybrid vector arm reliability under live DashScope + Milvus Lite (prior audit saw keyword_only).",
        "Report controller: length caps, peer-noise guards, claim→citation binding for every numeric assertion.",
        "CIK/ticker enrichment and statement hierarchy coverage beyond income-statement totals.",
    ]

    payload = {"score": score, "dimensions": dims, "gaps_to_10": gaps}
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "production_readiness.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
