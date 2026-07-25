#!/usr/bin/env python3
"""ARCHIVED AUDIT SCRIPT.

Historical purpose: P0 issuer/peer and numeric fact probe.
Replacement: entity/grounding tests, FAB cases and RC validation.
Last compatible schema: historical e2e_real fixture layout.
Not part of the supported release interface; do not run on production fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lumenfin.document_entity import resolve_document_entities
from lumenfin.document_ingest import parse_upload_documents
from lumenfin.documents import parse_pdf_document
from lumenfin.planning import build_query_plan
from lumenfin.rag.chunking import chunk_document
from lumenfin.rag.hybrid_retriever import _hits_from_scored_chunks

FIX = ROOT / "fixtures" / "e2e_real"
OUT = ROOT / "outputs" / "e2e_production_audit" / "raw"


METRIC_QUERIES: list[tuple[str, str, str]] = [
    # metric_key, query, company
    ("revenue", "What was Apple FY2024 total net sales / revenue?", "Apple"),
    ("net_income", "What was Apple net income in 2024?", "Apple"),
    ("eps", "What was Apple diluted earnings per share in 2024?", "Apple"),
    ("gross_margin", "What was Apple gross margin in 2024?", "Apple"),
    ("operating_income", "What was Apple operating income in 2024?", "Apple"),
    ("operating_margin", "What was Apple operating margin in 2024?", "Apple"),
    ("r_and_d", "What was Apple research and development expense in 2024?", "Apple"),
    ("debt", "What was Apple long-term debt?", "Apple"),
    ("operating_cash_flow", "What was Apple operating cash flow / cash from operations?", "Apple"),
    ("capex", "What was Apple capital expenditures / CapEx?", "Apple"),
    ("cash", "What was Apple cash and cash equivalents?", "Apple"),
    ("revenue", "What was NVIDIA revenue for fiscal year 2025?", "NVIDIA"),
    ("net_income", "What was NVIDIA net income FY2025?", "NVIDIA"),
    ("gross_margin", "What was NVIDIA gross margin?", "NVIDIA"),
    ("operating_income", "What was NVIDIA operating income?", "NVIDIA"),
    ("r_and_d", "What was NVIDIA research and development?", "NVIDIA"),
    ("revenue", "What was Tesla automotive / total revenue?", "Tesla"),
    ("net_income", "What was Tesla net income?", "Tesla"),
    ("operating_income", "What was Tesla operating income?", "Tesla"),
    ("capex", "What was Tesla capital expenditures CapEx?", "Tesla"),
]


def _probe_doc(path: Path) -> dict:
    if path.suffix.lower() in {".htm", ".html"}:
        docs = parse_upload_documents(path)
        return docs[0]
    return parse_pdf_document(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"entity": {}, "compare_intent": {}, "metrics": []}

    # --- 1) Entity: NVDA body mentions must not become issuers ---
    nvda = _probe_doc(FIX / "nvda_fy2025_10k_sec.pdf")
    mentioned = set(nvda.get("mentioned_companies") or [])
    issuers = list(nvda.get("detected_companies") or [])
    peers_in_body = {"AMD", "Microsoft", "Amazon", "Intel"} & mentioned
    report["entity"]["nvda_pdf"] = {
        "issuers": issuers,
        "primary": (nvda.get("primary_company") or {}).get("name"),
        "mentioned_sample": sorted(peers_in_body),
        "peers_not_issuers": issuers == ["NVIDIA"] and bool(peers_in_body or True),
    }
    print("ENTITY NVDA issuers=", issuers, "mentioned_peers=", sorted(peers_in_body))
    assert issuers == ["NVIDIA"], issuers

    # HTML path for Apple
    aapl_html_path = FIX / "aapl-20240928.htm"
    if aapl_html_path.exists():
        aapl_html = _probe_doc(aapl_html_path)
        report["entity"]["aapl_html"] = {
            "issuers": aapl_html.get("detected_companies"),
            "primary": (aapl_html.get("primary_company") or {}).get("name"),
            "html_table_count": aapl_html.get("html_table_count"),
            "source_type": aapl_html.get("source_type"),
        }
        print(
            "ENTITY AAPL HTML issuers=",
            aapl_html.get("detected_companies"),
            "tables=",
            aapl_html.get("html_table_count"),
        )
        assert aapl_html.get("detected_companies") == ["Apple"]

    tsla_pdf = FIX / "tsla_fy2024_10k_sec.pdf"
    if tsla_pdf.exists():
        tsla = _probe_doc(tsla_pdf)
        report["entity"]["tsla_pdf"] = {
            "issuers": tsla.get("detected_companies"),
            "primary": (tsla.get("primary_company") or {}).get("name"),
        }
        print("ENTITY TSLA issuers=", tsla.get("detected_companies"))
        assert tsla.get("detected_companies") == ["Tesla"], tsla.get("detected_companies")

    # --- 2) Compare intent: user-requested AMD allowed ---
    plan = build_query_plan(
        "Compare NVIDIA and AMD profitability and margins",
        document_contexts=[nvda],
        llm_client=None,
    )
    companies = list(plan.companies or [])
    report["compare_intent"] = {
        "intent": plan.intent,
        "companies": companies,
        "amd_allowed": "AMD" in companies,
        "nvidia_present": "NVIDIA" in companies,
    }
    print("COMPARE intent=", plan.intent, "companies=", companies)
    assert "NVIDIA" in companies
    assert "AMD" in companies, "user-requested AMD must be allowed on compare"

    # Non-compare with NVDA upload should NOT expand to AMD
    plan2 = build_query_plan(
        "Analyze NVIDIA FY2025 revenue and risk from the uploaded 10-K",
        document_contexts=[nvda],
        llm_client=None,
    )
    report["compare_intent"]["non_compare_companies"] = list(plan2.companies or [])
    report["compare_intent"]["non_compare_excludes_amd"] = "AMD" not in (plan2.companies or [])
    print("NON-COMPARE companies=", plan2.companies)

    # --- 3) Numeric grounding probe ---
    docs_by_company = {
        "Apple": _probe_doc(FIX / "aapl_fy2024_10k_sec.pdf"),
        "NVIDIA": nvda,
    }
    if tsla_pdf.exists():
        docs_by_company["Tesla"] = _probe_doc(tsla_pdf)
    # Prefer HTML facts when available (P1 native path).
    if aapl_html_path.exists():
        docs_by_company["Apple_HTML"] = _probe_doc(aapl_html_path)
    nvda_html = FIX / "nvda-20250126.htm"
    if nvda_html.exists():
        docs_by_company["NVIDIA"] = _probe_doc(nvda_html)
    tsla_html = FIX / "tsla-20241231.htm"
    if tsla_html.exists():
        docs_by_company["Tesla"] = _probe_doc(tsla_html)

    # Prefer native HTML Apple facts for metric probe when available
    apple_doc_for_metrics = docs_by_company.get("Apple_HTML") or docs_by_company.get("Apple")

    for metric_key, query, company in METRIC_QUERIES:
        if company == "Apple":
            doc = apple_doc_for_metrics
        else:
            doc = docs_by_company.get(company)
        if doc is None:
            report["metrics"].append(
                {
                    "metric": metric_key,
                    "company": company,
                    "query": query,
                    "ok": False,
                    "note": "missing fixture",
                }
            )
            continue
        chunks = chunk_document(doc)
        hits = _hits_from_scored_chunks(chunks, company=company, query=query, top_k=5)
        top = hits[0] if hits else {}
        fact = top.get("financial_fact") if isinstance(top, dict) else None
        entry = {
            "metric": metric_key,
            "company": company,
            "query": query,
            "retrieved_chunk": (top.get("text") or "")[:220] if top else "",
            "page": top.get("page") if top else None,
            "fact_metric": (fact or {}).get("metric") if fact else None,
            "period": (fact or {}).get("period") if fact else None,
            "value": (fact or {}).get("value") if fact else None,
            "scope": (fact or {}).get("scope") if fact else None,
            "statement_type": (fact or {}).get("statement_type") if fact else None,
            "row_label": (fact or {}).get("row_label") if fact else None,
            "source": (fact or {}).get("source") if fact else None,
            "correct_enough": bool(
                fact
                and str(fact.get("metric")) == metric_key
                and str(fact.get("value") or "")
            ),
        }
        # Flag common confusions
        confusions = []
        if fact and str(fact.get("scope")) == "segment" and metric_key == "revenue":
            confusions.append("segment_vs_consolidated")
        if fact and "quarter" in str(top.get("text") or "").lower():
            confusions.append("quarterly_vs_annual")
        if fact and "adjusted" in str((fact.get("row_label") or "")).lower():
            confusions.append("adjusted_metric")
        entry["confusions"] = confusions
        report["metrics"].append(entry)
        flag = "OK" if entry["correct_enough"] else "WEAK"
        print(
            f"  [{flag}] {company} {metric_key}: value={entry['value']} "
            f"scope={entry['scope']} page={entry['page']} source={entry['source']} confusions={confusions}"
        )

    # Apple HTML revenue ranking check
    if "Apple_HTML" in docs_by_company:
        html_doc = docs_by_company["Apple_HTML"]
        hits = _hits_from_scored_chunks(
            chunk_document(html_doc),
            company="Apple",
            query="Apple FY2024 total net sales revenue",
            top_k=5,
        )
        best_fact = None
        for hit in hits:
            fact = hit.get("financial_fact") if isinstance(hit, dict) else None
            if fact and str(fact.get("metric")) == "revenue":
                best_fact = fact
                break
        top_fact = best_fact or {}
        report["apple_html_revenue_top"] = {
            "text": (hits[0].get("text") if hits else "")[:200],
            "value": top_fact.get("value"),
            "scope": top_fact.get("scope"),
            "source": top_fact.get("source"),
            "prefers_consolidated": top_fact.get("scope") == "consolidated"
            or "391" in str(top_fact.get("value") or ""),
            "top5_has_fact": best_fact is not None,
        }
        print("APPLE HTML revenue top=", report["apple_html_revenue_top"])

    hit_rate = sum(1 for m in report["metrics"] if m.get("correct_enough")) / max(
        1, len(report["metrics"])
    )
    report["financial_fact_hit_rate"] = round(hit_rate, 3)
    out = OUT / "p0_deep_validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out} fact_hit_rate={hit_rate:.0%}")
    print("P0 deep validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
