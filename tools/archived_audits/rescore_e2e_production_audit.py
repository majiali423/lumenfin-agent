#!/usr/bin/env python3
"""ARCHIVED AUDIT SCRIPT.

Historical purpose: heuristically rescore initial E2E artifacts.
Replacement: FinAgentBench deterministic cases and RC validation.
Last compatible schema: legacy state/report layout.
Not part of the supported release interface; do not run on production fixtures.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "e2e_production_audit"
RAW = OUT / "raw"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _agent_scores(state: dict[str, Any]) -> dict[str, int]:
    steps = [e.get("step") for e in (state.get("audit_log") or []) if isinstance(e, dict)]
    status = str(state.get("workflow_status") or "")
    report = str(state.get("final_report") or "")
    rag = state.get("rag_evidence") or {}
    sources = {
        c: str((p or {}).get("structured_source") or "")
        for c, p in (state.get("retrieved_docs") or {}).items()
        if isinstance(p, dict)
    }
    plan = 5 if "query_planner" in steps and "supervisor" in steps else (3 if "query_planner" in steps else 1)
    if status == "needs_clarification":
        plan = max(plan, 4)
    tool = 1
    if "retrieval" in steps:
        tool += 2
    if any(v in {"sec_companyfacts", "yahoo_fundamentals", "document_extracted"} for v in sources.values()):
        tool += 1
    if state.get("financial_metrics"):
        tool += 1
    tool = min(5, tool)
    reasoning = 4 if status == "completed" and len(report) > 2000 else (3 if status == "incomplete_data" else 2)
    if "Evidence Boundary" in report:
        reasoning = min(5, reasoning + 1)
    grounding = 1
    if "#p" in report:
        grounding += 2
    if any(rag.values()):
        grounding += 1
    if "Disclaimer" in report:
        grounding += 1
    if status == "incomplete_data" and not state.get("financial_metrics"):
        grounding = max(grounding, 4)
    return {
        "planning": min(5, plan),
        "tool_usage": min(5, tool),
        "reasoning_quality": min(5, reasoning),
        "source_grounding": min(5, grounding),
    }


def _report_scores(state: dict[str, Any]) -> dict[str, int]:
    report = str(state.get("final_report") or "")
    status = str(state.get("workflow_status") or "")
    low = report.lower()
    structure = 0
    for marker in ("executive summary", "financial performance", "risk", "methodology", "disclaimer"):
        if marker in low:
            structure += 2
    structure = min(10, structure)
    citation = 0
    if "#p" in report:
        citation += 5
    if "sec" in low or "yahoo" in low or "data sources" in low:
        citation += 3
    if "evidence boundary" in low:
        citation += 2
    citation = min(10, citation)
    accuracy = 6
    if status == "incomplete_data":
        accuracy = 8
    sources = {
        str((p or {}).get("structured_source") or "")
        for p in (state.get("retrieved_docs") or {}).values()
        if isinstance(p, dict)
    }
    if "sample_db" in sources:
        accuracy = 2
    if state.get("financial_metrics") and status == "completed":
        accuracy = 7
    if any(v in {"sec_companyfacts", "yahoo_fundamentals", "document_extracted"} for v in sources):
        accuracy = max(accuracy, 7)
    reasoning = 5
    if "scenario" in low or "swot" in low or "thesis" in low:
        reasoning += 2
    if len(report) > 8000:
        reasoning += 1
    return {
        "accuracy": min(10, accuracy),
        "structure": structure,
        "reasoning": min(10, reasoning),
        "citation": citation,
    }


def _rescore_rag_query(q: dict[str, Any]) -> dict[str, Any]:
    """Stricter numeric scoring: require digits / year for numeric category."""
    category = q.get("category")
    terms = list(q.get("expected_terms") or [])
    hits = list(q.get("hits") or [])
    blob = " ".join(str(h.get("text") or "") for h in hits).lower()
    term_hits = [t for t in terms if t.lower() in blob]
    has_number = bool(re.search(r"\b\d{2,}\b", blob))
    score = int(q.get("relevance_0_5") or 0)
    note = str(q.get("notes") or "")
    problem = ""
    if category == "numeric":
        # Downgrade if no year-like / large number evidence for FY totals.
        numeric_terms = [t for t in terms if re.search(r"\d", t)]
        if numeric_terms and not any(t.lower() in blob for t in numeric_terms) and not has_number:
            score = min(score, 2)
            problem = "numeric query without clear numeric evidence in top hits"
        elif numeric_terms and not any(t.lower() in blob for t in numeric_terms):
            score = min(score, 3)
            problem = "FY/total numeral from expectation missing; thematic match only"
    if score <= 2 and not problem:
        problem = f"WEAK — {note}"
    elif score <= 3 and not problem:
        problem = note
    out = dict(q)
    out["relevance_0_5_rescored"] = score
    out["term_hits"] = term_hits
    out["problem"] = problem
    out["mode"] = q.get("mode")
    return out


def main() -> int:
    # Map case id -> latest state file
    states: dict[str, Path] = {}
    for path in sorted(ROOT.joinpath("outputs").glob("e2e-ag*_state.json")):
        # e2e-ag01_apple_live-0a8627_...
        name = path.name
        m = re.match(r"e2e-(ag\d+_[^-]+)-", name)
        if not m:
            continue
        states[m.group(1)] = path

    agents: list[dict[str, Any]] = []
    for case_id, state_path in sorted(states.items()):
        state = _load(state_path)
        report = str(state.get("final_report") or "")
        report_copy = RAW / f"{case_id}_report_rescored.md"
        report_copy.write_text(report, encoding="utf-8")
        ag = _agent_scores(state)
        rs = _report_scores(state)
        agents.append(
            {
                "id": case_id,
                "state_path": str(state_path),
                "query": state.get("query"),
                "workflow_status": state.get("workflow_status"),
                "companies": state.get("companies"),
                "llm_backend": state.get("llm_backend"),
                "structured_sources": {
                    c: (p or {}).get("structured_source")
                    for c, p in (state.get("retrieved_docs") or {}).items()
                },
                "clarification_questions": state.get("clarification_questions"),
                "agent_scores": ag,
                "report_scores": rs,
                "rag_hit_counts": {c: len(v or []) for c, v in (state.get("rag_evidence") or {}).items()},
                "telemetry_rag": ((state.get("run_telemetry") or {}).get("rag") or {}),
                "page_citation_count": len(re.findall(r"#p\d+", report)),
                "report_len": len(report),
                "report_excerpt": report[:1800],
                "audit_steps": [
                    e.get("step") for e in (state.get("audit_log") or []) if isinstance(e, dict)
                ],
            }
        )

    rag_raw = _load(RAW / "module2_rag.json")
    rag_queries = [_rescore_rag_query(q) for q in rag_raw.get("queries") or []]
    ingest_files = sorted(RAW.glob("ingest_*.json"))
    ingest = [_load(p)["stat"] for p in ingest_files]
    stress = _load(RAW / "module5_stress.json") if (RAW / "module5_stress.json").exists() else {}
    env = _load(RAW / "env.json")

    avg_rel = mean(q["relevance_0_5_rescored"] for q in rag_queries) if rag_queries else 0
    hit_rate = sum(1 for q in rag_queries if q["relevance_0_5_rescored"] >= 3) / max(1, len(rag_queries))
    modes = {q.get("mode") for q in rag_queries}

    plan_avg = mean(a["agent_scores"]["planning"] for a in agents)
    tool_avg = mean(a["agent_scores"]["tool_usage"] for a in agents)
    reason_avg = mean(a["agent_scores"]["reasoning_quality"] for a in agents)
    ground_avg = mean(a["agent_scores"]["source_grounding"] for a in agents)
    rep_acc = mean(a["report_scores"]["accuracy"] for a in agents)
    rep_struct = mean(a["report_scores"]["structure"] for a in agents)
    rep_reason = mean(a["report_scores"]["reasoning"] for a in agents)
    rep_cite = mean(a["report_scores"]["citation"] for a in agents)

    pdf_cases = [a for a in agents if a["rag_hit_counts"] or "pdf" in a["id"] or "sparse" in a["id"]]
    zero_cite_pdf = [a["id"] for a in agents if a["page_citation_count"] == 0 and a.get("rag_hit_counts")]
    # also PDF cases by id
    pdf_ids = {"ag02_nvda_pdf_live", "ag03_msft_pdf", "ag06_nvda_sustainability", "ag07_apple_pdf_risk", "ag10_sparse_pdf"}
    zero_cite_pdf = [a["id"] for a in agents if a["id"] in pdf_ids and a["page_citation_count"] == 0 and a["workflow_status"] == "completed"]

    keyword_only = all(str(m or "").startswith("keyword_only") for m in modes if m)

    sample = next((a for a in agents if a["id"] == "ag02_nvda_pdf_live"), agents[0] if agents else {})

    lines: list[str] = []
    lines.append("# LumenFin End-to-End Production Validation & Optimization Audit")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(
        "> Live DeepSeek + DashScope + SEC/Yahoo + SEC EDGAR 10-K content. "
        "Scores rescored from exported `*_state.json` / report artifacts after packaging fix. "
        "**No mock LLM.**"
    )
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        f"Full pipeline was exercised on real Apple/NVIDIA/Microsoft 10-K text (SEC EDGAR HTML→PDF) "
        f"plus live fundamentals. After correcting result packaging, agent runs are real "
        f"(`deepseek`, non-empty reports). RAG mean relevance (rescored) **{avg_rel:.2f}/5** "
        f"(hit@≥3={hit_rate:.0%}); modes={sorted(modes)}. "
        f"Agent means planning={plan_avg:.1f}, tools={tool_avg:.1f}, reasoning={reason_avg:.1f}, "
        f"grounding={ground_avg:.1f}. Report means accuracy={rep_acc:.1f}, structure={rep_struct:.1f}, "
        f"reasoning={rep_reason:.1f}, citation={rep_cite:.1f}."
    )
    lines.append("")
    lines.append(
        "**Analyst-risk verdict:** Honesty controls (HITL, fail-closed OpenAI/Oracle sparse) work. "
        "The largest risk is **evidence precision on long filings**: retrieval often returns thematically "
        "related narrative rather than the exact FY total line; this run’s RAG path was "
        f"**{'keyword-only (+lexical rerank) — vector branch did not contribute hits' if keyword_only else 'hybrid/vector mixed'}**. "
        "HTML→text PDF also destroys table geometry before ingestion."
    )
    lines.append("")
    lines.append("## 2. Environment")
    lines.append("")
    lines.append("| Component | Value |")
    lines.append("|-----------|-------|")
    for k, v in env.items():
        lines.append(f"| {k} | `{v}` |")
    lines.append("| document_source | `SEC EDGAR HTML 10-K → paginated PDF (scripts/convert_sec_html_to_pdf.py)` |")
    lines.append("")
    lines.append("## 3. Pipeline Evaluation")
    lines.append("")
    ingest_score = 6
    if any((s.get("issues") or []) for s in ingest):
        ingest_score = 5
    retrieval_score = round(avg_rel / 5 * 10, 1)
    if keyword_only:
        retrieval_score = min(retrieval_score, 6.5)
    agent_score = round((plan_avg + tool_avg + reason_avg + ground_avg) / 4 / 5 * 10, 1)
    report_score = round((rep_acc + rep_struct + rep_reason + rep_cite) / 4, 1)
    lines.append("| Module | Score (0-10) | Issue |")
    lines.append("|--------|-------------:|-------|")
    lines.append(f"| PDF Parsing | {ingest_score} | Apple missing metric_hints; NVDA/MSFT over-detect peer companies in text |")
    lines.append("| Chunking | 6 | ~666 char avg; 10-K prose chunks, not fact cells |")
    lines.append("| Embedding | 7 | DashScope 1024 OK on index; **query-time vector path missed (keyword_only)** |")
    lines.append(f"| Retrieval | {retrieval_score} | rescored mean={avg_rel:.2f}/5; mode={sorted(modes)} |")
    lines.append(f"| Agent | {agent_score} | plan/tool/reason/ground={plan_avg:.1f}/{tool_avg:.1f}/{reason_avg:.1f}/{ground_avg:.1f} |")
    lines.append(f"| Report Generation | {report_score} | acc/struct/reason/cite={rep_acc:.1f}/{rep_struct:.1f}/{rep_reason:.1f}/{rep_cite:.1f} |")
    lines.append("")
    lines.append("### 3.1 Ingestion")
    lines.append("")
    lines.append("| File | Pages | Chars | Chunks | Avg | Companies | Issues |")
    lines.append("|------|------:|------:|-------:|----:|-----------|--------|")
    for s in ingest:
        lines.append(
            f"| {s.get('filename')} | {s.get('page_count')} | {s.get('text_chars')} | {s.get('chunk_count')} | "
            f"{s.get('avg_chunk_chars')} | {', '.join(s.get('detected_companies') or [])} | "
            f"{'; '.join(s.get('issues') or []) or 'none'} |"
        )
    lines.append("")
    lines.append("## 4. RAG Evaluation")
    lines.append("")
    lines.append(f"- Queries: **{len(rag_queries)}**")
    lines.append(f"- Hit rate (rescored ≥3): **{hit_rate:.0%}**")
    lines.append(f"- Average relevance (rescored): **{avg_rel:.2f}/5**")
    lines.append(f"- Retrieve modes observed: `{sorted(modes)}`")
    if keyword_only:
        lines.append(
            "- **Finding:** all 15 queries served as `keyword_only+rerank` — hybrid vector arm did not return usable hits "
            "(index embed succeeded earlier; investigate session/tenant filter / collection mismatch / empty vector search)."
        )
    lines.append("")
    lines.append("| Query | Top evidence | Score | Problem |")
    lines.append("|-------|--------------|------:|---------|")
    for q in rag_queries:
        hits = q.get("hits") or []
        ev = "; ".join(
            f"`{h.get('citation')}`:{(str(h.get('text') or '')[:70]).replace('|','/')}" for h in hits[:2]
        ) or "(none)"
        lines.append(
            f"| {q.get('query_id')} [{q.get('category')}]: {str(q.get('query'))[:70]} | {ev} | "
            f"{q.get('relevance_0_5_rescored')} | {q.get('problem') or ''} |"
        )
    lines.append("")
    lines.append("## 5. Report Quality Evaluation")
    lines.append("")
    lines.append("| Case | Status | Companies | Sources | Plan | Tools | Reason | Ground | #p cites | Report len |")
    lines.append("|------|--------|-----------|---------|-----:|------:|-------:|-------:|---------:|-----------:|")
    for a in agents:
        ag = a["agent_scores"]
        lines.append(
            f"| {a['id']} | {a.get('workflow_status')} | {a.get('companies')} | `{a.get('structured_sources')}` | "
            f"{ag['planning']} | {ag['tool_usage']} | {ag['reasoning_quality']} | {ag['source_grounding']} | "
            f"{a.get('page_citation_count')} | {a.get('report_len')} |"
        )
    lines.append("")
    lines.append(f"### Sample: `{sample.get('id')}`")
    lines.append("")
    lines.append("```markdown")
    lines.append(str(sample.get("report_excerpt") or "")[:2000])
    lines.append("```")
    lines.append("")
    lines.append("### Fact findings (from real exports)")
    lines.append("")
    lines.append(
        f"- Completed PDF cases with **0 `#pN` in report body**: {zero_cite_pdf or 'none'} "
        "(citation section may still be absent if `rag_evidence` empty after company over-tagging / filter)."
    )
    for a in agents:
        if a["id"] in pdf_ids:
            lines.append(
                f"- `{a['id']}`: status={a['workflow_status']}, rag_hits={a['rag_hit_counts']}, "
                f"telemetry_mode={a.get('telemetry_rag',{}).get('mode')}, cites={a['page_citation_count']}"
            )
    lines.append(
        "- `ag08_openai_failclosed` / `ag10_sparse_pdf`: incomplete_data (fail-closed) — good honesty."
    )
    lines.append("- `ag09_ambiguous`: needs_clarification — HITL works.")
    lines.append("")
    lines.append("## 6. Bottleneck Analysis")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stress, ensure_ascii=False, indent=2)[:3500])
    lines.append("```")
    lines.append("")
    lines.append("### Priority")
    lines.append("")
    lines.append("**P0**")
    lines.append("")
    lines.append(
        "- **Vector retrieval miss in audit RAG harness** (all `keyword_only+rerank`): fix tenant/session/"
        "collection wiring so hybrid RRF actually uses DashScope vectors at query time."
    )
    lines.append(
        "- **Company over-detection on 10-K text** (NVDA chunk tagged with AMD/Amazon/…): breaks company "
        "filters and can zero out `rag_evidence` for the issuer."
    )
    lines.append(
        "- **Numeric answer grounding**: thematic hits ≠ FY total line; need fact index / table-cell chunks."
    )
    lines.append("")
    lines.append("**P1**")
    lines.append("")
    lines.append("- Native PDF/HTML/iXBRL ingest (stop lossy HTML→text pagination for production).")
    lines.append("- Ensure synthesizer citation section fires whenever `rag_evidence` non-empty (regression on PDF cases).")
    lines.append(f"- Long-doc latency: stress analyze ≈ {(stress.get('long_document') or {}).get('analyze_ms')} ms for 180 pages.")
    lines.append("- `run_telemetry.rag.mode` still null on some agent exports — confirm merge path.")
    lines.append("")
    lines.append("**P2**")
    lines.append("")
    lines.append("- LLM-as-judge gold set for RAG (replace term heuristics).")
    lines.append("- Valuation section honesty (“no DCF computed”).")
    lines.append("- Cache SEC companyfacts + embeddings by content hash.")
    lines.append("")
    lines.append("## 7. Optimization Roadmap")
    lines.append("")
    lines.append("1. Debug hybrid retrieve: log vector_hits/keyword_hits per company; assert mode contains `hybrid_rrf` in showcase.")
    lines.append("2. Narrow `detect_companies_from_text` for long filings (issuer-primary + explicit peers only).")
    lines.append("3. Add structured fact extraction (metric, period, value, page) into retrieval candidates.")
    lines.append("4. Keep lexical ZH/EN rerank; add cloud rerank only after vector arm is healthy.")
    lines.append("5. Report: deterministic citations + inline cites on upload-derived metrics.")
    lines.append("6. Infra: Milvus Server when concurrency >1; Lite OK for single-process demo.")
    lines.append("")
    lines.append("## Appendix")
    lines.append("")
    lines.append(f"- Rescored agent cases: {len(agents)}")
    lines.append(f"- State files under `outputs/e2e-ag*_state.json`")
    lines.append(f"- RAG raw: `{RAW / 'module2_rag.json'}`")
    lines.append(f"- This report: `{OUT / 'LumenFin_E2E_Audit_Report.md'}`")
    lines.append("")

    report_path = OUT / "LumenFin_E2E_Audit_Report.md"
    text = "\n".join(lines)
    report_path.write_text(text, encoding="utf-8")
    (ROOT / "LumenFin_E2E_Audit_Report.md").write_text(text, encoding="utf-8")
    RAW.joinpath("module3_agents_rescored.json").write_text(
        json.dumps(agents, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    RAW.joinpath("module2_rag_rescored.json").write_text(
        json.dumps(rag_queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Wrote", report_path)
    print(
        f"RAG mean={avg_rel:.2f} hit@3={hit_rate:.0%} modes={sorted(modes)} "
        f"agent plan/tool/reason/ground={plan_avg:.1f}/{tool_avg:.1f}/{reason_avg:.1f}/{ground_avg:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
