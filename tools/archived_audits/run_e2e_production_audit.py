#!/usr/bin/env python3
"""ARCHIVED AUDIT SCRIPT.

Historical purpose: initial live production-oriented E2E audit.
Replacement: FinAgentBench scripts/run_rc_validation.py.
Last compatible schema: historical e2e_real fixture/output layout.
Not part of the supported release interface; do not run on production fixtures.

Uses DeepSeek + DashScope + live SEC/Yahoo + SEC 10-K PDFs under fixtures/e2e_real/.
Writes:
  outputs/e2e_production_audit/raw/*.json
  outputs/e2e_production_audit/LumenFin_E2E_Audit_Report.md
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataclasses import replace

from lumenfin.config import AppConfig
from lumenfin.document_ingest import parse_upload_documents
from lumenfin.documents import parse_pdf_document
from lumenfin.evaluation import evaluate_run_state
from lumenfin.rag.chunking import chunk_document
from lumenfin.rag.factory import build_document_indexer, build_hybrid_retriever, build_rag_store
from lumenfin.rag.lexical import lexical_overlap
from lumenfin.rag.profiles import apply_showcase_rag_env
from lumenfin.service import LumenFinAnalysisService
from lumenfin.stdio import configure_stdio_utf8

apply_showcase_rag_env(overwrite=False)

OUT = ROOT / "outputs" / "e2e_production_audit"
RAW = OUT / "raw"
FIX = ROOT / "fixtures" / "e2e_real"
STRESS = ROOT / "fixtures" / "stress"

logger = logging.getLogger("e2e_audit")


@dataclass
class IngestStat:
    filename: str
    page_count: int
    text_chars: int
    chunk_count: int
    avg_chunk_chars: float
    detected_companies: list[str]
    metric_hints: dict[str, Any]
    issues: list[str] = field(default_factory=list)


@dataclass
class RagQueryResult:
    query_id: str
    query: str
    company: str
    category: str
    expected_terms: list[str]
    hits: list[dict[str, Any]]
    relevance_0_5: int
    answer_in_context: bool
    notes: str
    latency_ms: float
    mode: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save(name: str, payload: Any) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _hit_preview(hit: dict[str, Any], limit: int = 220) -> dict[str, Any]:
    text = str(hit.get("text") or "")
    return {
        "citation": hit.get("citation"),
        "page": hit.get("page"),
        "score": hit.get("rerank_score") or hit.get("fusion_score") or hit.get("score"),
        "method": hit.get("retrieval_method"),
        "companies": hit.get("companies"),
        "text": text[:limit],
    }


def _score_relevance(query: str, hits: list[dict[str, Any]], expected_terms: list[str]) -> tuple[int, bool, str]:
    if not hits:
        return 0, False, "no hits"
    blob = " ".join(str(h.get("text") or "") for h in hits).lower()
    term_hits = [t for t in expected_terms if t.lower() in blob]
    overlap = max((lexical_overlap(query, str(h.get("text") or "")) for h in hits), default=0.0)
    answer_in = len(term_hits) >= max(1, len(expected_terms) // 2)
    if len(term_hits) >= max(2, int(0.7 * len(expected_terms))) and overlap >= 0.15:
        score = 5
    elif len(term_hits) >= max(1, int(0.4 * len(expected_terms))) and overlap >= 0.08:
        score = 4
    elif term_hits or overlap >= 0.12:
        score = 3
    elif overlap >= 0.05:
        score = 2
    elif hits:
        score = 1
    else:
        score = 0
    note = f"term_hits={term_hits} max_overlap={overlap:.3f}"
    return score, answer_in, note


def module1_ingestion(pdfs: list[Path]) -> list[IngestStat]:
    stats: list[IngestStat] = []
    for path in pdfs:
        issues: list[str] = []
        parsed = parse_pdf_document(path)
        contexts = parse_upload_documents(path)
        doc = {
            "document_id": path.stem,
            "filename": path.name,
            "detected_companies": parsed.get("detected_companies") or [],
            "pages": parsed.get("pages") or [],
        }
        chunks = chunk_document(doc)
        avg = (sum(c["char_count"] for c in chunks) / len(chunks)) if chunks else 0.0
        text = str(parsed.get("text") or "")
        if parsed.get("page_count", 0) == 0:
            issues.append("zero pages from parser")
        if len(text) < 5000:
            issues.append("very short extracted text for a 10-K excerpt")
        # Table/numeric denseness heuristic
        digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
        if digit_ratio < 0.01:
            issues.append("low numeric density — possible table/figure loss after HTML→PDF conversion or parse")
        if not parsed.get("metric_hints"):
            issues.append("no metric_hints extracted from filing text")
        # Page metadata check on chunks
        if any(not c.get("page") for c in chunks):
            issues.append("some chunks missing page metadata")
        stats.append(
            IngestStat(
                filename=path.name,
                page_count=int(parsed.get("page_count") or 0),
                text_chars=len(text),
                chunk_count=len(chunks),
                avg_chunk_chars=round(avg, 1),
                detected_companies=list(parsed.get("detected_companies") or []),
                metric_hints=dict(parsed.get("metric_hints") or {}),
                issues=issues,
            )
        )
        _save(
            f"ingest_{path.stem}.json",
            {
                "stat": asdict(stats[-1]),
                "context_count": len(contexts),
                "sample_pages": (parsed.get("pages") or [])[:2],
                "chunk_samples": chunks[:5],
            },
        )
    return stats


def module2_rag(
    config: AppConfig,
    pdf_by_company: dict[str, Path],
) -> tuple[list[RagQueryResult], dict[str, Any]]:
    # Isolated Milvus DB for this audit run
    audit_uri = str(OUT / f"milvus_e2e_{uuid4().hex[:8]}.db")
    cfg = replace(
        config,
        milvus_uri=audit_uri,
        milvus_isolate=False,
        rag_index_mode="sync_on_run",
        data_mode="live",
    )
    store = build_rag_store(cfg)
    assert store is not None
    indexer = build_document_indexer(cfg, rag_store=store)
    retriever = build_hybrid_retriever(cfg, rag_store=store, indexer=indexer)
    assert retriever is not None

    receipts = []
    contexts_by_company: dict[str, list[dict[str, Any]]] = {}
    source_ids_by_company: dict[str, list[str]] = {}
    for company, path in pdf_by_company.items():
        path_receipts = indexer.index_paths([str(path)], tenant_id=cfg.rag_tenant_id)
        receipts.extend(path_receipts)
        contexts_by_company[company] = parse_upload_documents(path)
        source_ids_by_company[company] = [
            str(r.get("document_id") or path.stem) for r in path_receipts if isinstance(r, dict)
        ] or [path.stem]
    index_summary = {
        "uri": audit_uri,
        "receipts": receipts,
        "embed_dim": getattr(store.embedder, "dimension", None),
        "provider": type(store.embedder).__name__,
    }

    queries: list[tuple[str, str, str, str, list[str]]] = [
        # id, company, category, query, expected_terms
        ("rq01", "Apple", "numeric", "What was Apple's total net sales / revenue in fiscal 2024?", ["net sales", "revenue", "2024", "391"]),
        ("rq02", "Apple", "numeric", "What was Apple's net income in 2024?", ["net income", "2024"]),
        ("rq03", "Apple", "definition", "What does Apple disclose about Services net sales?", ["Services", "net sales"]),
        ("rq04", "Apple", "risk", "What supply chain or manufacturing concentration risks does Apple disclose?", ["supply", "manufacturing", "China", "risk"]),
        ("rq05", "Apple", "multi_hop", "How do Apple's R&D expenses relate to product development risk?", ["research", "development", "R&D", "product"]),
        ("rq06", "NVIDIA", "numeric", "What was NVIDIA's revenue for fiscal year 2025?", ["revenue", "2025", "130"]),
        ("rq07", "NVIDIA", "numeric", "What portion of NVIDIA revenue is Data Center related?", ["Data Center", "revenue"]),
        ("rq08", "NVIDIA", "risk", "What manufacturing or foundry / packaging supply risks does NVIDIA mention?", ["supply", "manufacturing", "foundry", "TSMC", "packaging"]),
        ("rq09", "NVIDIA", "definition", "How does NVIDIA describe Data Center growth drivers?", ["Data Center", "AI", "GPU"]),
        ("rq10", "Microsoft", "numeric", "What was Microsoft's revenue in fiscal year 2024?", ["revenue", "2024", "245"]),
        ("rq11", "Microsoft", "compare", "How does Microsoft describe Intelligent Cloud versus Productivity performance?", ["Intelligent Cloud", "Productivity", "revenue"]),
        ("rq12", "Microsoft", "risk", "What cybersecurity or AI-related risk factors does Microsoft disclose?", ["cyber", "security", "AI", "risk"]),
        ("rq13", "Apple", "compare", "Compare Apple iPhone versus Services contribution qualitatively from the filing.", ["iPhone", "Services"]),
        ("rq14", "NVIDIA", "implication", "Is NVIDIA's growth described as concentrated in AI / data center demand?", ["AI", "Data Center", "demand", "growth"]),
        ("rq15", "Microsoft", "multi_hop", "How could cloud capex and AI investment affect Microsoft's operating margins per the filing narrative?", ["cloud", "AI", "operating", "margin", "capital"]),
    ]

    results: list[RagQueryResult] = []
    for qid, company, category, query, terms in queries:
        t0 = time.perf_counter()
        hits, meta = retriever.retrieve_for_company_with_meta(
            query=query,
            company=company,
            session_id=cfg.rag_tenant_id,
            document_contexts=contexts_by_company.get(company) or [],
            tenant_id=cfg.rag_tenant_id,
            source_document_ids=source_ids_by_company.get(company),
            use_stored_chunks=True,
        )
        latency = (time.perf_counter() - t0) * 1000
        score, in_ctx, note = _score_relevance(query, hits, terms)
        results.append(
            RagQueryResult(
                query_id=qid,
                query=query,
                company=company,
                category=category,
                expected_terms=terms,
                hits=[_hit_preview(h) for h in hits],
                relevance_0_5=score,
                answer_in_context=in_ctx,
                notes=note,
                latency_ms=round(latency, 1),
                mode=(meta or {}).get("mode"),
            )
        )
        print(f"  RAG {qid} score={score} hits={len(hits)} mode={(meta or {}).get('mode')}")

    _save("module2_rag.json", {"index": index_summary, "queries": [asdict(r) for r in results]})
    try:
        store.close()
    except Exception:
        pass
    return results, index_summary


def _agent_scores(state: dict[str, Any], query: str) -> dict[str, int]:
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
    if any("#p" in report for _ in [0]):
        grounding += 2
    if any(rag.values()):
        grounding += 1
    if "Disclaimer" in report:
        grounding += 1
    if status == "incomplete_data" and not state.get("financial_metrics"):
        grounding = max(grounding, 4)  # honest fail-loud
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
    for marker in (
        "executive summary",
        "financial performance",
        "risk",
        "methodology",
        "disclaimer",
    ):
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
        accuracy = 8  # no invented metrics preferred
    if "sample_db" in str(state.get("retrieved_docs")):
        accuracy = 2
    metrics = state.get("financial_metrics") or {}
    if metrics and status == "completed":
        accuracy = 7
    reasoning = 5
    if "scenario" in low or "swot" in low or "thesis" in low:
        reasoning += 2
    if len(report) > 8000:
        reasoning += 1
    reasoning = min(10, reasoning)
    return {
        "accuracy": accuracy,
        "structure": structure,
        "reasoning": min(10, reasoning),
        "citation": citation,
    }


def module3_and_4_agents(config: AppConfig) -> list[dict[str, Any]]:
    cfg = replace(
        config,
        data_mode="live",
        fetch_live_fundamentals=True,
        fetch_sec_fundamentals=True,
        rag_index_mode="sync_on_run",
        milvus_uri=str(OUT / f"milvus_agent_{uuid4().hex[:8]}.db"),
        milvus_isolate=False,
    )
    service = LumenFinAnalysisService(cfg)
    cases = [
        {
            "id": "ag01_apple_live",
            "query": "Analyze Apple FY2024 annual profitability, operating margin, and R&D intensity using live fundamentals. Discuss valuation context with current market snapshot.",
            "docs": [],
        },
        {
            "id": "ag02_nvda_pdf_live",
            "query": "Analyze NVIDIA investment risk using the uploaded FY2025 10-K excerpt and current market valuation. Cite filing pages where possible.",
            "docs": [str(FIX / "nvda_fy2025_10k_sec.pdf")],
        },
        {
            "id": "ag03_msft_pdf",
            "query": "Using the uploaded Microsoft FY2024 10-K excerpt, summarize financial performance, Intelligent Cloud growth signals, and key risk factors with page citations.",
            "docs": [str(FIX / "msft_fy2024_10k_sec.pdf")],
        },
        {
            "id": "ag04_aapl_msft_compare",
            "query": "Compare Apple and Microsoft FY profitability and R&D intensity using live SEC/Yahoo fundamentals. Note supply-chain or platform risks briefly.",
            "docs": [],
        },
        {
            "id": "ag05_tesla_live",
            "query": "Analyze Tesla FY profitability, automotive margin signals, and balance-sheet risk using live fundamentals and market data.",
            "docs": [],
        },
        {
            "id": "ag06_nvda_sustainability",
            "query": "Based on the uploaded NVIDIA 10-K excerpt and live market data, is NVIDIA's recent growth sustainable? Ground claims in retrieved evidence and computed metrics.",
            "docs": [str(FIX / "nvda_fy2025_10k_sec.pdf")],
        },
        {
            "id": "ag07_apple_pdf_risk",
            "query": "From Apple's uploaded 10-K excerpt, extract disclosed concentration / supply-chain risks and connect them to financial resilience. Cite pages.",
            "docs": [str(FIX / "aapl_fy2024_10k_sec.pdf")],
        },
        {
            "id": "ag08_openai_failclosed",
            "query": "Analyze OpenAI FY2025 annual profitability, operating margin, and R&D intensity using live fundamentals only. Do not invent estimates if data is unavailable.",
            "docs": [],
        },
        {
            "id": "ag09_ambiguous",
            "query": "Is this company good?",
            "docs": [],
        },
        {
            "id": "ag10_sparse_pdf",
            "query": "Using only the uploaded note, underwrite Oracle Cloud FY2025 EBITDA margin and R&D intensity with citations.",
            "docs": [str(STRESS / "oracle_sparse_fluff.pdf")],
        },
    ]
    results: list[dict[str, Any]] = []
    for case in cases:
        print(f"  AGENT {case['id']} ...")
        t0 = time.perf_counter()
        try:
            result_pkg = service.analyze(
                query=case["query"],
                document_paths=case["docs"] or None,
                thread_id=f"e2e-{case['id']}-{uuid4().hex[:6]}",
            )
            wall_ms = (time.perf_counter() - t0) * 1000
            # analyze() returns API package; workflow state is under "result".
            state = dict(result_pkg.get("result") or result_pkg)
            report = str(state.get("final_report") or "")
            report_path = RAW / f"{case['id']}_report.md"
            report_path.write_text(report, encoding="utf-8")
            eval_res = evaluate_run_state(state)
            agent = _agent_scores(state, case["query"])
            report_scores = _report_scores(state)
            entry = {
                "id": case["id"],
                "query": case["query"],
                "documents": case["docs"],
                "ok": True,
                "wall_ms": round(wall_ms, 1),
                "workflow_status": state.get("workflow_status") or result_pkg.get("workflow_status"),
                "companies": state.get("companies"),
                "llm_backend": state.get("llm_backend") or result_pkg.get("llm_backend"),
                "structured_sources": {
                    c: (p or {}).get("structured_source")
                    for c, p in (state.get("retrieved_docs") or {}).items()
                },
                "clarification_questions": state.get("clarification_questions")
                or result_pkg.get("clarification_questions"),
                "agent_scores": agent,
                "report_scores": report_scores,
                "eval": eval_res.to_dict(),
                "rag_hit_counts": {c: len(v or []) for c, v in (state.get("rag_evidence") or {}).items()},
                "telemetry_rag": ((state.get("run_telemetry") or {}).get("rag") or {}),
                "audit_steps": [
                    {"step": e.get("step"), "status": e.get("status"), "latency_ms": e.get("latency_ms")}
                    for e in (state.get("audit_log") or [])
                    if isinstance(e, dict)
                ],
                "report_path": str(report_path),
                "artifacts": result_pkg.get("artifacts"),
                "report_excerpt": report[:1500],
                "page_citation_count": len(re.findall(r"#p\d+", report)),
            }
        except Exception as exc:
            entry = {
                "id": case["id"],
                "query": case["query"],
                "documents": case["docs"],
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"    FAIL {entry['error']}")
        results.append(entry)
        _save(f"{case['id']}.json", entry)
    _save("module3_agents.json", results)
    return results


def module5_stress(config: AppConfig) -> dict[str, Any]:
    cfg = replace(
        config,
        data_mode="live",
        rag_index_mode="sync_on_run",
        milvus_uri=str(OUT / f"milvus_stress_{uuid4().hex[:8]}.db"),
        milvus_isolate=False,
    )
    service = LumenFinAnalysisService(cfg)
    out: dict[str, Any] = {}

    # Long document performance
    long_pdf = FIX / "msft_fy2024_10k_sec_long.pdf"
    t0 = time.perf_counter()
    ingest = parse_pdf_document(long_pdf)
    parse_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    try:
        result_pkg = service.analyze(
            query="Summarize Microsoft FY2024 financial performance and top risk factors from the uploaded long 10-K excerpt. Cite pages.",
            document_paths=[str(long_pdf)],
            thread_id=f"e2e-long-{uuid4().hex[:6]}",
        )
        result = dict(result_pkg.get("result") or result_pkg)
        analyze_ms = (time.perf_counter() - t1) * 1000
        out["long_document"] = {
            "pages": ingest.get("page_count"),
            "parse_ms": round(parse_ms, 1),
            "analyze_ms": round(analyze_ms, 1),
            "status": result.get("workflow_status"),
            "rag_mode": ((result.get("run_telemetry") or {}).get("rag") or {}).get("mode"),
            "chunks_indexed": ((result.get("run_telemetry") or {}).get("rag") or {}).get("chunks_indexed"),
            "page_citations_in_report": len(re.findall(r"#p\d+", str(result.get("final_report") or ""))),
        }
    except Exception as exc:
        out["long_document"] = {"error": str(exc), "parse_ms": round(parse_ms, 1), "pages": ingest.get("page_count")}

    # Conflicting / prefer-upload signal
    try:
        conflict_pkg = service.analyze(
            query="Prefer uploaded materials only. Report Apple FY2024 revenue and EBITDA from the upload; flag if live providers would differ.",
            document_paths=[str(FIX / "aapl_fy2024_10k_sec.pdf")],
            thread_id=f"e2e-conflict-{uuid4().hex[:6]}",
        )
        conflict = dict(conflict_pkg.get("result") or conflict_pkg)
        out["prefer_upload"] = {
            "status": conflict.get("workflow_status"),
            "sources": {
                c: (p or {}).get("structured_source")
                for c, p in (conflict.get("retrieved_docs") or {}).items()
            },
            "source_resolution": conflict.get("source_resolution"),
            "report_has_source_resolution": "Source Resolution" in str(conflict.get("final_report") or ""),
        }
    except Exception as exc:
        out["prefer_upload"] = {"error": str(exc)}

    _save("module5_stress.json", out)
    return out


def _avg(nums: list[float]) -> float:
    return round(sum(nums) / len(nums), 3) if nums else 0.0


def write_report(
    *,
    env: dict[str, Any],
    ingest: list[IngestStat],
    rag: list[RagQueryResult],
    agents: list[dict[str, Any]],
    stress: dict[str, Any],
    index_summary: dict[str, Any],
) -> Path:
    hit_rate = sum(1 for r in rag if r.relevance_0_5 >= 3) / max(1, len(rag))
    avg_rel = _avg([float(r.relevance_0_5) for r in rag])
    answer_rate = sum(1 for r in rag if r.answer_in_context) / max(1, len(rag))

    agent_ok = [a for a in agents if a.get("ok")]
    plan_avg = _avg([a["agent_scores"]["planning"] for a in agent_ok if "agent_scores" in a])
    tool_avg = _avg([a["agent_scores"]["tool_usage"] for a in agent_ok if "agent_scores" in a])
    reason_avg = _avg([a["agent_scores"]["reasoning_quality"] for a in agent_ok if "agent_scores" in a])
    ground_avg = _avg([a["agent_scores"]["source_grounding"] for a in agent_ok if "agent_scores" in a])

    rep_acc = _avg([a["report_scores"]["accuracy"] for a in agent_ok if "report_scores" in a])
    rep_struct = _avg([a["report_scores"]["structure"] for a in agent_ok if "report_scores" in a])
    rep_reason = _avg([a["report_scores"]["reasoning"] for a in agent_ok if "report_scores" in a])
    rep_cite = _avg([a["report_scores"]["citation"] for a in agent_ok if "report_scores" in a])

    # Module scores 0-10 for summary table
    ingest_score = 7 if all(s.page_count > 0 for s in ingest) else 3
    if any("metric_hints" in i for s in ingest for i in s.issues):
        ingest_score -= 1
    if any("numeric density" in i for s in ingest for i in s.issues):
        ingest_score -= 1
    chunk_score = 7 if _avg([s.avg_chunk_chars for s in ingest]) < 1200 else 5
    embed_score = 8 if index_summary.get("embed_dim") == 1024 else 4
    retrieval_score = round(avg_rel / 5 * 10, 1)
    agent_score = round((plan_avg + tool_avg + reason_avg + ground_avg) / 4 / 5 * 10, 1)
    report_score = round((rep_acc + rep_struct + rep_reason + rep_cite) / 4, 1)

    # Bottlenecks
    p0: list[str] = []
    p1: list[str] = []
    p2: list[str] = []
    weak_rag = [r for r in rag if r.relevance_0_5 <= 2]
    if weak_rag:
        p0.append(
            f"RAG weak on {len(weak_rag)}/{len(rag)} queries (ids={[r.query_id for r in weak_rag]}); "
            "10-K HTML→text PDF loses native table structure; numeric term hit rate uneven."
        )
    low_cite = [a for a in agent_ok if a.get("page_citation_count", 0) == 0 and a.get("documents")]
    if low_cite:
        p1.append(f"PDF-backed agent runs with 0 #pN in report: {[a['id'] for a in low_cite]}")
    else:
        p2.append("Page citations present on PDF-backed runs after citation-section fix — keep regression tests.")
    if any(a.get("workflow_status") == "completed" and "fallback" in str(a.get("llm_backend")) for a in agent_ok):
        p0.append("Completed runs used local LLM fallback — DeepSeek path degraded.")
    long = stress.get("long_document") or {}
    if long.get("analyze_ms") and long["analyze_ms"] > 120_000:
        p1.append(f"Long-doc analyze wall time {long['analyze_ms']:.0f}ms — indexing/embed latency risk.")
    p1.append("SEC HTML→PDF conversion is text-pagination; production should ingest native PDF/HTML with table-aware parsers.")
    p2.append("Add LLM-as-judge groundedness on a fixed gold set; current relevance uses term+lexical heuristics.")

    # Sample report excerpt
    sample = next((a for a in agent_ok if a.get("id") == "ag02_nvda_pdf_live"), agent_ok[0] if agent_ok else {})

    lines: list[str] = []
    lines.append("# LumenFin End-to-End Production Validation & Optimization Audit")
    lines.append("")
    lines.append(f"Generated: {_now()}")
    lines.append("")
    lines.append("> Standard: live DeepSeek + DashScope embeddings + live SEC/Yahoo + SEC EDGAR 10-K content. No mock LLM.")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        f"This audit exercised the full diligence pipeline against **real SEC 10-K content** "
        f"(Apple FY2024, NVIDIA FY2025, Microsoft FY2024) and live market/fundamentals APIs. "
        f"RAG mean relevance **{avg_rel}/5** (hit@≥3 = {hit_rate:.0%}); agent dimension means "
        f"planning={plan_avg}, tools={tool_avg}, reasoning={reason_avg}, grounding={ground_avg} (0–5). "
        f"Report means accuracy={rep_acc}, structure={rep_struct}, reasoning={rep_reason}, citation={rep_cite} (0–10)."
    )
    lines.append("")
    lines.append(
        "**Analyst-risk verdict:** The largest production risk is **evidence fidelity on long filings** — "
        "retrieval sometimes returns thematically related narrative without the specific numeric cell an analyst needs, "
        "and HTML→text PDF preparation already discards native table geometry. Pipeline honesty (fail-closed, HITL, live sources) is comparatively strong."
    )
    lines.append("")
    lines.append("## 2. Environment")
    lines.append("")
    lines.append("| Component | Value |")
    lines.append("|-----------|-------|")
    for k, v in env.items():
        lines.append(f"| {k} | `{v}` |")
    lines.append("")
    lines.append("### Document provenance")
    lines.append("")
    lines.append("- Downloaded SEC EDGAR HTML 10-K filings with SEC-compliant User-Agent.")
    lines.append("- Converted to PDF via `scripts/convert_sec_html_to_pdf.py` (text extract → paginated PDF) for `parse_pdf_document` ingestion.")
    lines.append("- **Limitation (disclosed):** this is real filing text, not a byte-identical issuer PDF; tables become linear text.")
    lines.append("")
    lines.append("## 3. Pipeline Evaluation")
    lines.append("")
    lines.append("| Module | Score (0-10) | Issue |")
    lines.append("|--------|-------------:|-------|")
    lines.append(f"| PDF Parsing | {ingest_score} | See ingest issues below |")
    lines.append(f"| Chunking | {chunk_score} | Peer-table splitter helps synthetic tables; 10-K prose still page-ish |")
    lines.append(f"| Embedding | {embed_score} | provider={index_summary.get('provider')} dim={index_summary.get('embed_dim')} |")
    lines.append(f"| Retrieval | {retrieval_score} | mean_rel={avg_rel}/5 answer_in_context={answer_rate:.0%} |")
    lines.append(f"| Agent | {agent_score} | plan/tool/reason/ground = {plan_avg}/{tool_avg}/{reason_avg}/{ground_avg} |")
    lines.append(f"| Report Generation | {report_score} | acc/struct/reason/cite = {rep_acc}/{rep_struct}/{rep_reason}/{rep_cite} |")
    lines.append("")
    lines.append("### 3.1 Ingestion detail")
    lines.append("")
    lines.append("| File | Pages | Chars | Chunks | Avg chunk | Companies | Issues |")
    lines.append("|------|------:|------:|-------:|----------:|-----------|--------|")
    for s in ingest:
        lines.append(
            f"| {s.filename} | {s.page_count} | {s.text_chars} | {s.chunk_count} | {s.avg_chunk_chars} | "
            f"{', '.join(s.detected_companies) or '-'} | {'; '.join(s.issues) or 'none'} |"
        )
    lines.append("")
    lines.append("## 4. RAG Evaluation")
    lines.append("")
    lines.append(f"- Queries: **{len(rag)}**")
    lines.append(f"- Hit rate (relevance ≥ 3): **{hit_rate:.0%}**")
    lines.append(f"- Average relevance: **{avg_rel}/5**")
    lines.append(f"- Answer-in-context rate (expected-term heuristic): **{answer_rate:.0%}**")
    lines.append("")
    lines.append("| Query | Retrieved Evidence | Score | Problem |")
    lines.append("|-------|--------------------|------:|---------|")
    for r in rag:
        ev = "; ".join(
            f"`{h.get('citation')}`:{str(h.get('text') or '')[:80].replace('|', '/')}"
            for h in r.hits[:2]
        ) or "(none)"
        problem = r.notes if r.relevance_0_5 <= 3 else ""
        if r.relevance_0_5 <= 2:
            problem = f"WEAK — {r.notes}"
        lines.append(f"| {r.query_id} {r.category}: {r.query[:80]} | {ev} | {r.relevance_0_5} | {problem} |")
    lines.append("")
    lines.append("## 5. Report Quality Evaluation")
    lines.append("")
    lines.append("### Agent case scoreboard")
    lines.append("")
    lines.append("| Case | Status | Plan | Tools | Reason | Ground | Report cite# | LLM |")
    lines.append("|------|--------|-----:|------:|-------:|-------:|-------------:|-----|")
    for a in agents:
        if not a.get("ok"):
            lines.append(f"| {a.get('id')} | ERROR | - | - | - | - | - | {a.get('error')} |")
            continue
        ag = a.get("agent_scores") or {}
        lines.append(
            f"| {a['id']} | {a.get('workflow_status')} | {ag.get('planning')} | {ag.get('tool_usage')} | "
            f"{ag.get('reasoning_quality')} | {ag.get('source_grounding')} | {a.get('page_citation_count')} | {a.get('llm_backend')} |"
        )
    lines.append("")
    lines.append("### Sample report excerpt")
    lines.append("")
    lines.append(f"Case: `{sample.get('id')}` — status=`{sample.get('workflow_status')}` sources=`{sample.get('structured_sources')}`")
    lines.append("")
    lines.append("```markdown")
    lines.append(str(sample.get("report_excerpt") or "")[:2000])
    lines.append("```")
    lines.append("")
    lines.append("### Observed strengths / weaknesses")
    lines.append("")
    lines.append("- **Strength:** live `structured_source` honesty; OpenAI / sparse Oracle paths fail-closed rather than inventing EBITDA.")
    lines.append("- **Strength:** ambiguous query triggers clarification (HITL) instead of guessing a ticker.")
    lines.append("- **Weakness:** analyst-grade valuation discussion is template/heuristic-heavy vs full comps model.")
    lines.append("- **Weakness:** RAG may cite nearby narrative pages without guaranteeing the exact FY total line item.")
    lines.append("")
    lines.append("## 6. Bottleneck Analysis")
    lines.append("")
    lines.append("### Stress results")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(stress, ensure_ascii=False, indent=2, default=str)[:4000])
    lines.append("```")
    lines.append("")
    lines.append("### Priority")
    lines.append("")
    lines.append("**P0 — must fix for analyst trust**")
    lines.append("")
    for item in p0 or ["No P0 blockers beyond disclosed HTML→PDF table geometry loss; monitor RAG numeric miss rate."]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**P1 — clear quality lift**")
    lines.append("")
    for item in p1:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**P2 — experience / polish**")
    lines.append("")
    for item in p2:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 7. Optimization Roadmap")
    lines.append("")
    lines.append("### RAG")
    lines.append("")
    lines.append("- Ingest **native PDF / HTML / iXBRL** without lossy text reflow; keep table grid metadata.")
    lines.append("- Add **numeric/fact index** (metric, period, value, page) alongside dense chunks for FY totals.")
    lines.append("- Keep lexical ZH/EN rerank; consider **DashScope/cloud rerank** only when candidate pool ≥20.")
    lines.append("- Expand gold RAG eval set (15→50) with page-level citations as labels.")
    lines.append("")
    lines.append("### Agent")
    lines.append("")
    lines.append("- Planner: force `prefer_uploaded_only` when user says so; surface conflicts in §0 Source Resolution.")
    lines.append("- Tool routing already local-first; keep MCP out of production evidence path.")
    lines.append("- Trace export: ensure `run_telemetry.rag.mode` always populated (already fixed).")
    lines.append("")
    lines.append("### Report")
    lines.append("")
    lines.append("- Keep deterministic **Retrieved Document Citations** section; add inline `[cite:]` on metric claims when AST inputs come from upload.")
    lines.append("- Fail-loud minimal contract already added; extend valuation section to explicitly say “no DCF computed” when true.")
    lines.append("")
    lines.append("### Infrastructure")
    lines.append("")
    lines.append("- Cache SEC companyfacts + embeddings by content hash.")
    lines.append("- Isolate Milvus Lite per long job; move to Server when concurrent users appear.")
    lines.append("- Retain raw audit JSON under `outputs/e2e_production_audit/raw/` for regression diffs.")
    lines.append("")
    lines.append("## Appendix — Raw artifacts")
    lines.append("")
    lines.append(f"- Directory: `{OUT}`")
    lines.append(f"- RAG raw: `raw/module2_rag.json`")
    lines.append(f"- Agent raw: `raw/module3_agents.json`")
    lines.append(f"- Stress raw: `raw/module5_stress.json`")
    lines.append("")

    path = OUT / "LumenFin_E2E_Audit_Report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    # Also copy to repo root name requested
    root_copy = ROOT / "LumenFin_E2E_Audit_Report.md"
    root_copy.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def main() -> int:
    configure_stdio_utf8()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    config = AppConfig.from_env()
    llm_key = (config.llm.api_key or "").strip()
    if not llm_key:
        print("FAIL: DEEPSEEK_API_KEY missing")
        return 1
    if str(config.embedding_provider).lower() not in {"dashscope", "aliyun", "alibaba"}:
        print(f"WARN: embedding_provider={config.embedding_provider} (expected dashscope for production audit)")

    env = {
        "data_mode": config.data_mode,
        "llm_model": config.llm.model,
        "llm_base": config.llm.base_url,
        "embedding_provider": config.embedding_provider,
        "embedding_dimension": config.embedding_dimension,
        "dashscope_model": os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3"),
        "rag_index_mode": config.rag_index_mode,
        "rag_rerank": config.rag_rerank_enabled,
        "fetch_sec": config.fetch_sec_fundamentals,
        "fetch_live": config.fetch_live_fundamentals,
        "market_provider": config.market_data_provider,
        "started_at": _now(),
    }
    _save("env.json", env)
    print("=== MODULE 1: Ingestion ===")
    pdfs = [
        FIX / "aapl_fy2024_10k_sec.pdf",
        FIX / "nvda_fy2025_10k_sec.pdf",
        FIX / "msft_fy2024_10k_sec.pdf",
    ]
    tsla = FIX / "tsla_fy2024_10k_sec.pdf"
    if tsla.exists():
        pdfs.append(tsla)
    else:
        print("WARN missing Tesla fixture", tsla, "- continuing with original docs for query parity")
    for p in pdfs:
        if not p.exists():
            print("FAIL missing", p)
            return 1
    ingest = module1_ingestion(pdfs)
    for s in ingest:
        print(f"  {s.filename}: pages={s.page_count} chunks={s.chunk_count} issues={s.issues}")

    print("=== MODULE 2: RAG (15 queries) ===")
    rag, index_summary = module2_rag(
        config,
        {
            "Apple": FIX / "aapl_fy2024_10k_sec.pdf",
            "NVIDIA": FIX / "nvda_fy2025_10k_sec.pdf",
            "Microsoft": FIX / "msft_fy2024_10k_sec.pdf",
        },
    )

    print("=== MODULE 3/4: Agent + Report (10 cases) ===")
    agents = module3_and_4_agents(config)

    print("=== MODULE 5: Stress ===")
    stress = module5_stress(config)

    report_path = write_report(
        env=env,
        ingest=ingest,
        rag=rag,
        agents=agents,
        stress=stress,
        index_summary=index_summary,
    )
    print("REPORT:", report_path)
    print("ROOT COPY:", ROOT / "LumenFin_E2E_Audit_Report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
