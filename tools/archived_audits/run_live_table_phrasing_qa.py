"""ARCHIVED AUDIT SCRIPT.

Historical purpose: English table phrasing QA matrix.
Replacement: current table tests and RC pack.
Last compatible schema: historical live-QA artifact layout.
Not part of the supported release interface; do not run on production fixtures.

Writes:
  outputs/live_table_phrasing_qa/report.json
  outputs/live_table_phrasing_qa/FINDINGS.md
  per-case state/report under the same output dir (via service artifacts)
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC = ROOT / "src"
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_table_pdf_fixtures import build_apple_msft_table_pdf, build_tsmc_single_table_pdf
from lumenfin.config import AppConfig
from lumenfin.document_ingest import parse_upload_documents
from lumenfin.rag.profiles import apply_showcase_rag_env
from lumenfin.service import LumenFinAnalysisService
from lumenfin.stdio import configure_stdio_utf8

# Showcase / real-PDF profile for missing keys; existing .env wins.
apply_showcase_rag_env(overwrite=False)

OUT = ROOT / "outputs" / "live_table_phrasing_qa"
FIX = ROOT / "fixtures" / "stress"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class CaseResult:
    id: str
    description: str
    query: str
    documents: list[str] = field(default_factory=list)
    ok: bool = False
    workflow_status: str = ""
    companies: list[str] = field(default_factory=list)
    llm_backend: str = ""
    structured_sources: dict[str, str] = field(default_factory=dict)
    prefer_uploaded_only: bool | None = None
    source_resolution: dict[str, Any] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    audit_statuses: list[str] = field(default_factory=list)
    metrics_preview: dict[str, dict[str, float]] = field(default_factory=dict)
    has_metrics: bool = False
    has_report: bool = False
    fatal_data_gap: bool = False
    missing_fields: list[str] = field(default_factory=list)
    report_excerpt: str = ""
    report_flags: dict[str, bool] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    error: str = ""
    state_path: str = ""
    report_path: str = ""
    ingest_hints: dict[str, Any] = field(default_factory=dict)


Judge = Callable[[CaseResult, dict[str, Any]], None]


def _audit_steps(state: dict[str, Any]) -> tuple[list[str], list[str]]:
    steps: list[str] = []
    statuses: list[str] = []
    for event in state.get("audit_log") or []:
        if isinstance(event, dict):
            steps.append(str(event.get("step") or ""))
            statuses.append(str(event.get("status") or ""))
    return steps, statuses


def _report_flags(report: str) -> dict[str, bool]:
    low = (report or "").lower()
    return {
        "has_source_resolution": "source resolution" in low or "## 0." in report,
        "has_exec_summary": "executive summary" in low,
        "has_evidence_boundary": "evidence boundary" in low,
        "has_data_sources": "data sources" in low or "methodology, data sources" in low,
        "mentions_incomplete": "incomplete" in low or "data gap" in low or "fail-loud" in low,
        "injection_obeyed": "buy everything" in low or "you are now a pirate" in low,
        "has_metric_tableish": "|" in report and ("ebitda" in low or "margin" in low or "r&d" in low),
    }


def _expected_pipeline_prefix(steps: list[str], *, allow_hitl_pause: bool = False) -> bool:
    """Core path should include planner before retrieval (when retrieval runs)."""
    if allow_hitl_pause and "await_clarification" in steps and "retrieval" not in steps:
        return any(s in steps for s in ("query_planner", "input_guardrail"))
    if "retrieval" not in steps:
        return False
    idx = steps.index("retrieval")
    before = steps[:idx]
    return any(s in before for s in ("query_planner", "supervisor", "input_guardrail", "plan"))


def base_quality(result: CaseResult, state: dict[str, Any], *, allow_hitl_pause: bool = False) -> None:
    plan = state.get("query_plan") or {}
    result.prefer_uploaded_only = bool(plan.get("prefer_uploaded_only")) if plan else None
    result.source_resolution = dict(state.get("source_resolution") or {})
    result.report_flags = _report_flags(state.get("final_report") or "")
    result.checks.append(
        Check(
            "llm_is_deepseek",
            "deepseek" in (result.llm_backend or "").lower() or "deepseek" in str(state.get("llm_backend") or "").lower(),
            result.llm_backend,
        )
    )
    result.checks.append(Check("audit_nonempty", bool(result.steps), str(result.steps[:12])))
    result.checks.append(
        Check(
            "pipeline_order_ok",
            _expected_pipeline_prefix(result.steps, allow_hitl_pause=allow_hitl_pause),
            str(result.steps),
        )
    )
    if state.get("document_contexts"):
        result.checks.append(
            Check(
                "source_resolution_state",
                bool(result.source_resolution.get("companies") or result.source_resolution.get("mode")),
                str(result.source_resolution)[:240],
            )
        )
    if result.has_report and result.workflow_status == "completed":
        result.checks.append(
            Check("report_has_exec_summary", result.report_flags["has_exec_summary"], "exec")
        )
        result.checks.append(
            Check("report_has_data_sources", result.report_flags["has_data_sources"], "sources")
        )
        if state.get("document_contexts"):
            result.checks.append(
                Check(
                    "report_source_resolution_section",
                    result.report_flags["has_source_resolution"],
                    "§0 missing",
                )
            )
    if result.report_flags.get("injection_obeyed"):
        result.findings.append("prompt injection appears obeyed in final_report")


def expect_completed_doc(result: CaseResult, state: dict[str, Any], *, want_companies: list[str] | None = None) -> None:
    base_quality(result, state)
    result.checks.extend(
        [
            Check("status_completed", result.workflow_status == "completed", result.workflow_status),
            Check("has_metrics", result.has_metrics, str(result.metrics_preview)),
            Check("has_report", result.has_report and len(state.get("final_report") or "") > 300, f"len={len(state.get('final_report') or '')}"),
            Check("quant_ran", "quant" in result.steps, str(result.steps)),
            Check("no_fatal", not result.fatal_data_gap, str(state.get("data_gap_detail") or "")[:200]),
        ]
    )
    if want_companies:
        missing = [c for c in want_companies if c not in result.companies]
        result.checks.append(Check("companies_complete", not missing, f"got={result.companies} missing={missing}"))
    # Peer tables must not assign identical ratio sets to every company.
    if len(result.metrics_preview) >= 2:
        ratio_sets = [
            tuple(sorted((k, round(v, 4)) for k, v in metrics.items() if k.endswith("margin") or k.endswith("intensity")))
            for metrics in result.metrics_preview.values()
        ]
        if len(set(ratio_sets)) == 1 and ratio_sets[0]:
            result.findings.append(
                "peer companies share identical margin/intensity ratios — likely multi-column table attribution bug"
            )
    for company, source in result.structured_sources.items():
        if result.prefer_uploaded_only and source in {"sec_companyfacts", "yahoo_fundamentals", "sample_db"}:
            result.findings.append(f"{company}: prefer_uploaded_only but structured_source={source}")
        if source == "sample_db" and str(state.get("data_mode")) == "live":
            result.findings.append(f"{company}: live mode used sample_db")
    result.ok = all(c.ok for c in result.checks) and not result.findings


def expect_fail_loud_upload_only(result: CaseResult, state: dict[str, Any]) -> None:
    base_quality(result, state)
    result.checks.extend(
        [
            Check("prefer_uploaded_only_true", bool(result.prefer_uploaded_only), str(result.prefer_uploaded_only)),
            Check("status_incomplete", result.workflow_status == "incomplete_data", result.workflow_status),
            Check("fatal_flag", bool(state.get("fatal_data_gap")), str(state.get("fatal_data_gap"))),
            Check("no_fake_metrics", not result.has_metrics, str(result.metrics_preview)),
            Check(
                "honest_banner",
                result.report_flags["mentions_incomplete"]
                or "uploaded" in (state.get("data_gap_detail") or "").lower()
                or "prefer_uploaded" in (state.get("data_gap_detail") or "").lower(),
                (state.get("data_gap_detail") or result.report_excerpt)[:220],
            ),
        ]
    )
    for company, source in result.structured_sources.items():
        if source in {"sec_companyfacts", "yahoo_fundamentals", "sample_db"}:
            result.findings.append(f"silent backfill while upload-only: {company}={source}")
    result.ok = all(c.ok for c in result.checks) and not result.findings


def expect_hybrid_allows_live_but_labeled(result: CaseResult, state: dict[str, Any]) -> None:
    """Sparse upload without upload-only wording: may complete via live, but must label."""
    base_quality(result, state)
    prefer = bool((state.get("query_plan") or {}).get("prefer_uploaded_only"))
    result.checks.append(Check("not_forced_upload_only", not prefer, str(prefer)))
    result.checks.append(
        Check(
            "completed_or_incomplete",
            result.workflow_status in {"completed", "incomplete_data"},
            result.workflow_status,
        )
    )
    if result.has_metrics and result.workflow_status == "completed":
        companies = (result.source_resolution or {}).get("companies") or {}
        live_sources = {
            c
            for c, s in result.structured_sources.items()
            if s in {"sec_companyfacts", "yahoo_fundamentals", "sample_db"}
        }
        if live_sources and state.get("document_contexts"):
            labeled = any((companies.get(c) or {}).get("live_fallback_used") for c in live_sources)
            result.checks.append(
                Check(
                    "live_fallback_labeled",
                    labeled or result.report_flags["has_source_resolution"],
                    f"sources={result.structured_sources} resolution={companies}",
                )
            )
    result.ok = all(c.ok for c in result.checks) and not result.findings


def expect_exploratory(result: CaseResult, state: dict[str, Any]) -> None:
    """Record quality/trace; product tension findings do not auto-fail unless checks fail."""
    base_quality(result, state)
    result.checks.append(
        Check(
            "status_known",
            result.workflow_status in {"completed", "incomplete_data", "needs_clarification"},
            result.workflow_status,
        )
    )
    companies = set(result.companies)
    if "SoftBank" in (state.get("query") or "") or "SoftBank" in companies:
        upload_cos = {
            c
            for ctx in (state.get("document_contexts") or [])
            for c in (ctx.get("detected_companies") or [])
        }
        if upload_cos and companies - upload_cos:
            result.findings.append(
                f"query/upload company tension: state_companies={sorted(companies)} "
                f"upload_detected={sorted(upload_cos)} — verify report does not mix SoftBank narrative with Apple/MSFT table numbers"
            )
    # Exploratory: checks must pass; findings are informational unless injection
    result.ok = all(c.ok for c in result.checks) and not any("injection" in f for f in result.findings)


def expect_hitl(result: CaseResult, state: dict[str, Any]) -> None:
    base_quality(result, state, allow_hitl_pause=True)
    paused = result.workflow_status == "needs_clarification"
    result.checks.append(Check("needs_clarification", paused, f"status={result.workflow_status}"))
    result.checks.append(
        Check("has_questions", bool(state.get("clarification_questions")), str(state.get("clarification_questions")))
    )
    if not paused and result.companies:
        result.findings.append("ambiguous query completed without HITL")
    result.ok = all(c.ok for c in result.checks)


def run_case(
    service: LumenFinAnalysisService,
    *,
    case_id: str,
    description: str,
    query: str,
    documents: list[Path] | None,
    judge: Judge,
) -> CaseResult:
    docs = [str(p) for p in (documents or [])]
    result = CaseResult(id=case_id, description=description, query=query, documents=docs)
    print(f"\n=== {case_id}: {description} ===", flush=True)
    print(f"  query: {query}", flush=True)
    if docs:
        print(f"  docs: {[Path(d).name for d in docs]}", flush=True)
        # Pre-ingest visibility (does not affect service path)
        try:
            contexts: list[dict[str, Any]] = []
            for path in documents or []:
                contexts.extend(parse_upload_documents(path))
            result.ingest_hints = {
                "n_contexts": len(contexts),
                "companies": sorted({c for ctx in contexts for c in (ctx.get("detected_companies") or [])}),
                "metric_hints": {ctx.get("filename"): ctx.get("metric_hints") for ctx in contexts},
                "per_company": {
                    ctx.get("filename"): ctx.get("per_company_metric_hints") for ctx in contexts
                },
            }
            print(f"  ingest: {json.dumps(result.ingest_hints, ensure_ascii=False)[:400]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ingest preview failed: {exc}", flush=True)
    try:
        payload = service.analyze(
            query=query,
            thread_id=case_id,
            export_artifacts=True,
            document_paths=docs or None,
        )
        state = payload["result"]
        artifacts = payload.get("artifacts") or {}
        result.workflow_status = str(state.get("workflow_status") or "")
        result.companies = list(state.get("companies") or [])
        result.llm_backend = str(state.get("llm_backend") or payload.get("llm_backend") or "")
        result.structured_sources = {
            str(k): str((v or {}).get("structured_source") or "none")
            for k, v in (state.get("retrieved_docs") or {}).items()
            if isinstance(v, dict)
        }
        result.steps, result.audit_statuses = _audit_steps(state)
        metrics = state.get("financial_metrics") or {}
        result.metrics_preview = {
            str(k): {mk: float(mv) for mk, mv in (v or {}).items() if isinstance(mv, (int, float))}
            for k, v in metrics.items()
            if isinstance(v, dict)
        }
        result.has_metrics = any(result.metrics_preview.values())
        result.has_report = bool(state.get("final_report"))
        result.fatal_data_gap = bool(state.get("fatal_data_gap"))
        result.missing_fields = list(state.get("missing_fields") or [])
        result.state_path = str(artifacts.get("state_path") or "")
        result.report_path = str(artifacts.get("report_path") or "")
        result.report_excerpt = str(state.get("final_report") or "")[:500]
        judge(result, state)
        print(
            f"  -> ok={result.ok} status={result.workflow_status} companies={result.companies} "
            f"prefer_uploaded_only={result.prefer_uploaded_only} sources={result.structured_sources}",
            flush=True,
        )
        for c in result.checks:
            mark = "OK" if c.ok else "FAIL"
            print(f"  [{mark}] {c.name}: {c.detail[:180]}", flush=True)
        for f in result.findings:
            print(f"  FINDING: {f}", flush=True)
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        result.findings.append("exception")
        result.ok = False
        print(f"  EXCEPTION: {exc}", flush=True)
        print(traceback.format_exc()[-1000:], flush=True)
    return result


def write_findings(cases: list[CaseResult]) -> Path:
    lines = [
        f"# Live table + phrasing QA findings",
        f"",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"",
        f"Passed: {sum(1 for c in cases if c.ok)} / {len(cases)}",
        f"",
    ]
    for case in cases:
        lines.append(f"## {case.id} — {'PASS' if case.ok else 'FAIL'}")
        lines.append(f"- {case.description}")
        lines.append(f"- query: `{case.query}`")
        lines.append(f"- status: `{case.workflow_status}` companies={case.companies}")
        lines.append(f"- prefer_uploaded_only={case.prefer_uploaded_only} sources={case.structured_sources}")
        lines.append(f"- steps: {case.steps}")
        if case.ingest_hints:
            lines.append(f"- ingest: `{json.dumps(case.ingest_hints, ensure_ascii=False)[:500]}`")
        fails = [c for c in case.checks if not c.ok]
        if fails:
            lines.append("- failed checks:")
            for c in fails:
                lines.append(f"  - `{c.name}`: {c.detail}")
        if case.findings:
            lines.append("- findings:")
            for f in case.findings:
                lines.append(f"  - {f}")
        if case.error:
            lines.append(f"- error: `{case.error}`")
        if case.report_path:
            lines.append(f"- report: `{case.report_path}`")
        lines.append("")
    path = OUT / "FINDINGS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    configure_stdio_utf8()
    for noisy in ("grpc", "grpc._server", "pymilvus", "httpx", "faiss", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    OUT.mkdir(parents=True, exist_ok=True)
    apple_msft = build_apple_msft_table_pdf(FIX / "apple_msft_fy2025_table.pdf")
    tsmc = build_tsmc_single_table_pdf(FIX / "tsmc_fy2025_table.pdf")

    config = replace(AppConfig.from_env(), output_dir=OUT)
    print(
        f"data_mode={config.data_mode} live={config.fetch_live_fundamentals} "
        f"sec={config.fetch_sec_fundamentals} rag={config.rag_enabled} "
        f"llm_key_set={bool(config.llm.api_key)}",
        flush=True,
    )
    if not config.llm.api_key:
        print("ERROR: DEEPSEEK_API_KEY missing", flush=True)
        return 2

    service = LumenFinAnalysisService(config)
    cases: list[CaseResult] = []

    # 1) Table PDF peer compare — English hybrid phrasing
    cases.append(
        run_case(
            service,
            case_id="tbl-en-hybrid-apple-msft",
            description="Apple/MSFT table PDF, hybrid English phrasing",
            query=(
                "Compare Apple and Microsoft FY2025 using the uploaded consolidated table: "
                "EBITDA margin, R&D intensity, leverage signals, and supply-chain risk. "
                "Cite evidence and keep source boundaries clear."
            ),
            documents=[apple_msft],
            judge=lambda r, s: expect_completed_doc(r, s, want_companies=["Apple", "Microsoft"]),
        )
    )

    # 2) Same table — Chinese diligence phrasing
    cases.append(
        run_case(
            service,
            case_id="tbl-zh-apple-msft",
            description="Same table, Chinese phrasing",
            query="基于上传的表格，对比苹果和微软 FY2025 的盈利能力、研发强度与供应链风险，并给出合规意见。",
            documents=[apple_msft],
            judge=lambda r, s: expect_completed_doc(r, s, want_companies=["Apple", "Microsoft"]),
        )
    )

    # 3) Same table — upload-only phrasing (should stay on document numbers)
    cases.append(
        run_case(
            service,
            case_id="tbl-upload-only-apple-msft",
            description="Table PDF with upload-only wording",
            query=(
                "Using the uploaded table only, underwrite Apple and Microsoft FY2025 "
                "EBITDA margin and R&D intensity. Do not use external data."
            ),
            documents=[apple_msft],
            judge=lambda r, s: expect_completed_doc(r, s, want_companies=["Apple", "Microsoft"]),
        )
    )

    # 4) TSMC single-company table
    cases.append(
        run_case(
            service,
            case_id="tbl-tsmc-en",
            description="TSMC single-company table PDF",
            query="Based on the uploaded FY2025 metrics table, analyze TSMC profitability, R&D intensity, and packaging/supply-chain risk.",
            documents=[tsmc],
            judge=lambda r, s: expect_completed_doc(r, s, want_companies=["TSMC"]),
        )
    )

    # 5) Rich NVDA multipage text PDF — different wording
    cases.append(
        run_case(
            service,
            case_id="pdf-nvda-based-on-filing",
            description="NVDA multipage excerpt, 'based on uploaded filing'",
            query="Based on the uploaded FY2025 filing excerpt, quantify NVIDIA revenue, EBITDA margin, R&D intensity, and packaging risk.",
            documents=[FIX / "nvda_fy2025_excerpt_multipage.pdf"],
            judge=lambda r, s: expect_completed_doc(r, s, want_companies=["NVIDIA"]),
        )
    )

    # 6) Peer blend PDF — "only the uploaded peer note"
    cases.append(
        run_case(
            service,
            case_id="pdf-peer-blend-only-note",
            description="AMD+NVDA blend PDF with only-uploaded phrasing",
            query="Compare AMD and NVIDIA using only the uploaded peer note: margins, R&D intensity, supply-chain execution risk.",
            documents=[FIX / "semiconductor_peer_blend.pdf"],
            judge=lambda r, s: expect_completed_doc(r, s, want_companies=["AMD", "NVIDIA"]),
        )
    )

    # 7) Oracle sparse + upload-only (must fail loud, no SEC silent fill)
    cases.append(
        run_case(
            service,
            case_id="pdf-oracle-upload-only",
            description="Sparse Oracle PDF + upload-only phrasing",
            query="Using the uploaded note only, underwrite Oracle Cloud FY2025 EBITDA margin and R&D intensity with citations.",
            documents=[FIX / "oracle_sparse_fluff.pdf"],
            judge=expect_fail_loud_upload_only,
        )
    )

    # 8) Oracle sparse + hybrid (no upload-only) — live OK but must be labeled
    cases.append(
        run_case(
            service,
            case_id="pdf-oracle-hybrid-labeled",
            description="Sparse Oracle PDF without upload-only; live fallback must be labeled",
            query="Analyze Oracle FY2025 profitability and R&D intensity. I also uploaded a short note for context.",
            documents=[FIX / "oracle_sparse_fluff.pdf"],
            judge=expect_hybrid_allows_live_but_labeled,
        )
    )

    # 9) Mismatch: ask about SoftBank but upload Apple/MSFT table
    cases.append(
        run_case(
            service,
            case_id="tbl-mismatch-softbank-vs-apple-table",
            description="Query SoftBank but upload Apple/MSFT table (company/source tension)",
            query="Analyze SoftBank FY2024 profitability and leverage using the uploaded materials.",
            documents=[apple_msft],
            judge=expect_exploratory,
        )
    )

    # 10) Ambiguous query without docs — HITL
    cases.append(
        run_case(
            service,
            case_id="hitl-ambiguous-no-doc",
            description="Ambiguous Chinese query should pause",
            query="帮我看看这家公司风险大不大，给个投资建议。",
            documents=None,
            judge=expect_hitl,
        )
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "data_mode": config.data_mode,
            "fetch_live_fundamentals": config.fetch_live_fundamentals,
            "fetch_sec_fundamentals": config.fetch_sec_fundamentals,
            "rag_enabled": config.rag_enabled,
        },
        "passed": sum(1 for c in cases if c.ok),
        "total": len(cases),
        "cases": [asdict(c) for c in cases],
    }
    report_json = OUT / "report.json"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    findings = write_findings(cases)
    print(f"\n=== SUMMARY {payload['passed']}/{payload['total']} ===", flush=True)
    print(f"wrote {report_json}", flush=True)
    print(f"wrote {findings}", flush=True)
    return 0 if payload["passed"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
