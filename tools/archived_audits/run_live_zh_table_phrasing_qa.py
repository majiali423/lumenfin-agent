"""ARCHIVED AUDIT SCRIPT.

Historical purpose: Chinese table phrasing and mismatch QA.
Replacement: current table/mismatch tests and RC pack.
Last compatible schema: historical live-QA artifact layout.
Not part of the supported release interface; do not run on production fixtures.

Also covers SoftBank↔Apple mismatch HITL pause/resume.
"""

from __future__ import annotations

import json
import logging
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

from build_table_pdf_fixtures import build_apple_msft_zh_table_pdf
from lumenfin.config import AppConfig
from lumenfin.document_ingest import parse_upload_documents
from lumenfin.rag.profiles import apply_showcase_rag_env
from lumenfin.service import LumenFinAnalysisService
from lumenfin.stdio import configure_stdio_utf8

apply_showcase_rag_env(overwrite=False)

OUT = ROOT / "outputs" / "live_zh_table_phrasing_qa"
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
    missing_fields: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    prefer_uploaded_only: bool | None = None
    structured_sources: dict[str, str] = field(default_factory=dict)
    metrics_preview: dict[str, dict[str, float]] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    error: str = ""
    report_path: str = ""
    ingest_hints: dict[str, Any] = field(default_factory=dict)


Judge = Callable[[CaseResult, dict[str, Any]], None]


def expect_completed(result: CaseResult, state: dict[str, Any], *, want: list[str] | None = None) -> None:
    result.checks.append(Check("status_completed", result.workflow_status == "completed", result.workflow_status))
    result.checks.append(Check("has_metrics", bool(result.metrics_preview), str(result.metrics_preview)[:200]))
    result.checks.append(Check("llm_deepseek", "deepseek" in str(state.get("llm_backend") or "").lower(), str(state.get("llm_backend"))))
    if want:
        missing = [c for c in want if c not in result.companies]
        result.checks.append(Check("companies_ok", not missing, f"got={result.companies} missing={missing}"))
    if len(result.metrics_preview) >= 2:
        ratios = [
            tuple(sorted((k, round(v, 4)) for k, v in m.items() if "margin" in k or "intensity" in k))
            for m in result.metrics_preview.values()
        ]
        if len(set(ratios)) == 1 and ratios[0]:
            result.findings.append("identical peer ratios — possible table attribution bug")
    result.ok = all(c.ok for c in result.checks) and not result.findings


def expect_hitl_mismatch(result: CaseResult, state: dict[str, Any]) -> None:
    result.checks.append(
        Check("needs_clarification", result.workflow_status == "needs_clarification", result.workflow_status)
    )
    result.checks.append(
        Check(
            "mismatch_field",
            "company_upload_mismatch" in result.missing_fields,
            str(result.missing_fields),
        )
    )
    result.checks.append(
        Check("has_question", bool(result.clarification_questions), str(result.clarification_questions)[:200])
    )
    result.checks.append(Check("no_supervisor_yet", "supervisor" not in result.steps, str(result.steps)))
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
        try:
            contexts = []
            for path in documents or []:
                contexts.extend(parse_upload_documents(path))
            result.ingest_hints = {
                "companies": sorted({c for ctx in contexts for c in (ctx.get("detected_companies") or [])}),
                "per_company": {ctx.get("filename"): ctx.get("per_company_metric_hints") for ctx in contexts},
            }
            print(f"  ingest: {json.dumps(result.ingest_hints, ensure_ascii=False)[:360]}", flush=True)
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
        result.missing_fields = list(state.get("missing_fields") or [])
        result.clarification_questions = list(state.get("clarification_questions") or [])
        result.prefer_uploaded_only = bool((state.get("query_plan") or {}).get("prefer_uploaded_only"))
        result.structured_sources = {
            str(k): str((v or {}).get("structured_source") or "none")
            for k, v in (state.get("retrieved_docs") or {}).items()
            if isinstance(v, dict)
        }
        metrics = state.get("financial_metrics") or {}
        result.metrics_preview = {
            str(k): {mk: float(mv) for mk, mv in (v or {}).items() if isinstance(mv, (int, float))}
            for k, v in metrics.items()
            if isinstance(v, dict)
        }
        result.steps = [str(a.get("step") or "") for a in (state.get("audit_log") or [])]
        result.report_path = str(artifacts.get("report_path") or "")
        judge(result, state)
        print(
            f"  -> ok={result.ok} status={result.workflow_status} companies={result.companies} "
            f"missing={result.missing_fields} prefer={result.prefer_uploaded_only}",
            flush=True,
        )
        for c in result.checks:
            print(f"  [{'OK' if c.ok else 'FAIL'}] {c.name}: {c.detail[:160]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        result.ok = False
        print(f"  EXCEPTION: {exc}", flush=True)
        print(traceback.format_exc()[-800:], flush=True)
    return result


def main() -> int:
    configure_stdio_utf8()
    for noisy in ("grpc", "grpc._server", "pymilvus", "httpx", "faiss", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    OUT.mkdir(parents=True, exist_ok=True)
    zh_table = build_apple_msft_zh_table_pdf(FIX / "apple_msft_fy2025_table_zh.pdf")
    en_table = FIX / "apple_msft_fy2025_table.pdf"
    if not en_table.exists():
        from build_table_pdf_fixtures import build_apple_msft_table_pdf

        build_apple_msft_table_pdf(en_table)

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

    cases.append(
        run_case(
            service,
            case_id="zh-table-compare",
            description="中文表格：对比苹果微软",
            query="基于上传的中文表格，对比苹果和微软 FY2025 的盈利能力、研发强度与供应链风险。",
            documents=[zh_table],
            judge=lambda r, s: expect_completed(r, s, want=["Apple", "Microsoft"]),
        )
    )
    cases.append(
        run_case(
            service,
            case_id="zh-table-upload-only",
            description="中文表格：仅用上传",
            query="仅用上传表格分析苹果与微软 FY2025 EBITDA 利润率与研发强度，不要用外部数据。",
            documents=[zh_table],
            judge=lambda r, s: expect_completed(r, s, want=["Apple", "Microsoft"]),
        )
    )
    cases.append(
        run_case(
            service,
            case_id="zh-table-relative-time",
            description="中文表格：相对时间这两年",
            query="根据上传表格，分析苹果这两年盈利能力与研发投入。",
            documents=[zh_table],
            judge=lambda r, s: expect_completed(r, s, want=["Apple"]),
        )
    )
    cases.append(
        run_case(
            service,
            case_id="zh-byd-memo",
            description="比亚迪双语备忘录",
            query="基于上传的比亚迪备忘录，评估 FY2025 盈利能力、研发强度与供应链风险。",
            documents=[FIX / "byd_zh_en_memo.pdf"],
            judge=lambda r, s: expect_completed(r, s, want=["BYD"]),
        )
    )
    cases.append(
        run_case(
            service,
            case_id="zh-mismatch-softbank",
            description="问软银但上传苹果微软表 → HITL",
            query="请用上传材料分析软银 FY2024 盈利能力与杠杆。",
            documents=[zh_table],
            judge=expect_hitl_mismatch,
        )
    )

    # Resume mismatch with uploaded scope
    print("\n=== zh-mismatch-resume-uploaded ===", flush=True)
    resume = CaseResult(
        id="zh-mismatch-resume-uploaded",
        description="Resume mismatch with company_scope=uploaded",
        query="(clarify)",
    )
    try:
        payload = service.clarify(
            thread_id="zh-mismatch-softbank",
            clarification={"company_scope": "uploaded", "time_range": "FY2025"},
            export_artifacts=True,
        )
        state = payload["result"]
        resume.workflow_status = str(state.get("workflow_status") or "")
        resume.companies = list(state.get("companies") or [])
        resume.missing_fields = list(state.get("missing_fields") or [])
        resume.steps = [str(a.get("step") or "") for a in (state.get("audit_log") or [])]
        metrics = state.get("financial_metrics") or {}
        resume.metrics_preview = {
            str(k): {mk: float(mv) for mk, mv in (v or {}).items() if isinstance(mv, (int, float))}
            for k, v in metrics.items()
            if isinstance(v, dict)
        }
        expect_completed(resume, state, want=["Apple", "Microsoft"])
        print(f"  -> ok={resume.ok} status={resume.workflow_status} companies={resume.companies}", flush=True)
    except Exception as exc:  # noqa: BLE001
        resume.error = str(exc)
        resume.ok = False
        print(f"  EXCEPTION clarify: {exc}", flush=True)
    cases.append(resume)

    cases.append(
        run_case(
            service,
            case_id="zh-no-company-use-upload",
            description="未点名公司，应直接用上传表中的苹果微软",
            query="基于上传表格做 FY2025 盈利能力与研发强度对比，并给出合规意见。",
            documents=[zh_table],
            judge=lambda r, s: expect_completed(r, s, want=["Apple", "Microsoft"]),
        )
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": sum(1 for c in cases if c.ok),
        "total": len(cases),
        "cases": [asdict(c) for c in cases],
    }
    report = OUT / "report.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    findings = OUT / "FINDINGS.md"
    lines = [
        f"# ZH table phrasing QA",
        f"",
        f"Passed: {payload['passed']}/{payload['total']}",
        f"",
    ]
    for case in cases:
        lines.append(f"## {case.id} — {'PASS' if case.ok else 'FAIL'}")
        lines.append(f"- {case.description}")
        lines.append(f"- status=`{case.workflow_status}` companies={case.companies}")
        fails = [c for c in case.checks if not c.ok]
        if fails:
            for c in fails:
                lines.append(f"- FAIL `{c.name}`: {c.detail}")
        if case.findings:
            for f in case.findings:
                lines.append(f"- finding: {f}")
        if case.error:
            lines.append(f"- error: `{case.error}`")
        lines.append("")
    findings.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n=== SUMMARY {payload['passed']}/{payload['total']} ===", flush=True)
    print(f"wrote {report}", flush=True)
    print(f"wrote {findings}", flush=True)
    return 0 if payload["passed"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
