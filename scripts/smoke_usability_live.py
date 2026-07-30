"""Live usability smoke: DeepSeek brief vs full + mismatch HITL resume.

Does not print secrets. Writes a short markdown summary under outputs/.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin import LumenFinAgentSystem
from lumenfin.config import AppConfig
from lumenfin.document_ingest import parse_upload_documents
from lumenfin.env_bootstrap import bootstrap_dotenv
from lumenfin.reporting import export_run_artifacts


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
    cfg = AppConfig.from_env()
    if not cfg.llm.api_key:
        print("FAIL: DEEPSEEK_API_KEY not configured")
        return 2

    out_dir = ROOT / "outputs" / f"usability_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    findings: list[str] = []

    system = LumenFinAgentSystem(app_config=cfg)
    query = "Compare Apple and Microsoft FY2024 profitability, operating margin and R&D intensity."

    print("=== LIVE FULL (default, no output_format) ===", flush=True)
    full = system.run(query, thread_id="live-full-usability")
    full_report = full.get("final_report") or ""
    _assert("Research Thesis" in full_report or "Strategic Analysis (SWOT)" in full_report, "full missing long sections")
    _assert("Financial Performance Analysis" in full_report, "full missing financial table section")
    findings.append(f"FULL ok status={full.get('workflow_status')} len={len(full_report)} backend={full.get('llm_backend')}")
    export_run_artifacts(full, out_dir, "live-full", llm_backend=full.get("llm_backend"))

    print("=== LIVE BRIEF (explicit executive_summary) ===", flush=True)
    brief = system.run(query, thread_id="live-brief-usability", output_format="executive_summary")
    brief_report = brief.get("final_report") or ""
    _assert("Brief Diligence" in brief_report or "Report Mode" in brief_report, "brief missing mode banner")
    _assert("Research Thesis" not in brief_report, "brief still has Thesis")
    _assert("Strategic Analysis (SWOT)" not in brief_report, "brief still has SWOT")
    _assert("Financial Performance Analysis" in brief_report, "brief missing financial tables")
    findings.append(
        f"BRIEF ok status={brief.get('workflow_status')} len={len(brief_report)} "
        f"requested={brief.get('requested_output_format')} backend={brief.get('llm_backend')}"
    )
    arts = export_run_artifacts(brief, out_dir, "live-brief", llm_backend=brief.get("llm_backend"))
    _assert(arts.get("metrics_csv_path"), "brief export missing metrics_csv_path")
    findings.append(f"CSV path={arts.get('metrics_csv_path')}")

    # Keyword alone must NOT trim when no explicit format
    print("=== KEYWORD-ONLY (简版 in query, no output_format) ===", flush=True)
    kw = system.run(
        "简版对比 Apple and Microsoft FY2024 profitability and R&D",
        thread_id="live-keyword-usability",
    )
    kw_report = kw.get("final_report") or ""
    _assert("Research Thesis" in kw_report or "Strategic Analysis (SWOT)" in kw_report, "keyword-only incorrectly trimmed")
    findings.append(f"KEYWORD-ONLY stayed full len={len(kw_report)}")

    # Mismatch HITL with real PDF upload
    print("=== MISMATCH HITL (SoftBank query + NVDA PDF) ===", flush=True)
    pdf = ROOT / "fixtures" / "stress" / "nvda_fy2025_excerpt_multipage.pdf"
    docs = parse_upload_documents(pdf)
    system2 = LumenFinAgentSystem(app_config=cfg)
    paused = system2.run(
        "Analyze SoftBank FY2024 profitability using the uploaded materials.",
        thread_id="live-mismatch-usability",
        document_contexts=docs,
    )
    _assert(paused.get("workflow_status") == "needs_clarification", f"expected pause, got {paused.get('workflow_status')}")
    qs = paused.get("clarification_questions") or []
    _assert(any("uploaded" in q and "query" in q for q in qs), f"questions not optionized: {qs}")
    findings.append(f"PAUSE ok questions={qs}")

    resumed = system2.resume_with_clarification(
        "live-mismatch-usability",
        {"company_scope": "uploaded"},
    )
    findings.append(
        f"RESUME uploaded scope status={resumed.get('workflow_status')} "
        f"companies={resumed.get('companies')} report_len={len(resumed.get('final_report') or '')}"
    )
    export_run_artifacts(resumed, out_dir, "live-mismatch-resume", llm_backend=resumed.get("llm_backend"))

    summary = out_dir / "SUMMARY.md"
    summary.write_text(
        "# Usability live smoke\n\n" + "\n".join(f"- {line}" for line in findings) + "\n",
        encoding="utf-8",
    )
    print("SUMMARY:", summary, flush=True)
    for line in findings:
        print(" -", line, flush=True)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("FAIL:", type(exc).__name__, exc, flush=True)
        raise
