"""Live e2e report-quality smoke (DeepSeek).

Asserts analyst-facing quality gates on brief compare output.
Does not print secrets. Writes SUMMARY under outputs/.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin import LumenFinAgentSystem
from lumenfin.config import AppConfig
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

    out_dir = ROOT / "outputs" / f"report_quality_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    findings: list[str] = []

    system = LumenFinAgentSystem(app_config=cfg)
    query = "Compare Apple and Microsoft FY2024 profitability, operating margin and R&D intensity."

    print("=== LIVE BRIEF QUALITY ===", flush=True)
    brief = system.run(query, thread_id="rq-live-brief", output_format="executive_summary")
    report = brief.get("final_report") or ""
    _assert(brief.get("llm_backend") == "deepseek" or "deepseek" in str(brief.get("llm_backend")), "expected deepseek backend")
    _assert("Period Alignment" in report, "missing Period Alignment")
    _assert("Peer Metric Matrix" in report, "missing Peer Metric Matrix")
    _assert("Comparison capsule" in report or "Operating Margin" in report, "missing compare capsule signal")
    _assert("Company Profiles & Business Overview" not in report, "profiles should be skipped without upload")
    _assert("No uploaded company profile document was provided" not in report, "profile filler present")
    summary_block = report.split("## 1. Executive Summary", 1)[-1].split("## 4.", 1)[0]
    _assert("supply-chain risk signal is 'unknown'" not in summary_block, "unknown supply noise in brief summary")
    _assert("quality-screening research thesis" not in summary_block, "thesis filler in brief summary")
    # Matrix should make MSFT EBITDA asymmetry visible when missing
    if "EBITDA Margin" in report and "n/a" in report:
        findings.append("EBITDA asymmetry disclosed via n/a")
    findings.append(f"BRIEF ok len={len(report)} status={brief.get('workflow_status')}")
    arts = export_run_artifacts(brief, out_dir, "rq-brief", llm_backend=brief.get("llm_backend"))
    findings.append(f"artifacts={arts.get('report_path')}")

    print("=== LIVE FULL (no upload → no profile filler) ===", flush=True)
    full = system.run(query, thread_id="rq-live-full")
    full_report = full.get("final_report") or ""
    _assert("Period Alignment" in full_report, "full missing Period Alignment")
    _assert("Peer Metric Matrix" in full_report, "full missing matrix")
    _assert("No uploaded company profile document was provided" not in full_report, "full still has profile filler")
    findings.append(f"FULL ok len={len(full_report)} status={full.get('workflow_status')}")
    export_run_artifacts(full, out_dir, "rq-full", llm_backend=full.get("llm_backend"))

    summary = out_dir / "SUMMARY.md"
    summary.write_text("# Report quality live smoke\n\n" + "\n".join(f"- {x}" for x in findings) + "\n", encoding="utf-8")
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
