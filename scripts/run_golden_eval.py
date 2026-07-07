from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin import LumenFinAgentSystem
from lumenfin.evaluation import evaluate_run_state
from lumenfin.logging_utils import install_secret_redaction_filter
from lumenfin.reporting import export_run_artifacts
from lumenfin.stdio import configure_stdio_utf8


def main() -> None:
    configure_stdio_utf8()
    install_secret_redaction_filter()
    parser = argparse.ArgumentParser(description="Run golden evaluation cases for LumenFin.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--min-score", type=int, default=70)
    args = parser.parse_args()

    cases_path = ROOT / "data" / "eval_golden" / "golden_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    results: list[dict] = []

    for case in cases:
        app = LumenFinAgentSystem()
        thread_id = f"golden-{case['id']}-{int(time.time())}"
        print(f"Running {case['id']}...", flush=True)
        state = app.run(case["query"], thread_id=thread_id)
        eval_result = evaluate_run_state(state).to_dict()
        steps = [e.get("step") for e in state.get("audit_log", [])]
        companies = state.get("companies", [])
        report = state.get("final_report", "")

        steps_ok = all(s in steps for s in case.get("expect_steps", []))
        companies_ok = all(c in companies for c in case.get("expect_companies", []))
        markers_ok = all(m in report for m in case.get("report_markers", []))
        score_ok = eval_result["score"] >= args.min_score
        passed = steps_ok and companies_ok and markers_ok and score_ok

        artifacts = export_run_artifacts(state, ROOT / "outputs", thread_id)
        results.append(
            {
                "id": case["id"],
                "passed": passed,
                "score": eval_result["score"],
                "grade": eval_result["grade"],
                "steps_ok": steps_ok,
                "companies_ok": companies_ok,
                "markers_ok": markers_ok,
                "observed_steps": steps,
                "companies": companies,
                "report_path": artifacts.get("report_path"),
            }
        )
        print(f"  {'PASS' if passed else 'FAIL'} score={eval_result['score']}")

    passed_count = sum(1 for r in results if r["passed"])
    summary = {
        "total": len(results),
        "passed": passed_count,
        "pass_rate": round(passed_count / max(1, len(results)), 2),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.write:
        out = ROOT / "outputs" / "golden_eval_report.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
