"""Short live showcase: LumenFin -> FinRun -> FinAgentBench.

This script is intentionally smaller than run_live_multi_stability.py. It is meant
for interviews and demos where you need real API calls, concise output, and one
clear example for each important behavior:

1. Public US filer succeeds through SEC companyfacts.
2. Non-US company can use live Yahoo fundamentals fallback.
3. Private company with no public structured fundamentals fails closed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.profiles import apply_showcase_rag_env
from lumenfin.stdio import configure_stdio_utf8
from repo_paths import finagentbench_root

FAB = finagentbench_root()
GENERIC_CASE = FAB / "fixtures" / "case_lumenfin_generic.json"
OUT = ROOT / "outputs" / "live_showcase"
REPORT = OUT / "live_showcase_report.json"

CASES: dict[str, dict[str, Any]] = {
    "apple": {
        "id": "showcase-apple-sec",
        "label": "US public company success",
        "query": "Analyze Apple FY2025 annual profitability, operating margin, and R&D intensity using live fundamentals only.",
        "expect_companies": ["Apple"],
        "expect": "completed_live",
    },
    "tsmc": {
        "id": "showcase-tsmc-yahoo",
        "label": "Non-US live fallback",
        "query": "Analyze TSMC FY2025 annual profitability and R&D intensity using live fundamentals only.",
        "expect_companies": ["TSMC"],
        "expect": "completed_live_or_incomplete",
    },
    "openai": {
        "id": "showcase-openai-failclosed",
        "label": "Private company fail-closed",
        "query": (
            "Analyze OpenAI FY2025 annual profitability, operating margin, and R&D intensity using live "
            "fundamentals only. Do not use estimates if source financial statements are unavailable."
        ),
        "expect_companies": ["OpenAI"],
        "expect": "incomplete_required",
    },
    "peer": {
        "id": "showcase-peer-aapl-msft",
        "label": "Two-company comparison",
        "query": "Compare Apple and Microsoft FY2025 annual operating margins and R&D intensity using live fundamentals only.",
        "expect_companies": ["Apple", "Microsoft"],
        "expect": "completed_live",
    },
}


@dataclass
class ShowcaseResult:
    id: str
    label: str
    ok: bool = False
    workflow_status: str = ""
    companies: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    gate_passed: bool | None = None
    gate_score: float | None = None
    notes: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    state_path: str = ""
    report_path: str = ""
    finrun_path: str = ""
    eval_report_path: str = ""
    error: str = ""


def _demo_env() -> dict[str, str]:
    env = os.environ.copy()
    env["DATA_MODE"] = "live"
    env["ALLOW_LOCAL_FALLBACK"] = "false"
    env.setdefault("APP_ENV", "dev")
    env.setdefault("PYTHONUTF8", "1")
    # Fill missing RAG keys with showcase profile; explicit .env / shell wins.
    apply_showcase_rag_env(env, overwrite=False)
    return env


def _run(cmd: list[str], cwd: Path, *, timeout: int = 700) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=_demo_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _latest_state(thread_id: str) -> Path | None:
    matches = sorted(OUT.glob(f"{thread_id}_*_state.json"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _artifact_path(state_path: Path, suffix: str) -> Path:
    return state_path.with_name(state_path.name.replace("_state.json", suffix))


def _evaluate_with_bench(state_path: Path, case_id: str) -> tuple[bool | None, float | None, Path | None, str]:
    if not FAB.exists() or not GENERIC_CASE.exists():
        return None, None, None, f"FinAgentBench not found at {FAB}"

    finrun = _artifact_path(state_path, "-finrun.json")
    exp = _run(
        [sys.executable, str(ROOT / "scripts" / "export_finrun.py"), str(state_path), "--out", str(finrun)],
        cwd=ROOT,
    )
    if exp.returncode != 0:
        return None, None, None, (exp.stderr or exp.stdout)[-1000:]

    eval_dir = FAB / "outputs" / "live_showcase" / case_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    ev = _run(
        [
            sys.executable,
            "-m",
            "finagentbench",
            "evaluate",
            str(finrun),
            "--case",
            str(GENERIC_CASE),
            "--profile",
            "ci",
            "--out",
            str(eval_dir),
        ],
        cwd=FAB,
    )
    reports = sorted(eval_dir.glob("*_eval_report.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        return None, None, None, (ev.stderr or ev.stdout)[-1000:]
    payload = json.loads(reports[-1].read_text(encoding="utf-8"))
    return bool(payload.get("passed")), float(payload.get("score") or 0), reports[-1], ""


def _judge(case: dict[str, Any], result: ShowcaseResult) -> None:
    expected = set(case["expect_companies"])
    got = set(result.companies)
    sources = set(result.sources.values())
    expect = case["expect"]

    if expect == "incomplete_required":
        result.ok = (
            result.workflow_status == "incomplete_data"
            and expected.issubset(got)
            and all(result.sources.get(c) == "none" for c in expected)
            and result.gate_passed is False
        )
        if result.ok:
            result.notes.append("expected fail-closed: no public structured fundamentals")
        return

    if expect == "completed_live_or_incomplete" and result.workflow_status == "incomplete_data":
        result.ok = result.gate_passed is False
        result.notes.append("accepted fail-closed for unavailable live fundamentals")
        return

    live_sources = {"sec_companyfacts", "yahoo_fundamentals", "document_upload", "uploaded_json", "uploaded_csv"}
    result.ok = (
        result.workflow_status == "completed"
        and expected.issubset(got)
        and bool(sources)
        and sources <= live_sources
        and result.gate_passed is True
    )


def run_case(case: dict[str, Any], *, bench: bool) -> ShowcaseResult:
    result = ShowcaseResult(id=case["id"], label=case["label"])
    print(f"\n=== {case['id']} | {case['label']} ===", flush=True)
    started = time.perf_counter()
    proc = _run(
        [
            sys.executable,
            "run_demo.py",
            "--query",
            case["query"],
            "--thread-id",
            case["id"],
            "--output-dir",
            str(OUT),
        ],
        cwd=ROOT,
    )
    result.elapsed_sec = round(time.perf_counter() - started, 2)
    if proc.returncode != 0:
        result.error = (proc.stderr or proc.stdout)[-1500:]
        print(f"  ERROR run_demo rc={proc.returncode}", flush=True)
        return result

    state_path = _latest_state(case["id"])
    if state_path is None:
        result.error = "state file missing"
        return result

    state = json.loads(state_path.read_text(encoding="utf-8"))
    result.state_path = str(state_path)
    result.report_path = str(_artifact_path(state_path, "_report.md"))
    result.workflow_status = str(state.get("workflow_status") or "")
    result.companies = [str(c) for c in state.get("companies") or []]
    docs = state.get("retrieved_docs") or {}
    result.sources = {
        company: str((docs.get(company) or {}).get("structured_source") or "none")
        for company in result.companies
    }

    if bench:
        gate, score, eval_report, err = _evaluate_with_bench(state_path, case["id"])
        result.gate_passed = gate
        result.gate_score = score
        result.finrun_path = str(_artifact_path(state_path, "-finrun.json"))
        result.eval_report_path = str(eval_report) if eval_report else ""
        if err:
            result.notes.append(f"bench_error={err[:240]}")
    else:
        result.notes.append("bench skipped by --no-bench")

    _judge(case, result)
    print(
        f"  ok={result.ok} status={result.workflow_status} companies={result.companies} "
        f"sources={result.sources} gate={result.gate_passed} score={result.gate_score} "
        f"t={result.elapsed_sec}s",
        flush=True,
    )
    if result.notes:
        for note in result.notes:
            print(f"  note: {note}", flush=True)
    print(f"  report={result.report_path}", flush=True)
    if result.eval_report_path:
        print(f"  bench={result.eval_report_path}", flush=True)
    return result


def _selected_cases(selection: str) -> list[dict[str, Any]]:
    if selection == "all":
        return [CASES["apple"], CASES["tsmc"], CASES["openai"]]
    return [CASES[selection]]


def main() -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Run a short real-API LumenFin + FinAgentBench showcase.")
    parser.add_argument(
        "--case",
        choices=["all", *CASES.keys()],
        default="all",
        help="Which showcase case to run. Default: all = apple + tsmc + openai.",
    )
    parser.add_argument("--no-bench", action="store_true", help="Run LumenFin only; skip FinAgentBench evaluation.")
    parser.add_argument("--list", action="store_true", help="Print available cases and exit.")
    args = parser.parse_args()

    if args.list:
        for key, case in CASES.items():
            print(f"{key:7s} {case['id']:34s} {case['label']}")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    print("Live showcase settings:", flush=True)
    print("  DATA_MODE=live", flush=True)
    print("  ALLOW_LOCAL_FALLBACK=false", flush=True)
    print(f"  output={OUT}", flush=True)
    print(f"  finagentbench={FAB}", flush=True)

    results = [run_case(case, bench=not args.no_bench) for case in _selected_cases(args.case)]
    passed = sum(1 for result in results if result.ok)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / max(len(results), 1), 3),
        "bench_enabled": not args.no_bench,
        "cases": [asdict(result) for result in results],
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== SHOWCASE SUMMARY ===", flush=True)
    print(f"passed={passed}/{len(results)} pass_rate={payload['pass_rate']}", flush=True)
    print(f"report={REPORT}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
