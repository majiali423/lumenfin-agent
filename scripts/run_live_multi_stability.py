"""Multi-company live full-chain stability check (SEC/Yahoo → LumenFin → FinAgentBench)."""
from __future__ import annotations

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

from lumenfin.stdio import configure_stdio_utf8

FAB = Path(r"C:\a_project\Projects\finagentbench-demo")
GENERIC_CASE = FAB / "fixtures/case_lumenfin_generic.json"
OUT = ROOT / "outputs" / "e2e_live_multi"
REPORT = OUT / "live_multi_stability_report.json"

# Mix: mega-cap, semis, software, consumer, one non-US fallback, and one private-company fail-closed case.
CASES: list[dict[str, Any]] = [
    {
        "id": "live-nvda",
        "query": "Analyze NVIDIA FY2025/FY2026 annual profitability, operating margin, and R&D intensity.",
        "expect_companies": ["NVIDIA"],
        "expect_source": "sec_companyfacts",
    },
    {
        "id": "live-aapl",
        "query": "Analyze Apple FY2025 annual profitability, operating margin, and R&D intensity.",
        "expect_companies": ["Apple"],
        "expect_source": "sec_companyfacts",
    },
    {
        "id": "live-msft",
        "query": "Analyze Microsoft FY2025 annual profitability and R&D intensity with a short risk note.",
        "expect_companies": ["Microsoft"],
        "expect_source": "sec_companyfacts",
    },
    {
        "id": "live-amd",
        "query": "Analyze AMD FY2025 annual operating profitability and R&D intensity.",
        "expect_companies": ["AMD"],
        "expect_source": "sec_companyfacts",
    },
    {
        "id": "live-meta",
        "query": "Analyze Meta Platforms FY2025 annual profitability and R&D intensity.",
        "expect_companies": ["Meta"],
        "expect_source": "sec_companyfacts",
    },
    {
        "id": "live-amzn",
        "query": "Analyze Amazon FY2025 annual profitability and R&D intensity.",
        "expect_companies": ["Amazon"],
        "expect_source": "sec_companyfacts",
    },
    {
        "id": "live-peer-aapl-msft",
        "query": (
            "Compare Apple and Microsoft FY2025 annual operating margins and R&D intensity; "
            "rank which looks stronger on profitability."
        ),
        "expect_companies": ["Apple", "Microsoft"],
        "expect_source": "sec_companyfacts",
    },
    {
        "id": "live-peer-nvda-amd",
        "query": (
            "Compare NVIDIA and AMD FY2025/FY2026 annual operating margins and R&D intensity; "
            "note supply-chain concentration risk briefly."
        ),
        "expect_companies": ["NVIDIA", "AMD"],
        "expect_source": "sec_companyfacts",
    },
    {
        "id": "live-tsmc-fallback",
        "query": "Analyze TSMC FY2025 annual profitability and R&D intensity.",
        "expect_companies": ["TSMC"],
        # TSMC may miss SEC (non-US). Accept yahoo_fundamentals or incomplete_data.
        "expect_source": "any_live_or_incomplete",
    },
    {
        "id": "live-openai-private-negative",
        "query": (
            "Analyze OpenAI FY2025 annual profitability, operating margin, and R&D intensity using live "
            "fundamentals only. Do not use estimates if source financial statements are unavailable."
        ),
        "expect_companies": ["OpenAI"],
        # Private-company financial statements are not public structured fundamentals; require fail-closed.
        "expect_source": "incomplete_required",
    },
]


@dataclass
class CaseResult:
    id: str
    ok: bool = False
    workflow_status: str = ""
    companies: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    gate_passed: bool | None = None
    gate_score: float | None = None
    gate_failures: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    notes: list[str] = field(default_factory=list)
    error: str = ""
    state_path: str = ""


def _run(cmd: list[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("MAS_MILVUS_ISOLATE", "true")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def _latest_state(thread_id: str) -> Path | None:
    matches = sorted(OUT.glob(f"{thread_id}_*_state.json"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _gate(state_path: Path, out_name: str) -> tuple[bool | None, float | None, list[str], str]:
    finrun = state_path.with_name(state_path.name.replace("_state.json", f"-{out_name}-finrun.json"))
    exp = _run(
        [sys.executable, str(ROOT / "scripts/export_finrun.py"), str(state_path), "--out", str(finrun)],
        cwd=ROOT,
    )
    if exp.returncode != 0:
        return None, None, [], (exp.stderr or exp.stdout)[-800:]
    eval_dir = FAB / "outputs" / "e2e_live_multi" / out_name
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
    reports = list(eval_dir.glob("*_eval_report.json"))
    if not reports:
        return None, None, [], (ev.stderr or ev.stdout)[-800:]
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    fails = [
        f"{m['name']}: {[x.get('message') for x in m.get('findings', [])][:2]}"
        for m in report.get("metrics", [])
        if not m.get("passed")
    ]
    return bool(report.get("passed")), float(report.get("score") or 0), fails, ""


def _judge(case: dict[str, Any], result: CaseResult, state: dict[str, Any]) -> None:
    expected = set(case["expect_companies"])
    got = set(result.companies)
    source_mode = case["expect_source"]
    sources = set(result.sources.values())

    if result.workflow_status == "needs_clarification":
        result.notes.append("unexpected_hitl")
        result.ok = False
        return

    if source_mode == "incomplete_required":
        result.ok = (
            result.workflow_status == "incomplete_data"
            and expected.issubset(got)
            and all(result.sources.get(c) == "none" for c in expected if c in got)
            and result.gate_passed is False
        )
        if result.ok:
            result.notes.append("fail_closed_ok_for_unavailable_private_company_fundamentals")
        return

    if source_mode == "any_live_or_incomplete":
        if result.workflow_status == "incomplete_data":
            result.ok = result.gate_passed is False
            result.notes.append("incomplete_ok_for_non_us_or_missing_fundamentals")
            return
        ok_source = sources <= {"sec_companyfacts", "yahoo_fundamentals"} and bool(sources)
        result.ok = (
            result.workflow_status == "completed"
            and expected.issubset(got)
            and ok_source
            and bool(result.gate_passed)
        )
        return

    ok_companies = expected.issubset(got)
    ok_source = all(result.sources.get(c) == source_mode for c in expected if c in result.sources)
    # If SEC miss for a US name, Yahoo is acceptable but annotated.
    if not ok_source and all(
        result.sources.get(c) in {"sec_companyfacts", "yahoo_fundamentals"} for c in expected if c in result.sources
    ):
        result.notes.append(f"source_fallback_used={result.sources}")
        ok_source = True
    has_metrics = all(
        bool((state.get("financial_metrics") or {}).get(c)) for c in expected if c in got
    )
    result.ok = (
        result.workflow_status == "completed"
        and ok_companies
        and ok_source
        and has_metrics
        and bool(result.gate_passed)
    )


def run_one(case: dict[str, Any]) -> CaseResult:
    result = CaseResult(id=case["id"])
    print(f"\n=== {case['id']} ===", flush=True)
    t0 = time.perf_counter()
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
        timeout=700,
    )
    result.elapsed_sec = round(time.perf_counter() - t0, 2)
    if proc.returncode != 0:
        result.error = (proc.stderr or proc.stdout)[-1500:]
        print(f"  FAIL rc={proc.returncode}", flush=True)
        return result
    state_path = _latest_state(case["id"])
    if not state_path:
        result.error = "state missing"
        return result
    state = json.loads(state_path.read_text(encoding="utf-8"))
    result.state_path = str(state_path)
    result.workflow_status = str(state.get("workflow_status") or "")
    result.companies = list(state.get("companies") or [])
    docs = state.get("retrieved_docs") or {}
    result.sources = {
        str(c): str((docs.get(c) or {}).get("structured_source") or "none") for c in result.companies
    }
    result.metrics = {
        str(c): {
            k: float(v)
            for k, v in ((docs.get(c) or {}).get("market_data") or {}).items()
            if isinstance(v, (int, float))
        }
        for c in result.companies
    }
    if result.workflow_status in {"completed", "incomplete_data"}:
        passed, score, fails, err = _gate(state_path, case["id"])
        result.gate_passed = passed
        result.gate_score = score
        result.gate_failures = fails
        if err:
            result.notes.append(f"gate_err={err[:200]}")
    _judge(case, result, state)
    print(
        f"  ok={result.ok} status={result.workflow_status} companies={result.companies} "
        f"sources={result.sources} gate={result.gate_passed} score={result.gate_score} "
        f"t={result.elapsed_sec}s",
        flush=True,
    )
    for note in result.notes:
        print(f"  note: {note}", flush=True)
    if not result.ok and result.gate_failures:
        print(f"  gate_fails: {result.gate_failures[:3]}", flush=True)
    # Be kind to SEC rate limits between cases.
    time.sleep(1.5)
    return result


def main() -> int:
    configure_stdio_utf8()
    OUT.mkdir(parents=True, exist_ok=True)
    results = [run_one(case) for case in CASES]
    passed = sum(1 for r in results if r.ok)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / max(len(results), 1), 3),
        "cases": [asdict(r) for r in results],
    }
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== LIVE MULTI SUMMARY ===", flush=True)
    print(f"passed={passed}/{len(results)} pass_rate={payload['pass_rate']}", flush=True)
    print(f"report={REPORT}", flush=True)
    for r in results:
        if not r.ok:
            print(f" - FAIL {r.id}: status={r.workflow_status} sources={r.sources} err={r.error[:160]}", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
