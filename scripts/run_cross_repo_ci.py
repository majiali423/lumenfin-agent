#!/usr/bin/env python3
"""LumenFin-owned cross-repository CI orchestrator.

Generates a real sample FinRun via ``export_finrun_state``, then invokes the
pinned FinAgentBench gate and mutation suite with absolute paths. Does not
change FinAgentBench evaluators or thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repo_paths import finagentbench_root, lumenfin_root  # noqa: E402

_EXPECTED_MUTATIONS = ("wrong_number", "wrong_entity", "missing_citation", "missing_risk")


class CrossRepoCiError(RuntimeError):
    """Fail-fast orchestration error with non-sensitive path details."""


def _resolve_path(raw: Path) -> Path:
    return raw.expanduser().resolve()


def _git_revision(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unavailable"


def _fab_package_version(fab: Path) -> str:
    pyproject = fab / "pyproject.toml"
    if not pyproject.is_file():
        return "unavailable"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("version"):
            # version = "0.1.0rc2"
            parts = line.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip("\"'")
    try:
        from importlib import metadata

        return metadata.version("finagentbench")
    except Exception:  # noqa: BLE001 - best-effort CI metadata only
        return "unavailable"


def _case_hash(case_path: Path) -> str:
    if not case_path.is_file():
        return "unavailable"
    try:
        fab_root = case_path.resolve().parents[1]
        fab_root_str = str(fab_root)
        if fab_root_str not in sys.path:
            sys.path.insert(0, fab_root_str)
        from finagentbench.provenance import case_hash

        payload = json.loads(case_path.read_text(encoding="utf-8"))
        return case_hash(payload)
    except Exception as exc:  # noqa: BLE001 - report metadata, do not fail gate
        return f"unavailable:{exc.__class__.__name__}"


def _git_porcelain(repo: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise CrossRepoCiError(f"git status failed in {repo}")
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _is_dependency_noise(line: str, *, dependency_names: set[str]) -> bool:
    """Ignore an explicitly checked-out sibling dependency nested by mistake."""
    path = line[3:].strip() if len(line) >= 4 else line.strip()
    # Untracked dirs look like "?? finagentbench-demo/"
    name = path.rstrip("/").split("/", 1)[0]
    return name in dependency_names


def lumenfin_unexpected_dirty(
    lumen: Path,
    *,
    dependency_names: set[str] | None = None,
) -> list[str]:
    dependency_names = dependency_names or {"finagentbench-demo"}
    return [
        line
        for line in _git_porcelain(lumen)
        if not _is_dependency_noise(line, dependency_names=dependency_names)
    ]


def require_finrun_file(path: Path) -> dict:
    if not path.is_file():
        raise CrossRepoCiError(
            f"FinRun missing before gate. expected_file={path} exists=False"
        )
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise CrossRepoCiError(f"FinRun empty before gate. expected_file={path}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CrossRepoCiError(
            f"FinRun JSON invalid before gate. expected_file={path} error={exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CrossRepoCiError(f"FinRun schema invalid: root must be object at {path}")
    if not payload.get("schema_version"):
        raise CrossRepoCiError(
            f"FinRun schema invalid: missing schema_version at {path}"
        )
    return payload


def require_mutation_report(path: Path) -> dict:
    if not path.is_file():
        raise CrossRepoCiError(
            f"Mutation report missing after suite. expected_file={path}"
        )
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise CrossRepoCiError(f"Mutation report empty. expected_file={path}")
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CrossRepoCiError(
            f"Mutation report JSON invalid. expected_file={path} error={exc}"
        ) from exc
    items = report.get("items")
    if not isinstance(items, list) or not items:
        raise CrossRepoCiError(
            f"Mutation report missing items evidence. expected_file={path}"
        )
    results = {
        item["failure_type"]: (
            not item["actual_passed"] and not item["missing_expected_findings"]
        )
        for item in items
        if item.get("failure_type") and item.get("failure_type") != "none"
    }
    missing = [name for name in _EXPECTED_MUTATIONS if name not in results]
    if missing:
        raise CrossRepoCiError(
            f"Mutation report incomplete; missing failure_type={missing} "
            f"expected_file={path}"
        )
    if not all(results[name] for name in _EXPECTED_MUTATIONS):
        raise CrossRepoCiError(
            f"Mutation gate incomplete detection; results={results} "
            f"expected_file={path}"
        )
    return report


def generate_sample_finrun(*, lumen: Path, fab: Path, finrun_path: Path) -> dict:
    state_path = fab / "fixtures" / "lumenfin_state_sample.json"
    if not state_path.is_file():
        raise CrossRepoCiError(f"sample LumenFin state missing at {state_path}")
    sys.path.insert(0, str(lumen / "src"))
    from lumenfin.finrun import export_finrun_state

    state = json.loads(state_path.read_text(encoding="utf-8"))
    finrun = export_finrun_state(state)
    finrun_path.parent.mkdir(parents=True, exist_ok=True)
    finrun_path.write_text(json.dumps(finrun, indent=2), encoding="utf-8")
    return require_finrun_file(finrun_path)


def run_gate(*, fab: Path, finrun_path: Path, case_path: Path, profile: str, gate_out: Path) -> int:
    gate_out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "finagentbench",
            "gate",
            str(finrun_path),
            "--case",
            str(case_path),
            "--profile",
            profile,
            "--out",
            str(gate_out),
        ],
        cwd=str(fab),
        check=False,
    )
    return proc.returncode


def run_mutation_suite(*, fab: Path, report_path: Path) -> int:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            str(fab / "scripts" / "run_mutation_suite.py"),
            "--out",
            str(report_path),
        ],
        cwd=str(fab),
        check=False,
    )
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lumenfin-root",
        type=Path,
        default=None,
        help="Absolute/relative LumenFin root (default: discovery / this repo)",
    )
    parser.add_argument(
        "--finagentbench-root",
        type=Path,
        default=None,
        help="Absolute/relative FinAgentBench root (default: FINAGENTBENCH_DIR / sibling)",
    )
    parser.add_argument("--profile", default="ci", choices=("ci", "audit", "default"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Absolute output directory for FinRun + reports",
    )
    parser.add_argument(
        "--require-clean-lumenfin",
        action="store_true",
        help="Fail if LumenFin has unexpected dirty tracked/untracked files",
    )
    args = parser.parse_args(argv)

    lumen = _resolve_path(args.lumenfin_root) if args.lumenfin_root else lumenfin_root()
    fab = (
        _resolve_path(args.finagentbench_root)
        if args.finagentbench_root
        else finagentbench_root()
    )

    out_dir = _resolve_path(
        args.out_dir
        if args.out_dir is not None
        else Path(os.environ.get("CROSS_REPO_OUT_DIR", lumen / "outputs" / "cross_repo_validation"))
    )
    finrun_path = out_dir / "sample_finrun.json"
    mutation_path = out_dir / "mutation_detection_report.json"
    gate_dir = out_dir / "gate"
    case_path = fab / "fixtures" / "case_lumenfin_diligence.json"

    print(f"LUMENFIN_ROOT={lumen}", flush=True)
    print(f"FAB_ROOT={fab}", flush=True)
    print(f"FINRUN_PATH={finrun_path}", flush=True)
    print(f"OUT_DIR={out_dir}", flush=True)
    print(f"cwd={Path.cwd()}", flush=True)

    if not (lumen / "src" / "lumenfin").is_dir():
        raise SystemExit(f"LumenFin root invalid: {lumen}")
    if not (fab / "finagentbench").is_dir():
        raise SystemExit(f"FinAgentBench root invalid: {fab}")
    if not case_path.is_file():
        raise SystemExit(f"FinAgentBench case missing: {case_path}")

    try:
        dirty = lumenfin_unexpected_dirty(lumen)
        if args.require_clean_lumenfin and dirty:
            raise CrossRepoCiError(
                "LumenFin unexpected dirty files before cross-repo gate: "
                + "; ".join(dirty[:20])
            )

        finrun = generate_sample_finrun(lumen=lumen, fab=fab, finrun_path=finrun_path)
        print(
            f"sample_finrun generated path={finrun_path} "
            f"schema_version={finrun.get('schema_version')} bytes={finrun_path.stat().st_size}",
            flush=True,
        )

        gate_rc = run_gate(
            fab=fab,
            finrun_path=finrun_path,
            case_path=case_path,
            profile=args.profile,
            gate_out=gate_dir,
        )
        mutation_rc = run_mutation_suite(fab=fab, report_path=mutation_path)
        mutation_report = require_mutation_report(mutation_path)
        mutation_results = {
            item["failure_type"]: (
                not item["actual_passed"] and not item["missing_expected_findings"]
            )
            for item in mutation_report.get("items", [])
            if item.get("failure_type") != "none"
        }
    except CrossRepoCiError as exc:
        print(f"CROSS_REPO_CI_FAIL: {exc}", flush=True)
        return 2

    summary = {
        "lumenfin_root": str(lumen),
        "finagentbench_root": str(fab),
        "lumenfin_commit": _git_revision(lumen),
        "finagentbench_requested_ref": os.environ.get("FINAGENTBENCH_REF", ""),
        "finagentbench_commit": _git_revision(fab),
        "finagentbench_package_version": _fab_package_version(fab),
        "lumenfin_unexpected_dirty": dirty,
        "lumenfin_worktree_dirty": bool(dirty),
        "finagentbench_worktree_dirty": bool(_git_porcelain(fab)),
        "finrun_schema_version": finrun.get("schema_version", "legacy-0"),
        "case_path": str(case_path),
        "case_hash": _case_hash(case_path),
        "benchmark_profile": args.profile,
        "sample_finrun": str(finrun_path),
        "finrun_bytes": finrun_path.stat().st_size,
        "finagentbench_gate_passed": gate_rc == 0,
        "mutation_gate_passed": mutation_rc == 0 and all(mutation_results.get(n) for n in _EXPECTED_MUTATIONS),
        "mutation_detection_rate": mutation_report.get("detection_rate"),
        "mutation_results": mutation_results,
        "mutation_report": str(mutation_path),
    }
    summary["passed"] = bool(
        summary["finagentbench_gate_passed"] and summary["mutation_gate_passed"]
    )
    summary_path = out_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
