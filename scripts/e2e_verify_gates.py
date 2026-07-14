from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAB = Path(r"C:\a_project\Projects\finagentbench-demo")
CASE = FAB / "fixtures/case_lumenfin_generic.json"


def main() -> int:
    state_dir = ROOT / "outputs/e2e_jul14"
    states = sorted(state_dir.glob("*_state.json"), key=lambda p: p.stat().st_mtime)
    print("=== STATE FIELD CHECK ===")
    for state_path in states:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        docs = state.get("retrieved_docs") or {}
        prov_ok = all(isinstance(bundle.get("provenance"), dict) for bundle in docs.values()) if docs else False
        print(f"\n{state_path.name}")
        print(f"  workflow: {state.get('workflow_status')} llm: {state.get('llm_backend')}")
        print(f"  companies: {state.get('companies')}")
        print(f"  guardrail: {state.get('input_guardrail_summary')}")
        print(f"  violations: {state.get('compliance_violations')}")
        print(f"  retrieval_provenance keys: {list((state.get('retrieval_provenance') or {}).keys())}")
        print(f"  all doc provenance: {prov_ok}")
        print(f"  rag_chunks: {sum(len(v) for v in (state.get('rag_evidence') or {}).values())}")

    print("\n=== FINAGENTBENCH GATE ===")
    eval_root = FAB / "outputs/e2e_jul14"
    eval_root.mkdir(parents=True, exist_ok=True)
    for state_path in states:
        finrun_path = state_path.with_name(state_path.name.replace("_state.json", "-finrun.json"))
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/export_finrun.py"), str(state_path), "--out", str(finrun_path)],
            check=True,
            cwd=str(ROOT),
        )
        out_dir = eval_root / state_path.stem.replace("_state", "-eval")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "finagentbench",
                "evaluate",
                str(finrun_path),
                "--case",
                str(CASE),
                "--profile",
                "ci",
                "--out",
                str(out_dir),
            ],
            cwd=str(FAB),
            capture_output=True,
            text=True,
        )
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            report = json.loads(summary_path.read_text(encoding="utf-8"))
            failed = [metric for metric in report.get("metrics", []) if not metric.get("passed")]
            print(f"\n{state_path.name} -> passed={report.get('passed')} score={report.get('score')}")
            for metric in failed:
                messages = [finding.get("message", "")[:120] for finding in metric.get("findings", [])]
                print(f"  FAIL {metric['name']}: {messages}")
        else:
            tail = (proc.stderr or proc.stdout)[-800:]
            print(f"\n{state_path.name} eval error:\n{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
