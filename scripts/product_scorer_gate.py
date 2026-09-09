#!/usr/bin/env python3
"""Fail-closed product v3 scorer gate.

Empty or frozen FinAgentBench refs are configuration failures, not skipped
successes. Frozen rc.3/rc.4 remain the FinRun contract pins only.
``passed`` is True only when joint tests actually ran and succeeded.
``joint_tests=not_run`` is never reported as a passed product gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONTRACT_PINS = frozenset({"v0.1.0-rc.3", "v0.1.0-rc.4"})
DEFAULT_BRANCHES = frozenset({"main", "master", "HEAD"})


def classify_product_ref(ref: str | None) -> str:
    text = str(ref or "").strip()
    if not text:
        return "empty_ref"
    if text in DEFAULT_BRANCHES:
        return "invalid_ref"
    if text in FROZEN_CONTRACT_PINS:
        return "invalid_ref"
    return "compatible_ref"


def v3_scorer_present(search_paths: list[Path] | None = None) -> bool:
    if search_paths:
        for path in search_paths:
            if path is None:
                continue
            resolved = str(path)
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
    try:
        from finagentbench.metrics.visible_supported_claims import visible_supported_claims
        from finagentbench.schema import SUPPORTED_SCORING_VERSIONS
    except Exception:
        return False
    if not callable(visible_supported_claims):
        return False
    versions = {str(item) for item in SUPPORTED_SCORING_VERSIONS}
    return "3" in versions


def evaluate(
    *,
    product_ref: str | None,
    fab_root: Path | None = None,
    check_module: bool = False,
    run_joint: bool = False,
    joint_runner: Callable[[], int] | None = None,
) -> dict[str, Any]:
    status = classify_product_ref(product_ref)
    result: dict[str, Any] = {
        "status": status,
        "config_ok": False,
        "passed": False,
        "joint_tests": "not_run",
        "product_ref": str(product_ref or ""),
        "frozen_pins_are_not_v3": True,
    }
    if status != "compatible_ref":
        result["reason"] = (
            "FINAGENTBENCH_PRODUCT_REF must be a published v3 scorer commit/tag. "
            "Empty, default-branch, and frozen rc.3/rc.4 refs are configuration failures."
        )
        return result
    paths = [fab_root] if fab_root is not None else None
    if check_module or run_joint:
        if not v3_scorer_present(paths):
            result["status"] = "missing_scorer"
            result["reason"] = "visible_supported_claims v3 module is missing"
            return result
    result["config_ok"] = True
    if not run_joint:
        result["reason"] = "configuration accepted; joint tests not_run (not a passed product gate)"
        return result
    code = int(joint_runner() if joint_runner is not None else 1)
    result["joint_tests"] = "failed" if code else "passed"
    if code:
        result["status"] = "tests_failed"
        result["reason"] = "joint product tests failed"
        return result
    result["status"] = "passed"
    result["passed"] = True
    result["reason"] = "v3 product scorer gate passed"
    return result


def _run_joint_unittest() -> int:
    from run_tests import main as run_tests_main

    argv = sys.argv
    sys.argv = ["run_tests.py", "--joint-only"]
    try:
        return int(run_tests_main())
    finally:
        sys.argv = argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed v3 product scorer configuration gate.")
    parser.add_argument("--require-ref", action="store_true", help="Exit 1 when PRODUCT_REF is unusable.")
    parser.add_argument("--require-v3-module", action="store_true", help="Exit 1 when v3 scorer import fails.")
    parser.add_argument("--run-joint", action="store_true", help="Also run scripts/run_tests.py --joint-only.")
    parser.add_argument("--json", action="store_true", help="Print the evaluation payload.")
    args = parser.parse_args()
    ref = os.environ.get("FINAGENTBENCH_PRODUCT_REF")
    fab = os.environ.get("FINAGENTBENCH_DIR", "").strip()
    fab_root = Path(fab) if fab else None
    payload = evaluate(
        product_ref=ref,
        fab_root=fab_root,
        check_module=args.require_v3_module or args.run_joint,
        run_joint=args.run_joint,
        joint_runner=_run_joint_unittest if args.run_joint else None,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"product_scorer_gate status={payload['status']} "
            f"config_ok={payload['config_ok']} passed={payload['passed']} "
            f"joint={payload['joint_tests']}"
        )
        if payload.get("reason"):
            print(payload["reason"])
    if args.run_joint:
        return 0 if payload.get("passed") else 1
    if args.require_v3_module:
        return 0 if payload.get("config_ok") else 1
    if args.require_ref:
        return 0 if classify_product_ref(ref) == "compatible_ref" else 1
    return 0


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    raise SystemExit(main())
