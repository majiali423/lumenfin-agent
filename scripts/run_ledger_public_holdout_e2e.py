#!/usr/bin/env python3
"""LEDGER public_holdout E2E runner. Default-deny. Does not parse holdout text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.ledger_public_holdout_e2e import (  # noqa: E402
    HoldoutE2EError,
    blocked_preflight_payload,
    load_contract,
    refuse_unauthorized,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot LEDGER public_holdout held-out E2E. "
            "Currently blocked: no compatible prebuilt index."
        )
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirm-public-holdout-consumption", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        parsed = parser.parse_args(args)
        refuse_unauthorized(repo_root=ROOT, argv=args)
    except HoldoutE2EError as exc:
        contract = load_contract(repo_root=ROOT)
        payload = blocked_preflight_payload(contract)
        payload["error"] = str(exc)
        if parsed.preflight_only if "parsed" in locals() else "--preflight-only" in args:
            payload["status"] = "PREFLIGHT_BLOCKED"
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
