#!/usr/bin/env python3
"""Synthetic remote alias-compliance canary CLI.

Independent of LEDGER, FinanceBench, and public_holdout. This phase keeps
official preflight and remote execution default-deny. Formal future runs
require both --confirm-synthetic-alias-compliance and --allow-remote.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.synthetic_alias_compliance import (
    CanaryError,
    NetworkProbe,
    load_frozen_config,
    parse_cli_guard,
    run_canary,
)
from lumenfin.stdio import configure_stdio_utf8
from lumenfin.structured_answer import redact_structured_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic remote alias-compliance canary. "
            "Not product accuracy, not LEDGER/FinanceBench, not rc5."
        )
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--confirm-synthetic-alias-compliance",
        action="store_true",
        help="Required acknowledgement for a future official remote run",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Required together with --confirm-synthetic-alias-compliance.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    probe = NetworkProbe()
    probe.install()
    try:
        parse_cli_guard(argv)
        parser = build_parser()
        args = parser.parse_args(argv)
        config = load_frozen_config(
            ROOT / "data" / "eval_rag" / "synthetic_alias_compliance_config.json",
            repo_root=ROOT,
            require_published=True,
        )
        run_canary(
            repo_root=ROOT,
            frozen_config=config,
            confirm_synthetic_alias_compliance=bool(args.confirm_synthetic_alias_compliance),
            allow_remote=bool(args.allow_remote),
            preflight_only=bool(args.preflight_only),
            resume=bool(args.resume),
            official=True,
        )
        return 0
    except CanaryError as exc:
        print(redact_structured_error(str(exc)), file=sys.stderr)
        return 2
    finally:
        probe.remove()
        if probe.remote_request_count:
            print("synthetic alias compliance made a remote call", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
