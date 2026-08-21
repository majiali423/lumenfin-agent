#!/usr/bin/env python3
"""Offline synthetic structured-citation contract canary.

Not product accuracy, RAG recall, FinanceBench, or a LEDGER benchmark.
Refuses public_holdout, remote providers, and non-empty output overwrite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.structured_citation_canary import (
    DEFAULT_RAW_OUTPUT_DIR,
    CanaryError,
    canonical_config,
    parse_cli_guard,
    run_canary,
)
from lumenfin.stdio import configure_stdio_utf8
from lumenfin.structured_answer import redact_structured_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline synthetic structured-citation contract canary. "
            "This is not a product-accuracy or LEDGER/FinanceBench score."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RAW_OUTPUT_DIR),
        help="Empty gitignored raw output directory",
    )
    parser.add_argument(
        "--seal-path",
        default="",
        help="Optional slim tracked result path (refuses overwrite)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty worktree for local development runs",
    )
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args_list = list(sys.argv[1:] if argv is None else argv)
    try:
        parse_cli_guard(args_list)
        parser = build_parser()
        args = parser.parse_args(args_list)
        if args.allow_remote:
            raise CanaryError("structured citation canary refuses --allow-remote")
        seal = args.seal_path.strip() or None
        result = run_canary(
            output_dir=args.output_dir,
            repo_root=ROOT,
            require_clean_worktree=not args.allow_dirty,
            seal_path=seal,
            config=canonical_config(),
        )
        print(
            "structured citation canary passed "
            f"(cases_failed={result['metrics']['cases_failed']}, "
            f"remote_request_count={result['remote_request_count']}, "
            f"config_hash={result['config_hash'][:12]})"
        )
        print(
            "synthetic contract canary only; not product accuracy; "
            "not LEDGER/FinanceBench; rc5 unpublished"
        )
        return 0
    except CanaryError as exc:
        print(f"structured citation canary failed: {redact_structured_error(str(exc))}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"structured citation canary failed: {redact_structured_error(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
