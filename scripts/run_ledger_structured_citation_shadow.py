#!/usr/bin/env python3
"""LEDGER public/dev structured-citation shadow CLI.

Exposed public/dev shadow only. Not held-out, not product accuracy, not a
LEDGER benchmark, and not rc5. Formal scoring requires both
--confirm-exposed-shadow and --allow-remote. Official live scoring binds the
verified candidate-cache prefix; it does not rebuild the cache and does not
open public_holdout. Preflight refuses remote authorization and makes no
provider calls. Official preflight writes only
outputs/ledger_structured_citation_shadow_preflight_v3/. The accepted v2
preflight cannot authorize a later execution commit. This stage does not
run official preflight or the paid public/dev shadow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.ledger_structured_citation_shadow import (
    DEFAULT_FROZEN_CONFIG_PATH,
    DEFAULT_OFFICIAL_OUTPUT_DIR,
    DEFAULT_PREFLIGHT_OUTPUT_DIR,
    CLI_SPLIT,
    ShadowError,
    load_frozen_config,
    parse_cli_guard,
    run_shadow,
)
from lumenfin.stdio import configure_stdio_utf8
from lumenfin.structured_answer import redact_structured_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LEDGER public/dev structured-citation shadow. "
            "Not held-out, not product accuracy, not a benchmark, not rc5."
        )
    )
    parser.add_argument("--split", required=True, help="Must be public-dev")
    parser.add_argument(
        "--frozen-config",
        required=True,
        help="Path to the published frozen shadow config",
    )
    parser.add_argument(
        "--confirm-exposed-shadow",
        action="store_true",
        required=True,
        help="Required acknowledgement that this is an exposed public/dev shadow",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Required for a paid public/dev run together with --confirm-exposed-shadow.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OFFICIAL_OUTPUT_DIR),
        help="Official score directory; default is fixed",
    )
    parser.add_argument(
        "--preflight-dir",
        default=str(DEFAULT_PREFLIGHT_OUTPUT_DIR),
        help="Preflight directory; separate from official scores",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    try:
        parse_cli_guard(argv)
        parser = build_parser()
        args = parser.parse_args(argv)
        if args.split != CLI_SPLIT:
            raise ShadowError("structured citation shadow only allows --split public-dev")
        require_published = Path(args.frozen_config) == DEFAULT_FROZEN_CONFIG_PATH or Path(
            args.frozen_config
        ).as_posix().endswith(DEFAULT_FROZEN_CONFIG_PATH.as_posix())
        config = load_frozen_config(
            args.frozen_config,
            require_published=require_published,
        )
        if args.preflight_only and args.allow_remote:
            raise ShadowError("refusing --allow-remote with --preflight-only")
        if not args.preflight_only and not args.allow_remote:
            raise ShadowError("formal scoring requires --allow-remote")
        run_shadow(
            repo_root=ROOT,
            frozen_config=config,
            split=args.split,
            confirm_exposed_shadow=bool(args.confirm_exposed_shadow),
            output_dir=Path(args.output_dir),
            preflight_output_dir=Path(args.preflight_dir),
            allow_remote=bool(args.allow_remote),
            preflight_only=bool(args.preflight_only),
            resume=bool(args.resume),
            live_generate=bool(args.allow_remote) and not bool(args.preflight_only),
            strict_paths=True,
        )
        return 0
    except ShadowError as exc:
        print(redact_structured_error(str(exc)), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
