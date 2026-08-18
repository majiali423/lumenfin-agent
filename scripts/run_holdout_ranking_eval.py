#!/usr/bin/env python3
"""Validate a new holdout dataset; scoring and remote calls are disabled."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout import (
    HOLDOUT_SPLIT,
    HoldoutError,
    holdout_file_sha256,
    load_holdout_questions,
    resolve_holdout_questions_path,
    validate_holdout_request,
)
from lumenfin.stdio import configure_stdio_utf8

DEFAULT_DATASET_DIR = ROOT / "data" / "eval_rag" / "holdout"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the unseen-holdout ranking schema. Scoring is not enabled; "
            "FinanceBench splits and remote calls are refused."
        )
    )
    parser.add_argument("--split", default=HOLDOUT_SPLIT)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--questions-path",
        default=None,
        help="Defaults to <dataset-dir>/schema_example.jsonl and cannot escape dataset-dir.",
    )
    parser.add_argument("--allow-remote", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        dataset_dir = validate_holdout_request(
            split=args.split,
            dataset_dir=args.dataset_dir,
            repo_root=ROOT,
            allow_remote=bool(args.allow_remote),
        )
        questions_path = resolve_holdout_questions_path(
            dataset_dir,
            args.questions_path,
        )
        rows = load_holdout_questions(questions_path)
        dataset_hash = holdout_file_sha256(questions_path)
    except HoldoutError as exc:
        print(f"blocked: {exc}", flush=True)
        return 2

    print(
        "[holdout-ranking] VALIDATE_OK "
        f"split={HOLDOUT_SPLIT} questions={len(rows)} "
        f"dataset_sha256={dataset_hash} scoring=NOT_ENABLED remote_calls=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
