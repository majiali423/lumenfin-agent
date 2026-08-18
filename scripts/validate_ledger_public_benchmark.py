#!/usr/bin/env python3
"""Validate a pinned local LEDGER snapshot and build a split manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout import (
    HoldoutError,
    build_ledger_split_manifest,
    iter_ledger_parquet_rows,
    ledger_snapshot_sha256,
)
from lumenfin.stdio import configure_stdio_utf8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a pinned LEDGER parquet snapshot and create a company-disjoint "
            "public_dev/public_holdout manifest. No retrieval or remote calls."
        )
    )
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--allow-remote", action="store_true")
    return parser


def _write_new_manifest(path: Path, payload: dict) -> None:
    target = path.expanduser().resolve()
    if target.exists():
        raise HoldoutError(f"refusing to overwrite existing manifest: {target}")
    if not target.parent.is_dir():
        raise HoldoutError(f"manifest parent directory not found: {target.parent}")
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, target)


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        if args.allow_remote:
            raise HoldoutError(
                "LEDGER validation is offline-only; refusing --allow-remote"
            )
        snapshot_hash = ledger_snapshot_sha256(args.parquet_path)
        manifest = build_ledger_split_manifest(
            iter_ledger_parquet_rows(args.parquet_path),
            source_revision=args.source_revision,
            source_artifact_sha256=snapshot_hash,
            salt=args.split_salt,
            holdout_fraction=float(args.holdout_fraction),
        )
        if args.output_manifest:
            _write_new_manifest(Path(args.output_manifest), manifest)
    except (HoldoutError, ValueError) as exc:
        print(f"blocked: {exc}", flush=True)
        return 2
    print(
        "[ledger-public-benchmark] VALIDATE_OK "
        f"rows={manifest['rows']} "
        f"dev={manifest['splits']['public_dev']['queries']} "
        f"holdout={manifest['splits']['public_holdout']['queries']} "
        "scoring=NOT_ENABLED remote_calls=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
