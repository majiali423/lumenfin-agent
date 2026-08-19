#!/usr/bin/env python3
"""Validate and publish the LEDGER parent-pack probe counts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_ledger_public_dev_parent_pack_probe as probe_cli
import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_parent_probe import SCHEMA_VERSION
from lumenfin.stdio import configure_stdio_utf8

SEALED_OUTPUT = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_parent_pack_probe_5x10.json"
).resolve()


def _load_json_bytes(path: Path, *, label: str) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutError(f"cannot read {label}") from exc
    if not isinstance(payload, dict):
        raise HoldoutError(f"{label} must be an object")
    return payload, raw


def _script_sha256() -> str:
    return hashlib.sha256(
        (ROOT / "scripts" / "run_ledger_public_dev_parent_pack_probe.py")
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .encode()
    ).hexdigest()


def _validate(*, aggregate_path: Path, per_case_path: Path) -> bytes:
    aggregate, raw = _load_json_bytes(aggregate_path, label="parent probe aggregate")
    try:
        per_case_raw = per_case_path.read_bytes()
        rows = paired_cli._parse_jsonl_text(
            per_case_raw.decode("utf-8"),
            label="parent probe per-case",
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise HoldoutError("cannot read parent probe per-case") from exc
    expected_recovered = {
        strategy: sum(bool(row["recovered"][strategy]) for row in rows)
        for strategy in (
            "chunk_final",
            "retrieved_page_full",
            "retrieved_page_window_1",
            "gold_page_full",
        )
    }
    serialized = raw.decode("utf-8")
    if (
        aggregate.get("schema_version") != SCHEMA_VERSION
        or aggregate.get("probe_source_sha256") != _script_sha256()
        or aggregate.get("cases") != 50
        or len(rows) != 50
        or aggregate.get("recovered_cases") != expected_recovered
        or aggregate.get("recovered_by_leak_class")
        != probe_cli._leak_recovery(rows)
        or aggregate.get("recommended_next_workstream")
        != probe_cli.recommend_next(rows)
        or aggregate.get("remote_calls") != 0
        or aggregate.get("product_accuracy_claim") is not False
        or aggregate.get("financebench_phase4") != "NOT_RUN"
        or aggregate.get("per_case_sha256")
        != hashlib.sha256(per_case_raw).hexdigest()
        or '"query_text"' in serialized
        or '"mmd_text"' in serialized
    ):
        raise HoldoutError("parent pack probe artifact failed sealed validation")
    for row in rows:
        if row.get("row_sha256") != probe_cli._row_sha256(row):
            raise HoldoutError("parent probe per-case identity failed sealed validation")
    return raw


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--per-case", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).expanduser().resolve()
        if output != SEALED_OUTPUT:
            raise HoldoutError("sealed output path is not the fixed tracked path")
        if output.exists():
            raise HoldoutError("refusing to overwrite sealed parent probe artifact")
        raw = _validate(
            aggregate_path=Path(args.aggregate).expanduser().resolve(),
            per_case_path=Path(args.per_case).expanduser().resolve(),
        )
        tmp = output.with_name(output.name + ".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, output)
        print(f"sealed: {output}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - redact CLI boundary failures
        safe = (
            exc
            if isinstance(exc, HoldoutError)
            else HoldoutError(f"parent probe seal failed: {type(exc).__name__}")
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
