#!/usr/bin/env python3
"""Validate and publish the parent-page suffix generate failure taxonomy."""
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

import run_ledger_public_dev_e2e_failure_taxonomy as tax_cli
import run_ledger_public_dev_parent_page_e2e as parent_e2e_cli
import run_ledger_public_dev_parent_page_e2e_taxonomy as parent_tax_cli
import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_parent_return import LOCKED_STRATEGY
from lumenfin.stdio import configure_stdio_utf8

SEALED_OUTPUT = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_parent_page_e2e_taxonomy_5x40.json"
).resolve()
SUFFIX_QUERY_IDS_SHA256 = (
    "7bd0906ea034b7c2a679957ac0ad82f1583934872f689c82385d5cc7a9aa9c33"
)


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
        (
            ROOT
            / "scripts"
            / "run_ledger_public_dev_parent_page_e2e_taxonomy.py"
        )
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .encode()
    ).hexdigest()


def _validate(
    *,
    aggregate_path: Path,
    per_case_path: Path,
    parent_e2e_path: Path,
) -> bytes:
    aggregate, raw = _load_json_bytes(
        aggregate_path,
        label="parent-page taxonomy aggregate",
    )
    parent_e2e, parent_raw = _load_json_bytes(
        parent_e2e_path,
        label="sealed parent-page e2e aggregate",
    )
    try:
        per_case_raw = per_case_path.read_bytes()
        rows = paired_cli._parse_jsonl_text(
            per_case_raw.decode("utf-8"),
            label="parent-page taxonomy per-case",
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise HoldoutError("cannot read parent-page taxonomy per-case") from exc
    chunk = tax_cli._summarize(rows, "chunk")
    parent = tax_cli._summarize(rows, "parent_page")
    serialized = raw.decode("utf-8")
    if (
        aggregate.get("schema_version") != parent_tax_cli.SCHEMA_VERSION
        or aggregate.get("taxonomy_source_sha256") != _script_sha256()
        or aggregate.get("cases") != 200
        or len(rows) != 200
        or aggregate.get("e2e_run_identity_sha256")
        != parent_e2e.get("run_identity_sha256")
        or aggregate.get("e2e_aggregate_sha256")
        != hashlib.sha256(parent_raw).hexdigest()
        or aggregate.get("e2e_per_case_sha256")
        != parent_e2e.get("per_case_sha256")
        or aggregate.get("selection", {}).get("query_ids_sha256")
        != SUFFIX_QUERY_IDS_SHA256
        or aggregate.get("locked_strategy") != LOCKED_STRATEGY
        or aggregate.get("comparison")
        != {"chunk": chunk, "parent_page": parent}
        or aggregate.get("recommended_next_workstream")
        != parent["recommended_next_workstream"]
        or aggregate.get("remote_calls") != 0
        or aggregate.get("qwen3_calls") != 0
        or aggregate.get("generate_calls") != 0
        or aggregate.get("product_accuracy_claim") is not False
        or aggregate.get("financebench_phase4") != "NOT_RUN"
        or aggregate.get("per_case_sha256")
        != hashlib.sha256(per_case_raw).hexdigest()
        or '"query_text"' in serialized
        or '"mmd_text"' in serialized
    ):
        raise HoldoutError(
            "parent-page taxonomy artifact failed sealed validation"
        )
    for row in rows:
        if row.get("row_sha256") != parent_tax_cli._row_sha256(row):
            raise HoldoutError(
                "parent-page taxonomy per-case identity failed sealed validation"
            )
        arms = row.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(parent_e2e_cli.ARMS):
            raise HoldoutError(
                "parent-page taxonomy arm identity failed sealed validation"
            )
    return raw


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--per-case", required=True)
    parser.add_argument("--parent-e2e-aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).expanduser().resolve()
        if output != SEALED_OUTPUT:
            raise HoldoutError("sealed output path is not the fixed tracked path")
        if output.exists():
            raise HoldoutError(
                "refusing to overwrite sealed parent-page taxonomy artifact"
            )
        parent_e2e_path = Path(args.parent_e2e_aggregate).expanduser().resolve()
        if parent_e2e_path != parent_tax_cli.TRACKED_PARENT_E2E:
            raise HoldoutError(
                "taxonomy seal must use the tracked parent-page e2e aggregate"
            )
        raw = _validate(
            aggregate_path=Path(args.aggregate).expanduser().resolve(),
            per_case_path=Path(args.per_case).expanduser().resolve(),
            parent_e2e_path=parent_e2e_path,
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
            else HoldoutError(
                f"parent-page taxonomy seal failed: {type(exc).__name__}"
            )
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
