#!/usr/bin/env python3
"""Validate and publish the LEDGER parent-page vs chunk generation canary."""
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

import run_ledger_public_dev_parent_page_e2e as e2e_cli
import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_parent_return import (
    HOLDOUT_CASES_PER_COMPANY,
    LOCKED_STRATEGY,
)
from lumenfin.stdio import configure_stdio_utf8

SEALED_OUTPUT = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_parent_page_e2e_5x40.json"
).resolve()
SUFFIX_QUERY_IDS_SHA256 = (
    "7bd0906ea034b7c2a679957ac0ad82f1583934872f689c82385d5cc7a9aa9c33"
)
PREFIX_QUERY_IDS_SHA256 = (
    "6fbe540fa4cca45f298950b7728d769beee8bb43a9711c3bece01a2b62a8f9aa"
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
        (ROOT / "scripts" / "run_ledger_public_dev_parent_page_e2e.py")
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .encode()
    ).hexdigest()


def _parent_return_sha256() -> str:
    return hashlib.sha256(
        (
            ROOT
            / "src"
            / "lumenfin"
            / "eval"
            / "holdout"
            / "ledger_parent_return.py"
        )
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .encode()
    ).hexdigest()


def _validate(
    *,
    aggregate_path: Path,
    per_case_path: Path,
    plan_path: Path,
) -> bytes:
    aggregate, raw = _load_json_bytes(
        aggregate_path,
        label="parent-page e2e aggregate",
    )
    plan, _ = _load_json_bytes(plan_path, label="parent-page e2e plan")
    try:
        per_case_raw = per_case_path.read_bytes()
        rows = paired_cli._parse_jsonl_text(
            per_case_raw.decode("utf-8"),
            label="parent-page e2e per-case",
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise HoldoutError("cannot read parent-page e2e per-case") from exc
    comparison = e2e_cli.build_comparison(rows)
    chunk = comparison["chunk"]
    parent = comparison["parent_page"]
    generate_errors = (
        chunk["generate_error_cases"] + parent["generate_error_cases"]
    )
    generate_attempts = chunk["generate_attempts"] + parent["generate_attempts"]
    expected_calls = {
        "qwen3_calls": 0,
        "generate_logical_calls": len(rows) * 2,
        "generate_physical_attempts": generate_attempts,
        "generate_errors": int(generate_errors),
        "billing_semantics": "persisted_complete_cases_at_least_once",
        "unobserved_inflight_remote_calls_possible": True,
        "latency_scope": "generate_not_production_e2e",
    }
    serialized = raw.decode("utf-8")
    if (
        aggregate.get("schema_version") != e2e_cli.SCHEMA_VERSION
        or aggregate.get("e2e_source_sha256") != _script_sha256()
        or aggregate.get("e2e_source_sha256") != plan.get("e2e_source_sha256")
        or aggregate.get("parent_return_source_sha256") != _parent_return_sha256()
        or aggregate.get("parent_return_source_sha256")
        != plan.get("parent_return_source_sha256")
        or aggregate.get("cases") != 200
        or len(rows) != 200
        or plan.get("cases") != 200
        or plan.get("companies") != 5
        or plan.get("cases_per_company") != HOLDOUT_CASES_PER_COMPANY
        or plan.get("arm") != "A_prod"
        or plan.get("locked_strategy") != LOCKED_STRATEGY
        or plan.get("generate_arms") != list(e2e_cli.ARMS)
        or plan.get("qwen3_calls") != 0
        or aggregate.get("qwen3_calls") != 0
        or aggregate.get("locked_strategy") != LOCKED_STRATEGY
        or aggregate.get("arm") != "A_prod"
        or aggregate.get("selection", {}).get("query_ids_sha256")
        != SUFFIX_QUERY_IDS_SHA256
        or aggregate.get("selection", {}).get("prefix_query_ids_sha256")
        != PREFIX_QUERY_IDS_SHA256
        or aggregate.get("selection", {}).get("query_ids_sha256")
        != plan.get("query_ids_sha256")
        or aggregate.get("selection", {}).get("gold_identity_sha256")
        != plan.get("gold_identity_sha256")
        or aggregate.get("candidate_manifest", {}).get("candidate_cache_sha256")
        != plan.get("candidate_cache_sha256")
        or plan.get("candidate_cache_sha256")
        != e2e_cli.EXPECTED_CANDIDATE_CACHE_SHA256
        or aggregate.get("comparison") != comparison
        or aggregate.get("call_accounting") != expected_calls
        or aggregate.get("generate_calls") != generate_attempts
        or aggregate.get("per_case_sha256")
        != hashlib.sha256(per_case_raw).hexdigest()
        or aggregate.get("primary_comparison_valid") is not (generate_errors == 0)
        or aggregate.get("product_accuracy_claim") is not False
        or aggregate.get("financebench_phase4") != "NOT_RUN"
        or plan.get("product_accuracy_claim") is not False
        or plan.get("financebench_phase4") != "NOT_RUN"
        or '"query_text"' in serialized
        or '"mmd_text"' in serialized
    ):
        raise HoldoutError("parent-page e2e artifact failed sealed validation")
    for row in rows:
        if row.get("row_sha256") != e2e_cli._row_sha256(row):
            raise HoldoutError(
                "parent-page e2e per-case identity failed sealed validation"
            )
        arms = row.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(e2e_cli.ARMS):
            raise HoldoutError(
                "parent-page e2e per-case arm identity failed sealed validation"
            )
    return raw


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--per-case", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).expanduser().resolve()
        if output != SEALED_OUTPUT:
            raise HoldoutError("sealed output path is not the fixed tracked path")
        if output.exists():
            raise HoldoutError("refusing to overwrite sealed parent-page e2e artifact")
        raw = _validate(
            aggregate_path=Path(args.aggregate).expanduser().resolve(),
            per_case_path=Path(args.per_case).expanduser().resolve(),
            plan_path=Path(args.plan).expanduser().resolve(),
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
            else HoldoutError(f"parent-page e2e seal failed: {type(exc).__name__}")
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
