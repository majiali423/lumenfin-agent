#!/usr/bin/env python3
"""Validate and publish the fixed LEDGER public-dev e2e canary artifact."""
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

import run_ledger_public_dev_e2e_canary as e2e_cli
import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.eval.holdout import HoldoutError
from lumenfin.stdio import configure_stdio_utf8

SEALED_OUTPUT = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_e2e_canary_5x10.json"
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
        (ROOT / "scripts" / "run_ledger_public_dev_e2e_canary.py")
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
    aggregate, raw = _load_json_bytes(aggregate_path, label="e2e aggregate")
    plan, _ = _load_json_bytes(plan_path, label="e2e plan")
    try:
        per_case_raw = per_case_path.read_bytes()
        rows = paired_cli._parse_jsonl_text(
            per_case_raw.decode("utf-8"),
            label="e2e per-case",
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise HoldoutError("cannot read e2e per-case") from exc
    per_case_sha256 = hashlib.sha256(per_case_raw).hexdigest()
    lexical = e2e_cli._summarize(rows, "lexical")
    qwen3 = e2e_cli._summarize(rows, "qwen3")
    paired = e2e_cli._paired_numeric(rows)
    fallbacks = qwen3["fallback_cases"]
    generate_errors = (
        lexical["generate_error_cases"] + qwen3["generate_error_cases"]
    )
    generate_attempts = (
        lexical["generate_attempts"] + qwen3["generate_attempts"]
    )
    expected_comparison = {
        "lexical": lexical,
        "qwen3": qwen3,
        "delta_qwen3_minus_lexical": {
            "numeric_accuracy": round(
                qwen3["numeric_accuracy"] - lexical["numeric_accuracy"],
                4,
            ),
            "citation_support_rate": round(
                qwen3["citation_support_rate"] - lexical["citation_support_rate"],
                4,
            ),
        },
        "paired_numeric_match": paired,
    }
    expected_calls = {
        "candidate_embedding_remote_calls": 0,
        "qwen3_logical_calls": len(rows),
        "qwen3_physical_attempts": qwen3["qwen3_attempts"],
        "generate_logical_calls": len(rows) * 2,
        "generate_physical_attempts": generate_attempts,
        "rerank_fallbacks": int(fallbacks),
        "generate_errors": int(generate_errors),
        "billing_semantics": "persisted_complete_cases_at_least_once",
        "unobserved_inflight_remote_calls_possible": True,
        "latency_scope": "rerank_plus_generate_not_production_e2e",
    }
    serialized = raw.decode("utf-8")
    if (
        aggregate.get("schema_version") != e2e_cli.SCHEMA_VERSION
        or aggregate.get("e2e_source_sha256") != _script_sha256()
        or aggregate.get("e2e_source_sha256") != plan.get("e2e_source_sha256")
        or aggregate.get("cases") != 50
        or len(rows) != 50
        or plan.get("cases") != 50
        or plan.get("companies") != 5
        or plan.get("cases_per_company") != 10
        or plan.get("arm") != "A_prod"
        or plan.get("parent_query_ids_sha256")
        != e2e_cli.PARENT_QUERY_IDS_SHA256
        or aggregate.get("selection", {}).get("parent_query_ids_sha256")
        != e2e_cli.PARENT_QUERY_IDS_SHA256
        or aggregate.get("selection", {}).get("query_ids_sha256")
        != plan.get("selected_query_ids_sha256")
        or aggregate.get("selection", {}).get("gold_identity_sha256")
        != plan.get("gold_identity_sha256")
        or aggregate.get("comparison") != expected_comparison
        or aggregate.get("call_accounting") != expected_calls
        or aggregate.get("per_case_sha256") != per_case_sha256
        or aggregate.get("qwen3_calls") != qwen3["qwen3_attempts"]
        or aggregate.get("generate_calls") != generate_attempts
        or aggregate.get("primary_comparison_valid")
        is not (fallbacks == 0 and generate_errors == 0)
        or aggregate.get("product_accuracy_claim") is not False
        or aggregate.get("financebench_phase4") != "NOT_RUN"
        or plan.get("product_accuracy_claim") is not False
        or plan.get("financebench_phase4") != "NOT_RUN"
        or '"query_text"' in serialized
        or '"text"' in serialized
    ):
        raise HoldoutError("e2e canary artifact failed sealed validation")
    for row in rows:
        if row.get("row_sha256") != e2e_cli._row_sha256(row):
            raise HoldoutError("e2e per-case row identity failed sealed validation")
        arms = row.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(e2e_cli.ARMS):
            raise HoldoutError("e2e per-case arm identity failed sealed validation")
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
            raise HoldoutError("refusing to overwrite sealed e2e artifact")
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
            else HoldoutError(f"e2e seal failed: {type(exc).__name__}")
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
