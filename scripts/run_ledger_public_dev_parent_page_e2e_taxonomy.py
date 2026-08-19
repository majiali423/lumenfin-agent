#!/usr/bin/env python3
"""Local-only failure taxonomy for the sealed parent-page suffix generate canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_ledger_public_dev_e2e_failure_taxonomy as tax_cli
import run_ledger_public_dev_parent_pack_probe as pack_probe
import run_ledger_public_dev_parent_pack_suffix as suffix_cli
import run_ledger_public_dev_parent_page_e2e as parent_e2e_cli
import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_e2e_taxonomy import (
    classify_e2e_case,
    classify_parent_page_generate_case,
)
from lumenfin.eval.holdout.ledger_parent_return import (
    LOCKED_STRATEGY,
    build_parent_page_hits,
    parent_prompt_char_cap,
)
from lumenfin.stdio import configure_stdio_utf8

SCHEMA_VERSION = "lumenfin_ledger_parent_page_e2e_taxonomy.v1"
ARMS = parent_e2e_cli.ARMS
ARM = parent_e2e_cli.ARM
TRACKED_PARENT_E2E = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_parent_page_e2e_5x40.json"
).resolve()
TRACKED_SUFFIX = parent_e2e_cli.TRACKED_SUFFIX


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify sealed parent-page suffix generate failures locally. "
            "Does not call remote providers or change production RAG."
        )
    )
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--paired-aggregate", required=True)
    parser.add_argument("--paired-per-case", required=True)
    parser.add_argument("--parent-e2e-aggregate", required=True)
    parser.add_argument("--parent-e2e-per-case", required=True)
    parser.add_argument("--suffix-aggregate", required=True)
    parser.add_argument("--e2e-aggregate", required=True)
    parser.add_argument("--baseline-aggregate", required=True)
    parser.add_argument("--baseline-per-case", required=True)
    parser.add_argument("--prerank-aggregate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-remote", action="store_true")
    return parser


def _script_sha256() -> str:
    return hashlib.sha256(
        Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _row_sha256(row: dict) -> str:
    return paired_cli._canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        if args.allow_remote:
            raise HoldoutError("parent-page e2e taxonomy is local-only")
        parent_path = Path(args.parent_e2e_aggregate).expanduser().resolve()
        suffix_path = Path(args.suffix_aggregate).expanduser().resolve()
        if parent_path != TRACKED_PARENT_E2E:
            raise HoldoutError(
                "taxonomy must use the tracked parent-page e2e aggregate"
            )
        if suffix_path != TRACKED_SUFFIX:
            raise HoldoutError("taxonomy must use the tracked suffix probe")
        parent_aggregate = paired_cli._load_json(
            parent_path,
            label="sealed parent-page e2e aggregate",
        )
        suffix = paired_cli._load_json(suffix_path, label="sealed suffix probe")
        parent_rows = paired_cli._read_jsonl(
            Path(args.parent_e2e_per_case),
            label="sealed parent-page e2e per-case",
        )
        per_case_sha256 = hashlib.sha256(
            Path(args.parent_e2e_per_case).read_bytes()
        ).hexdigest()
        if (
            parent_aggregate.get("schema_version") != parent_e2e_cli.SCHEMA_VERSION
            or parent_aggregate.get("cases") != 200
            or len(parent_rows) != 200
            or parent_aggregate.get("primary_comparison_valid") is not True
            or parent_aggregate.get("product_accuracy_claim") is not False
            or parent_aggregate.get("financebench_phase4") != "NOT_RUN"
            or parent_aggregate.get("qwen3_calls") != 0
            or per_case_sha256 != parent_aggregate.get("per_case_sha256")
            or parent_aggregate.get("selection", {}).get("query_ids_sha256")
            != suffix.get("selection", {}).get("query_ids_sha256")
        ):
            raise HoldoutError("sealed parent-page e2e identity is incompatible")
        frozen = suffix_cli._load_frozen(args)
        if (
            paired_cli._ids_sha256(tuple(frozen["suffix_ids"]))
            != parent_aggregate["selection"]["query_ids_sha256"]
            or frozen["candidate_manifest"]["candidate_cache_sha256"]
            != parent_aggregate["candidate_manifest"]["candidate_cache_sha256"]
        ):
            raise HoldoutError("taxonomy suffix identity diverged")
        paired_rows = paired_cli._read_jsonl(
            Path(args.paired_per_case),
            label="paired Qwen3 per-case",
        )
        paired_by_id = {str(row["query_id"]): row for row in paired_rows}
        generate_by_id = {str(row["query_id"]): row for row in parent_rows}
        pages = pack_probe._page_by_id(frozen["dataset"])
        chunk_max = int(parent_aggregate["chunk_max_document_chars"])
        classified: list[dict[str, Any]] = []
        for candidate in frozen["suffix_rows"]:
            query_id = str(candidate["query_id"])
            generated = generate_by_id.get(query_id)
            paired_row = paired_by_id.get(query_id)
            query = frozen["query_by_id"].get(query_id)
            if generated is None or paired_row is None or query is None:
                raise HoldoutError("taxonomy suffix row is missing generate or rank")
            if (
                generated.get("run_identity_sha256")
                != parent_aggregate.get("run_identity_sha256")
                or generated.get("row_sha256")
                != parent_e2e_cli._row_sha256(generated)
            ):
                raise HoldoutError("taxonomy generate row identity mismatch")
            ranked = paired_row["reranked_arms"][ARM]
            if bool(ranked.get("rerank_fallback")):
                raise HoldoutError("taxonomy refuses a Qwen3 fallback rank")
            identity = list(ranked["final_identity"])
            if ranked["final_identity_sha256"] != generated.get(
                "final_identity_sha256"
            ):
                raise HoldoutError("taxonomy Qwen3 identity diverged from generate")
            parent_hits = build_parent_page_hits(identity, pages)
            chunk_arm = classify_e2e_case(
                pool_hits=list(candidate["hits"]),
                final_identity=identity,
                qrels=list(query["qrels"]),
                gold_value=float(generated["arms"]["chunk"]["gold_value"]),
                numeric_matched=bool(generated["arms"]["chunk"]["numeric_match"]),
                abstain=bool(generated["arms"]["chunk"]["abstain"]),
                max_document_chars=chunk_max,
            )
            parent_arm = classify_parent_page_generate_case(
                pool_hits=list(candidate["hits"]),
                final_identity=identity,
                parent_hits=parent_hits,
                qrels=list(query["qrels"]),
                gold_value=float(generated["arms"]["parent_page"]["gold_value"]),
                numeric_matched=bool(
                    generated["arms"]["parent_page"]["numeric_match"]
                ),
                abstain=bool(generated["arms"]["parent_page"]["abstain"]),
                chunk_max_document_chars=chunk_max,
                parent_max_document_chars=parent_prompt_char_cap(parent_hits),
            )
            chunk_arm["outcome"] = generated["arms"]["chunk"]["outcome"]
            chunk_arm["numeric_match"] = bool(
                generated["arms"]["chunk"]["numeric_match"]
            )
            chunk_arm["abstain"] = bool(generated["arms"]["chunk"]["abstain"])
            parent_arm["outcome"] = generated["arms"]["parent_page"]["outcome"]
            parent_arm["numeric_match"] = bool(
                generated["arms"]["parent_page"]["numeric_match"]
            )
            parent_arm["abstain"] = bool(generated["arms"]["parent_page"]["abstain"])
            row = {
                "query_id": query_id,
                "e2e_run_identity_sha256": parent_aggregate["run_identity_sha256"],
                "shared_candidate_identity_sha256": candidate[
                    "candidate_identity_sha256"
                ],
                "final_identity_sha256": ranked["final_identity_sha256"],
                "locked_strategy": LOCKED_STRATEGY,
                "arms": {"chunk": chunk_arm, "parent_page": parent_arm},
            }
            row["row_sha256"] = _row_sha256(row)
            classified.append(row)
        if tuple(row["query_id"] for row in classified) != tuple(
            frozen["suffix_ids"]
        ):
            raise HoldoutError("taxonomy coverage is not the frozen company suffix")
        chunk = tax_cli._summarize(classified, "chunk")
        parent = tax_cli._summarize(classified, "parent_page")
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        per_case_path = output_dir / "per_case.jsonl"
        _atomic_jsonl(per_case_path, classified)
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "taxonomy_source_sha256": _script_sha256(),
            "e2e_run_identity_sha256": parent_aggregate["run_identity_sha256"],
            "e2e_aggregate_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            "e2e_per_case_sha256": per_case_sha256,
            "candidate_cache_sha256": frozen["candidate_manifest"][
                "candidate_cache_sha256"
            ],
            "cases": 200,
            "selection": parent_aggregate["selection"],
            "arm": ARM,
            "locked_strategy": LOCKED_STRATEGY,
            "comparison": {
                "chunk": chunk,
                "parent_page": parent,
            },
            "recommended_next_workstream": parent["recommended_next_workstream"],
            "remote_calls": 0,
            "qwen3_calls": 0,
            "generate_calls": 0,
            "product_accuracy_claim": False,
            "financebench_phase4": "NOT_RUN",
            "per_case_sha256": hashlib.sha256(per_case_path.read_bytes()).hexdigest(),
        }
        _atomic_json(output_dir / "aggregate.json", aggregate)
        print(json.dumps(aggregate, indent=2), flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - redact CLI boundary failures
        safe = (
            exc
            if isinstance(exc, HoldoutError)
            else HoldoutError(
                f"LEDGER parent-page taxonomy failed: {type(exc).__name__}"
            )
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
