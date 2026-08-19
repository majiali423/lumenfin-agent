#!/usr/bin/env python3
"""Local-only failure taxonomy for the sealed LEDGER e2e canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_ledger_public_dev_e2e_canary as e2e_cli
import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_e2e_taxonomy import (
    LEAK_CLASSES,
    SCHEMA_VERSION,
    classify_e2e_case,
    recommend_next_workstream,
)
from lumenfin.stdio import configure_stdio_utf8

TRACKED_E2E = (
    ROOT / "data" / "eval_rag" / "holdout" / "ledger_public_dev_e2e_canary_5x10.json"
).resolve()
ARMS = ("lexical", "qwen3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify sealed LEDGER e2e canary failures locally. "
            "Does not call remote providers or change production RAG."
        )
    )
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--e2e-aggregate", required=True)
    parser.add_argument("--e2e-per-case", required=True)
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


def _summarize(rows: list[dict], arm: str) -> dict[str, Any]:
    counts = Counter(str(row["arms"][arm]["leak_class"]) for row in rows)
    ranked = Counter(str(row["arms"][arm]["ranking_class"]) for row in rows)
    work = Counter(str(row["arms"][arm]["next_workstream"]) for row in rows)
    leak_counts = {name: int(counts.get(name, 0)) for name in LEAK_CLASSES}
    return {
        "arm": arm,
        "cases": len(rows),
        "leak_class_counts": leak_counts,
        "ranking_class_counts": dict(ranked),
        "next_workstream_counts": dict(work),
        "recommended_next_workstream": recommend_next_workstream(leak_counts),
        "pool_hit_cases": sum(bool(row["arms"][arm]["pool_hit"]) for row in rows),
        "hit_at_10_cases": sum(bool(row["arms"][arm]["hit_at_10"]) for row in rows),
        "number_in_final_context_cases": sum(
            bool(row["arms"][arm]["number_in_final_context"]) for row in rows
        ),
    }


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        if args.allow_remote:
            raise HoldoutError("e2e failure taxonomy is local-only")
        aggregate_path = Path(args.e2e_aggregate).expanduser().resolve()
        if aggregate_path != TRACKED_E2E:
            raise HoldoutError("taxonomy must use the tracked sealed e2e aggregate")
        e2e_aggregate = paired_cli._load_json(
            aggregate_path,
            label="sealed e2e aggregate",
        )
        e2e_rows = paired_cli._read_jsonl(
            Path(args.e2e_per_case),
            label="sealed e2e per-case",
        )
        if (
            e2e_aggregate.get("schema_version") != e2e_cli.SCHEMA_VERSION
            or e2e_aggregate.get("cases") != 50
            or len(e2e_rows) != 50
            or e2e_aggregate.get("product_accuracy_claim") is not False
            or e2e_aggregate.get("financebench_phase4") != "NOT_RUN"
        ):
            raise HoldoutError("sealed e2e identity is incompatible")
        per_case_sha256 = hashlib.sha256(
            Path(args.e2e_per_case).read_bytes()
        ).hexdigest()
        if per_case_sha256 != e2e_aggregate.get("per_case_sha256"):
            raise HoldoutError("e2e per-case hash does not match the sealed aggregate")
        proxy = argparse.Namespace(
            parquet_path=args.parquet_path,
            manifest=args.manifest,
            split_salt=args.split_salt,
            candidate_dir=args.candidate_dir,
            baseline_aggregate=args.baseline_aggregate,
            baseline_per_case=args.baseline_per_case,
            prerank_aggregate=args.prerank_aggregate,
        )
        bundle = e2e_cli._load_plan_inputs(proxy)
        if (
            bundle["candidate_manifest"]["candidate_cache_sha256"]
            != e2e_aggregate["candidate_manifest"]["candidate_cache_sha256"]
            or paired_cli._ids_sha256(tuple(bundle["selected_ids"]))
            != e2e_aggregate["selection"]["query_ids_sha256"]
        ):
            raise HoldoutError("taxonomy candidate identity diverged from sealed e2e")
        candidate_by_id = {
            str(row["query_id"]): row for row in bundle["selected"]
        }
        max_document_chars = int(
            e2e_aggregate["rerank_settings"]["max_document_chars"]
        )
        classified: list[dict[str, Any]] = []
        for e2e_row in e2e_rows:
            query_id = str(e2e_row["query_id"])
            candidate = candidate_by_id.get(query_id)
            query = bundle["query_by_id"].get(query_id)
            if (
                candidate is None
                or query is None
                or e2e_row.get("run_identity_sha256")
                != e2e_aggregate.get("run_identity_sha256")
            ):
                raise HoldoutError("taxonomy e2e row identity mismatch")
            arms: dict[str, Any] = {}
            for arm in ARMS:
                generated = e2e_row["arms"][arm]
                classified_arm = classify_e2e_case(
                    pool_hits=list(candidate["hits"]),
                    final_identity=list(generated["final_identity"]),
                    qrels=list(query["qrels"]),
                    gold_value=float(generated["gold_value"]),
                    numeric_matched=bool(generated["numeric_match"]),
                    abstain=bool(generated["abstain"]),
                    max_document_chars=max_document_chars,
                )
                classified_arm["outcome"] = generated["outcome"]
                classified_arm["numeric_match"] = bool(generated["numeric_match"])
                classified_arm["abstain"] = bool(generated["abstain"])
                arms[arm] = classified_arm
            row = {
                "query_id": query_id,
                "e2e_run_identity_sha256": e2e_aggregate["run_identity_sha256"],
                "shared_candidate_identity_sha256": candidate[
                    "candidate_identity_sha256"
                ],
                "arms": arms,
            }
            row["row_sha256"] = _row_sha256(row)
            classified.append(row)
        if tuple(row["query_id"] for row in classified) != tuple(
            bundle["selected_ids"]
        ):
            raise HoldoutError("taxonomy coverage is not the frozen company prefix")
        lexical = _summarize(classified, "lexical")
        qwen3 = _summarize(classified, "qwen3")
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        per_case_path = output_dir / "per_case.jsonl"
        _atomic_jsonl(per_case_path, classified)
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "taxonomy_source_sha256": _script_sha256(),
            "e2e_run_identity_sha256": e2e_aggregate["run_identity_sha256"],
            "e2e_aggregate_sha256": hashlib.sha256(
                aggregate_path.read_bytes()
            ).hexdigest(),
            "e2e_per_case_sha256": per_case_sha256,
            "candidate_cache_sha256": bundle["candidate_manifest"][
                "candidate_cache_sha256"
            ],
            "cases": 50,
            "selection": {
                "cases": 50,
                "companies": 5,
                "cases_per_company": 10,
                "parent_query_ids_sha256": e2e_cli.PARENT_QUERY_IDS_SHA256,
                "query_ids_sha256": e2e_aggregate["selection"]["query_ids_sha256"],
                "strategy": "frozen_5x50_company_prefix_v1",
            },
            "arm": "A_prod",
            "comparison": {
                "lexical": lexical,
                "qwen3": qwen3,
            },
            "recommended_next_workstream": qwen3["recommended_next_workstream"],
            "remote_calls": 0,
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
                f"LEDGER e2e taxonomy failed: {type(exc).__name__}"
            )
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
