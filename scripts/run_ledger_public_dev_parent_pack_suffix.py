#!/usr/bin/env python3
"""Local parent-page packing on the frozen 5x40 suffix. No generation."""
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

import run_ledger_public_dev_e2e_canary as e2e_cli
import run_ledger_public_dev_e2e_failure_taxonomy as tax_cli
import run_ledger_public_dev_parent_pack_probe as pack_probe
import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_e2e import load_ledger_gold_values
from lumenfin.eval.holdout.ledger_parent_probe import PACK_STRATEGIES, recoverability
from lumenfin.eval.holdout.ledger_parent_return import (
    HOLDOUT_CASES_PER_COMPANY,
    LOCKED_STRATEGY,
    PREFIX_CASES_PER_COMPANY,
    SCHEMA_VERSION,
    assert_disjoint_from_prefix,
    select_frozen_slice,
)
from lumenfin.stdio import configure_stdio_utf8

PARENT_QUERY_IDS_SHA256 = e2e_cli.PARENT_QUERY_IDS_SHA256
PREFIX_QUERY_IDS_SHA256 = (
    "6fbe540fa4cca45f298950b7728d769beee8bb43a9711c3bece01a2b62a8f9aa"
)
TRACKED_QWEN3 = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_qwen3_paired_5x50.json"
).resolve()
ARM = "A_prod"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure parent-page recoverability on the frozen 5x40 suffix. "
            "Local-only; does not generate answers or change production RAG."
        )
    )
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--paired-aggregate", required=True)
    parser.add_argument("--paired-per-case", required=True)
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


def recommend_next(rows: list[dict]) -> str:
    chunk = sum(bool(row["recovered"]["chunk_final"]) for row in rows)
    parent = sum(bool(row["recovered"][LOCKED_STRATEGY]) for row in rows)
    if parent <= chunk:
        return "do_not_generate_parent_page_on_this_split"
    return "generate_chunk_vs_parent_page_on_suffix"


def _load_frozen(args: argparse.Namespace) -> dict[str, Any]:
    proxy = argparse.Namespace(
        parquet_path=args.parquet_path,
        manifest=args.manifest,
        split_salt=args.split_salt,
        baseline_aggregate=args.baseline_aggregate,
        baseline_per_case=args.baseline_per_case,
        prerank_aggregate=args.prerank_aggregate,
        batch_size=64,
        embedding_dimension=1024,
    )
    manifest, dataset, plans, qrel_audit, prerank = paired_cli._load_context(proxy)
    candidate_rows, candidate_manifest = paired_cli._validate_candidate_cache(
        Path(args.candidate_dir).expanduser().resolve(),
        manifest=manifest,
        plans=plans,
        qrel_audit=qrel_audit,
        prerank=prerank,
    )
    if len(plans) != 5:
        raise HoldoutError("parent-page suffix requires the frozen 5-company plan")
    parent_ids = tuple(
        str(query_id) for plan in plans for query_id in plan["query_ids"]
    )
    if paired_cli._ids_sha256(parent_ids) != PARENT_QUERY_IDS_SHA256:
        raise HoldoutError("parent query identity mismatch")
    prefix_rows = e2e_cli._prefix_candidate_rows(candidate_rows, plans)
    suffix_rows = select_frozen_slice(
        candidate_rows,
        plans,
        start=PREFIX_CASES_PER_COMPANY,
        count=HOLDOUT_CASES_PER_COMPANY,
    )
    prefix_ids = [str(row["query_id"]) for row in prefix_rows]
    suffix_ids = [str(row["query_id"]) for row in suffix_rows]
    if paired_cli._ids_sha256(tuple(prefix_ids)) != PREFIX_QUERY_IDS_SHA256:
        raise HoldoutError("e2e prefix identity diverged")
    assert_disjoint_from_prefix(suffix_ids, prefix_ids)
    return {
        "dataset": dataset,
        "plans": plans,
        "candidate_manifest": candidate_manifest,
        "query_by_id": {str(query["query_id"]): query for query in dataset.queries},
        "prefix_ids": prefix_ids,
        "suffix_rows": suffix_rows,
        "suffix_ids": suffix_ids,
        "gold": load_ledger_gold_values(args.parquet_path, query_ids=suffix_ids),
    }


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        if args.allow_remote:
            raise HoldoutError("parent-page suffix probe is local-only")
        e2e_path = Path(args.e2e_aggregate).expanduser().resolve()
        if e2e_path != tax_cli.TRACKED_E2E:
            raise HoldoutError("suffix probe must use the tracked e2e aggregate")
        paired_path = Path(args.paired_aggregate).expanduser().resolve()
        if paired_path != TRACKED_QWEN3:
            raise HoldoutError("suffix probe must use the tracked Qwen3 aggregate")
        e2e_aggregate = paired_cli._load_json(e2e_path, label="sealed e2e aggregate")
        qwen3_aggregate = paired_cli._load_json(
            paired_path,
            label="sealed Qwen3 aggregate",
        )
        if (
            e2e_aggregate.get("selection", {}).get("query_ids_sha256")
            != PREFIX_QUERY_IDS_SHA256
            or qwen3_aggregate.get("selection", {}).get("query_ids_sha256")
            != PARENT_QUERY_IDS_SHA256
            or qwen3_aggregate.get("primary_comparison_valid") is not True
            or qwen3_aggregate.get("product_accuracy_claim") is not False
        ):
            raise HoldoutError("suffix probe source aggregates are incompatible")
        paired_rows = paired_cli._read_jsonl(
            Path(args.paired_per_case),
            label="paired Qwen3 per-case",
        )
        paired_by_id = {str(row["query_id"]): row for row in paired_rows}
        if len(paired_by_id) != 250:
            raise HoldoutError("paired Qwen3 per-case coverage is not 250")
        frozen = _load_frozen(args)
        pages = pack_probe._page_by_id(frozen["dataset"])
        max_document_chars = int(
            e2e_aggregate["rerank_settings"]["max_document_chars"]
        )
        classified: list[dict[str, Any]] = []
        for candidate in frozen["suffix_rows"]:
            query_id = str(candidate["query_id"])
            paired_row = paired_by_id.get(query_id)
            query = frozen["query_by_id"].get(query_id)
            if paired_row is None or query is None:
                raise HoldoutError("suffix query missing paired rank or qrels")
            ranked = paired_row["reranked_arms"][ARM]
            if bool(ranked.get("rerank_fallback")):
                raise HoldoutError("suffix probe refuses a Qwen3 fallback rank")
            identity = list(ranked["final_identity"])
            recovered = recoverability(
                gold_value=float(frozen["gold"][query_id]),
                chunk_final_text=pack_probe._final_chunk_text(
                    candidate=candidate,
                    final_identity=identity,
                    max_document_chars=max_document_chars,
                ),
                retrieved_page_ids=pack_probe.unique_document_ids(identity),
                gold_page_ids=pack_probe._positive_pages(list(query["qrels"])),
                page_by_id=pages,
            )
            row = {
                "query_id": query_id,
                "paired_run_identity_sha256": paired_row["run_identity_sha256"],
                "final_identity_sha256": ranked["final_identity_sha256"],
                "locked_strategy": LOCKED_STRATEGY,
                **recovered,
            }
            row["row_sha256"] = _row_sha256(row)
            classified.append(row)
        if tuple(row["query_id"] for row in classified) != tuple(frozen["suffix_ids"]):
            raise HoldoutError("suffix probe coverage mismatch")
        totals = {
            strategy: sum(bool(row["recovered"][strategy]) for row in classified)
            for strategy in PACK_STRATEGIES
        }
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        per_case_path = output_dir / "per_case.jsonl"
        _atomic_jsonl(per_case_path, classified)
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "probe_source_sha256": _script_sha256(),
            "selection": {
                "strategy": "frozen_5x50_company_suffix_v1",
                "companies": 5,
                "cases": 200,
                "cases_per_company": HOLDOUT_CASES_PER_COMPANY,
                "start_index": PREFIX_CASES_PER_COMPANY,
                "parent_query_ids_sha256": PARENT_QUERY_IDS_SHA256,
                "prefix_query_ids_sha256": PREFIX_QUERY_IDS_SHA256,
                "query_ids_sha256": paired_cli._ids_sha256(
                    tuple(frozen["suffix_ids"])
                ),
            },
            "arm": ARM,
            "locked_strategy": LOCKED_STRATEGY,
            "recovered_cases": totals,
            "recommended_next_workstream": recommend_next(classified),
            "remote_calls": 0,
            "qwen3_calls": 0,
            "generate_calls": 0,
            "product_accuracy_claim": False,
            "financebench_phase4": "NOT_RUN",
            "latency_scope": "text_packing_not_generation",
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
                f"LEDGER parent suffix probe failed: {type(exc).__name__}"
            )
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
