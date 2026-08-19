#!/usr/bin/env python3
"""Chunk vs parent-page generation on the frozen 5x40 suffix. Reuses Qwen3 ranks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
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
import run_ledger_public_dev_parent_pack_suffix as suffix_cli
import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.env_bootstrap import bootstrap_dotenv
from lumenfin.eval.financebench.metrics import percentile_sorted
from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_e2e import (
    RELATIVE_TOLERANCE,
    SYSTEM_PROMPT,
    build_generation_prompt,
    parse_answer_payload,
    score_generated_answer,
)
from lumenfin.eval.holdout.ledger_parent_return import (
    HOLDOUT_CASES_PER_COMPANY,
    LOCKED_STRATEGY,
    build_parent_page_hits,
    parent_prompt_char_cap,
)
from lumenfin.llm import DeepSeekChatClient, LLMSettings
from lumenfin.stdio import configure_stdio_utf8

SCHEMA_VERSION = "lumenfin_ledger_parent_page_e2e.v1"
ARMS = ("chunk", "parent_page")
TRACKED_SUFFIX = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_parent_pack_suffix_5x40.json"
).resolve()
TRACKED_E2E = tax_cli.TRACKED_E2E
TRACKED_QWEN3 = suffix_cli.TRACKED_QWEN3
ARM = "A_prod"
EXPECTED_CANDIDATE_CACHE_SHA256 = (
    "c49d06665376b769950492cecd41cb3d7ad144509e57d0cdf09493aeab52e65a"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate numeric answers from frozen Qwen3 Top-10 chunks vs "
            "parent pages on the 5x40 suffix. Does not change production RAG."
        )
    )
    parser.add_argument("--phase", choices=("plan", "run"), required=True)
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--paired-aggregate", required=True)
    parser.add_argument("--paired-per-case", required=True)
    parser.add_argument("--e2e-aggregate", required=True)
    parser.add_argument("--suffix-aggregate", required=True)
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


def _parent_return_sha256() -> str:
    path = ROOT / "src" / "lumenfin" / "eval" / "holdout" / "ledger_parent_return.py"
    return hashlib.sha256(
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
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


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return round(percentile_sorted(values, q), 2)


def _chunk_hits(candidate: dict, identity: list[dict]) -> list[dict]:
    by_chunk = {str(hit["chunk_id"]): hit for hit in candidate["hits"]}
    hits: list[dict] = []
    for item in identity:
        hit = by_chunk.get(str(item.get("chunk_id") or ""))
        if hit is None:
            raise HoldoutError("parent e2e chunk is outside the frozen pool")
        hits.append(hit)
    if not hits:
        raise HoldoutError("parent e2e chunk context is empty")
    return hits


def _run_generate(
    *,
    arm: str,
    query: dict,
    hits: list[dict],
    gold_value: float,
    llm: DeepSeekChatClient,
    max_document_chars: int,
) -> dict[str, Any]:
    prompt = build_generation_prompt(
        query_text=str(query["query_text"]),
        hits=hits,
        max_document_chars=max_document_chars,
    )
    llm.mark_usage_start()
    started = time.perf_counter()
    raw = llm.chat(SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=200)
    generate_ms = round((time.perf_counter() - started) * 1000.0, 2)
    parsed = parse_answer_payload(raw)
    scored = score_generated_answer(
        gold_value=gold_value,
        parsed=parsed,
        hits=hits,
        qrels={
            str(item["doc_id"]): int(item["relevance"])
            for item in query["qrels"]
        },
    )
    usage = llm.usage_since_mark()
    return {
        "arm": arm,
        "final_identity_sha256": paired_cli._hash_hits(hits),
        "predicted_value": parsed["value"],
        "gold_value": gold_value,
        "cited_chunk_ids": parsed["cited_chunk_ids"],
        "prompt_chars": len(prompt),
        "generate_attempts": int(getattr(llm, "last_attempts", 1) or 1),
        "generate_tokens": int(usage["prompt_tokens"] + usage["completion_tokens"]),
        "generate_latency_ms": generate_ms,
        "generate_error_type": "",
        **scored,
    }


def _summarize(rows: list[dict], arm: str) -> dict[str, Any]:
    arm_rows = [row["arms"][arm] for row in rows]
    matches = [bool(item["numeric_match"]) for item in arm_rows]
    totals = [float(item["generate_latency_ms"]) for item in arm_rows]
    return {
        "arm": arm,
        "cases": len(arm_rows),
        "numeric_accuracy": round(mean(float(item) for item in matches), 4),
        "abstain_rate": round(mean(float(item["abstain"]) for item in arm_rows), 4),
        "citation_support_rate": round(
            mean(float(item["citation_supported"]) for item in arm_rows),
            4,
        ),
        "generate_error_cases": sum(
            bool(item.get("generate_error_type")) for item in arm_rows
        ),
        "generate_attempts": sum(int(item["generate_attempts"]) for item in arm_rows),
        "generate_tokens": sum(int(item["generate_tokens"]) for item in arm_rows),
        "prompt_chars": sum(int(item["prompt_chars"]) for item in arm_rows),
        "latency_ms": {
            "p50_generate": _percentile(totals, 0.5),
            "p95_generate": _percentile(totals, 0.95),
            "mean_generate": round(mean(totals), 2) if totals else 0.0,
        },
    }


def _paired_numeric(rows: list[dict]) -> dict[str, int]:
    counts = {"gain": 0, "loss": 0, "unchanged": 0}
    for row in rows:
        before = bool(row["arms"]["chunk"]["numeric_match"])
        after = bool(row["arms"]["parent_page"]["numeric_match"])
        if after and not before:
            counts["gain"] += 1
        elif before and not after:
            counts["loss"] += 1
        else:
            counts["unchanged"] += 1
    return counts


def build_comparison(rows: list[dict]) -> dict[str, Any]:
    chunk_summary = _summarize(rows, "chunk")
    parent_summary = _summarize(rows, "parent_page")
    return {
        "chunk": chunk_summary,
        "parent_page": parent_summary,
        "delta_parent_minus_chunk": {
            "numeric_accuracy": round(
                parent_summary["numeric_accuracy"]
                - chunk_summary["numeric_accuracy"],
                4,
            ),
            "citation_support_rate": round(
                parent_summary["citation_support_rate"]
                - chunk_summary["citation_support_rate"],
                4,
            ),
        },
        "paired_numeric_match": _paired_numeric(rows),
    }


def _public_plan(
    *,
    frozen: dict,
    paired_by_id: dict,
    pages: dict[str, str],
    llm_settings: dict,
    chunk_max_chars: int,
) -> dict:
    chunk_chars = 0
    parent_chars = 0
    parent_pages = 0
    for candidate in frozen["suffix_rows"]:
        query = frozen["query_by_id"][str(candidate["query_id"])]
        identity = list(
            paired_by_id[str(candidate["query_id"])]["reranked_arms"][ARM][
                "final_identity"
            ]
        )
        chunk_hits = _chunk_hits(candidate, identity)
        parent_hits = build_parent_page_hits(identity, pages)
        parent_pages += len(parent_hits)
        chunk_chars += len(
            build_generation_prompt(
                query_text=str(query["query_text"]),
                hits=chunk_hits,
                max_document_chars=chunk_max_chars,
            )
        )
        parent_chars += len(
            build_generation_prompt(
                query_text=str(query["query_text"]),
                hits=parent_hits,
                max_document_chars=parent_prompt_char_cap(parent_hits),
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "cases": len(frozen["suffix_ids"]),
        "companies": 5,
        "cases_per_company": HOLDOUT_CASES_PER_COMPANY,
        "arm": ARM,
        "locked_strategy": LOCKED_STRATEGY,
        "generate_arms": list(ARMS),
        "qwen3_calls": 0,
        "generate_requests_without_retries": len(frozen["suffix_ids"]) * 2,
        "parent_pages": parent_pages,
        "chunk_max_document_chars": chunk_max_chars,
        "chunk_prompt_chars_upper_bound": chunk_chars,
        "parent_prompt_chars_upper_bound": parent_chars,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "query_ids_sha256": paired_cli._ids_sha256(tuple(frozen["suffix_ids"])),
        "gold_identity_sha256": e2e_cli._gold_identity_sha256(frozen["gold"]),
        "candidate_cache_sha256": frozen["candidate_manifest"][
            "candidate_cache_sha256"
        ],
        "llm_settings": llm_settings,
        "parent_return_source_sha256": _parent_return_sha256(),
        "e2e_source_sha256": _script_sha256(),
        "product_accuracy_claim": False,
        "financebench_phase4": "NOT_RUN",
    }


def _require_output(path: Path, *, phase: str) -> Path:
    target = path.expanduser().resolve()
    if phase == "plan":
        if not target.parent.is_dir():
            raise HoldoutError(f"output parent directory not found: {target.parent}")
        target.mkdir(exist_ok=True)
        marker = target / ".incomplete"
        if not marker.exists() and not (target / "aggregate.json").exists():
            marker.write_text(
                "LEDGER parent-page e2e has not completed.\n",
                encoding="utf-8",
            )
        return target
    if not target.is_dir() or not (target / "plan.json").is_file():
        raise HoldoutError("parent-page e2e plan phase is not complete")
    if (target / "aggregate.json").exists():
        raise HoldoutError("refusing to rerun a completed parent-page e2e output")
    return target


def _validate_completed(
    rows: list[dict],
    *,
    candidate_by_id: dict[str, dict],
    run_identity: str,
) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        candidate = candidate_by_id.get(query_id)
        if (
            not query_id
            or query_id in completed
            or candidate is None
            or row.get("run_identity_sha256") != run_identity
            or row.get("shared_candidate_identity_sha256")
            != candidate["candidate_identity_sha256"]
            or row.get("row_sha256") != _row_sha256(row)
            or set(row.get("arms") or {}) != set(ARMS)
        ):
            raise HoldoutError("completed parent-page e2e row identity mismatch")
        completed[query_id] = row
    return completed


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "plan" and args.allow_remote:
            raise HoldoutError("parent-page e2e plan is local-only")
        suffix_path = Path(args.suffix_aggregate).expanduser().resolve()
        if suffix_path != TRACKED_SUFFIX:
            raise HoldoutError("parent-page e2e must use the tracked suffix probe")
        suffix = paired_cli._load_json(suffix_path, label="sealed suffix probe")
        if suffix.get("recommended_next_workstream") != (
            "generate_chunk_vs_parent_page_on_suffix"
        ):
            raise HoldoutError("suffix probe did not authorize generation")
        e2e_path = Path(args.e2e_aggregate).expanduser().resolve()
        paired_path = Path(args.paired_aggregate).expanduser().resolve()
        if e2e_path != TRACKED_E2E:
            raise HoldoutError("parent-page e2e must use the tracked e2e aggregate")
        if paired_path != TRACKED_QWEN3:
            raise HoldoutError("parent-page e2e must use the tracked Qwen3 aggregate")
        e2e_aggregate = paired_cli._load_json(
            e2e_path,
            label="sealed e2e aggregate",
        )
        qwen3_aggregate = paired_cli._load_json(
            paired_path,
            label="sealed Qwen3 aggregate",
        )
        if (
            e2e_aggregate.get("selection", {}).get("query_ids_sha256")
            != suffix["selection"]["prefix_query_ids_sha256"]
            or qwen3_aggregate.get("selection", {}).get("query_ids_sha256")
            != suffix["selection"]["parent_query_ids_sha256"]
            or qwen3_aggregate.get("primary_comparison_valid") is not True
            or e2e_aggregate.get("product_accuracy_claim") is not False
        ):
            raise HoldoutError("parent-page e2e source aggregates are incompatible")
        frozen = suffix_cli._load_frozen(args)
        if (
            paired_cli._ids_sha256(tuple(frozen["suffix_ids"]))
            != suffix["selection"]["query_ids_sha256"]
        ):
            raise HoldoutError("suffix query identity diverged")
        paired_rows = paired_cli._read_jsonl(
            Path(args.paired_per_case),
            label="paired Qwen3 per-case",
        )
        paired_by_id = {str(row["query_id"]): row for row in paired_rows}
        if len(paired_by_id) != 250:
            raise HoldoutError("paired Qwen3 per-case coverage is not 250")
        cache_sha = frozen["candidate_manifest"]["candidate_cache_sha256"]
        if (
            cache_sha != EXPECTED_CANDIDATE_CACHE_SHA256
            or cache_sha
            != e2e_aggregate["candidate_manifest"]["candidate_cache_sha256"]
        ):
            raise HoldoutError("parent-page e2e candidate cache identity diverged")
        pages = pack_probe._page_by_id(frozen["dataset"])
        chunk_max = int(e2e_aggregate["rerank_settings"]["max_document_chars"])
        bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
        llm_settings_obj = LLMSettings.from_env()
        if not llm_settings_obj.api_key:
            raise HoldoutError("parent-page e2e requires DEEPSEEK_API_KEY")
        llm_public = e2e_cli._llm_settings_public(llm_settings_obj)
        plan = _public_plan(
            frozen=frozen,
            paired_by_id=paired_by_id,
            pages=pages,
            llm_settings=llm_public,
            chunk_max_chars=chunk_max,
        )
        if args.phase == "plan":
            output_dir = _require_output(Path(args.output_dir), phase="plan")
            plan_path = output_dir / "plan.json"
            if plan_path.exists():
                existing = json.loads(plan_path.read_text(encoding="utf-8"))
                if existing != plan:
                    raise HoldoutError("existing parent-page e2e plan has diverged")
            else:
                _atomic_json(plan_path, plan)
            print(json.dumps(plan, indent=2), flush=True)
            return 0
        if not args.allow_remote:
            raise HoldoutError("parent-page e2e run requires explicit --allow-remote")
        if not str(llm_settings_obj.base_url).startswith("https://"):
            raise HoldoutError("parent-page generator base URL must use HTTPS")
        output_dir = _require_output(Path(args.output_dir), phase="run")
        planned = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
        if planned != plan:
            raise HoldoutError("parent-page e2e plan changed after local freeze")
        run_identity = paired_cli._canonical_sha256(
            {
                "plan": plan,
                "suffix_query_ids_sha256": suffix["selection"]["query_ids_sha256"],
            }
        )
        per_case_path = output_dir / "per_case.jsonl"
        existing = (
            paired_cli._read_jsonl(per_case_path, label="parent-page e2e per-case")
            if per_case_path.exists()
            else []
        )
        candidate_by_id = {
            str(row["query_id"]): row for row in frozen["suffix_rows"]
        }
        completed = _validate_completed(
            existing,
            candidate_by_id=candidate_by_id,
            run_identity=run_identity,
        )
        ordered = list(existing)
        llm = DeepSeekChatClient(llm_settings_obj)
        for index, candidate in enumerate(frozen["suffix_rows"]):
            query_id = str(candidate["query_id"])
            if query_id in completed:
                continue
            query = frozen["query_by_id"][query_id]
            ranked = paired_by_id[query_id]["reranked_arms"][ARM]
            if bool(ranked.get("rerank_fallback")):
                raise HoldoutError("parent-page e2e refuses a Qwen3 fallback rank")
            identity = list(ranked["final_identity"])
            chunk_hits = _chunk_hits(candidate, identity)
            if ranked["final_identity_sha256"] != paired_cli._hash_hits(chunk_hits):
                raise HoldoutError("parent-page e2e Qwen3 identity diverged")
            parent_hits = build_parent_page_hits(identity, pages)
            row = {
                "query_id": query_id,
                "run_identity_sha256": run_identity,
                "shared_candidate_identity_sha256": candidate[
                    "candidate_identity_sha256"
                ],
                "final_identity_sha256": ranked["final_identity_sha256"],
                "arms": {
                    "chunk": _run_generate(
                        arm="chunk",
                        query=query,
                        hits=chunk_hits,
                        gold_value=float(frozen["gold"][query_id]),
                        llm=llm,
                        max_document_chars=chunk_max,
                    ),
                    "parent_page": _run_generate(
                        arm="parent_page",
                        query=query,
                        hits=parent_hits,
                        gold_value=float(frozen["gold"][query_id]),
                        llm=llm,
                        max_document_chars=parent_prompt_char_cap(parent_hits),
                    ),
                },
            }
            row["row_sha256"] = _row_sha256(row)
            ordered.append(row)
            _atomic_jsonl(per_case_path, ordered)
            completed[query_id] = row
            print(
                f"[parent-e2e] case={index + 1}/{len(frozen['suffix_rows'])} OK",
                flush=True,
            )
        rows = paired_cli._read_jsonl(per_case_path, label="parent-page e2e per-case")
        completed = _validate_completed(
            rows,
            candidate_by_id=candidate_by_id,
            run_identity=run_identity,
        )
        if tuple(completed) != tuple(frozen["suffix_ids"]):
            raise HoldoutError("parent-page e2e coverage mismatch")
        chunk_summary = _summarize(rows, "chunk")
        parent_summary = _summarize(rows, "parent_page")
        comparison = build_comparison(rows)
        generate_errors = (
            chunk_summary["generate_error_cases"]
            + parent_summary["generate_error_cases"]
        )
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "e2e_source_sha256": _script_sha256(),
            "parent_return_source_sha256": _parent_return_sha256(),
            "run_identity_sha256": run_identity,
            "split": "public_dev",
            "cases": len(rows),
            "selection": {
                **suffix["selection"],
                "gold_identity_sha256": plan["gold_identity_sha256"],
            },
            "candidate_manifest": {
                "candidate_cache_sha256": plan["candidate_cache_sha256"],
                "candidate_set_identity_sha256": frozen["candidate_manifest"][
                    "candidate_set_identity_sha256"
                ],
            },
            "llm_settings": llm_public,
            "chunk_max_document_chars": chunk_max,
            "comparison": comparison,
            "call_accounting": {
                "qwen3_calls": 0,
                "generate_logical_calls": len(rows) * 2,
                "generate_physical_attempts": (
                    chunk_summary["generate_attempts"]
                    + parent_summary["generate_attempts"]
                ),
                "generate_errors": int(generate_errors),
                "billing_semantics": "persisted_complete_cases_at_least_once",
                "unobserved_inflight_remote_calls_possible": True,
                "latency_scope": "generate_not_production_e2e",
            },
            "qwen3_calls": 0,
            "generate_calls": (
                chunk_summary["generate_attempts"]
                + parent_summary["generate_attempts"]
            ),
            "locked_strategy": LOCKED_STRATEGY,
            "arm": ARM,
            "primary_comparison_valid": generate_errors == 0,
            "product_accuracy_claim": False,
            "financebench_phase4": "NOT_RUN",
            "per_case_sha256": hashlib.sha256(per_case_path.read_bytes()).hexdigest(),
        }
        _atomic_json(output_dir / "aggregate.json", aggregate)
        (output_dir / ".incomplete").unlink(missing_ok=True)
        print(
            "[parent-e2e] E2E_OK "
            f"cases={len(rows)} "
            f"chunk={chunk_summary['numeric_accuracy']} "
            f"parent={parent_summary['numeric_accuracy']}",
            flush=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - redact CLI boundary failures
        safe = (
            exc
            if isinstance(exc, HoldoutError)
            else HoldoutError(
                f"LEDGER parent-page e2e failed: {type(exc).__name__}"
            )
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
