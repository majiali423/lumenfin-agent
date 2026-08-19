#!/usr/bin/env python3
"""Limited LEDGER e2e canary: frozen Hybrid candidates, A_prod, Qwen3 vs lexical."""
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

import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.env_bootstrap import bootstrap_dotenv
from lumenfin.eval.financebench.candidate_pool_ablation import (
    build_locked_qwen3_reranker,
    public_rerank_settings,
    snapshot_rerank_settings,
)
from lumenfin.eval.financebench.metrics import percentile_sorted
from lumenfin.eval.holdout import ARM_SPECS, HoldoutError, prepare_rerank_pool
from lumenfin.eval.holdout.ledger_e2e import (
    RELATIVE_TOLERANCE,
    SYSTEM_PROMPT,
    build_generation_prompt,
    load_ledger_gold_values,
    parse_answer_payload,
    score_generated_answer,
)
from lumenfin.llm import DeepSeekChatClient, LLMSettings
from lumenfin.rag.rerank import LexicalReranker
from lumenfin.stdio import configure_stdio_utf8

SCHEMA_VERSION = "lumenfin_ledger_e2e_canary.v1"
CASES_PER_COMPANY = 10
ARMS = ("lexical", "qwen3")
PARENT_QUERY_IDS_SHA256 = (
    "cb1654a41dec7ae04efd6666dd5ddfbcf29862631b1d0acbad884fb0402de044"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Limited LEDGER public-dev e2e canary on frozen Hybrid candidates. "
            "Generates numeric answers for A_prod after lexical and Qwen3 ranking."
        )
    )
    parser.add_argument("--phase", choices=("plan", "run"), required=True)
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-aggregate", required=True)
    parser.add_argument("--baseline-per-case", required=True)
    parser.add_argument("--prerank-aggregate", required=True)
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


def _llm_settings_public(settings: LLMSettings) -> dict[str, Any]:
    return {
        "model": settings.model,
        "timeout_seconds": settings.timeout_seconds,
        "max_retries": settings.max_retries,
        "temperature": 0.0,
        "max_tokens": 200,
        "base_url_sha256": hashlib.sha256(
            settings.base_url.encode()
        ).hexdigest(),
    }


def _prefix_candidate_rows(
    candidate_rows: list[dict],
    plans: list[dict],
    *,
    cases_per_company: int = CASES_PER_COMPANY,
) -> list[dict]:
    by_id = {str(row["query_id"]): row for row in candidate_rows}
    selected: list[dict] = []
    for plan in plans:
        query_ids = list(plan["query_ids"])
        if len(query_ids) < cases_per_company:
            raise HoldoutError("frozen company prefix is shorter than the canary")
        prefix = query_ids[:cases_per_company]
        seen: set[str] = set()
        for query_id in prefix:
            if query_id in seen:
                raise HoldoutError("e2e canary prefix contains duplicate query ids")
            seen.add(query_id)
            row = by_id.get(query_id)
            if row is None:
                raise HoldoutError("e2e canary query is missing from the candidate cache")
            selected.append(row)
    if len(selected) != cases_per_company * len(plans):
        raise HoldoutError("e2e canary coverage mismatch")
    return selected


def _gold_identity_sha256(values: dict[str, float]) -> str:
    payload = [
        {"query_id": query_id, "value": values[query_id]}
        for query_id in sorted(values)
    ]
    return paired_cli._canonical_sha256(payload)


def _arm_pool(hits: list[dict]) -> list[dict]:
    return prepare_rerank_pool(hits, arm="A_prod")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return round(percentile_sorted(values, q), 2)


def _summarize(rows: list[dict], arm: str) -> dict[str, Any]:
    arm_rows = [row["arms"][arm] for row in rows]
    matches = [bool(item["numeric_match"]) for item in arm_rows]
    citations = [bool(item["citation_supported"]) for item in arm_rows]
    fallbacks = [bool(item["rerank_fallback"]) for item in arm_rows]
    totals = [float(item["total_latency_ms"]) for item in arm_rows]
    generate = [float(item["generate_latency_ms"]) for item in arm_rows]
    rerank = [float(item["rerank_latency_ms"]) for item in arm_rows]
    return {
        "arm": arm,
        "cases": len(arm_rows),
        "numeric_accuracy": round(mean(float(item) for item in matches), 4),
        "citation_support_rate": round(
            mean(float(item) for item in citations),
            4,
        ),
        "abstain_rate": round(
            mean(float(item["abstain"]) for item in arm_rows),
            4,
        ),
        "fallback_cases": sum(fallbacks),
        "generate_error_cases": sum(
            bool(item.get("generate_error_type")) for item in arm_rows
        ),
        "qwen3_attempts": sum(int(item["rerank_attempts"]) for item in arm_rows)
        if arm == "qwen3"
        else 0,
        "generate_attempts": sum(int(item["generate_attempts"]) for item in arm_rows),
        "generate_tokens": sum(int(item["generate_tokens"]) for item in arm_rows),
        "latency_ms": {
            "mean_total": round(mean(totals), 2) if totals else 0.0,
            "p50_total": _percentile(totals, 0.5),
            "p95_total": _percentile(totals, 0.95),
            "mean_rerank": round(mean(rerank), 2) if rerank else 0.0,
            "mean_generate": round(mean(generate), 2) if generate else 0.0,
        },
    }


def _paired_numeric(rows: list[dict]) -> dict[str, int]:
    counts = {"gain": 0, "loss": 0, "unchanged": 0}
    for row in rows:
        before = bool(row["arms"]["lexical"]["numeric_match"])
        after = bool(row["arms"]["qwen3"]["numeric_match"])
        if after and not before:
            counts["gain"] += 1
        elif before and not after:
            counts["loss"] += 1
        else:
            counts["unchanged"] += 1
    return counts


def _row_sha256(row: dict) -> str:
    return paired_cli._canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )


def _run_arm(
    *,
    arm: str,
    query: dict,
    hits: list[dict],
    gold_value: float,
    reranker: Any,
    llm: DeepSeekChatClient,
    max_document_chars: int,
) -> dict[str, Any]:
    pool = _arm_pool(hits)
    started = time.perf_counter()
    ranked, rerank_meta = reranker.rerank(
        str(query["query_text"]),
        pool,
        top_k=ARM_SPECS["A_prod"].final_k,
    )
    rerank_ms = round((time.perf_counter() - started) * 1000.0, 2)
    prompt = build_generation_prompt(
        query_text=str(query["query_text"]),
        hits=ranked,
        max_document_chars=max_document_chars,
    )
    llm.mark_usage_start()
    generate_started = time.perf_counter()
    raw = llm.chat(SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=200)
    generate_ms = round((time.perf_counter() - generate_started) * 1000.0, 2)
    parsed = parse_answer_payload(raw)
    scored = score_generated_answer(
        gold_value=gold_value,
        parsed=parsed,
        hits=ranked,
        qrels={
            str(item["doc_id"]): int(item["relevance"])
            for item in query["qrels"]
        },
    )
    usage = llm.usage_since_mark()
    return {
        "arm": arm,
        "pool_identity_sha256": paired_cli._hash_hits(pool),
        "final_identity_sha256": paired_cli._hash_hits(ranked),
        "final_identity": paired_cli._hit_identity(ranked),
        "predicted_value": parsed["value"],
        "gold_value": gold_value,
        "cited_chunk_ids": parsed["cited_chunk_ids"],
        "rerank_attempts": int(rerank_meta.get("rerank_attempts") or 0),
        "rerank_tokens": int(rerank_meta.get("rerank_tokens") or 0),
        "rerank_fallback": bool(rerank_meta.get("rerank_fallback")),
        "rerank_error_type": str(rerank_meta.get("rerank_error_type") or ""),
        "rerank_latency_ms": float(rerank_meta.get("rerank_latency_ms") or rerank_ms),
        "generate_attempts": int(getattr(llm, "last_attempts", 1) or 1),
        "generate_tokens": int(usage["prompt_tokens"] + usage["completion_tokens"]),
        "generate_latency_ms": generate_ms,
        "total_latency_ms": round(
            float(rerank_meta.get("rerank_latency_ms") or rerank_ms) + generate_ms,
            2,
        ),
        "generate_error_type": "",
        **scored,
    }


def _validate_completed_rows(
    rows: list[dict],
    *,
    candidate_by_id: dict[str, dict],
    run_identity_sha256: str,
) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        candidate = candidate_by_id.get(query_id)
        if (
            not query_id
            or query_id in completed
            or candidate is None
            or row.get("run_identity_sha256") != run_identity_sha256
            or row.get("shared_candidate_identity_sha256")
            != candidate["candidate_identity_sha256"]
            or row.get("row_sha256") != _row_sha256(row)
        ):
            raise HoldoutError("completed e2e row identity mismatch")
        arms = row.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARMS):
            raise HoldoutError("completed e2e arm identity mismatch")
        completed[query_id] = row
    return completed


def _build_run_identity(
    *,
    candidate_manifest: dict,
    selected_ids: list[str],
    gold_identity_sha256: str,
    rerank_settings: dict,
    llm_settings: dict,
) -> str:
    return paired_cli._canonical_sha256(
        {
            "candidate_cache_sha256": candidate_manifest[
                "candidate_cache_sha256"
            ],
            "candidate_set_identity_sha256": candidate_manifest[
                "candidate_set_identity_sha256"
            ],
            "selected_query_ids_sha256": paired_cli._ids_sha256(tuple(selected_ids)),
            "gold_identity_sha256": gold_identity_sha256,
            "rerank_settings": rerank_settings,
            "llm_settings": llm_settings,
            "reranker_source_sha256": paired_cli._reranker_source_sha256(),
            "e2e_source_sha256": _script_sha256(),
            "relative_tolerance": RELATIVE_TOLERANCE,
            "arm": "A_prod",
        }
    )


def _require_output(path: Path, *, phase: str) -> Path:
    target = path.expanduser().resolve()
    if phase == "plan":
        if not target.parent.is_dir():
            raise HoldoutError(f"output parent directory not found: {target.parent}")
        target.mkdir(exist_ok=True)
        marker = target / ".incomplete"
        if not marker.exists() and not (target / "aggregate.json").exists():
            marker.write_text(
                "LEDGER e2e canary has not completed.\n",
                encoding="utf-8",
            )
        return target
    if not target.is_dir() or not (target / "plan.json").is_file():
        raise HoldoutError("e2e plan phase is not complete")
    if (target / "aggregate.json").exists():
        raise HoldoutError("refusing to rerun a completed e2e output")
    return target


def _load_plan_inputs(args: argparse.Namespace):
    class _Proxy:
        pass

    proxy = _Proxy()
    proxy.parquet_path = args.parquet_path
    proxy.manifest = args.manifest
    proxy.split_salt = args.split_salt
    proxy.baseline_aggregate = args.baseline_aggregate
    proxy.baseline_per_case = args.baseline_per_case
    proxy.prerank_aggregate = args.prerank_aggregate
    proxy.batch_size = 64
    proxy.embedding_dimension = 1024
    manifest, dataset, plans, qrel_audit, prerank = paired_cli._load_context(proxy)
    candidate_dir = Path(args.candidate_dir).expanduser().resolve()
    candidate_rows, candidate_manifest = paired_cli._validate_candidate_cache(
        candidate_dir,
        manifest=manifest,
        plans=plans,
        qrel_audit=qrel_audit,
        prerank=prerank,
    )
    if len(plans) != 5:
        raise HoldoutError("e2e canary requires the frozen 5-company plan")
    parent_ids = tuple(
        str(query_id) for plan in plans for query_id in plan["query_ids"]
    )
    if paired_cli._ids_sha256(parent_ids) != PARENT_QUERY_IDS_SHA256:
        raise HoldoutError("e2e parent query identity mismatch")
    selected = _prefix_candidate_rows(candidate_rows, plans)
    selected_ids = [str(row["query_id"]) for row in selected]
    gold = load_ledger_gold_values(args.parquet_path, query_ids=selected_ids)
    query_by_id = {str(query["query_id"]): query for query in dataset.queries}
    return {
        "manifest": manifest,
        "plans": plans,
        "prerank": prerank,
        "candidate_manifest": candidate_manifest,
        "selected": selected,
        "selected_ids": selected_ids,
        "gold": gold,
        "query_by_id": query_by_id,
        "gold_identity_sha256": _gold_identity_sha256(gold),
    }


def _public_plan(bundle: dict, *, rerank_settings: dict, llm_settings: dict) -> dict:
    selected = bundle["selected"]
    request_chars = 0
    document_slots = 0
    prompt_chars = 0
    for row in selected:
        query = bundle["query_by_id"][str(row["query_id"])]
        pool = _arm_pool(list(row["hits"]))
        document_slots += len(pool)
        request_chars += len(str(query["query_text"])) * len(pool) + sum(
            len(str(hit["text"])[: int(rerank_settings["max_document_chars"])])
            for hit in pool
        )
        prompt_chars += len(
            build_generation_prompt(
                query_text=str(query["query_text"]),
                hits=pool[: ARM_SPECS["A_prod"].final_k],
                max_document_chars=int(rerank_settings["max_document_chars"]),
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "parent_query_ids_sha256": PARENT_QUERY_IDS_SHA256,
        "cases": len(selected),
        "companies": 5,
        "cases_per_company": CASES_PER_COMPANY,
        "arm": "A_prod",
        "ranking_arms": list(ARMS),
        "qwen3_requests_without_retries": len(selected),
        "lexical_requests": len(selected),
        "generate_requests_without_retries": len(selected) * 2,
        "document_slots": document_slots,
        "qwen3_request_chars": request_chars,
        "generate_prompt_chars_upper_bound": prompt_chars * 2,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "candidate_cache_sha256": bundle["candidate_manifest"][
            "candidate_cache_sha256"
        ],
        "selected_query_ids_sha256": paired_cli._ids_sha256(
            tuple(bundle["selected_ids"])
        ),
        "gold_identity_sha256": bundle["gold_identity_sha256"],
        "rerank_settings": rerank_settings,
        "llm_settings": llm_settings,
        "reranker_source_sha256": paired_cli._reranker_source_sha256(),
        "e2e_source_sha256": _script_sha256(),
        "product_accuracy_claim": False,
        "financebench_phase4": "NOT_RUN",
    }


def _run_canary(
    *,
    output_dir: Path,
    bundle: dict,
    rerank_settings: dict,
    llm_settings_obj: LLMSettings,
) -> dict:
    llm_public = _llm_settings_public(llm_settings_obj)
    run_identity = _build_run_identity(
        candidate_manifest=bundle["candidate_manifest"],
        selected_ids=bundle["selected_ids"],
        gold_identity_sha256=bundle["gold_identity_sha256"],
        rerank_settings=rerank_settings,
        llm_settings=llm_public,
    )
    planned = _load_json_obj(output_dir / "plan.json")
    expected_plan = _public_plan(
        bundle,
        rerank_settings=rerank_settings,
        llm_settings=llm_public,
    )
    if planned != expected_plan:
        raise HoldoutError("e2e plan changed after local freeze")
    paired_cli._validate_candidate_queries(
        bundle["selected"],
        bundle["query_by_id"],
    )
    lexical = LexicalReranker()
    qwen3 = build_locked_qwen3_reranker(
        snapshot_rerank_settings()
    )
    llm = DeepSeekChatClient(llm_settings_obj)
    candidate_by_id = {
        str(row["query_id"]): row for row in bundle["selected"]
    }
    per_case_path = output_dir / "per_case.jsonl"
    existing = (
        paired_cli._read_jsonl(per_case_path, label="e2e per-case")
        if per_case_path.exists()
        else []
    )
    completed = _validate_completed_rows(
        existing,
        candidate_by_id=candidate_by_id,
        run_identity_sha256=run_identity,
    )
    ordered = list(existing)
    for index, candidate in enumerate(bundle["selected"]):
        query_id = str(candidate["query_id"])
        if query_id in completed:
            continue
        query = bundle["query_by_id"][query_id]
        row = {
            "query_id": query_id,
            "run_identity_sha256": run_identity,
            "shared_candidate_identity_sha256": candidate[
                "candidate_identity_sha256"
            ],
            "arms": {
                "lexical": _run_arm(
                    arm="lexical",
                    query=query,
                    hits=list(candidate["hits"]),
                    gold_value=bundle["gold"][query_id],
                    reranker=lexical,
                    llm=llm,
                    max_document_chars=int(
                        rerank_settings["max_document_chars"]
                    ),
                ),
                "qwen3": _run_arm(
                    arm="qwen3",
                    query=query,
                    hits=list(candidate["hits"]),
                    gold_value=bundle["gold"][query_id],
                    reranker=qwen3,
                    llm=llm,
                    max_document_chars=int(
                        rerank_settings["max_document_chars"]
                    ),
                ),
            },
        }
        row["row_sha256"] = _row_sha256(row)
        ordered.append(row)
        _atomic_jsonl(per_case_path, ordered)
        completed[query_id] = row
        print(
            f"[ledger-e2e] case={index + 1}/{len(bundle['selected'])} OK",
            flush=True,
        )
    rows = paired_cli._read_jsonl(per_case_path, label="e2e per-case")
    completed = _validate_completed_rows(
        rows,
        candidate_by_id=candidate_by_id,
        run_identity_sha256=run_identity,
    )
    if tuple(completed) != tuple(bundle["selected_ids"]):
        raise HoldoutError("e2e per-case coverage mismatch")
    lexical_summary = _summarize(rows, "lexical")
    qwen3_summary = _summarize(rows, "qwen3")
    qwen3_attempts = qwen3_summary["qwen3_attempts"]
    generate_attempts = (
        lexical_summary["generate_attempts"] + qwen3_summary["generate_attempts"]
    )
    fallbacks = qwen3_summary["fallback_cases"]
    generate_errors = (
        lexical_summary["generate_error_cases"]
        + qwen3_summary["generate_error_cases"]
    )
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "e2e_source_sha256": _script_sha256(),
        "reranker_source_sha256": paired_cli._reranker_source_sha256(),
        "run_identity_sha256": run_identity,
        "split": "public_dev",
        "cases": len(rows),
        "selection": {
            "strategy": "frozen_5x50_company_prefix_v1",
            "companies": 5,
            "cases": len(rows),
            "cases_per_company": CASES_PER_COMPANY,
            "parent_query_ids_sha256": PARENT_QUERY_IDS_SHA256,
            "query_ids_sha256": expected_plan["selected_query_ids_sha256"],
            "gold_identity_sha256": bundle["gold_identity_sha256"],
        },
        "candidate_manifest": {
            "candidate_cache_sha256": bundle["candidate_manifest"][
                "candidate_cache_sha256"
            ],
            "candidate_set_identity_sha256": bundle["candidate_manifest"][
                "candidate_set_identity_sha256"
            ],
        },
        "rerank_settings": rerank_settings,
        "llm_settings": llm_public,
        "comparison": {
            "lexical": lexical_summary,
            "qwen3": qwen3_summary,
            "delta_qwen3_minus_lexical": {
                "numeric_accuracy": round(
                    qwen3_summary["numeric_accuracy"]
                    - lexical_summary["numeric_accuracy"],
                    4,
                ),
                "citation_support_rate": round(
                    qwen3_summary["citation_support_rate"]
                    - lexical_summary["citation_support_rate"],
                    4,
                ),
            },
            "paired_numeric_match": _paired_numeric(rows),
        },
        "call_accounting": {
            "candidate_embedding_remote_calls": 0,
            "qwen3_logical_calls": len(rows),
            "qwen3_physical_attempts": qwen3_attempts,
            "generate_logical_calls": len(rows) * 2,
            "generate_physical_attempts": generate_attempts,
            "rerank_fallbacks": int(fallbacks),
            "generate_errors": int(generate_errors),
            "billing_semantics": "persisted_complete_cases_at_least_once",
            "unobserved_inflight_remote_calls_possible": True,
            "latency_scope": "rerank_plus_generate_not_production_e2e",
        },
        "per_case_sha256": hashlib.sha256(per_case_path.read_bytes()).hexdigest(),
        "qwen3_calls": qwen3_attempts,
        "generate_calls": generate_attempts,
        "primary_comparison_valid": fallbacks == 0 and generate_errors == 0,
        "product_accuracy_claim": False,
        "financebench_phase4": "NOT_RUN",
    }
    _atomic_json(output_dir / "aggregate.json", aggregate)
    (output_dir / ".incomplete").unlink(missing_ok=True)
    print(
        "[ledger-e2e] E2E_OK "
        f"cases={len(rows)} qwen3_attempts={qwen3_attempts} "
        f"fallbacks={fallbacks}",
        flush=True,
    )
    return aggregate


def _load_json_obj(path: Path) -> dict:
    return paired_cli._load_json(path, label="e2e plan")


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "plan" and args.allow_remote:
            raise HoldoutError("e2e plan is local-only")
        bundle = _load_plan_inputs(args)
        if args.phase == "plan":
            bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
            rerank_settings = public_rerank_settings(snapshot_rerank_settings())
            llm_settings = LLMSettings.from_env()
            if not llm_settings.api_key:
                raise HoldoutError("e2e plan requires DEEPSEEK_API_KEY to freeze identity")
            output_dir = _require_output(Path(args.output_dir), phase="plan")
            frozen = _public_plan(
                bundle,
                rerank_settings=rerank_settings,
                llm_settings=_llm_settings_public(llm_settings),
            )
            plan_path = output_dir / "plan.json"
            if plan_path.exists():
                if _load_json_obj(plan_path) != frozen:
                    raise HoldoutError("existing e2e plan has diverged")
            else:
                _atomic_json(plan_path, frozen)
            print(json.dumps(frozen, indent=2), flush=True)
            return 0
        if not args.allow_remote:
            raise HoldoutError("e2e run requires explicit --allow-remote")
        bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
        paired_cli._validate_remote_rerank_configuration(
            snapshot_rerank_settings()
        )
        llm_settings = LLMSettings.from_env()
        if not llm_settings.api_key:
            raise HoldoutError("e2e run requires DEEPSEEK_API_KEY")
        if not str(llm_settings.base_url).startswith("https://"):
            raise HoldoutError("e2e generator base URL must use HTTPS")
        output_dir = _require_output(Path(args.output_dir), phase="run")
        _run_canary(
            output_dir=output_dir,
            bundle=bundle,
            rerank_settings=public_rerank_settings(snapshot_rerank_settings()),
            llm_settings_obj=llm_settings,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - redact CLI boundary failures
        safe = (
            exc
            if isinstance(exc, HoldoutError)
            else HoldoutError(f"LEDGER e2e canary failed: {type(exc).__name__}")
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
