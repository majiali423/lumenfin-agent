#!/usr/bin/env python3
"""Freeze Hybrid candidates, then run paired Qwen3 reranking on LEDGER public-dev."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_ledger_public_dev_hybrid_stratified as stratified_cli
import run_ledger_public_dev_ranking as retrieval_cli

from lumenfin.env_bootstrap import bootstrap_dotenv
from lumenfin.eval.financebench.candidate_pool_ablation import (
    build_locked_qwen3_reranker,
    public_rerank_settings,
    snapshot_rerank_settings,
)
from lumenfin.eval.financebench.retrieval import build_eval_store
from lumenfin.eval.holdout import (
    ARM_SPECS,
    HoldoutError,
    build_ledger_public_dev_dataset,
    iter_ledger_parquet_rows,
    ledger_public_dev_qrel_audit,
    ledger_snapshot_sha256,
    prepare_rerank_pool,
    score_ledger_public_dev,
    summarize_ranking_cases,
)
from lumenfin.stdio import configure_stdio_utf8

SCHEMA_VERSION = "lumenfin_ledger_qwen3_paired.v1"
CANDIDATE_SCHEMA = "lumenfin_ledger_hybrid_candidates.v1"
TRACKED_PRERANK = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_hybrid_stratified_5x50.json"
).resolve()
PINNED_PRERANK_CANONICAL_SHA256 = (
    "d28f4548394ad9aaa7942533ee44a1643cad8d465ade9ab227f676585b3276e6"
)
ALLOWED_HIT_FIELDS = (
    "chunk_id",
    "document_id",
    "source_document_id",
    "filename",
    "page",
    "text",
    "companies",
    "chunk_type",
    "score",
    "fusion_score",
    "retrieval_method",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Three-phase LEDGER Qwen3 evaluation: freeze Hybrid candidates, "
            "inspect rerank cost, then run paired A_prod/R_page reranking."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("candidates", "plan-rerank", "rerank"),
        required=True,
    )
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-aggregate", required=True)
    parser.add_argument("--baseline-per-case", required=True)
    parser.add_argument("--prerank-aggregate", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--allow-remote", action="store_true")
    return parser


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def _reranker_source_paths() -> tuple[Path, ...]:
    return (
        ROOT / "src" / "lumenfin" / "provider_retry.py",
        ROOT / "src" / "lumenfin" / "provider_resilience.py",
        ROOT / "src" / "lumenfin" / "rag" / "rerank.py",
        ROOT
        / "src"
        / "lumenfin"
        / "eval"
        / "financebench"
        / "candidate_pool_ablation.py",
        Path(__file__),
    )


def _reranker_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in _reranker_source_paths():
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(
            path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _ordered_identity_sha256(values: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(
            values,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _ids_sha256(values: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutError(f"cannot read {label}") from exc
    if not isinstance(payload, dict):
        raise HoldoutError(f"{label} must be an object")
    return payload


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


def _read_jsonl(path: Path, *, label: str) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HoldoutError(f"cannot read {label}") from exc
    return _parse_jsonl_text("\n".join(lines), label=label)


def _parse_jsonl_text(text: str, *, label: str) -> list[dict]:
    rows: list[dict] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HoldoutError(f"{label} contains invalid JSON") from exc
        if not isinstance(row, dict):
            raise HoldoutError(f"{label} row must be an object")
        rows.append(row)
    return rows


def _prepare_fresh_output(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists():
        raise HoldoutError(f"refusing to reuse output for candidate phase: {target}")
    if not target.parent.is_dir():
        raise HoldoutError(f"output parent directory not found: {target.parent}")
    target.mkdir()
    (target / ".incomplete").write_text(
        "LEDGER Qwen3 paired run has not completed.\n",
        encoding="utf-8",
    )
    return target


def _require_candidate_output(path: Path) -> Path:
    target = path.expanduser().resolve()
    if not target.is_dir() or not (target / ".candidates_ready").is_file():
        raise HoldoutError("candidate phase is not complete")
    if (target / "aggregate.json").exists():
        raise HoldoutError("refusing to rerun a completed Qwen3 output")
    return target


def _load_context(args: argparse.Namespace):
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = retrieval_cli._load_manifest(manifest_path)
    snapshot = ledger_snapshot_sha256(args.parquet_path)
    holdout_fraction = retrieval_cli._validate_snapshot_and_salt(
        manifest=manifest,
        snapshot_sha256=snapshot,
        salt=str(args.split_salt),
    )
    dataset = build_ledger_public_dev_dataset(
        iter_ledger_parquet_rows(args.parquet_path),
        salt=str(args.split_salt),
        holdout_fraction=holdout_fraction,
        manifest=manifest,
    )
    plans = stratified_cli._build_plans(
        dataset,
        company_count=5,
        cases_per_company=50,
        batch_size=int(args.batch_size),
    )
    source_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    qrel_audit = ledger_public_dev_qrel_audit(
        dataset,
        source_queries=int(manifest["splits"]["public_dev"]["queries"]),
    )
    stratified_cli._validate_sealed_baseline(
        aggregate_path=Path(args.baseline_aggregate),
        per_case_path=Path(args.baseline_per_case),
        manifest=manifest,
        source_manifest_sha256=source_manifest_sha256,
        expected_qrel_audit=qrel_audit,
    )
    prerank_path = Path(args.prerank_aggregate).expanduser().resolve()
    if prerank_path != TRACKED_PRERANK:
        raise HoldoutError("prerank aggregate must be the tracked sealed artifact")
    prerank = _load_json(prerank_path, label="sealed Hybrid prerank aggregate")
    if (
        _canonical_sha256(prerank) != PINNED_PRERANK_CANONICAL_SHA256
        or prerank.get("dataset_snapshot_sha256")
        != manifest["dataset_snapshot_sha256"]
        or prerank.get("selection", {}).get("query_ids_sha256")
        != stratified_cli._public_plan(plans)["query_ids_sha256"]
        or prerank.get("qwen3_calls") != 0
    ):
        raise HoldoutError("sealed Hybrid prerank identity is incompatible")
    child_config = prerank.get("child_run_config") or {}
    if (
        int(args.batch_size) != 64
        or int(args.embedding_dimension)
        != int(child_config.get("embedding_dimension") or 0)
        or child_config.get("embedding_provider") != "dashscope"
        or child_config.get("embedding_model") != "text-embedding-v4"
        or child_config.get("retrieval_mode") != "hybrid"
        or child_config.get("evaluator_source_sha256")
        != retrieval_cli._evaluator_source_sha256()
    ):
        raise HoldoutError("candidate retrieval configuration is not locked")
    return manifest, dataset, plans, qrel_audit, prerank


def _sanitize_hit(hit: dict) -> dict:
    sanitized = {
        field: hit[field]
        for field in ALLOWED_HIT_FIELDS
        if field in hit
    }
    if not str(sanitized.get("chunk_id") or ""):
        raise HoldoutError("candidate cache hit is missing chunk_id")
    if not str(sanitized.get("document_id") or ""):
        raise HoldoutError("candidate cache hit is missing document_id")
    if not str(sanitized.get("text") or "").strip():
        raise HoldoutError("candidate cache hit is missing text")
    return sanitized


def _candidate_row(
    *,
    query: dict,
    hits: list[dict],
) -> dict:
    sanitized = [_sanitize_hit(hit) for hit in hits]
    identity = [
        {
            "chunk_id": str(hit["chunk_id"]),
            "document_id": str(hit["document_id"]),
        }
        for hit in sanitized
    ]
    return {
        "query_id": str(query["query_id"]),
        "query_text_sha256": hashlib.sha256(
            str(query["query_text"]).encode()
        ).hexdigest(),
        "company_key_sha256": hashlib.sha256(
            str(query["company_key"]).encode()
        ).hexdigest(),
        "candidate_identity_sha256": _ordered_identity_sha256(identity),
        "hits": sanitized,
    }


def _candidate_set_identity(rows: list[dict]) -> str:
    return _ordered_identity_sha256(
        [
            {
                "query_id": str(row["query_id"]),
                "candidate_identity_sha256": str(
                    row["candidate_identity_sha256"]
                ),
            }
            for row in rows
        ]
    )


def _validate_candidate_call_accounting(
    candidate_manifest: dict,
    *,
    selection: dict,
) -> None:
    calls = candidate_manifest.get("call_accounting") or {}
    document_calls = calls.get("document_embedding_remote_calls")
    query_calls = calls.get("query_embedding_remote_calls")
    if (
        calls.get("documents_indexed") != selection["documents"]
        or calls.get("chunks_indexed") != selection["chunks"]
        or calls.get("embed_chars") != selection["embed_chars"]
        or not isinstance(document_calls, int)
        or isinstance(document_calls, bool)
        or document_calls < selection["estimated_document_http_calls"]
        or not isinstance(query_calls, int)
        or isinstance(query_calls, bool)
        or query_calls < selection["expected_query_http_calls_minimum"]
        or calls.get("remote_calls") != document_calls + query_calls
    ):
        raise HoldoutError("candidate cache call accounting is incompatible")


def _build_candidates(
    args: argparse.Namespace,
    *,
    manifest: dict,
    dataset,
    plans: list[dict],
    qrel_audit: dict,
    prerank: dict,
) -> dict:
    if not args.allow_remote:
        raise HoldoutError("candidate phase requires explicit --allow-remote")
    bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
    output_dir = _prepare_fresh_output(Path(args.output_dir))
    rows: list[dict] = []
    totals = {
        "documents_indexed": 0,
        "chunks_indexed": 0,
        "embed_chars": 0,
        "document_embedding_remote_calls": 0,
        "query_embedding_remote_calls": 0,
        "remote_calls": 0,
    }
    try:
        for plan_index, plan in enumerate(plans):
            scoped = retrieval_cli._subset_for_company(
                dataset,
                company_key=str(plan["company_key"]),
                max_cases=int(plan["selected_cases"]),
            )
            index_dir = output_dir / f"_index_{plan_index:03d}_{uuid4().hex[:8]}"
            index_dir.mkdir()
            store = None
            try:
                store = build_eval_store(
                    uri=str(index_dir / "ledger_public_dev.db"),
                    embedding_provider="dashscope",
                    embedding_dimension=int(args.embedding_dimension),
                    collection_name="lumenfin_ledger_qwen3_candidates",
                    allow_remote=True,
                    mode="hybrid",
                    embedding_model="text-embedding-v4",
                )
                session_id = (
                    f"ledger-qwen3-candidates-"
                    f"{manifest['dataset_snapshot_sha256'][:12]}-{plan_index}"
                )
                index_stats = retrieval_cli._index_documents(
                    store,
                    scoped,
                    session_id=session_id,
                    batch_size=int(args.batch_size),
                )
                if (
                    index_stats["documents_indexed"] != plan["documents"]
                    or index_stats["chunks_indexed"] != plan["chunks"]
                    or index_stats["embed_chars"] != plan["embed_chars"]
                    or index_stats["embed_physical_calls"]
                    < plan["estimated_document_http_calls"]
                ):
                    raise HoldoutError("candidate index workload identity mismatch")
                totals["documents_indexed"] += index_stats["documents_indexed"]
                totals["chunks_indexed"] += index_stats["chunks_indexed"]
                totals["embed_chars"] += index_stats["embed_chars"]
                totals["document_embedding_remote_calls"] += index_stats[
                    "embed_physical_calls"
                ]
                for query in scoped.queries:
                    hits, meta = retrieval_cli._retrieve_candidates(
                        store,
                        query=str(query["query_text"]),
                        company=str(query["company_key"]),
                        top_k=ARM_SPECS["A_prod"].source_k,
                        session_id=session_id,
                        mode="hybrid",
                    )
                    query_calls = int(meta["remote_calls"])
                    totals["query_embedding_remote_calls"] += query_calls
                    rows.append(_candidate_row(query=query, hits=hits))
            finally:
                if store is not None:
                    store.close()
                shutil.rmtree(index_dir, ignore_errors=True)
            print(
                f"[ledger-qwen3] candidates company={plan_index + 1}/{len(plans)} OK",
                flush=True,
            )
    except Exception as exc:
        if isinstance(exc, HoldoutError):
            raise
        raise HoldoutError(
            f"candidate generation failed: {type(exc).__name__}"
        ) from None
    totals["remote_calls"] = (
        totals["document_embedding_remote_calls"]
        + totals["query_embedding_remote_calls"]
    )
    expected_query_ids = tuple(
        str(query_id)
        for plan in plans
        for query_id in plan["query_ids"]
    )
    if (
        tuple(str(row["query_id"]) for row in rows) != expected_query_ids
        or totals["query_embedding_remote_calls"]
        < sum(int(plan["expected_query_http_calls"]) for plan in plans)
    ):
        raise HoldoutError("candidate query coverage or call accounting mismatch")
    cache_path = output_dir / "candidates.jsonl"
    _atomic_jsonl(cache_path, rows)
    cache_sha256 = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    manifest_payload = {
        "schema_version": CANDIDATE_SCHEMA,
        "orchestrator_source_sha256": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode()
        ).hexdigest(),
        "retrieval_evaluator_source_sha256": (
            retrieval_cli._evaluator_source_sha256()
        ),
        "dataset_snapshot_sha256": manifest["dataset_snapshot_sha256"],
        "selection": stratified_cli._public_plan(plans),
        "source_prerank_canonical_sha256": _canonical_sha256(prerank),
        "source_prerank_child_run_config_sha256": prerank[
            "child_run_config_sha256"
        ],
        "retrieval_batch_size": int(args.batch_size),
        "qrel_corpus_audit": qrel_audit,
        "cases": len(rows),
        "candidate_cache_sha256": cache_sha256,
        "candidate_set_identity_sha256": _candidate_set_identity(rows),
        "call_accounting": totals,
        "qwen3_calls": 0,
        "contains_local_candidate_text": True,
        "tracked_artifact_allowed": False,
    }
    _atomic_json(output_dir / "candidate_manifest.json", manifest_payload)
    (output_dir / ".candidates_ready").write_text(
        "Candidate cache is complete and frozen before Qwen3.\n",
        encoding="utf-8",
    )
    print(
        "[ledger-qwen3] CANDIDATES_READY "
        f"cases={len(rows)} remote_calls={totals['remote_calls']} qwen3_calls=0",
        flush=True,
    )
    return manifest_payload


def _validate_candidate_cache(
    output_dir: Path,
    *,
    manifest: dict,
    plans: list[dict],
    qrel_audit: dict,
    prerank: dict,
) -> tuple[list[dict], dict]:
    candidate_manifest = _load_json(
        output_dir / "candidate_manifest.json",
        label="candidate manifest",
    )
    cache_path = output_dir / "candidates.jsonl"
    cache_bytes = cache_path.read_bytes()
    cache_hash = hashlib.sha256(cache_bytes).hexdigest()
    try:
        rows = _parse_jsonl_text(
            cache_bytes.decode("utf-8"),
            label="candidate cache",
        )
    except UnicodeDecodeError as exc:
        raise HoldoutError("candidate cache is not UTF-8") from exc
    expected_ids = tuple(
        str(query_id)
        for plan in plans
        for query_id in plan["query_ids"]
    )
    selection = stratified_cli._public_plan(plans)
    _validate_candidate_call_accounting(
        candidate_manifest,
        selection=selection,
    )
    if (
        candidate_manifest.get("schema_version") != CANDIDATE_SCHEMA
        or candidate_manifest.get("orchestrator_source_sha256")
        != hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode()
        ).hexdigest()
        or candidate_manifest.get("retrieval_evaluator_source_sha256")
        != retrieval_cli._evaluator_source_sha256()
        or candidate_manifest.get("dataset_snapshot_sha256")
        != manifest["dataset_snapshot_sha256"]
        or candidate_manifest.get("qrel_corpus_audit") != qrel_audit
        or candidate_manifest.get("source_prerank_canonical_sha256")
        != _canonical_sha256(prerank)
        or candidate_manifest.get("source_prerank_child_run_config_sha256")
        != prerank["child_run_config_sha256"]
        or candidate_manifest.get("retrieval_batch_size") != 64
        or candidate_manifest.get("candidate_cache_sha256") != cache_hash
        or candidate_manifest.get("candidate_set_identity_sha256")
        != _candidate_set_identity(rows)
        or tuple(str(row.get("query_id") or "") for row in rows) != expected_ids
        or candidate_manifest.get("cases") != len(expected_ids)
        or candidate_manifest.get("qwen3_calls") != 0
    ):
        raise HoldoutError("candidate cache identity is incompatible")
    for row in rows:
        if (
            len(row.get("hits") or []) > ARM_SPECS["A_prod"].source_k
            or not row.get("hits")
        ):
            raise HoldoutError("candidate cache has invalid source window")
        expected_identity = _candidate_row(
            query={
                "query_id": row["query_id"],
                "query_text": "",
                "company_key": "",
            },
            hits=list(row["hits"]),
        )["candidate_identity_sha256"]
        if row.get("candidate_identity_sha256") != expected_identity:
            raise HoldoutError("candidate row identity hash mismatch")
    return rows, candidate_manifest


def _rerank_plan(
    candidate_rows: list[dict],
    query_by_id: dict[str, dict],
    *,
    max_document_chars: int,
    max_attempts: int,
) -> dict:
    requests = 0
    document_slots = 0
    request_chars = 0
    for row in candidate_rows:
        query = query_by_id[str(row["query_id"])]
        query_text = str(query["query_text"])
        hits = list(row["hits"])
        for arm in ("A_prod", "R_page"):
            pool = prepare_rerank_pool(hits, arm=arm)
            requests += 1
            document_slots += len(pool)
            request_chars += len(query_text) * len(pool) + sum(
                len(str(hit["text"])[:max_document_chars])
                for hit in pool
            )
    return {
        "cases": len(candidate_rows),
        "arms_per_case": 2,
        "qwen3_requests_without_retries": requests,
        "qwen3_physical_attempts_ceiling": requests * max_attempts,
        "document_slots": document_slots,
        "request_chars": request_chars,
        "max_document_chars": max_document_chars,
        "max_attempts": max_attempts,
    }


def _freeze_rerank_plan(output_dir: Path, frozen_plan: dict) -> None:
    plan_path = output_dir / "rerank_plan.json"
    if (output_dir / "per_case.jsonl").exists():
        raise HoldoutError(
            "cannot rewrite rerank plan after reranking has started"
        )
    if plan_path.exists():
        if _load_json(plan_path, label="frozen rerank plan") != frozen_plan:
            raise HoldoutError("existing rerank plan has diverged")
    else:
        _atomic_json(plan_path, frozen_plan)


def _validate_remote_rerank_configuration(settings: dict[str, Any]) -> None:
    if settings.get("model") != "qwen3-rerank":
        raise HoldoutError("paired rerank model must be qwen3-rerank")
    if not str(settings.get("_base_url") or "").startswith("https://"):
        raise HoldoutError("paired rerank requires a locked HTTPS base URL")
    if not str(os.getenv("DASHSCOPE_API_KEY") or "").strip():
        raise HoldoutError("paired rerank credential is unavailable")
    numeric_bounds = (
        ("timeout_seconds", 0.1),
        ("backoff_seconds", 0.0),
        ("max_attempts", 1),
        ("max_inflight", 1),
        ("max_document_chars", 1),
    )
    for field, minimum in numeric_bounds:
        value = settings.get(field)
        if isinstance(value, bool):
            raise HoldoutError(f"paired rerank setting {field} is invalid")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise HoldoutError(
                f"paired rerank setting {field} is invalid"
            ) from exc
        if not math.isfinite(normalized) or normalized < minimum:
            raise HoldoutError(
                f"paired rerank setting {field} would be normalized"
            )
    if not str(settings.get("instruct") or "").strip():
        raise HoldoutError("paired rerank instruct must not be empty")


def _hash_hits(hits: list[dict]) -> str:
    return _ordered_identity_sha256(
        [
            {
                "chunk_id": str(hit["chunk_id"]),
                "document_id": str(hit["document_id"]),
            }
            for hit in hits
        ]
    )


def _hit_identity(hits: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": str(hit["chunk_id"]),
            "document_id": str(hit["document_id"]),
        }
        for hit in hits
    ]


def _row_sha256(row: dict) -> str:
    return _canonical_sha256(
        {key: value for key, value in row.items() if key != "row_sha256"}
    )


def _validate_candidate_queries(
    candidate_rows: list[dict],
    query_by_id: dict[str, dict],
) -> None:
    for candidate in candidate_rows:
        query = query_by_id.get(str(candidate.get("query_id") or ""))
        if (
            query is None
            or candidate.get("query_text_sha256")
            != hashlib.sha256(str(query["query_text"]).encode()).hexdigest()
            or candidate.get("company_key_sha256")
            != hashlib.sha256(str(query["company_key"]).encode()).hexdigest()
        ):
            raise HoldoutError("candidate query/company identity mismatch")


def _run_one_case(
    *,
    query: dict,
    candidate_row: dict,
    dataset,
    reranker,
    run_identity_sha256: str,
) -> dict:
    hits = list(candidate_row["hits"])
    rerank_records: dict[str, dict] = {}

    def retrieve(_query: str, _company: str, _top_k: int):
        return hits, {"remote_calls": 0}

    def rerank(query_text: str, pool: list[dict], top_k: int, arm: str):
        ranked, meta = reranker.rerank(query_text, pool, top_k=top_k)
        rerank_records[arm] = {
            "pool_identity_sha256": _hash_hits(pool),
            "final_identity_sha256": _hash_hits(ranked),
            "final_identity": _hit_identity(ranked),
            "rerank_attempts": int(meta.get("rerank_attempts") or 0),
            "rerank_tokens": int(meta.get("rerank_tokens") or 0),
            "rerank_fallback": bool(meta.get("rerank_fallback")),
            "rerank_error_type": str(meta.get("rerank_error_type") or ""),
            "rerank_latency_ms": float(meta.get("rerank_latency_ms") or 0.0),
        }
        return ranked, meta

    single = type(dataset)(
        queries=(query,),
        page_documents=dataset.page_documents,
        companies=dataset.companies,
        reports=dataset.reports,
    )
    prerank = score_ledger_public_dev(
        single,
        retrieve_candidates=retrieve,
        retrieval_requires_remote=False,
        allow_remote=False,
    )
    reranked = score_ledger_public_dev(
        single,
        retrieve_candidates=retrieve,
        retrieval_requires_remote=False,
        rerank=rerank,
        allow_remote=True,
    )
    if set(rerank_records) != {"A_prod", "R_page"}:
        raise HoldoutError("Qwen3 rerank did not produce both arm records")
    row = {
        "query_id": str(query["query_id"]),
        "run_identity_sha256": run_identity_sha256,
        "shared_candidate_identity_sha256": str(
            candidate_row["candidate_identity_sha256"]
        ),
        "prerank_arms": {
            arm: prerank["per_case"][arm][0]
            for arm in ("A_prod", "R_page")
        },
        "reranked_arms": {
            arm: {
                **reranked["per_case"][arm][0],
                **rerank_records[arm],
            }
            for arm in ("A_prod", "R_page")
        },
    }
    row["row_sha256"] = _row_sha256(row)
    return row


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
            or row.get("shared_candidate_identity_sha256")
            != candidate["candidate_identity_sha256"]
            or row.get("run_identity_sha256") != run_identity_sha256
            or row.get("row_sha256") != _row_sha256(row)
        ):
            raise HoldoutError("completed Qwen3 row identity mismatch")
        for group in ("prerank_arms", "reranked_arms"):
            arms = row.get(group)
            if not isinstance(arms, dict) or any(
                str((arms.get(arm) or {}).get("case_id") or "") != query_id
                for arm in ("A_prod", "R_page")
            ):
                raise HoldoutError("completed Qwen3 arm identity mismatch")
        for arm in ("A_prod", "R_page"):
            arm_row = row["reranked_arms"][arm]
            pool = prepare_rerank_pool(list(candidate["hits"]), arm=arm)
            final_identity = arm_row.get("final_identity")
            if not isinstance(final_identity, list):
                raise HoldoutError("completed Qwen3 final identity is invalid")
            pool_identity = {
                (str(item["chunk_id"]), str(item["document_id"]))
                for item in _hit_identity(pool)
            }
            normalized_final: list[dict[str, str]] = []
            for item in final_identity:
                if not isinstance(item, dict):
                    raise HoldoutError(
                        "completed Qwen3 final identity is invalid"
                    )
                identity = (
                    str(item.get("chunk_id") or ""),
                    str(item.get("document_id") or ""),
                )
                if (
                    not all(identity)
                    or identity not in pool_identity
                    or identity
                    in {
                        (
                            existing["chunk_id"],
                            existing["document_id"],
                        )
                        for existing in normalized_final
                    }
                ):
                    raise HoldoutError(
                        "completed Qwen3 final identity is invalid"
                    )
                normalized_final.append(
                    {
                        "chunk_id": identity[0],
                        "document_id": identity[1],
                    }
                )
            if (
                len(normalized_final) > ARM_SPECS[arm].final_k
                or arm_row.get("pool_identity_sha256") != _hash_hits(pool)
                or arm_row.get("final_identity_sha256")
                != _ordered_identity_sha256(normalized_final)
            ):
                raise HoldoutError("completed Qwen3 ranking identity mismatch")
            for field in ("rerank_attempts", "rerank_tokens"):
                value = arm_row.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise HoldoutError(
                        "completed Qwen3 accounting is invalid"
                    )
            if not isinstance(arm_row.get("rerank_fallback"), bool):
                raise HoldoutError("completed Qwen3 fallback is invalid")
            latency = arm_row.get("rerank_latency_ms")
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or latency < 0
            ):
                raise HoldoutError("completed Qwen3 latency is invalid")
        completed[query_id] = row
    return completed


def _paired_counts(
    rows: list[dict],
    *,
    arm: str,
    metric: str,
) -> dict[str, int]:
    counts = {"gain": 0, "loss": 0, "unchanged": 0}
    for row in rows:
        before = bool(row["prerank_arms"][arm].get(metric))
        after = bool(row["reranked_arms"][arm].get(metric))
        if after and not before:
            counts["gain"] += 1
        elif before and not after:
            counts["loss"] += 1
        else:
            counts["unchanged"] += 1
    return counts


def _summary(rows: list[dict], group: str) -> dict:
    summaries = {
        arm: summarize_ranking_cases(
            [dict(row[group][arm]) for row in rows],
            arm=arm,
        )
        for arm in ("A_prod", "R_page")
    }
    for summary in summaries.values():
        summary["scoring_status"] = (
            "public_dev_remote_rerank"
            if group == "reranked_arms"
            else "public_dev_offline_prerank"
        )
    if group == "reranked_arms":
        for arm, summary in summaries.items():
            summary["remote_calls"] = sum(
                int(row[group][arm]["rerank_attempts"])
                for row in rows
            )
    return summaries


def _metric_deltas(before: dict, after: dict) -> dict:
    fields = (
        "page_hit_at_5",
        "page_hit_at_10",
        "mrr",
        "ndcg_at_10",
        "mean_unique_pages_top10",
        "mean_duplicate_page_occupancy_top10",
    )
    return {
        arm: {
            field: round(
                float(after[arm][field]) - float(before[arm][field]),
                4,
            )
            for field in fields
        }
        for arm in ("A_prod", "R_page")
    }


def _all_qwen3_ok(rows: list[dict]) -> bool:
    return all(
        int(row["reranked_arms"][arm]["rerank_attempts"]) > 0
        and int(row["reranked_arms"][arm]["rerank_tokens"]) > 0
        and not bool(row["reranked_arms"][arm]["rerank_fallback"])
        and not str(row["reranked_arms"][arm]["rerank_error_type"])
        for row in rows
        for arm in ("A_prod", "R_page")
    )


def _run_rerank(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    dataset,
    candidate_rows: list[dict],
    candidate_manifest: dict,
    prerank: dict,
) -> dict:
    if not args.allow_remote:
        raise HoldoutError("rerank phase requires explicit --allow-remote")
    bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
    settings = snapshot_rerank_settings()
    _validate_remote_rerank_configuration(settings)
    planned = _load_json(
        output_dir / "rerank_plan.json",
        label="frozen rerank plan",
    )
    query_by_id = {
        str(query["query_id"]): query
        for query in dataset.queries
    }
    expected_plan = {
        **_rerank_plan(
            candidate_rows,
            query_by_id,
            max_document_chars=int(settings["max_document_chars"]),
            max_attempts=int(settings["max_attempts"]),
        ),
        "candidate_cache_sha256": candidate_manifest[
            "candidate_cache_sha256"
        ],
        "candidate_set_identity_sha256": candidate_manifest[
            "candidate_set_identity_sha256"
        ],
        "rerank_settings": public_rerank_settings(settings),
        "reranker_source_sha256": _reranker_source_sha256(),
    }
    if planned != expected_plan:
        raise HoldoutError("rerank plan or settings changed after local freeze")
    run_identity_sha256 = _canonical_sha256(
        {
            "candidate_cache_sha256": candidate_manifest[
                "candidate_cache_sha256"
            ],
            "candidate_set_identity_sha256": candidate_manifest[
                "candidate_set_identity_sha256"
            ],
            "rerank_settings": public_rerank_settings(settings),
            "reranker_source_sha256": _reranker_source_sha256(),
        }
    )
    _validate_candidate_queries(candidate_rows, query_by_id)
    reranker = build_locked_qwen3_reranker(settings)
    candidate_by_id = {
        str(row["query_id"]): row
        for row in candidate_rows
    }
    per_case_path = output_dir / "per_case.jsonl"
    existing = (
        _read_jsonl(per_case_path, label="Qwen3 per-case")
        if per_case_path.exists()
        else []
    )
    completed = _validate_completed_rows(
        existing,
        candidate_by_id=candidate_by_id,
        run_identity_sha256=run_identity_sha256,
    )
    ordered_rows = list(existing)
    for index, candidate in enumerate(candidate_rows):
        query_id = str(candidate["query_id"])
        if query_id in completed:
            continue
        query = query_by_id.get(query_id)
        if query is None:
            raise HoldoutError("candidate query is absent from frozen dataset")
        row = _run_one_case(
            query=query,
            candidate_row=candidate,
            dataset=dataset,
            reranker=reranker,
            run_identity_sha256=run_identity_sha256,
        )
        ordered_rows.append(row)
        _atomic_jsonl(per_case_path, ordered_rows)
        completed[query_id] = row
        print(
            f"[ledger-qwen3] rerank case={index + 1}/{len(candidate_rows)} OK",
            flush=True,
        )
    rows = _read_jsonl(per_case_path, label="Qwen3 per-case")
    completed = _validate_completed_rows(
        rows,
        candidate_by_id=candidate_by_id,
        run_identity_sha256=run_identity_sha256,
    )
    if tuple(completed) != tuple(str(row["query_id"]) for row in candidate_rows):
        raise HoldoutError("Qwen3 per-case coverage mismatch")
    before = _summary(rows, "prerank_arms")
    after = _summary(rows, "reranked_arms")
    attempts = sum(
        int(row["reranked_arms"][arm]["rerank_attempts"])
        for row in rows
        for arm in ("A_prod", "R_page")
    )
    tokens = sum(
        int(row["reranked_arms"][arm]["rerank_tokens"])
        for row in rows
        for arm in ("A_prod", "R_page")
    )
    fallbacks = sum(
        bool(row["reranked_arms"][arm]["rerank_fallback"])
        for row in rows
        for arm in ("A_prod", "R_page")
    )
    all_qwen3_ok = _all_qwen3_ok(rows)
    per_case_sha256 = hashlib.sha256(per_case_path.read_bytes()).hexdigest()
    candidate_calls = int(
        candidate_manifest["call_accounting"]["remote_calls"]
    )
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "orchestrator_source_sha256": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode()
        ).hexdigest(),
        "split": "public_dev",
        "cases": len(rows),
        "selection": candidate_manifest["selection"],
        "candidate_manifest": {
            "candidate_cache_sha256": candidate_manifest[
                "candidate_cache_sha256"
            ],
            "candidate_set_identity_sha256": candidate_manifest[
                "candidate_set_identity_sha256"
            ],
            "source_prerank_canonical_sha256": _canonical_sha256(prerank),
        },
        "run_identity_sha256": run_identity_sha256,
        "reranker_source_sha256": _reranker_source_sha256(),
        "rerank_settings": public_rerank_settings(settings),
        "comparison": {
            "prerank": before,
            "qwen3": after,
            "delta_qwen3_minus_prerank": _metric_deltas(before, after),
            "paired_counts": {
                arm: {
                    metric: _paired_counts(rows, arm=arm, metric=metric)
                    for metric in ("hit_at_5", "hit_at_10")
                }
                for arm in ("A_prod", "R_page")
            },
        },
        "call_accounting": {
            "candidate_embedding_remote_calls": candidate_calls,
            "qwen3_logical_calls": len(rows) * 2,
            "qwen3_physical_attempts": attempts,
            "qwen3_tokens": tokens,
            "rerank_fallbacks": int(fallbacks),
            "remote_calls": candidate_calls + attempts,
            "billing_semantics": "persisted_complete_cases_at_least_once",
            "unobserved_inflight_remote_calls_possible": True,
        },
        "per_case_sha256": per_case_sha256,
        "remote_calls": candidate_calls + attempts,
        "qwen3_calls": attempts,
        "primary_comparison_valid": all_qwen3_ok,
        "product_accuracy_claim": False,
    }
    _atomic_json(output_dir / "aggregate.json", aggregate)
    (output_dir / ".incomplete").unlink()
    print(
        "[ledger-qwen3] RERANK_OK "
        f"cases={len(rows)} qwen3_attempts={attempts} "
        f"fallbacks={fallbacks}",
        flush=True,
    )
    return aggregate


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "plan-rerank" and args.allow_remote:
            raise HoldoutError("plan-rerank is local-only")
        manifest, dataset, plans, qrel_audit, prerank = _load_context(args)
        if args.phase == "candidates":
            _build_candidates(
                args,
                manifest=manifest,
                dataset=dataset,
                plans=plans,
                qrel_audit=qrel_audit,
                prerank=prerank,
            )
            return 0
        output_dir = _require_candidate_output(Path(args.output_dir))
        candidate_rows, candidate_manifest = _validate_candidate_cache(
            output_dir,
            manifest=manifest,
            plans=plans,
            qrel_audit=qrel_audit,
            prerank=prerank,
        )
        query_by_id = {
            str(query["query_id"]): query
            for query in dataset.queries
        }
        bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
        settings = snapshot_rerank_settings()
        plan = _rerank_plan(
            candidate_rows,
            query_by_id,
            max_document_chars=int(settings["max_document_chars"]),
            max_attempts=int(settings["max_attempts"]),
        )
        if args.phase == "plan-rerank":
            frozen_plan = {
                **plan,
                "candidate_cache_sha256": candidate_manifest[
                    "candidate_cache_sha256"
                ],
                "candidate_set_identity_sha256": candidate_manifest[
                    "candidate_set_identity_sha256"
                ],
                "rerank_settings": public_rerank_settings(settings),
                "reranker_source_sha256": _reranker_source_sha256(),
            }
            _freeze_rerank_plan(output_dir, frozen_plan)
            print(json.dumps(frozen_plan, indent=2), flush=True)
            return 0
        _run_rerank(
            args,
            output_dir=output_dir,
            dataset=dataset,
            candidate_rows=candidate_rows,
            candidate_manifest=candidate_manifest,
            prerank=prerank,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - redact all CLI boundary failures
        safe = exc if isinstance(exc, HoldoutError) else HoldoutError(
            f"LEDGER Qwen3 paired run failed: {type(exc).__name__}"
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
