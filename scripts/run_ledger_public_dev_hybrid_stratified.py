#!/usr/bin/env python3
"""Run a small company-stratified LEDGER Hybrid retrieval experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_ledger_public_dev_ranking as child_cli

from lumenfin.env_bootstrap import bootstrap_dotenv
from lumenfin.eval.holdout import (
    HoldoutError,
    build_ledger_public_dev_dataset,
    iter_ledger_parquet_rows,
    ledger_public_dev_qrel_audit,
    ledger_snapshot_sha256,
    summarize_ranking_cases,
)
from lumenfin.provider_resilience import redact_provider_message
from lumenfin.rag.chunking import chunk_document
from lumenfin.rag.dashscope_defaults import resolved_dashscope_embedding_model
from lumenfin.stdio import configure_stdio_utf8

SCHEMA_VERSION = "lumenfin_ledger_hybrid_stratified.v1"
MAX_COMPANIES = 5
MAX_CASES_PER_COMPANY = 50
EXPECTED_BASELINE_CASES = 7615
TRACKED_BASELINE_AGGREGATE = (
    ROOT / "data" / "eval_rag" / "holdout" / "ledger_public_dev_bm25_baseline.json"
).resolve()
PINNED_BASELINE_CANONICAL_SHA256 = (
    "400fd88240829010b92f3b6866ffa0a8c7c0c4698ba029a178a60c5b80ea7ec2"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sequential five-company LEDGER Hybrid retrieval sample. "
            "DashScope embeddings are remote; Qwen3 remains disabled."
        )
    )
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-aggregate", required=True)
    parser.add_argument("--baseline-per-case", required=True)
    parser.add_argument("--company-count", type=int, default=MAX_COMPANIES)
    parser.add_argument(
        "--cases-per-company",
        type=int,
        default=MAX_CASES_PER_COMPANY,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dimension", type=int, default=1024)
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def _ids_sha256(values: list[str] | tuple[str, ...]) -> str:
    canonical = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _config_sha256(payload: dict) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _load_json(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutError(f"cannot read {label}") from exc
    if not isinstance(payload, dict):
        raise HoldoutError(f"{label} must be an object")
    return payload


def _strict_nonnegative_int(mapping: dict, field: str) -> int:
    value = mapping.get(field)
    if type(value) is not int or value < 0:
        raise HoldoutError(f"Hybrid accounting field {field} is invalid")
    return value


def select_stratified_company_keys(
    companies: tuple[str, ...],
    *,
    count: int,
) -> tuple[str, ...]:
    ordered = tuple(sorted(companies))
    if count < 2 or count > MAX_COMPANIES or count > len(ordered):
        raise HoldoutError(
            f"--company-count must be between 2 and {min(MAX_COMPANIES, len(ordered))}"
        )
    indexes = tuple(
        index * (len(ordered) - 1) // (count - 1)
        for index in range(count)
    )
    selected = tuple(ordered[index] for index in indexes)
    if len(set(selected)) != count:
        raise HoldoutError("stratified company selection is not unique")
    return selected


def _build_plans(
    dataset,
    *,
    company_count: int,
    cases_per_company: int,
    batch_size: int,
) -> list[dict]:
    if not 1 <= cases_per_company <= MAX_CASES_PER_COMPANY:
        raise HoldoutError(
            f"--cases-per-company must be between 1 and {MAX_CASES_PER_COMPANY}"
        )
    if batch_size <= 0:
        raise HoldoutError("--batch-size must be > 0")
    companies = select_stratified_company_keys(
        dataset.companies,
        count=company_count,
    )
    plans: list[dict] = []
    for company_key in companies:
        scoped = child_cli._subset_for_company(
            dataset,
            company_key=company_key,
            max_cases=cases_per_company,
        )
        chunks_per_document = [
            chunk_document(dict(document))
            for document in scoped.page_documents
        ]
        chunks = sum(len(items) for items in chunks_per_document)
        chars = sum(
            len(str(chunk.get("text") or ""))
            for items in chunks_per_document
            for chunk in items
        )
        estimated_http_calls = 0
        for start in range(0, len(chunks_per_document), batch_size):
            batch_chunks = sum(
                len(items)
                for items in chunks_per_document[start : start + batch_size]
            )
            estimated_http_calls += (batch_chunks + 9) // 10
        query_ids = tuple(str(query["query_id"]) for query in scoped.queries)
        expected_query_http_calls = len(
            {str(query["query_text"]) for query in scoped.queries}
        )
        plans.append(
            {
                "company_key": company_key,
                "company_key_sha256": _ids_sha256((company_key,)),
                "selected_cases": len(query_ids),
                "query_ids": query_ids,
                "query_ids_sha256": _ids_sha256(query_ids),
                "expected_query_http_calls": expected_query_http_calls,
                "documents": len(scoped.page_documents),
                "reports": scoped.reports,
                "chunks": chunks,
                "embed_chars": chars,
                "estimated_document_http_calls": estimated_http_calls,
            }
        )
    return plans


def _public_plan(plans: list[dict]) -> dict:
    return {
        "strategy": "sorted_company_even_spacing_v1",
        "companies": len(plans),
        "cases": sum(int(plan["selected_cases"]) for plan in plans),
        "documents": sum(int(plan["documents"]) for plan in plans),
        "chunks": sum(int(plan["chunks"]) for plan in plans),
        "embed_chars": sum(int(plan["embed_chars"]) for plan in plans),
        "estimated_document_http_calls": sum(
            int(plan["estimated_document_http_calls"])
            for plan in plans
        ),
        "query_http_calls_without_retries": sum(
            int(plan["selected_cases"]) for plan in plans
        ),
        "expected_query_http_calls_minimum": sum(
            int(plan["expected_query_http_calls"]) for plan in plans
        ),
        "estimated_total_http_calls_without_retries": sum(
            int(plan["estimated_document_http_calls"])
            + int(plan["expected_query_http_calls"])
            for plan in plans
        ),
        "company_keys_sha256": _ids_sha256(
            tuple(str(plan["company_key"]) for plan in plans)
        ),
        "query_ids_sha256": _ids_sha256(
            tuple(
                str(query_id)
                for plan in plans
                for query_id in plan["query_ids"]
            )
        ),
        "per_company": [
            {
                key: plan[key]
                for key in (
                    "company_key_sha256",
                    "selected_cases",
                    "query_ids_sha256",
                    "expected_query_http_calls",
                    "documents",
                    "reports",
                    "chunks",
                    "embed_chars",
                    "estimated_document_http_calls",
                )
            }
            for plan in plans
        ],
    }


def _prepare_output(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists():
        if (target / "aggregate.json").exists():
            raise HoldoutError(f"refusing to overwrite completed run: {target}")
        if not (target / ".incomplete").is_file():
            raise HoldoutError(f"refusing unknown existing output directory: {target}")
        return target
    if not target.parent.is_dir():
        raise HoldoutError(f"output parent directory not found: {target.parent}")
    target.mkdir()
    (target / ".incomplete").write_text(
        "LEDGER stratified Hybrid run has not completed.\n",
        encoding="utf-8",
    )
    return target


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _remove_indexes(path: Path) -> None:
    for index_path in path.glob("_index_*"):
        shutil.rmtree(index_path)


def _run_company(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    plan: dict,
    index: int,
) -> Path:
    run_dir = output_dir / "companies" / f"{index:03d}"
    aggregate_path = run_dir / "aggregate.json"
    if aggregate_path.is_file() and not (run_dir / ".incomplete").exists():
        return run_dir
    if run_dir.exists():
        _remove_indexes(run_dir)
        shutil.rmtree(run_dir)
    run_dir.parent.mkdir(exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_ledger_public_dev_ranking.py"),
        "--parquet-path",
        str(args.parquet_path),
        "--manifest",
        str(args.manifest),
        "--split-salt",
        str(args.split_salt),
        "--output-dir",
        str(run_dir),
        "--max-cases",
        str(plan["selected_cases"]),
        "--company-key",
        str(plan["company_key"]),
        "--batch-size",
        str(args.batch_size),
        "--mode",
        "hybrid",
        "--embedding-provider",
        "dashscope",
        "--embedding-dimension",
        str(args.embedding_dimension),
        "--embedding-model",
        str(args.embedding_model),
        "--allow-remote",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if run_dir.exists():
            _remove_indexes(run_dir)
        raw = completed.stdout or completed.stderr or "no child diagnostic"
        diagnostic = redact_provider_message(raw[-1000:], limit=240)
        raise HoldoutError(
            f"Hybrid company run {index} failed with {completed.returncode}: "
            f"{diagnostic}"
        )
    if not aggregate_path.is_file():
        raise HoldoutError(f"Hybrid company run {index} wrote no aggregate")
    return run_dir


def _validate_company_run(
    run_dir: Path,
    *,
    plan: dict,
    manifest: dict,
    expected_run_config_sha256: str,
    source_manifest_sha256: str,
    expected_qrel_audit: dict,
) -> dict:
    aggregate = _load_json(run_dir / "aggregate.json", label="company aggregate")
    per_case_path = run_dir / "per_case.jsonl"
    per_case_hash = hashlib.sha256(per_case_path.read_bytes()).hexdigest()
    if aggregate.get("per_case_sha256") != per_case_hash:
        raise HoldoutError("Hybrid company per-case hash mismatch")
    if aggregate.get("dataset_snapshot_sha256") != manifest["dataset_snapshot_sha256"]:
        raise HoldoutError("Hybrid company dataset identity mismatch")
    if aggregate.get("source_manifest_sha256") != source_manifest_sha256:
        raise HoldoutError("Hybrid company manifest identity mismatch")
    if aggregate.get("run_config_sha256") != expected_run_config_sha256:
        raise HoldoutError("Hybrid company run config mismatch")
    if child_cli._config_sha256(aggregate.get("run_config") or {}) != (
        expected_run_config_sha256
    ):
        raise HoldoutError("Hybrid company run config hash mismatch")
    if aggregate.get("qrel_corpus_audit") != expected_qrel_audit:
        raise HoldoutError("Hybrid company qrel audit mismatch")
    selection = aggregate.get("selection") or {}
    expected_selection = {
        "strategy": "single_company_prefix_v1",
        "selected_cases": plan["selected_cases"],
        "selected_companies": 1,
        "query_ids_sha256": plan["query_ids_sha256"],
        "company_keys_sha256": plan["company_key_sha256"],
    }
    if any(
        selection.get(key) != value
        for key, value in expected_selection.items()
    ):
        raise HoldoutError("Hybrid company selection identity mismatch")
    calls = aggregate.get("call_accounting") or {}
    retrieval_calls = _strict_nonnegative_int(calls, "retrieval_calls")
    query_calls = _strict_nonnegative_int(
        calls,
        "query_embedding_remote_calls",
    )
    document_calls = _strict_nonnegative_int(
        calls,
        "document_embedding_remote_calls",
    )
    total_calls = _strict_nonnegative_int(calls, "remote_calls")
    retrieval_remote_calls = _strict_nonnegative_int(
        calls,
        "retrieval_remote_calls",
    )
    index = aggregate.get("index") or {}
    if (
        retrieval_calls != plan["selected_cases"]
        or document_calls <= 0
        or query_calls < plan["expected_query_http_calls"]
        or document_calls < plan["estimated_document_http_calls"]
        or retrieval_remote_calls != query_calls
        or total_calls != query_calls + document_calls
        or aggregate.get("remote_calls") != total_calls
        or _strict_nonnegative_int(index, "documents_indexed") != plan["documents"]
        or _strict_nonnegative_int(index, "documents_in_scoped_corpus")
        != plan["documents"]
        or _strict_nonnegative_int(index, "chunks_indexed") != plan["chunks"]
        or _strict_nonnegative_int(index, "embed_chars") != plan["embed_chars"]
        or _strict_nonnegative_int(index, "embed_physical_calls")
        != document_calls
        or _strict_nonnegative_int(index, "estimated_dashscope_http_calls")
        != plan["estimated_document_http_calls"]
        or _strict_nonnegative_int(index, "reports") != plan["reports"]
    ):
        raise HoldoutError("Hybrid company remote-call accounting mismatch")
    if any(
        _strict_nonnegative_int(calls, field) != 0
        for field in ("rerank_calls", "rerank_attempts", "rerank_fallbacks")
    ):
        raise HoldoutError("Hybrid company unexpectedly called reranker")
    if aggregate.get("qwen3_calls") != 0:
        raise HoldoutError("Hybrid company unexpectedly called Qwen3")
    if index.get("companies") != 1:
        raise HoldoutError("Hybrid company index scope mismatch")
    return aggregate


def _parse_per_case_text(text: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise HoldoutError("per-case artifact contains invalid JSON") from exc
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in seen:
            raise HoldoutError("per-case artifact has missing or duplicate query IDs")
        arms = row.get("arms")
        if not isinstance(arms, dict) or any(
            not isinstance(arms.get(arm), dict)
            or str(arms[arm].get("case_id") or "") != query_id
            for arm in ("A_prod", "R_page")
        ):
            raise HoldoutError("per-case arm identity does not match query ID")
        seen.add(query_id)
        rows.append(row)
    return rows


def _load_per_case(path: Path) -> list[dict]:
    return _parse_per_case_text(path.read_text(encoding="utf-8"))


def _validate_sealed_baseline(
    *,
    aggregate_path: Path,
    per_case_path: Path,
    manifest: dict,
    source_manifest_sha256: str,
    expected_qrel_audit: dict,
) -> tuple[dict, list[dict], str]:
    resolved_aggregate = aggregate_path.expanduser().resolve()
    if resolved_aggregate != TRACKED_BASELINE_AGGREGATE:
        raise HoldoutError("baseline aggregate must be the tracked sealed artifact")
    aggregate = _load_json(resolved_aggregate, label="baseline aggregate")
    if _config_sha256(aggregate) != PINNED_BASELINE_CANONICAL_SHA256:
        raise HoldoutError("tracked baseline aggregate identity has changed")
    run_config = aggregate.get("run_config")
    if (
        aggregate.get("cases") != EXPECTED_BASELINE_CASES
        or aggregate.get("dataset_snapshot_sha256")
        != manifest["dataset_snapshot_sha256"]
        or aggregate.get("source_manifest_sha256") != source_manifest_sha256
        or aggregate.get("qrel_corpus_audit") != expected_qrel_audit
        or aggregate.get("remote_calls") != 0
        or aggregate.get("qwen3_calls") != 0
        or aggregate.get("primary_comparison_valid") is not False
        or aggregate.get("selection", {}).get("selected_cases")
        != EXPECTED_BASELINE_CASES
        or aggregate.get("selection", {}).get("selected_companies")
        != int(manifest["splits"]["public_dev"]["companies"])
        or aggregate.get("index", {}).get("embedding_provider")
        != "deterministic"
        or aggregate.get("index", {}).get("retrieval_mode") != "bm25"
        or not isinstance(run_config, dict)
        or run_config.get("embedding_provider") != "deterministic"
        or run_config.get("retrieval_mode") != "bm25"
        or run_config.get("evaluator_source_sha256")
        != child_cli._evaluator_source_sha256()
        or child_cli._config_sha256(run_config)
        != aggregate.get("run_config_sha256")
    ):
        raise HoldoutError("sealed BM25 baseline identity is incompatible")
    calls = aggregate.get("call_accounting") or {}
    if (
        _strict_nonnegative_int(calls, "retrieval_calls")
        != EXPECTED_BASELINE_CASES
        or any(
            _strict_nonnegative_int(calls, field) != 0
            for field in (
                "retrieval_remote_calls",
                "rerank_calls",
                "rerank_attempts",
                "rerank_fallbacks",
                "remote_calls",
            )
        )
    ):
        raise HoldoutError("sealed BM25 baseline call accounting is incompatible")
    per_case_bytes = per_case_path.expanduser().resolve().read_bytes()
    actual_per_case_hash = hashlib.sha256(per_case_bytes).hexdigest()
    if aggregate.get("per_case_sha256") != actual_per_case_hash:
        raise HoldoutError("sealed BM25 baseline per-case hash mismatch")
    try:
        per_case_text = per_case_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HoldoutError("sealed BM25 baseline per-case is not UTF-8") from exc
    rows = _parse_per_case_text(per_case_text)
    if len(rows) != EXPECTED_BASELINE_CASES:
        raise HoldoutError("sealed BM25 baseline per-case coverage mismatch")
    return aggregate, rows, actual_per_case_hash


def _paired_counts(
    baseline_rows: list[dict],
    hybrid_rows: list[dict],
    *,
    arm: str,
    metric: str,
) -> dict[str, int]:
    baseline = {
        str(row["query_id"]): bool(row["arms"][arm].get(metric))
        for row in baseline_rows
    }
    counts = {"gain": 0, "loss": 0, "unchanged": 0}
    for row in hybrid_rows:
        query_id = str(row["query_id"])
        before = baseline[query_id]
        after = bool(row["arms"][arm].get(metric))
        if after and not before:
            counts["gain"] += 1
        elif before and not after:
            counts["loss"] += 1
        else:
            counts["unchanged"] += 1
    return counts


def _summaries(rows: list[dict]) -> dict:
    summaries = {
        arm: summarize_ranking_cases(
            [dict(row["arms"][arm]) for row in rows],
            arm=arm,
        )
        for arm in ("A_prod", "R_page")
    }
    for summary in summaries.values():
        summary["scoring_status"] = "public_dev_offline_prerank"
    return summaries


def _metric_deltas(baseline: dict, hybrid: dict) -> dict:
    fields = (
        "pool_hit_rate",
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
                float(hybrid[arm][field]) - float(baseline[arm][field]),
                4,
            )
            for field in fields
        }
        for arm in ("A_prod", "R_page")
    }


def _aggregate(
    run_dirs: list[Path],
    *,
    plans: list[dict],
    output_dir: Path,
    manifest: dict,
    baseline_aggregate: dict,
    baseline_rows: list[dict],
    baseline_per_case_sha256: str,
) -> dict:
    hybrid_rows: list[dict] = []
    seen: set[str] = set()
    calls = {
        field: 0
        for field in (
            "retrieval_calls",
            "retrieval_remote_calls",
            "rerank_calls",
            "rerank_attempts",
            "rerank_fallbacks",
            "document_embedding_remote_calls",
            "query_embedding_remote_calls",
            "remote_calls",
        )
    }
    index_totals = {
        field: 0
        for field in (
            "documents_indexed",
            "chunks_indexed",
            "embed_calls",
            "embed_physical_calls",
            "embed_chars",
            "companies",
            "reports",
        )
    }
    child_run_config: dict | None = None
    child_run_config_sha256 = ""
    for run_dir in run_dirs:
        aggregate = _load_json(run_dir / "aggregate.json", label="company aggregate")
        if child_run_config is None:
            child_run_config = dict(aggregate["run_config"])
            child_run_config_sha256 = str(aggregate["run_config_sha256"])
        elif (
            aggregate.get("run_config") != child_run_config
            or aggregate.get("run_config_sha256") != child_run_config_sha256
        ):
            raise HoldoutError("Hybrid company run identities diverge")
        for field in calls:
            calls[field] += _strict_nonnegative_int(
                aggregate["call_accounting"],
                field,
            )
        for field in index_totals:
            index_totals[field] += _strict_nonnegative_int(
                aggregate["index"],
                field,
            )
        for row in _load_per_case(run_dir / "per_case.jsonl"):
            query_id = str(row["query_id"])
            if query_id in seen:
                raise HoldoutError("Hybrid company outputs duplicate query IDs")
            seen.add(query_id)
            hybrid_rows.append(row)
    expected_ids = {
        str(query_id)
        for plan in plans
        for query_id in plan["query_ids"]
    }
    if seen != expected_ids:
        raise HoldoutError("Hybrid aggregate query coverage mismatch")

    baseline_rows = [
        row for row in baseline_rows if str(row["query_id"]) in expected_ids
    ]
    if len(baseline_rows) != len(expected_ids):
        raise HoldoutError("BM25 baseline is missing selected Hybrid queries")
    baseline_rows.sort(key=lambda row: str(row["query_id"]))
    hybrid_rows.sort(key=lambda row: str(row["query_id"]))
    baseline_summaries = _summaries(baseline_rows)
    hybrid_summaries = _summaries(hybrid_rows)

    per_case_path = output_dir / "per_case.jsonl"
    tmp = per_case_path.with_name(per_case_path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in hybrid_rows),
        encoding="utf-8",
    )
    os.replace(tmp, per_case_path)
    per_case_sha256 = hashlib.sha256(per_case_path.read_bytes()).hexdigest()
    comparison = {
        "baseline": baseline_summaries,
        "hybrid": hybrid_summaries,
        "delta_hybrid_minus_bm25": _metric_deltas(
            baseline_summaries,
            hybrid_summaries,
        ),
        "paired_counts": {
            arm: {
                metric: _paired_counts(
                    baseline_rows,
                    hybrid_rows,
                    arm=arm,
                    metric=metric,
                )
                for metric in ("pool_hit", "hit_at_10")
            }
            for arm in ("A_prod", "R_page")
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "orchestrator_source_sha256": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode()
        ).hexdigest(),
        "split": "public_dev",
        "cases": len(hybrid_rows),
        "selection": _public_plan(plans),
        "comparison": comparison,
        "call_accounting": calls,
        "index": {
            **index_totals,
            "embedding_provider": "dashscope",
            "retrieval_mode": "hybrid",
            "indexes_retained": False,
        },
        "dataset_snapshot_sha256": manifest["dataset_snapshot_sha256"],
        "child_run_config": child_run_config,
        "child_run_config_sha256": child_run_config_sha256,
        "baseline": {
            "aggregate_canonical_sha256": _config_sha256(
                baseline_aggregate
            ),
            "per_case_sha256": baseline_per_case_sha256,
            "run_config_sha256": baseline_aggregate["run_config_sha256"],
        },
        "per_case_sha256": per_case_sha256,
        "remote_calls": calls["remote_calls"],
        "qwen3_calls": 0,
        "primary_comparison_valid": False,
    }


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        if bool(args.plan_only) == bool(args.allow_remote):
            raise HoldoutError(
                "choose exactly one of --plan-only or --allow-remote"
            )
        manifest_path = Path(args.manifest).expanduser().resolve()
        manifest = child_cli._load_manifest(manifest_path)
        snapshot_sha256 = ledger_snapshot_sha256(args.parquet_path)
        holdout_fraction = child_cli._validate_snapshot_and_salt(
            manifest=manifest,
            snapshot_sha256=snapshot_sha256,
            salt=str(args.split_salt),
        )
        dataset = build_ledger_public_dev_dataset(
            iter_ledger_parquet_rows(args.parquet_path),
            salt=str(args.split_salt),
            holdout_fraction=holdout_fraction,
            manifest=manifest,
        )
        plans = _build_plans(
            dataset,
            company_count=int(args.company_count),
            cases_per_company=int(args.cases_per_company),
            batch_size=int(args.batch_size),
        )
        public_plan = _public_plan(plans)
        source_manifest_sha256 = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        expected_qrel_audit = ledger_public_dev_qrel_audit(
            dataset,
            source_queries=int(manifest["splits"]["public_dev"]["queries"]),
        )
        (
            baseline_aggregate,
            baseline_rows,
            baseline_per_case_sha256,
        ) = _validate_sealed_baseline(
            aggregate_path=Path(args.baseline_aggregate),
            per_case_path=Path(args.baseline_per_case),
            manifest=manifest,
            source_manifest_sha256=source_manifest_sha256,
            expected_qrel_audit=expected_qrel_audit,
        )
        if args.plan_only:
            print(json.dumps(public_plan, indent=2), flush=True)
            return 0

        bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
        embedding_model = resolved_dashscope_embedding_model(
            str(args.embedding_model or "")
        )
        expected_run_config = child_cli._build_run_config(
            manifest,
            mode="hybrid",
            embedding_provider="dashscope",
            embedding_dimension=int(args.embedding_dimension),
            embedding_model=embedding_model,
        )
        expected_run_config_sha256 = child_cli._config_sha256(
            expected_run_config
        )
        output_dir = _prepare_output(Path(args.output_dir))
        run_dirs: list[Path] = []
        for index, plan in enumerate(plans):
            run_dir = _run_company(
                args=args,
                output_dir=output_dir,
                plan=plan,
                index=index,
            )
            _validate_company_run(
                run_dir,
                plan=plan,
                manifest=manifest,
                expected_run_config_sha256=expected_run_config_sha256,
                source_manifest_sha256=source_manifest_sha256,
                expected_qrel_audit=expected_qrel_audit,
            )
            _remove_indexes(run_dir)
            run_dirs.append(run_dir)
            print(
                f"[ledger-hybrid] company={index + 1}/{len(plans)} OK",
                flush=True,
            )
        aggregate = _aggregate(
            run_dirs,
            plans=plans,
            output_dir=output_dir,
            manifest=manifest,
            baseline_aggregate=baseline_aggregate,
            baseline_rows=baseline_rows,
            baseline_per_case_sha256=baseline_per_case_sha256,
        )
        _atomic_write_json(output_dir / "aggregate.json", aggregate)
        (output_dir / ".incomplete").unlink()
        print(
            "[ledger-hybrid] STRATIFIED_OK "
            f"cases={aggregate['cases']} companies={len(plans)} "
            f"remote_calls={aggregate['remote_calls']} qwen3_calls=0",
            flush=True,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - redact all CLI boundary failures
        safe = exc if isinstance(exc, HoldoutError) else HoldoutError(
            f"LEDGER stratified Hybrid failed: {type(exc).__name__}"
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
