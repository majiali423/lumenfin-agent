#!/usr/bin/env python3
"""Run and aggregate the full LEDGER public-dev BM25 baseline by company shard."""

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

import run_ledger_public_dev_ranking as shard_cli

from lumenfin.eval.holdout import (
    HoldoutError,
    build_ledger_public_dev_dataset,
    iter_ledger_parquet_rows,
    summarize_ranking_cases,
)
from lumenfin.provider_resilience import redact_provider_message
from lumenfin.stdio import configure_stdio_utf8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LEDGER public_dev BM25 in sequential company shards. "
            "Every child is offline-only and releases Milvus before the next shard."
        )
    )
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-count", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-remote", action="store_true")
    return parser


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
        "LEDGER company-sharded run has not completed.\n",
        encoding="utf-8",
    )
    return target


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutError(f"cannot read shard artifact: {path.name}") from exc
    if not isinstance(payload, dict):
        raise HoldoutError(f"shard artifact must be an object: {path.name}")
    return payload


def _ids_sha256(values: list[str] | tuple[str, ...]) -> str:
    canonical = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _strict_nonnegative_int(mapping: dict, field: str) -> int:
    value = mapping.get(field)
    if type(value) is not int or value < 0:
        raise HoldoutError(f"LEDGER shard accounting field {field} is invalid")
    return value


def _plan_shards(
    *,
    args: argparse.Namespace,
    manifest: dict,
) -> dict[int, dict]:
    dataset = build_ledger_public_dev_dataset(
        iter_ledger_parquet_rows(args.parquet_path),
        salt=str(args.split_salt),
        holdout_fraction=float(manifest["holdout_fraction"]),
        manifest=manifest,
    )
    companies = sorted(dataset.companies)
    plans: dict[int, dict] = {}
    for shard_index in range(args.shard_count):
        shard_companies = tuple(companies[shard_index::args.shard_count])
        company_set = set(shard_companies)
        query_ids = tuple(
            str(query["query_id"])
            for query in dataset.queries
            if str(query["company_key"]) in company_set
        )
        plans[shard_index] = {
            "selected_cases": len(query_ids),
            "selected_companies": len(shard_companies),
            "query_ids_sha256": _ids_sha256(query_ids),
            "company_keys_sha256": _ids_sha256(shard_companies),
        }
    return plans


def _validate_completed_shard(
    aggregate: dict,
    *,
    shard_index: int,
    shard_count: int,
    dataset_snapshot_sha256: str,
    expected_selection: dict,
    run_config_sha256: str,
    source_manifest_sha256: str,
    expected_qrel_audit: dict,
    per_case_sha256: str,
) -> None:
    selection = aggregate.get("selection")
    calls = aggregate.get("call_accounting")
    if not isinstance(selection, dict) or not isinstance(calls, dict):
        raise HoldoutError("completed LEDGER shard metadata is incomplete")
    expected = {
        "strategy": "company_modulo_shard_v1",
        "company_shard_index": shard_index,
        "company_shard_count": shard_count,
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise HoldoutError("completed LEDGER shard identity mismatch")
    if aggregate.get("dataset_snapshot_sha256") != dataset_snapshot_sha256:
        raise HoldoutError("completed LEDGER shard dataset mismatch")
    for field, expected_value in expected_selection.items():
        if selection.get(field) != expected_value:
            raise HoldoutError("completed LEDGER shard coverage identity mismatch")
    if aggregate.get("run_config_sha256") != run_config_sha256:
        raise HoldoutError("completed LEDGER shard run config mismatch")
    run_config = aggregate.get("run_config")
    if (
        not isinstance(run_config, dict)
        or shard_cli._config_sha256(run_config) != run_config_sha256
    ):
        raise HoldoutError("completed LEDGER shard run config hash mismatch")
    if aggregate.get("source_manifest_sha256") != source_manifest_sha256:
        raise HoldoutError("completed LEDGER shard manifest identity mismatch")
    if aggregate.get("qrel_corpus_audit") != expected_qrel_audit:
        raise HoldoutError("completed LEDGER shard qrel audit mismatch")
    if aggregate.get("per_case_sha256") != per_case_sha256:
        raise HoldoutError("completed LEDGER shard per-case hash mismatch")
    zero_call_fields = (
        "retrieval_remote_calls",
        "rerank_calls",
        "rerank_attempts",
        "rerank_fallbacks",
        "remote_calls",
    )
    if any(_strict_nonnegative_int(calls, field) != 0 for field in zero_call_fields):
        raise HoldoutError("completed LEDGER shard reports remote/rerank calls")
    if _strict_nonnegative_int(calls, "retrieval_calls") != int(
        expected_selection["selected_cases"]
    ):
        raise HoldoutError("completed LEDGER shard retrieval call count mismatch")
    if aggregate.get("qwen3_calls") != 0:
        raise HoldoutError("completed LEDGER shard reports Qwen3 calls")


def _remove_shard_indexes(shard_dir: Path) -> None:
    for path in shard_dir.glob("_index_*"):
        shutil.rmtree(path)


def _run_shard(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    shard_index: int,
) -> Path:
    shard_dir = output_dir / "shards" / f"{shard_index:03d}"
    aggregate_path = shard_dir / "aggregate.json"
    if aggregate_path.is_file() and not (shard_dir / ".incomplete").exists():
        return shard_dir
    if shard_dir.exists():
        _remove_shard_indexes(shard_dir)
        shutil.rmtree(shard_dir)
    shard_dir.parent.mkdir(exist_ok=True)
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
        str(shard_dir),
        "--company-shard-index",
        str(shard_index),
        "--company-shard-count",
        str(args.shard_count),
        "--batch-size",
        str(args.batch_size),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if shard_dir.exists():
            _remove_shard_indexes(shard_dir)
        raw_diagnostic = completed.stdout or completed.stderr or "no child diagnostic"
        diagnostic = redact_provider_message(
            raw_diagnostic[-1000:],
            limit=240,
        )
        raise HoldoutError(
            f"LEDGER company shard {shard_index} failed with exit "
            f"{completed.returncode}: {diagnostic}"
        )
    if not aggregate_path.is_file():
        if shard_dir.exists():
            _remove_shard_indexes(shard_dir)
        raise HoldoutError(f"LEDGER company shard {shard_index} wrote no aggregate")
    _remove_shard_indexes(shard_dir)
    return shard_dir


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _aggregate_shards(
    shard_dirs: list[Path],
    *,
    manifest: dict,
    output_dir: Path,
) -> dict:
    arm_rows = {"A_prod": [], "R_page": []}
    combined_rows: list[dict] = []
    seen_query_ids: set[str] = set()
    call_fields = (
        "retrieval_calls",
        "retrieval_remote_calls",
        "rerank_calls",
        "rerank_attempts",
        "rerank_fallbacks",
        "remote_calls",
    )
    calls = {field: 0 for field in call_fields}
    index_totals = {
        "documents_indexed": 0,
        "chunks_indexed": 0,
        "embed_calls": 0,
        "companies": 0,
        "reports": 0,
    }
    qrel_audit: dict | None = None
    run_config: dict | None = None
    run_config_sha256 = ""
    source_manifest_sha256 = ""
    qwen3_calls = 0
    for shard_dir in shard_dirs:
        aggregate = _load_json(shard_dir / "aggregate.json")
        if qrel_audit is None:
            qrel_audit = dict(aggregate["qrel_corpus_audit"])
            run_config = dict(aggregate["run_config"])
            run_config_sha256 = str(aggregate["run_config_sha256"])
            source_manifest_sha256 = str(aggregate["source_manifest_sha256"])
        elif aggregate.get("qrel_corpus_audit") != qrel_audit:
            raise HoldoutError("LEDGER shard qrel audits diverge")
        elif (
            aggregate.get("run_config") != run_config
            or aggregate.get("run_config_sha256") != run_config_sha256
            or aggregate.get("source_manifest_sha256") != source_manifest_sha256
        ):
            raise HoldoutError("LEDGER shard run identities diverge")
        qwen3_calls += _strict_nonnegative_int(aggregate, "qwen3_calls")
        for field in call_fields:
            calls[field] += _strict_nonnegative_int(
                aggregate["call_accounting"],
                field,
            )
        for field in index_totals:
            index_totals[field] += _strict_nonnegative_int(
                aggregate["index"],
                field,
            )
        shard_query_ids: list[str] = []
        per_case_path = shard_dir / "per_case.jsonl"
        if hashlib.sha256(per_case_path.read_bytes()).hexdigest() != str(
            aggregate.get("per_case_sha256") or ""
        ):
            raise HoldoutError("LEDGER shard per-case hash mismatch")
        for raw_line in per_case_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            query_id = str(row.get("query_id") or "")
            if not query_id or query_id in seen_query_ids:
                raise HoldoutError("LEDGER shards contain missing or duplicate query_id")
            seen_query_ids.add(query_id)
            shard_query_ids.append(query_id)
            combined_rows.append(row)
            for arm, rows in arm_rows.items():
                arm_row = dict(row["arms"][arm])
                if str(arm_row.get("case_id") or "") != query_id:
                    raise HoldoutError("LEDGER shard metric case_id mismatch")
                rows.append(arm_row)
        if _ids_sha256(tuple(shard_query_ids)) != str(
            aggregate["selection"]["query_ids_sha256"]
        ):
            raise HoldoutError("LEDGER shard per-case query identity mismatch")
    expected_cases = int(manifest["public_dev_corpus_audit"]["scorable_queries"])
    if len(seen_query_ids) != expected_cases:
        raise HoldoutError(
            "LEDGER sharded run does not cover the frozen scorable query count"
        )
    if _ids_sha256(tuple(seen_query_ids)) != str(
        manifest["public_dev_corpus_audit"]["scorable_query_ids_sha256"]
    ):
        raise HoldoutError("LEDGER sharded run query identity mismatch")
    summaries = {
        arm: summarize_ranking_cases(rows, arm=arm)
        for arm, rows in arm_rows.items()
    }
    for summary in summaries.values():
        summary["scoring_status"] = "public_dev_offline_prerank"
        summary["remote_calls"] = 0
    combined_rows.sort(key=lambda row: str(row["query_id"]))
    per_case_path = output_dir / "per_case.jsonl"
    tmp = per_case_path.with_name(per_case_path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined_rows),
        encoding="utf-8",
    )
    os.replace(tmp, per_case_path)
    per_case_sha256 = hashlib.sha256(per_case_path.read_bytes()).hexdigest()
    return {
        "schema_version": "lumenfin_ledger_public_dev_scoring.v1",
        "split": "public_dev",
        "cases": expected_cases,
        "arms": summaries,
        "call_accounting": calls,
        "primary_comparison_valid": False,
        "dataset_snapshot_sha256": manifest["dataset_snapshot_sha256"],
        "run_config": run_config,
        "run_config_sha256": run_config_sha256,
        "per_case_sha256": per_case_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "index": {
            **index_totals,
            "embedding_provider": "deterministic",
            "retrieval_mode": "bm25",
            "shards": len(shard_dirs),
            "indexes_retained": False,
        },
        "qrel_corpus_audit": qrel_audit,
        "selection": {
            "strategy": "company_modulo_shard_v1",
            "company_shards": len(shard_dirs),
            "selected_cases": expected_cases,
            "selected_companies": int(
                manifest["splits"]["public_dev"]["companies"]
            ),
        },
        "remote_calls": calls["remote_calls"],
        "qwen3_calls": qwen3_calls,
    }


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        if args.allow_remote:
            raise HoldoutError("LEDGER sharded BM25 run is offline-only")
        if args.shard_count <= 0:
            raise HoldoutError("--shard-count must be > 0")
        manifest = shard_cli._load_manifest(Path(args.manifest))
        snapshot_sha256 = shard_cli.ledger_snapshot_sha256(args.parquet_path)
        shard_cli._validate_snapshot_and_salt(
            manifest=manifest,
            snapshot_sha256=snapshot_sha256,
            salt=str(args.split_salt),
        )
        company_count = int(manifest["splits"]["public_dev"]["companies"])
        if args.shard_count > company_count:
            raise HoldoutError("--shard-count cannot exceed public_dev companies")
        shard_plans = _plan_shards(args=args, manifest=manifest)
        run_config_sha256 = shard_cli._config_sha256(
            shard_cli._build_run_config(manifest)
        )
        source_manifest_sha256 = hashlib.sha256(
            Path(args.manifest).expanduser().resolve().read_bytes()
        ).hexdigest()
        output_dir = _prepare_output(Path(args.output_dir))
        shard_dirs: list[Path] = []
        for shard_index in range(args.shard_count):
            shard_dir = _run_shard(
                args=args,
                output_dir=output_dir,
                shard_index=shard_index,
            )
            aggregate = _load_json(shard_dir / "aggregate.json")
            _validate_completed_shard(
                aggregate,
                shard_index=shard_index,
                shard_count=args.shard_count,
                dataset_snapshot_sha256=str(manifest["dataset_snapshot_sha256"]),
                expected_selection=shard_plans[shard_index],
                run_config_sha256=run_config_sha256,
                source_manifest_sha256=source_manifest_sha256,
                expected_qrel_audit=dict(manifest["public_dev_corpus_audit"]),
                per_case_sha256=hashlib.sha256(
                    (shard_dir / "per_case.jsonl").read_bytes()
                ).hexdigest(),
            )
            _remove_shard_indexes(shard_dir)
            shard_dirs.append(shard_dir)
            print(
                f"[ledger-public-dev] shard={shard_index + 1}/{args.shard_count} OK",
                flush=True,
            )
        aggregate = _aggregate_shards(
            shard_dirs,
            manifest=manifest,
            output_dir=output_dir,
        )
        _atomic_write_json(output_dir / "aggregate.json", aggregate)
        (output_dir / ".incomplete").unlink()
    except (HoldoutError, OSError, ValueError) as exc:
        print(f"blocked: {exc}", flush=True)
        return 2
    print(
        "[ledger-public-dev] FULL_BM25_OK "
        f"cases={aggregate['cases']} shards={args.shard_count} "
        "remote_calls=0 qwen3_calls=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
