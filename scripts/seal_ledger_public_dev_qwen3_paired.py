#!/usr/bin/env python3
"""Validate and publish the fixed LEDGER public-dev Qwen3 paired artifact."""
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

import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.eval.holdout import HoldoutError
from lumenfin.stdio import configure_stdio_utf8

SEALED_OUTPUT = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_qwen3_paired_5x50.json"
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
        (ROOT / "scripts" / "run_ledger_public_dev_qwen3_paired.py")
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .encode()
    ).hexdigest()


def _load_jsonl_bytes(path: Path, *, label: str) -> tuple[list[dict], bytes]:
    try:
        raw = path.read_bytes()
        rows = paired_cli._parse_jsonl_text(
            raw.decode("utf-8"),
            label=label,
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise HoldoutError(f"cannot read {label}") from exc
    return rows, raw


def _validate(
    *,
    aggregate_path: Path,
    per_case_path: Path,
    candidate_manifest_path: Path,
    candidate_cache_path: Path,
) -> bytes:
    aggregate, raw = _load_json_bytes(aggregate_path, label="paired aggregate")
    candidate_manifest, _ = _load_json_bytes(
        candidate_manifest_path,
        label="candidate manifest",
    )
    prerank, _ = _load_json_bytes(
        paired_cli.TRACKED_PRERANK,
        label="tracked Hybrid prerank",
    )
    per_case_rows, per_case_raw = _load_jsonl_bytes(
        per_case_path,
        label="paired per-case",
    )
    candidate_rows, candidate_raw = _load_jsonl_bytes(
        candidate_cache_path,
        label="candidate cache",
    )
    per_case_sha256 = hashlib.sha256(per_case_raw).hexdigest()
    cache_sha256 = hashlib.sha256(candidate_raw).hexdigest()
    candidate_set_sha256 = paired_cli._candidate_set_identity(candidate_rows)
    reranker_source_sha256 = paired_cli._reranker_source_sha256()
    run_identity_sha256 = paired_cli._canonical_sha256(
        {
            "candidate_cache_sha256": cache_sha256,
            "candidate_set_identity_sha256": candidate_set_sha256,
            "rerank_settings": aggregate.get("rerank_settings"),
            "reranker_source_sha256": reranker_source_sha256,
        }
    )
    candidate_by_id = {
        str(row.get("query_id") or ""): row
        for row in candidate_rows
    }
    completed = paired_cli._validate_completed_rows(
        per_case_rows,
        candidate_by_id=candidate_by_id,
        run_identity_sha256=run_identity_sha256,
    )
    before = paired_cli._summary(per_case_rows, "prerank_arms")
    after = paired_cli._summary(per_case_rows, "reranked_arms")
    expected_comparison = {
        "prerank": before,
        "qwen3": after,
        "delta_qwen3_minus_prerank": paired_cli._metric_deltas(
            before,
            after,
        ),
        "paired_counts": {
            arm: {
                metric: paired_cli._paired_counts(
                    per_case_rows,
                    arm=arm,
                    metric=metric,
                )
                for metric in ("hit_at_5", "hit_at_10")
            }
            for arm in ("A_prod", "R_page")
        },
    }
    attempts = sum(
        int(row["reranked_arms"][arm]["rerank_attempts"])
        for row in per_case_rows
        for arm in ("A_prod", "R_page")
    )
    tokens = sum(
        int(row["reranked_arms"][arm]["rerank_tokens"])
        for row in per_case_rows
        for arm in ("A_prod", "R_page")
    )
    fallbacks = sum(
        bool(row["reranked_arms"][arm]["rerank_fallback"])
        for row in per_case_rows
        for arm in ("A_prod", "R_page")
    )
    selection = candidate_manifest.get("selection") or {}
    query_ids = [str(row.get("query_id") or "") for row in candidate_rows]
    per_company = selection.get("per_company")
    if not isinstance(per_company, list):
        raise HoldoutError("candidate selection failed sealed validation")
    selection_offset = 0
    for company in per_company:
        if not isinstance(company, dict):
            raise HoldoutError("candidate selection failed sealed validation")
        selected_cases = company.get("selected_cases")
        if (
            isinstance(selected_cases, bool)
            or not isinstance(selected_cases, int)
            or selected_cases <= 0
        ):
            raise HoldoutError("candidate selection failed sealed validation")
        company_rows = candidate_rows[
            selection_offset : selection_offset + selected_cases
        ]
        company_query_ids = [
            str(row.get("query_id") or "") for row in company_rows
        ]
        company_key_hashes = {
            str(row.get("company_key_sha256") or "")
            for row in company_rows
        }
        if (
            len(company_rows) != selected_cases
            or paired_cli._ids_sha256(tuple(company_query_ids))
            != company.get("query_ids_sha256")
            or company_key_hashes != {company.get("company_key_sha256")}
        ):
            raise HoldoutError("candidate selection failed sealed validation")
        selection_offset += selected_cases
    candidate_calls = candidate_manifest.get("call_accounting") or {}
    document_calls = candidate_calls.get("document_embedding_remote_calls")
    query_calls = candidate_calls.get("query_embedding_remote_calls")
    candidate_remote_calls = candidate_calls.get("remote_calls")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (document_calls, query_calls, candidate_remote_calls)
    ):
        raise HoldoutError("candidate call accounting failed sealed validation")
    expected_aggregate_calls = {
        "candidate_embedding_remote_calls": candidate_remote_calls,
        "qwen3_logical_calls": len(per_case_rows) * 2,
        "qwen3_physical_attempts": attempts,
        "qwen3_tokens": tokens,
        "rerank_fallbacks": int(fallbacks),
        "remote_calls": candidate_remote_calls + attempts,
        "billing_semantics": "persisted_complete_cases_at_least_once",
        "unobserved_inflight_remote_calls_possible": True,
    }
    serialized = json.dumps(aggregate, ensure_ascii=False)
    if (
        aggregate.get("schema_version") != paired_cli.SCHEMA_VERSION
        or aggregate.get("orchestrator_source_sha256") != _script_sha256()
        or aggregate.get("reranker_source_sha256")
        != reranker_source_sha256
        or aggregate.get("run_identity_sha256") != run_identity_sha256
        or len(candidate_rows) != 250
        or len(per_case_rows) != 250
        or len(candidate_by_id) != 250
        or tuple(completed) != tuple(candidate_by_id)
        or aggregate.get("cases") != len(per_case_rows)
        or aggregate.get("selection") != selection
        or paired_cli._canonical_sha256(prerank)
        != paired_cli.PINNED_PRERANK_CANONICAL_SHA256
        or selection != prerank.get("selection")
        or selection_offset != len(candidate_rows)
        or paired_cli._ids_sha256(tuple(query_ids))
        != selection.get("query_ids_sha256")
        or selection.get("cases") != len(candidate_rows)
        or aggregate.get("per_case_sha256") != per_case_sha256
        or candidate_manifest.get("candidate_cache_sha256") != cache_sha256
        or candidate_manifest.get("candidate_set_identity_sha256")
        != candidate_set_sha256
        or candidate_manifest.get("cases") != len(candidate_rows)
        or candidate_manifest.get("qwen3_calls") != 0
        or candidate_manifest.get("source_prerank_canonical_sha256")
        != paired_cli.PINNED_PRERANK_CANONICAL_SHA256
        or candidate_manifest.get("retrieval_evaluator_source_sha256")
        != paired_cli.retrieval_cli._evaluator_source_sha256()
        or candidate_calls.get("documents_indexed")
        != selection.get("documents")
        or candidate_calls.get("chunks_indexed") != selection.get("chunks")
        or candidate_calls.get("embed_chars") != selection.get("embed_chars")
        or not isinstance(document_calls, int)
        or isinstance(document_calls, bool)
        or document_calls < selection.get("estimated_document_http_calls", 0)
        or not isinstance(query_calls, int)
        or isinstance(query_calls, bool)
        or query_calls
        < selection.get("expected_query_http_calls_minimum", 0)
        or candidate_remote_calls != document_calls + query_calls
        or aggregate.get("candidate_manifest")
        != {
            "candidate_cache_sha256": cache_sha256,
            "candidate_set_identity_sha256": candidate_set_sha256,
            "source_prerank_canonical_sha256": candidate_manifest.get(
                "source_prerank_canonical_sha256"
            ),
        }
        or aggregate.get("comparison") != expected_comparison
        or aggregate.get("call_accounting") != expected_aggregate_calls
        or aggregate.get("remote_calls")
        != candidate_remote_calls + attempts
        or aggregate.get("qwen3_calls") != attempts
        or aggregate.get("primary_comparison_valid")
        is not paired_cli._all_qwen3_ok(per_case_rows)
        or aggregate.get("product_accuracy_claim") is not False
        or '"query_text"' in serialized
        or '"text"' in serialized
    ):
        raise HoldoutError("paired Qwen3 artifact failed sealed validation")
    for candidate in candidate_rows:
        expected_identity = paired_cli._candidate_row(
            query={
                "query_id": candidate["query_id"],
                "query_text": "",
                "company_key": "",
            },
            hits=list(candidate.get("hits") or []),
        )["candidate_identity_sha256"]
        if candidate.get("candidate_identity_sha256") != expected_identity:
            raise HoldoutError("candidate row identity failed sealed validation")
    return raw


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--per-case", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        output = Path(args.output).expanduser().resolve()
        if output != SEALED_OUTPUT:
            raise HoldoutError("sealed output path is not the fixed tracked path")
        if output.exists():
            raise HoldoutError("refusing to overwrite sealed Qwen3 artifact")
        raw = _validate(
            aggregate_path=Path(args.aggregate).expanduser().resolve(),
            per_case_path=Path(args.per_case).expanduser().resolve(),
            candidate_manifest_path=Path(
                args.candidate_manifest
            ).expanduser().resolve(),
            candidate_cache_path=Path(args.candidate_cache).expanduser().resolve(),
        )
        tmp = output.with_name(output.name + ".tmp")
        tmp.write_bytes(raw)
        os.replace(tmp, output)
        print(f"sealed: {output}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001 - redact CLI boundary failures
        safe = exc if isinstance(exc, HoldoutError) else HoldoutError(
            f"Qwen3 seal failed: {type(exc).__name__}"
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
