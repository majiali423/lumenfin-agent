#!/usr/bin/env python3
"""Local-only parent-page packing probe for the sealed LEDGER e2e canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
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
import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_parent_probe import (
    PACK_STRATEGIES,
    SCHEMA_VERSION,
    recoverability,
    unique_document_ids,
)
from lumenfin.stdio import configure_stdio_utf8

TRACKED_TAXONOMY = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_e2e_failure_taxonomy_5x10.json"
).resolve()
ARM = "qwen3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether returning parent pages would put gold KPI digits "
            "into the generation context. Local-only; no production changes."
        )
    )
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--e2e-aggregate", required=True)
    parser.add_argument("--e2e-per-case", required=True)
    parser.add_argument("--taxonomy-aggregate", required=True)
    parser.add_argument("--taxonomy-per-case", required=True)
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


def _page_by_id(dataset: Any) -> dict[str, str]:
    pages: dict[str, str] = {}
    for document in dataset.page_documents:
        document_id = str(document.get("document_id") or "").strip()
        text = "".join(str(part) for part in (document.get("pages") or []))
        if not document_id or not text.strip():
            raise HoldoutError("public-dev page corpus has an empty page")
        if document_id in pages:
            raise HoldoutError("public-dev page corpus has duplicate document_id")
        pages[document_id] = text
    if not pages:
        raise HoldoutError("public-dev page corpus is empty")
    return pages


def _final_chunk_text(
    *,
    candidate: Mapping[str, Any],
    final_identity: Sequence[Mapping[str, Any]],
    max_document_chars: int,
) -> str:
    by_chunk = {
        str(hit["chunk_id"]): hit for hit in candidate["hits"]
    }
    parts: list[str] = []
    for item in final_identity:
        chunk_id = str(item.get("chunk_id") or "")
        hit = by_chunk.get(chunk_id)
        if hit is None:
            raise HoldoutError("parent probe final chunk is outside the frozen pool")
        parts.append(str(hit.get("text") or "")[: max(1, int(max_document_chars))])
    return "\n".join(parts)


def _positive_pages(qrels: Sequence[Mapping[str, Any]]) -> list[str]:
    pages: list[str] = []
    seen: set[str] = set()
    for item in qrels:
        if int(item["relevance"]) <= 0:
            continue
        doc_id = str(item["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        pages.append(doc_id)
    if not pages:
        raise HoldoutError("parent probe query has no positive qrels")
    return pages


def recommend_next(rows: list[dict]) -> str:
    evidence = [
        row
        for row in rows
        if row["leak_class"] == "evidence_gap_number_absent"
    ]
    pool_miss = [
        row for row in rows if row["leak_class"] == "retrieval_pool_miss"
    ]
    page_full_hits = sum(
        bool(row["recovered"]["retrieved_page_full"]) for row in evidence
    )
    gold_full_hits = sum(
        bool(row["recovered"]["gold_page_full"]) for row in pool_miss
    )
    if evidence and page_full_hits * 2 >= len(evidence):
        return "retrieve_child_return_parent_page"
    if pool_miss and gold_full_hits * 2 >= len(pool_miss):
        return "new_eval_index_for_recall"
    return "source_digit_gap"


def _leak_recovery(rows: list[dict]) -> dict[str, Any]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["leak_class"]), []).append(row)
    payload: dict[str, Any] = {}
    for leak, items in sorted(grouped.items()):
        payload[leak] = {
            "cases": len(items),
            **{
                strategy: sum(bool(item["recovered"][strategy]) for item in items)
                for strategy in PACK_STRATEGIES
            },
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    try:
        if args.allow_remote:
            raise HoldoutError("parent pack probe is local-only")
        taxonomy_path = Path(args.taxonomy_aggregate).expanduser().resolve()
        if taxonomy_path != TRACKED_TAXONOMY:
            raise HoldoutError("probe must use the tracked sealed taxonomy")
        if Path(args.e2e_aggregate).expanduser().resolve() != tax_cli.TRACKED_E2E:
            raise HoldoutError("probe must use the tracked sealed e2e aggregate")
        taxonomy = paired_cli._load_json(
            taxonomy_path,
            label="sealed taxonomy aggregate",
        )
        e2e_aggregate = paired_cli._load_json(
            Path(args.e2e_aggregate),
            label="sealed e2e aggregate",
        )
        taxonomy_rows = paired_cli._read_jsonl(
            Path(args.taxonomy_per_case),
            label="taxonomy per-case",
        )
        e2e_rows = paired_cli._read_jsonl(
            Path(args.e2e_per_case),
            label="e2e per-case",
        )
        if (
            hashlib.sha256(Path(args.taxonomy_per_case).read_bytes()).hexdigest()
            != taxonomy.get("per_case_sha256")
            or hashlib.sha256(Path(args.e2e_per_case).read_bytes()).hexdigest()
            != e2e_aggregate.get("per_case_sha256")
            or taxonomy.get("recommended_next_workstream")
            != "section_parent_retrieval"
            or taxonomy.get("remote_calls") != 0
        ):
            raise HoldoutError("parent probe source artifacts are incompatible")
        bundle = e2e_cli._load_plan_inputs(args)
        context_proxy = argparse.Namespace(
            parquet_path=args.parquet_path,
            manifest=args.manifest,
            split_salt=args.split_salt,
            baseline_aggregate=args.baseline_aggregate,
            baseline_per_case=args.baseline_per_case,
            prerank_aggregate=args.prerank_aggregate,
            batch_size=64,
            embedding_dimension=1024,
        )
        _manifest, dataset, _plans, _audit, _prerank = paired_cli._load_context(
            context_proxy
        )
        pages = _page_by_id(dataset)
        e2e_by_id = {str(row["query_id"]): row for row in e2e_rows}
        tax_by_id = {str(row["query_id"]): row for row in taxonomy_rows}
        candidate_by_id = {
            str(row["query_id"]): row for row in bundle["selected"]
        }
        max_document_chars = int(
            e2e_aggregate["rerank_settings"]["max_document_chars"]
        )
        classified: list[dict[str, Any]] = []
        for query_id in bundle["selected_ids"]:
            e2e_row = e2e_by_id.get(query_id)
            tax_row = tax_by_id.get(query_id)
            candidate = candidate_by_id.get(query_id)
            query = bundle["query_by_id"].get(query_id)
            if e2e_row is None or tax_row is None or candidate is None or query is None:
                raise HoldoutError("parent probe coverage mismatch")
            generated = e2e_row["arms"][ARM]
            leak = str(tax_row["arms"][ARM]["leak_class"])
            recovered = recoverability(
                gold_value=float(generated["gold_value"]),
                chunk_final_text=_final_chunk_text(
                    candidate=candidate,
                    final_identity=list(generated["final_identity"]),
                    max_document_chars=max_document_chars,
                ),
                retrieved_page_ids=unique_document_ids(
                    list(generated["final_identity"])
                ),
                gold_page_ids=_positive_pages(list(query["qrels"])),
                page_by_id=pages,
            )
            chunk_flag = bool(tax_row["arms"][ARM]["number_in_final_context"])
            if bool(recovered["recovered"]["chunk_final"]) != chunk_flag:
                raise HoldoutError("parent probe chunk_final diverged from taxonomy")
            row = {
                "query_id": query_id,
                "e2e_run_identity_sha256": e2e_aggregate["run_identity_sha256"],
                "taxonomy_leak_class": leak,
                "leak_class": leak,
                **recovered,
            }
            row["row_sha256"] = _row_sha256(row)
            classified.append(row)
        totals = Counter()
        for strategy in PACK_STRATEGIES:
            totals[strategy] = sum(
                bool(row["recovered"][strategy]) for row in classified
            )
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        per_case_path = output_dir / "per_case.jsonl"
        _atomic_jsonl(per_case_path, classified)
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "probe_source_sha256": _script_sha256(),
            "e2e_run_identity_sha256": e2e_aggregate["run_identity_sha256"],
            "taxonomy_source_sha256": taxonomy.get("taxonomy_source_sha256"),
            "cases": 50,
            "arm": "A_prod",
            "ranking_arm": ARM,
            "recovered_cases": dict(totals),
            "recovered_by_leak_class": _leak_recovery(classified),
            "recommended_next_workstream": recommend_next(classified),
            "remote_calls": 0,
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
            else HoldoutError(f"LEDGER parent probe failed: {type(exc).__name__}")
        )
        print(f"blocked: {safe}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
