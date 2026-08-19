#!/usr/bin/env python3
"""Local BM25 page-parent eval index on the frozen 5x40 suffix. No production RAG."""
from __future__ import annotations

import argparse
import hashlib
import json
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

import run_ledger_public_dev_parent_pack_suffix as suffix_cli
import run_ledger_public_dev_parent_page_e2e as parent_e2e_cli
import run_ledger_public_dev_qwen3_paired as paired_cli

from lumenfin.eval.financebench.retrieval import build_eval_store
from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_parent_return import HOLDOUT_CASES_PER_COMPANY
from lumenfin.eval.holdout.ledger_section_parent import (
    COLLECTION_NAME,
    LOCKED_INDEX_UNIT,
    SCHEMA_VERSION,
    parent_page_index_unit,
    pool_hit,
    recommend_next,
    select_company_pages,
)
from lumenfin.stdio import configure_stdio_utf8

TRACKED_TAXONOMY = (
    ROOT
    / "data"
    / "eval_rag"
    / "holdout"
    / "ledger_public_dev_parent_page_e2e_taxonomy_5x40.json"
).resolve()
SOURCE_K = 20
FINAL_K = 10
ARM = "P_page"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Index frozen-company LEDGER pages as BM25 parents and score "
            "suffix pool recall. Local-only; does not change production RAG."
        )
    )
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--paired-aggregate", required=True)
    parser.add_argument("--paired-per-case", required=True)
    parser.add_argument("--taxonomy-aggregate", required=True)
    parser.add_argument("--taxonomy-per-case", required=True)
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


def _index_pages(store: Any, units: list[dict], *, session_id: str) -> int:
    for unit in units:
        stats = store.index_chunks(
            [unit],
            tenant_id=session_id,
            source_document_id=str(unit["document_id"]),
            session_id=session_id,
            replace_existing=True,
        )
        if int(stats.get("chunks_indexed") or 0) != 1:
            raise HoldoutError("section-parent index did not write one page unit")
    return len(units)


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    store = None
    index_dir: Path | None = None
    try:
        if args.allow_remote:
            raise HoldoutError("section-parent BM25 preflight is local-only")
        taxonomy_path = Path(args.taxonomy_aggregate).expanduser().resolve()
        if taxonomy_path != TRACKED_TAXONOMY:
            raise HoldoutError("section-parent must use the tracked suffix taxonomy")
        suffix_path = Path(args.suffix_aggregate).expanduser().resolve()
        if suffix_path != parent_e2e_cli.TRACKED_SUFFIX:
            raise HoldoutError("section-parent must use the tracked suffix probe")
        taxonomy = paired_cli._load_json(
            taxonomy_path,
            label="sealed parent-page taxonomy",
        )
        if (
            taxonomy.get("recommended_next_workstream") != "section_parent_retrieval"
            or taxonomy.get("product_accuracy_claim") is not False
            or taxonomy.get("financebench_phase4") != "NOT_RUN"
            or taxonomy.get("remote_calls") != 0
        ):
            raise HoldoutError("taxonomy did not authorize a new eval index")
        taxonomy_rows = paired_cli._read_jsonl(
            Path(args.taxonomy_per_case),
            label="sealed parent-page taxonomy per-case",
        )
        if (
            hashlib.sha256(Path(args.taxonomy_per_case).read_bytes()).hexdigest()
            != taxonomy.get("per_case_sha256")
            or len(taxonomy_rows) != 200
        ):
            raise HoldoutError("taxonomy per-case identity diverged")
        frozen = suffix_cli._load_frozen(args)
        if (
            paired_cli._ids_sha256(tuple(frozen["suffix_ids"]))
            != taxonomy["selection"]["query_ids_sha256"]
        ):
            raise HoldoutError("section-parent suffix identity diverged")
        company_keys = [str(plan["company_key"]) for plan in frozen["plans"]]
        pages = select_company_pages(frozen["dataset"].page_documents, company_keys)
        units = [parent_page_index_unit(page) for page in pages]
        taxonomy_by_id = {str(row["query_id"]): row for row in taxonomy_rows}
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        index_dir = output_dir / f"_index_{uuid4().hex[:8]}"
        index_dir.mkdir()
        store = build_eval_store(
            uri=str(index_dir / "section_parent.db"),
            embedding_provider="deterministic",
            embedding_dimension=384,
            collection_name=COLLECTION_NAME,
            allow_remote=False,
            mode="bm25",
        )
        session_id = (
            "ledger-section-parent-"
            + taxonomy["selection"]["query_ids_sha256"][:12]
        )
        indexed = _index_pages(store, units, session_id=session_id)
        classified: list[dict[str, Any]] = []
        for candidate in frozen["suffix_rows"]:
            query_id = str(candidate["query_id"])
            query = frozen["query_by_id"][query_id]
            tax_row = taxonomy_by_id.get(query_id)
            if tax_row is None:
                raise HoldoutError("section-parent query missing taxonomy row")
            hits = store.bm25_search(
                str(query["query_text"]),
                session_id=session_id,
                companies=[str(query["company_key"])],
                top_k=SOURCE_K,
            )
            if not hits:
                raise HoldoutError("section-parent BM25 returned no hits")
            parent_pool = pool_hit(hits, list(query["qrels"]))
            parent_hit10 = pool_hit(hits[:FINAL_K], list(query["qrels"]))
            hybrid_pool = pool_hit(list(candidate["hits"]), list(query["qrels"]))
            leak = str(tax_row["arms"]["parent_page"]["leak_class"])
            row = {
                "query_id": query_id,
                "arm": ARM,
                "locked_index_unit": LOCKED_INDEX_UNIT,
                "hybrid_pool_hit": hybrid_pool,
                "parent_pool_hit": parent_pool,
                "parent_hit_at_10": parent_hit10,
                "parent_pool_size": len(hits),
                "taxonomy_parent_leak_class": leak,
                "recovered_pool_miss": bool(
                    leak == "retrieval_pool_miss" and parent_pool and not hybrid_pool
                ),
                "final_identity_sha256": paired_cli._hash_hits(hits[:FINAL_K]),
            }
            row["row_sha256"] = _row_sha256(row)
            classified.append(row)
        if tuple(row["query_id"] for row in classified) != tuple(frozen["suffix_ids"]):
            raise HoldoutError("section-parent coverage mismatch")
        hybrid_pool_hits = sum(bool(row["hybrid_pool_hit"]) for row in classified)
        parent_pool_hits = sum(bool(row["parent_pool_hit"]) for row in classified)
        parent_hit10 = sum(bool(row["parent_hit_at_10"]) for row in classified)
        pool_misses = [
            row
            for row in classified
            if row["taxonomy_parent_leak_class"] == "retrieval_pool_miss"
        ]
        recovered = sum(bool(row["recovered_pool_miss"]) for row in classified)
        gains = sum(
            bool(row["parent_pool_hit"] and not row["hybrid_pool_hit"])
            for row in classified
        )
        losses = sum(
            bool(row["hybrid_pool_hit"] and not row["parent_pool_hit"])
            for row in classified
        )
        per_case_path = output_dir / "per_case.jsonl"
        _atomic_jsonl(per_case_path, classified)
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "probe_source_sha256": _script_sha256(),
            "cases": len(classified),
            "companies": 5,
            "cases_per_company": HOLDOUT_CASES_PER_COMPANY,
            "selection": taxonomy["selection"],
            "arm": ARM,
            "locked_index_unit": LOCKED_INDEX_UNIT,
            "pages_indexed": indexed,
            "pool_hit_at_20": {
                "hybrid_chunk": hybrid_pool_hits,
                "parent_page_bm25": parent_pool_hits,
                "delta_parent_minus_hybrid": parent_pool_hits - hybrid_pool_hits,
            },
            "parent_hit_at_10": parent_hit10,
            "paired_pool_hit": {
                "gain": gains,
                "loss": losses,
                "unchanged": len(classified) - gains - losses,
            },
            "taxonomy_pool_miss_cases": len(pool_misses),
            "taxonomy_pool_miss_recovered": recovered,
            "recommended_next_workstream": recommend_next(
                hybrid_pool_hits=hybrid_pool_hits,
                parent_pool_hits=parent_pool_hits,
                cases=len(classified),
            ),
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
                f"LEDGER section-parent preflight failed: {type(exc).__name__}"
            )
        )
        print(f"blocked: {safe}", flush=True)
        return 2
    finally:
        if store is not None:
            closer = getattr(store, "close", None)
            if callable(closer):
                closer()
        if index_dir is not None:
            shutil.rmtree(index_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
