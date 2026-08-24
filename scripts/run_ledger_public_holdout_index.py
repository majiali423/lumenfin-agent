#!/usr/bin/env python3
"""Build or dry-run the LEDGER public_holdout rc5 document index. No query/gold/qrels."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.env_bootstrap import bootstrap_dotenv
from lumenfin.eval.ledger_public_holdout_index import (
    COLLECTION,
    DEFAULT_SNAPSHOT,
    DRYRUN_DIR,
    INDEX_DIR,
    PRODUCT_COMMIT,
    PRODUCT_TAG,
    SEAL_PATH,
    SESSION_ID,
    HoldoutIndexError,
    assert_rc5_sources,
    atomic_write_json,
    budget_ok,
    credential_present,
    materialize_documents,
    open_store,
    run_canary,
    run_dry_run,
    scan_holdout_reports,
    sha256_path,
    verify_official_identity,
)
from lumenfin.stdio import configure_stdio_utf8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Document-only LEDGER public_holdout index using rc5 chunker and "
            "DashScope text-embedding-v4. Never reads holdout questions, gold, "
            "or relevance labels."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--output-dir", default=str(INDEX_DIR))
    return parser


def _git_head() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        .decode()
        .strip()
    )


def _worktree_clean() -> bool:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
    return not status.strip()


def _index_documents(store, documents: list[dict], *, batch_size: int, checkpoint: Path, done: set[str]) -> dict:
    totals = {
        "documents_indexed": 0,
        "chunks_indexed": 0,
        "embed_logical_calls": 0,
        "embed_physical_calls": 0,
        "embed_chars": 0,
        "billing_semantics": "at_least_once",
        "exactly_once_claimed": False,
    }
    pending = [item for item in documents if str(item["document_id"]) not in done]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        stats = store.index_documents(batch, SESSION_ID)
        totals["documents_indexed"] += int(stats.get("documents_indexed") or 0)
        totals["chunks_indexed"] += int(stats.get("chunks_indexed") or 0)
        totals["embed_logical_calls"] += int(stats.get("embed_calls") or 0)
        totals["embed_chars"] += int(stats.get("embed_chars") or 0)
        totals["embed_physical_calls"] += int(
            getattr(store.embedder, "last_physical_calls", 0) or 0
        )
        for item in batch:
            done.add(str(item["document_id"]))
        atomic_write_json(
            checkpoint,
            {
                "completed_document_ids": sorted(done),
                "completed_count": len(done),
                "totals": totals,
            },
        )
    return totals


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    bootstrap_dotenv()
    args = build_parser().parse_args(argv)
    snapshot = (ROOT / args.snapshot).resolve() if not Path(args.snapshot).is_absolute() else Path(args.snapshot)
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    try:
        if args.dry_run:
            payload = run_dry_run(repo_root=ROOT, snapshot=snapshot, official=True)
            dest = ROOT / DRYRUN_DIR
            dest.mkdir(parents=True, exist_ok=True)
            atomic_write_json(dest / "estimate.json", payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0 if payload["status"] == "BUDGET_OK" else 3
        if args.canary_only:
            dry = run_dry_run(repo_root=ROOT, snapshot=snapshot, official=True)
            store = open_store(
                uri=str((output_dir / "milvus_lite.db").resolve()),
                allow_remote=args.allow_remote,
            )
            canary = run_canary(
                store=store,
                companies=list((dry.get("corpus") or {}).get("companies") or []),
                expected_chunks=int(dry["estimate"]["chunks"]),
            )
            print(json.dumps(canary, ensure_ascii=False, indent=2))
            return 0
        if not args.build:
            raise HoldoutIndexError("choose --dry-run or --build")
        if not args.allow_remote:
            raise HoldoutIndexError("official index build requires --allow-remote")
        cred = credential_present()
        if not cred["present"]:
            raise HoldoutIndexError("DASHSCOPE_API_KEY is missing")
        dry = run_dry_run(repo_root=ROOT, snapshot=snapshot, official=True)
        if not budget_ok(dry["estimate"]):
            raise HoldoutIndexError("dry-run exceeds the hard embedding budget")
        if output_dir.exists() and not args.resume:
            raise HoldoutIndexError("index directory already exists; use --resume or a new path")
        if args.resume and not output_dir.exists():
            raise HoldoutIndexError("resume requested but index directory is missing")
        output_dir.mkdir(parents=True, exist_ok=True)
        reports, _audit, _rows = scan_holdout_reports(snapshot)
        documents, stats = materialize_documents(reports)
        verify_official_identity(repo_root=ROOT, snapshot=snapshot, stats=stats)
        checkpoint = output_dir / "checkpoint.json"
        done: set[str] = set()
        if args.resume and checkpoint.exists():
            previous = json.loads(checkpoint.read_text(encoding="utf-8"))
            done = set(previous.get("completed_document_ids") or [])
        started = time.perf_counter()
        store = open_store(
            uri=str((output_dir / "milvus_lite.db").resolve()),
            allow_remote=True,
        )
        totals = _index_documents(
            store, documents, batch_size=8, checkpoint=checkpoint, done=done
        )
        canary = run_canary(
            store=store,
            companies=list(stats["companies"]),
            expected_chunks=int(dry["estimate"]["chunks"]),
        )
        elapsed = round(time.perf_counter() - started, 2)
        db_path = output_dir / "milvus_lite.db"
        seal = {
            "schema_version": "ledger_public_holdout_index_seal.v1",
            "status": "SEALED",
            "product_tag": PRODUCT_TAG,
            "product_commit": PRODUCT_COMMIT,
            "index_harness_commit": _git_head(),
            "worktree_clean": _worktree_clean(),
            "change_kind": "resource_authorization_before_holdout_consumption",
            "model_or_retrieval_tuning": False,
            "collection": COLLECTION,
            "session_id": SESSION_ID,
            "tenant_id": SESSION_ID,
            "index_dir": str(INDEX_DIR).replace("\\", "/"),
            "uri": str(Path("outputs") / "ledger_public_holdout_rc5_index_v1" / "milvus_lite.db").replace("\\", "/"),
            "schema": "dense_bm25_v1",
            "embedding": dry["embedding"],
            "chunker": dry["estimate"]["chunker"],
            "rc5_source_sha256": assert_rc5_sources(repo_root=ROOT),
            "corpus": {key: value for key, value in {**dry["corpus"], **stats}.items() if key != "companies"},
            "build": {
                **totals,
                "elapsed_seconds": elapsed,
                "resumed": bool(args.resume),
                "document_embedding_http_requests": totals["embed_physical_calls"],
                "deepseek_calls": 0,
                "qwen3_calls": 0,
            },
            "canary": canary,
            "input_access_audit": dry["input_access_audit"],
            "milvus_db_sha256": sha256_path(db_path) if db_path.exists() else "",
            "milvus_db_bytes": db_path.stat().st_size if db_path.exists() else 0,
            "holdout_consumed": False,
        }
        atomic_write_json(output_dir / "index_seal.json", seal)
        tracked = dict(seal)
        tracked["large_index_gitignored"] = True
        atomic_write_json(ROOT / SEAL_PATH, tracked)
        print(json.dumps({"status": "SEALED", "chunks": totals["chunks_indexed"]}, indent=2))
        return 0
    except HoldoutIndexError as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
