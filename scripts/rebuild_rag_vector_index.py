#!/usr/bin/env python3
"""Preflight or execute a controlled dense/BM25 RAG collection rebuild."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.config import AppConfig
from lumenfin.database import RagDocumentRepository
from lumenfin.rag.factory import build_rag_store
from lumenfin.rag.rebuild import build_rebuild_manifest, rebuild_vector_index
from lumenfin.stdio import configure_stdio_utf8


def main() -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        description="Rebuild the configured Milvus collection from durable RAG chunks."
    )
    parser.add_argument("--tenant-id", help="Limit the rebuild to one tenant.")
    parser.add_argument("--execute", action="store_true", help="Reset and rebuild the target collection.")
    parser.add_argument(
        "--confirm-reset",
        metavar="COLLECTION",
        help="Required with --execute; must exactly match MAS_MILVUS_COLLECTION.",
    )
    args = parser.parse_args()

    config = AppConfig.from_env()
    repository = RagDocumentRepository(config.database_url, db_path=config.db_path)
    manifest = build_rebuild_manifest(repository, tenant_id=args.tenant_id)
    preflight = {
        "mode": "preflight",
        "collection": config.milvus_collection,
        "documents": len(manifest),
        "chunks": sum(int(item["chunk_count"]) for item in manifest),
        "tenant_id": args.tenant_id,
        "bm25_enabled": config.rag_bm25_enabled,
    }
    if not args.execute:
        print(json.dumps(preflight, ensure_ascii=False, sort_keys=True))
        return 0

    if args.confirm_reset != config.milvus_collection:
        parser.error(
            "--execute requires --confirm-reset with the exact configured collection name "
            f"({config.milvus_collection})"
        )
    if args.tenant_id:
        parser.error("--tenant-id cannot be combined with --execute because reset is collection-wide")
    if not manifest:
        parser.error("refusing to reset the collection because no ready documents were found")

    store = build_rag_store(config)
    if store is None:
        parser.error("MAS_RAG_ENABLED must be true")
    try:
        result = rebuild_vector_index(
            repository=repository,
            store=store,
            tenant_id=args.tenant_id,
            reset_collection=True,
        )
    finally:
        store.close()
    print(json.dumps({"mode": "executed", **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
