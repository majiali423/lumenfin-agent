#!/usr/bin/env python3
"""Verify immediate dense and optional BM25 search for a durable RAG document."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify first dense/BM25 RAG search.")
    parser.add_argument("--expect-collection")
    args = parser.parse_args()

    config = AppConfig.from_env()
    repository = RagDocumentRepository(config.database_url, db_path=config.db_path)
    documents = repository.list_documents(index_status="ready")
    if not documents:
        raise RuntimeError("no ready RAG documents")
    document = documents[0]
    chunks = repository.list_chunks(
        tenant_id=document["tenant_id"],
        source_document_ids=[document["document_id"]],
    )
    if not chunks:
        raise RuntimeError("ready document has no durable chunks")

    store = build_rag_store(config)
    if store is None:
        raise RuntimeError("RAG store is disabled")
    try:
        if args.expect_collection and store.collection_name != args.expect_collection:
            raise RuntimeError(
                f"configured collection mismatch: expected={args.expect_collection}, "
                f"actual={store.collection_name}"
            )
        hits = store.vector_search(
            chunks[0]["text"],
            tenant_id=document["tenant_id"],
            source_document_ids=[document["document_id"]],
            top_k=3,
        )
        if not hits:
            raise RuntimeError("first application search returned no vector hits")
        bm25_hits = []
        if store.bm25_enabled:
            bm25_hits = store.bm25_search(
                chunks[0]["text"],
                tenant_id=document["tenant_id"],
                source_document_ids=[document["document_id"]],
                top_k=3,
            )
            if not bm25_hits:
                raise RuntimeError("first application search returned no BM25 hits")
        print(
            json.dumps(
                {
                    "backend": store.backend,
                    "collection": store.collection_name,
                    "document_id": document["document_id"],
                    "vector_hits": len(hits),
                    "bm25_enabled": store.bm25_enabled,
                    "bm25_hits": len(bm25_hits),
                },
                sort_keys=True,
            )
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
