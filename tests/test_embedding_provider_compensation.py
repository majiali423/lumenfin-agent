"""Embedding provider failure compensation and retry-layer separation."""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lumenfin.database import RagDocumentRepository
from lumenfin.provider_resilience import (
    InvalidProviderResponseError,
    ProviderCallContext,
    classify_provider_exception,
)
from lumenfin.rag.embeddings import DashScopeEmbeddingProvider, ResilientEmbeddingProvider
from lumenfin.rag.indexer import DocumentIndexer


def _md(path: Path) -> Path:
    path.write_text("# Apple\n\nRevenue grew in fiscal 2024.\n", encoding="utf-8")
    return path


class _EmbeddingAwareStore:
    """Minimal vector store that embeds via provider then can fail/cleanup."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.rows: dict[tuple[str, str, str], dict] = {}
        self.index_calls = 0
        self.delete_calls = 0

    def index_chunks(self, chunks, *, tenant_id: str, source_document_id: str, **_kwargs):
        self.index_calls += 1
        texts = [str(chunk.get("text") or "") for chunk in chunks]
        # Provider retry happens inside embedder; Redis job retry is a separate layer.
        vectors = self.embedder.embed(texts)
        for chunk, vector in zip(chunks, vectors):
            key = (tenant_id, source_document_id, str(chunk["chunk_id"]))
            self.rows[key] = {
                **chunk,
                "tenant_id": tenant_id,
                "source_document_id": source_document_id,
                "vector_dim": len(vector),
            }
        return {"chunks_indexed": len(chunks), "embed_calls": 1}

    def vector_search(self, query: str, *, tenant_id: str | None = None, **_kwargs) -> list[dict]:
        return [
            dict(row)
            for row in self.rows.values()
            if tenant_id is None or row["tenant_id"] == tenant_id
        ]

    def delete_by_source_document(self, *, tenant_id: str, source_document_id: str) -> int:
        self.delete_calls += 1
        before = len(self.rows)
        self.rows = {
            key: value
            for key, value in self.rows.items()
            if key[:2] != (tenant_id, source_document_id)
        }
        return before - len(self.rows)

    def close(self) -> None:
        return None


class EmbeddingProviderFailureTestCase(unittest.TestCase):
    def test_invalid_response_scenarios_do_not_transient_retry(self) -> None:
        from provider_stub.server import reset_state, serve

        reset_state()
        server = serve("127.0.0.1", 18091)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for scenario in (
                "embedding_count_mismatch",
                "embedding_dimension_mismatch",
                "malformed_json",
            ):
                http = httpx.Client(timeout=2.0, trust_env=False)
                real_post = http.post

                def post(url, headers=None, json=None, timeout=None, _s=scenario):
                    headers = dict(headers or {})
                    headers["X-LumenFin-Scenario"] = _s
                    return real_post(url, headers=headers, json=json, timeout=timeout)

                http.post = post  # type: ignore[method-assign]
                inner = DashScopeEmbeddingProvider(
                    api_key="stub",
                    dimension=64,
                    base_url="http://127.0.0.1:18091/v1",
                    timeout_seconds=2.0,
                    client=http,
                )
                provider = ResilientEmbeddingProvider(
                    inner, max_retries=3, backoff_seconds=0.01, sleep=lambda _: None
                )
                with self.assertRaises(Exception) as raised:
                    provider.embed(["a", "b"])
                self.assertEqual(classify_provider_exception(raised.exception), "invalid_response")
                self.assertEqual(provider.last_attempts, 1)
                http.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_timeout_until_deadline_stops_provider_retry(self) -> None:
        calls = {"n": 0}

        class SlowInner:
            dimension = 64

            def embed(self, texts):
                calls["n"] += 1
                raise TimeoutError("embedding timed out")

        ctx = ProviderCallContext.create(deadline_seconds=0.15)
        clock = {"t": 100.0}
        ctx.now = lambda: clock["t"]
        ctx.deadline_monotonic = clock["t"] + 0.15

        def sleep(delay: float) -> None:
            clock["t"] += float(delay)

        ctx.sleep = sleep
        provider = ResilientEmbeddingProvider(
            SlowInner(), max_retries=5, backoff_seconds=0.05, sleep=sleep, call_context=ctx
        )
        with self.assertRaises(Exception):
            provider.embed(["x"])
        self.assertGreaterEqual(calls["n"], 1)
        self.assertLess(calls["n"], 5)

    def test_indexer_compensation_on_invalid_embedding(self) -> None:
        class BoomEmbedder:
            dimension = 384
            calls = 0

            def embed(self, texts):
                BoomEmbedder.calls += 1
                # Simulate partial write then failure: pre-seed a vector row that cleanup must clear.
                raise InvalidProviderResponseError("embedding_count_mismatch")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            db = root / "rag.db"
            repo = RagDocumentRepository(f"sqlite:///{db.as_posix()}")
            embedder = BoomEmbedder()
            store = _EmbeddingAwareStore(embedder)
            # Pre-existing orphan-like row for same upcoming document is cleaned by _fail path.
            indexer = DocumentIndexer(rag_store=store, repository=repo)
            path = _md(root / "apple.md")
            # Inject a partial vector before failure by wrapping index_chunks.
            real_index = store.index_chunks

            def index_with_partial(chunks, *, tenant_id, source_document_id, **kwargs):
                store.rows[(tenant_id, source_document_id, "partial")] = {
                    "chunk_id": "partial",
                    "tenant_id": tenant_id,
                    "source_document_id": source_document_id,
                    "text": "partial",
                }
                return real_index(
                    chunks,
                    tenant_id=tenant_id,
                    source_document_id=source_document_id,
                    **kwargs,
                )

            store.index_chunks = index_with_partial  # type: ignore[method-assign]
            receipt = indexer.index_file(path, tenant_id="tenant-a")
            self.assertEqual(receipt["status"], "failed")
            persisted = repo.get_document(receipt["document_id"], tenant_id="tenant-a")
            self.assertEqual(persisted["index_status"], "failed")
            self.assertEqual(
                len(repo.list_chunks(tenant_id="tenant-a", source_document_ids=[receipt["document_id"]])),
                0,
            )
            self.assertEqual(store.vector_search("Apple", tenant_id="tenant-a"), [])
            self.assertGreaterEqual(store.delete_calls, 1)
            self.assertEqual(BoomEmbedder.calls, 1)


if __name__ == "__main__":
    unittest.main()
