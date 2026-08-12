"""Regression: tenant/stored-chunk harness must reach hybrid when vectors exist."""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.embeddings import DeterministicEmbeddingProvider
from lumenfin.rag.hybrid_retriever import HybridEvidenceRetriever
from lumenfin.rag.indexer import DocumentIndexer
from lumenfin.rag.milvus_store import MilvusRAGStore
from lumenfin.database import RagDocumentRepository


class HybridHarnessParityTestCase(unittest.TestCase):
    def test_tenant_index_then_retrieve_is_hybrid_not_keyword_only(self) -> None:
        root = ROOT / "test_artifacts" / f"hybrid-parity-{uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=True)
        uri = str(root / "rag.db")
        db_path = root / "meta.db"
        try:
            store = MilvusRAGStore(uri, DeterministicEmbeddingProvider(), collection_name="parity")
            repo = RagDocumentRepository(f"sqlite:///{db_path.as_posix()}", db_path=db_path)
            indexer = DocumentIndexer(rag_store=store, repository=repo, tenant_id="default")
            fixture = root / "nvda_note.pdf"
            # Minimal PDF via text file is not PDF; index markdown through indexer path using md.
            md = root / "nvda_note.md"
            md.write_text(
                "NVIDIA Form 10-K excerpt.\n"
                "Data Center revenue grew. Net revenue was 130497 million in fiscal 2025.\n"
                "Supply chain and foundry packaging risks remain.\n",
                encoding="utf-8",
            )
            receipts = indexer.index_paths([str(md)], tenant_id="default")
            self.assertTrue(receipts)
            self.assertEqual(receipts[0]["status"], "ready")
            doc_id = receipts[0]["document_id"]
            contexts = list(receipts[0].get("contexts") or [])

            def chunk_loader(*, tenant_id: str, source_document_ids: list[str]):
                return indexer.list_chunks(tenant_id=tenant_id, source_document_ids=source_document_ids)

            retriever = HybridEvidenceRetriever(
                store,
                top_k=3,
                chunk_loader=chunk_loader,
                rerank_enabled=True,
                rerank_candidates=8,
            )
            hits, meta = retriever.retrieve_for_company_with_meta(
                query="What was NVIDIA revenue in fiscal 2025?",
                company="NVIDIA",
                session_id="default",
                document_contexts=contexts,
                tenant_id="default",
                source_document_ids=[doc_id],
                use_stored_chunks=True,
            )
            self.assertGreaterEqual(len(hits), 1)
            mode = str(meta.get("mode") or "")
            self.assertIn("hybrid_dense_bm25_rrf", mode, msg=f"expected hybrid mode, got {meta}")
            self.assertGreater(int(meta.get("vector_hits") or 0), 0)
            self.assertGreater(int(meta.get("bm25_hits") or 0), 0)
        finally:
            try:
                store.close()
            except Exception:
                pass
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
