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

from lumenfin.rag.chunking import chunk_document
from lumenfin.rag.embeddings import DeterministicEmbeddingProvider
from lumenfin.rag.hybrid_retriever import HybridEvidenceRetriever, reciprocal_rank_fusion
from lumenfin.rag.milvus_store import MilvusRAGStore


def _make_temp_milvus_uri() -> tuple[Path, str]:
    tmp_dir = ROOT / "test_artifacts" / f"rag-{uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir, str(tmp_dir / "rag.db")


class RagModuleTestCase(unittest.TestCase):
    def test_chunking_tags_financial_signal(self) -> None:
        document = {
            "document_id": "apple-q1",
            "filename": "apple.pdf",
            "detected_companies": ["Apple"],
            "pages": [
                "Apple reported revenue of 400 billion and EBITDA margin expansion in 2025.",
                "Supply chain risk remains elevated due to concentration in Asia manufacturing.",
            ],
        }
        chunks = chunk_document(document)
        chunk_types = {chunk["chunk_type"] for chunk in chunks}
        self.assertIn("financial_metric", chunk_types)
        self.assertIn("risk_signal", chunk_types)

    def test_milvus_index_and_vector_search(self) -> None:
        tmp_dir, uri = _make_temp_milvus_uri()
        embedder = DeterministicEmbeddingProvider()
        store = MilvusRAGStore(uri, embedder, collection_name="rag_test")
        try:
            documents = [
                {
                    "document_id": "apple-q1",
                    "filename": "apple.pdf",
                    "detected_companies": ["Apple"],
                    "pages": [
                        "Apple revenue grew to 400 billion with strong services momentum.",
                        "Management warned about supply chain risk in key manufacturing regions.",
                    ],
                }
            ]
            stats = store.index_documents(documents, session_id="sess-1")
            self.assertEqual(stats["chunks_indexed"], 2)

            hits = store.vector_search(
                "Apple supply chain risk",
                session_id="sess-1",
                companies=["Apple"],
                top_k=2,
            )
            self.assertGreaterEqual(len(hits), 1)
            self.assertTrue(any("supply chain" in hit["text"].lower() for hit in hits))
        finally:
            store.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_hybrid_retriever_rrf_prefers_relevant_chunk(self) -> None:
        tmp_dir, uri = _make_temp_milvus_uri()
        embedder = DeterministicEmbeddingProvider()
        store = MilvusRAGStore(uri, embedder, collection_name="rag_hybrid")
        retriever = HybridEvidenceRetriever(store, top_k=3)
        try:
            documents = [
                {
                    "document_id": "apple-q1",
                    "filename": "apple.pdf",
                    "detected_companies": ["Apple"],
                    "pages": [
                        "Apple revenue 400 billion EBITDA 120 billion.",
                        "Supply chain risk remains a key concern for Apple operations.",
                    ],
                }
            ]
            store.index_documents(documents, session_id="sess-2")
            hits = retriever.retrieve_for_company(
                query="Apple supply chain risk assessment",
                company="Apple",
                session_id="sess-2",
                document_contexts=documents,
            )
            self.assertGreaterEqual(len(hits), 1)
            top_text = hits[0]["text"].lower()
            self.assertIn("supply chain", top_text)
        finally:
            store.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_reciprocal_rank_fusion_merges_lists(self) -> None:
        list_a = [{"chunk_id": "a", "text": "a"}]
        list_b = [{"chunk_id": "b", "text": "b"}, {"chunk_id": "a", "text": "a"}]
        merged = reciprocal_rank_fusion([list_a, list_b])
        self.assertEqual(merged[0]["chunk_id"], "a")
        self.assertIn("fusion_score", merged[0])


if __name__ == "__main__":
    unittest.main()
