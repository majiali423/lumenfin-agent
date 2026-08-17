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
        self._assert_all_chunks_bounded(chunks, 900)
        chunk_types = {chunk["chunk_type"] for chunk in chunks}
        self.assertIn("financial_metric", chunk_types)
        self.assertIn("risk_signal", chunk_types)

    def test_peer_table_splits_per_company_rows(self) -> None:
        document = {
            "document_id": "peer-table",
            "filename": "apple_msft_fy2025_table.pdf",
            "detected_companies": ["Apple", "Microsoft"],
            "pages": [
                "\n".join(
                    [
                        "Consolidated Peer Fundamentals Table",
                        "Metric Apple Microsoft",
                        "Revenue 383.3 245.1",
                        "EBITDA 130.1 128.4",
                        "Operating Income 118.2 109.4",
                        "R&D Expense 31.4 29.5",
                        "Apple supply chain risk remains medium due to assembly concentration.",
                        "Microsoft Azure remains a primary growth engine.",
                    ]
                )
            ],
        }
        chunks = chunk_document(document)
        self._assert_all_chunks_bounded(chunks, 900)
        self.assertGreaterEqual(len(chunks), 8)
        apple_only = [c for c in chunks if c["companies"] == ["Apple"]]
        msft_only = [c for c in chunks if c["companies"] == ["Microsoft"]]
        self.assertTrue(any("Revenue" in c["text"] and "383.3" in c["text"] for c in apple_only))
        self.assertTrue(any("Revenue" in c["text"] and "245.1" in c["text"] for c in msft_only))
        self.assertFalse(any("245.1" in c["text"] for c in apple_only if "Revenue" in c["text"]))

    def _assert_all_chunks_bounded(self, chunks: list[dict], max_chars: int) -> None:
        oversize = [
            (chunk.get("chunk_id"), len(chunk.get("text") or ""))
            for chunk in chunks
            if len(chunk.get("text") or "") > max_chars
        ]
        self.assertEqual(oversize, [], f"chunks exceeded max_chunk_chars={max_chars}: {oversize}")

    def test_narrative_chunks_share_true_character_overlap(self) -> None:
        first = "A" * 200 + " BOUNDARY_TOKEN "
        second = "OPERATING_INCOME_WAS_REPORTED_SEPARATELY " + ("B" * 80)
        chunks = chunk_document(
            {
                "document_id": "overlap-doc",
                "filename": "overlap.pdf",
                "detected_companies": ["Acme"],
                "pages": [f"{first}\n\n{second}"],
            },
            max_chunk_chars=220,
            overlap_chars=40,
        )
        self._assert_all_chunks_bounded(chunks, 220)
        narrative = [
            chunk
            for chunk in chunks
            if "BOUNDARY_TOKEN" in chunk["text"] or "OPERATING_INCOME" in chunk["text"]
        ]
        self.assertGreaterEqual(len(narrative), 2)
        left = narrative[0]["text"]
        right = narrative[1]["text"]
        self.assertIn("BOUNDARY_TOKEN", left)
        self.assertIn("BOUNDARY_TOKEN", right)
        self.assertTrue(right.startswith(left[-40:]))
        self.assertIn("OPERATING_INCOME_WAS_REPORTED_SEPARATELY", right)

    def test_oversize_paragraph_uses_sliding_window_not_hard_split(self) -> None:
        chunks = chunk_document(
            {
                "document_id": "long-para",
                "filename": "long.pdf",
                "detected_companies": ["Acme"],
                "pages": ["X" * 350],
            },
            max_chunk_chars=200,
            overlap_chars=50,
        )
        self._assert_all_chunks_bounded(chunks, 200)
        windows = [chunk["text"] for chunk in chunks if set(chunk["text"]) <= {"X"}]
        self.assertGreaterEqual(len(windows), 2)
        for left, right in zip(windows, windows[1:]):
            self.assertEqual(left[-50:], right[:50])
            self.assertLessEqual(len(left), 200)
            self.assertLessEqual(len(right), 200)

    def test_overlap_at_least_max_chars_is_clamped_and_bounded(self) -> None:
        chunks = chunk_document(
            {
                "document_id": "clamp-overlap",
                "filename": "clamp.pdf",
                "detected_companies": ["Acme"],
                "pages": ["Y" * 180],
            },
            max_chunk_chars=50,
            overlap_chars=50,
        )
        self._assert_all_chunks_bounded(chunks, 50)
        windows = [chunk["text"] for chunk in chunks if set(chunk["text"]) <= {"Y"}]
        self.assertGreaterEqual(len(windows), 2)
        for window in windows:
            self.assertLessEqual(len(window), 50)
        self.assertEqual(windows[0][-49:], windows[1][:49])

    def test_zero_overlap_does_not_strip_chunk_prefixes(self) -> None:
        chunks = chunk_document(
            {
                "document_id": "no-overlap",
                "filename": "plain.pdf",
                "detected_companies": ["Acme"],
                "pages": [
                    "FIRST_UNIQUE_PREFIX " + ("A" * 80) + "\n\nSECOND_UNIQUE_PREFIX " + ("B" * 80)
                ],
            },
            max_chunk_chars=120,
            overlap_chars=0,
        )
        self._assert_all_chunks_bounded(chunks, 120)
        texts = [chunk["text"] for chunk in chunks if "UNIQUE_PREFIX" in chunk["text"]]
        self.assertTrue(any(text.startswith("FIRST_UNIQUE_PREFIX") for text in texts))
        self.assertTrue(any(text.startswith("SECOND_UNIQUE_PREFIX") for text in texts))

    def test_peer_table_rows_are_not_character_overlapped(self) -> None:
        chunks = chunk_document(
            {
                "document_id": "peer-table-no-overlap",
                "filename": "apple_msft_table.pdf",
                "detected_companies": ["Apple", "Microsoft"],
                "pages": [
                    "\n".join(
                        [
                            "Metric Apple Microsoft",
                            "Revenue 383.3 245.1",
                            "EBITDA 130.1 128.4",
                            "Operating Income 118.2 109.4",
                        ]
                    )
                ],
            },
            max_chunk_chars=220,
            overlap_chars=40,
        )
        self._assert_all_chunks_bounded(chunks, 220)
        apple_revenue = next(
            chunk for chunk in chunks if "383.3" in chunk["text"] and "Revenue" in chunk["text"]
        )
        microsoft_revenue = next(
            chunk for chunk in chunks if "245.1" in chunk["text"] and "Revenue" in chunk["text"]
        )
        self.assertFalse(microsoft_revenue["text"].startswith(apple_revenue["text"][-40:]))
        self.assertNotIn("383.3", microsoft_revenue["text"])
        self.assertNotIn("245.1", apple_revenue["text"])

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

    def test_weighted_rrf_breaks_cross_rank_tie_for_bm25(self) -> None:
        dense = [
            {"chunk_id": "noise", "text": "noise"},
            {"chunk_id": "relevant", "text": "relevant"},
        ]
        bm25 = [
            {"chunk_id": "relevant", "text": "relevant"},
            {"chunk_id": "noise", "text": "noise"},
        ]

        merged = reciprocal_rank_fusion([dense, bm25], weights=[1.0, 1.1])

        self.assertEqual(merged[0]["chunk_id"], "relevant")

    def test_rrf_rejects_weight_count_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "weights"):
            reciprocal_rank_fusion([[{"chunk_id": "a"}]], weights=[1.0, 1.1])


if __name__ == "__main__":
    unittest.main()
