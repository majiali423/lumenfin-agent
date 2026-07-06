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
from lumenfin.rag.metrics import (
    citation_coverage,
    citation_recall_at_k,
    evaluate_retrieval_case,
    find_relevant_chunks,
    groundedness_score,
    mean_reciprocal_rank,
    recall_at_k,
    report_citation_coverage,
    summarize_eval_results,
)
from lumenfin.rag.milvus_store import MilvusRAGStore


class RagMetricsTestCase(unittest.TestCase):
    def test_recall_mrr_and_citation_metrics(self) -> None:
        retrieved = [
            {"chunk_id": "a", "citation": "doc.pdf#p1", "text": "revenue growth"},
            {"chunk_id": "b", "citation": "doc.pdf#p2", "text": "margin trend"},
            {"chunk_id": "c", "citation": "bad-citation", "text": "noise"},
        ]
        relevant_ids = {"a", "b"}
        self.assertEqual(recall_at_k(retrieved, relevant_ids, k=2), 1.0)
        self.assertEqual(recall_at_k(retrieved, relevant_ids, k=1), 0.5)
        self.assertEqual(mean_reciprocal_rank(retrieved, relevant_ids), 1.0)
        self.assertAlmostEqual(citation_coverage(retrieved), 2 / 3)
        self.assertEqual(
            citation_recall_at_k(retrieved, {"doc.pdf#p1", "doc.pdf#p2"}, k=2),
            1.0,
        )

    def test_groundedness_and_report_citation_coverage(self) -> None:
        hits = [
            {"text": "Apple supply chain risk and revenue trend in FY2025"},
            {"text": "Unrelated operating commentary"},
        ]
        score = groundedness_score(
            hits,
            query="Apple supply chain risk",
            relevant_terms=["supply chain", "revenue"],
        )
        self.assertGreater(score, 0.3)
        report = "Evidence from Apple_eval.pdf#p2 supports supply chain risk."
        coverage = report_citation_coverage(
            report,
            ["Apple_eval.pdf#p1", "Apple_eval.pdf#p2"],
        )
        self.assertEqual(coverage, 0.5)

    def test_evaluate_retrieval_case_with_hybrid_retriever(self) -> None:
        tmp_dir = ROOT / "test_artifacts" / f"rag-metrics-{uuid4().hex[:8]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        uri = str(tmp_dir / "rag.db")
        store = MilvusRAGStore(uri, DeterministicEmbeddingProvider(), collection_name="rag_metrics")
        retriever = HybridEvidenceRetriever(store, top_k=3)
        document = {
            "document_id": "apple-q1",
            "filename": "apple.pdf",
            "detected_companies": ["Apple"],
            "pages": [
                "Apple revenue 400 billion EBITDA 120 billion.",
                "Supply chain risk remains a key concern for Apple operations.",
            ],
        }
        try:
            store.index_documents([document], session_id="sess-metrics")
            hits = retriever.retrieve_for_company(
                query="Apple supply chain risk assessment",
                company="Apple",
                session_id="sess-metrics",
                document_contexts=[document],
            )
            relevant = find_relevant_chunks(document, ["supply chain"])
            result = evaluate_retrieval_case(
                case_id="apple-supply-chain",
                company="Apple",
                query="Apple supply chain risk assessment",
                document=document,
                retrieved=hits,
                relevant_terms=["supply chain"],
                k_values=[1, 3],
            )
            self.assertGreaterEqual(result.relevant_chunk_count, 1)
            self.assertTrue(result.passed)
            self.assertEqual(result.citation_coverage, 1.0)
            self.assertGreater(result.mrr, 0.0)
            summary = summarize_eval_results([result])
            self.assertEqual(summary["cases"], 1)
            self.assertEqual(summary["passed"], 1)
        finally:
            store.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
