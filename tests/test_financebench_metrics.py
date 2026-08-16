from __future__ import annotations

import unittest

from lumenfin.eval.financebench.metrics import (
    bootstrap_mean_ci,
    hit_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from lumenfin.eval.financebench.scoring import score_retrieval_case
from lumenfin.eval.financebench.qrels import map_chunks_to_qrels
from tests.financebench_fixtures import parsed_question


class FinanceBenchMetricsTestCase(unittest.TestCase):
    def test_hit_recall_mrr_ndcg(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = {"c"}
        self.assertEqual(hit_at_k(retrieved, relevant, k=1), 0.0)
        self.assertEqual(hit_at_k(retrieved, relevant, k=3), 1.0)
        self.assertEqual(recall_at_k(retrieved, relevant, k=3), 1.0)
        self.assertAlmostEqual(mean_reciprocal_rank(retrieved, relevant), 1.0 / 3.0)
        self.assertGreater(ndcg_at_k(retrieved, relevant, k=3), 0.0)
        self.assertEqual(hit_at_k(retrieved, set(), k=3), 0.0)

    def test_bootstrap_is_deterministic(self) -> None:
        values = [1.0, 0.0, 1.0, 0.0, 1.0]
        first = bootstrap_mean_ci(values, n_bootstrap=200, seed=20260816)
        second = bootstrap_mean_ci(values, n_bootstrap=200, seed=20260816)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "computed")
        self.assertLessEqual(float(first["ci95_low"]), float(first["mean"]))
        self.assertGreaterEqual(float(first["ci95_high"]), float(first["mean"]))

    def test_score_retrieval_case_page_and_chunk(self) -> None:
        question = parsed_question(
            financebench_id="financebench_id_00001",
            company="3M",
            doc_name="3M_2018_10K",
            question="capex?",
            page_zero=59,
            evidence_text="Purchases of property, plant and equipment (PP&E) (1,577)",
        )
        chunks = [
            {
                "chunk_id": "3M_2018_10K:p60:c0",
                "document_id": "3M_2018_10K",
                "filename": "3M_2018_10K.pdf",
                "page": 60,
                "text": "Purchases of property, plant and equipment (PP&E) (1,577)",
            }
        ]
        qrels = map_chunks_to_qrels(question, chunks)
        hits = [
            {
                "chunk_id": "other:p1:c0",
                "document_id": "other",
                "filename": "other.pdf",
                "page": 1,
                "text": "no",
            },
            chunks[0],
        ]
        row = score_retrieval_case(
            question=question, qrels=qrels, hits=hits, mode="bm25", top_k=10
        )
        self.assertEqual(row["page"]["hit_at"]["1"], 0.0)
        self.assertEqual(row["page"]["hit_at"]["5"], 1.0)
        self.assertEqual(row["page"]["first_relevant_rank"], 2)
        self.assertNotIn("text", row["retrieved"][0])
        self.assertEqual(row["failure_class"], "rank_gt_1")


if __name__ == "__main__":
    unittest.main()
