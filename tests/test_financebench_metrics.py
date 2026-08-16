from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.financebench.metrics import (
    bootstrap_mean_ci,
    hit_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    single_gold_recall_equals_hit,
)


class FinanceBenchMetricsTests(unittest.TestCase):
    def test_hit_recall_mrr_ndcg_and_empty_results(self) -> None:
        retrieved = ["p2", "p9", "p1"]
        relevant = {"p1", "p2"}
        self.assertEqual(hit_at_k(retrieved, relevant, k=1), 1.0)
        self.assertEqual(recall_at_k(retrieved, relevant, k=1), 0.5)
        self.assertEqual(mean_reciprocal_rank(retrieved, relevant), 1.0)
        self.assertGreater(ndcg_at_k(retrieved, relevant, k=5), 0.0)
        self.assertEqual(hit_at_k([], relevant, k=5), 0.0)
        self.assertEqual(recall_at_k([], relevant, k=5), 0.0)
        self.assertEqual(mean_reciprocal_rank([], relevant), 0.0)
        self.assertEqual(ndcg_at_k([], relevant, k=10), 0.0)

    def test_single_gold_page_recall_equals_hit(self) -> None:
        retrieved = ["noise", "gold"]
        relevant = {"gold"}
        self.assertTrue(single_gold_recall_equals_hit(len(relevant)))
        for k in (1, 3, 5, 10):
            self.assertEqual(
                hit_at_k(retrieved, relevant, k=k),
                recall_at_k(retrieved, relevant, k=k),
            )

    def test_multi_gold_recall_can_differ_from_hit(self) -> None:
        retrieved = ["a"]
        relevant = {"a", "b"}
        self.assertEqual(hit_at_k(retrieved, relevant, k=5), 1.0)
        self.assertEqual(recall_at_k(retrieved, relevant, k=5), 0.5)
        self.assertFalse(single_gold_recall_equals_hit(len(relevant)))

    def test_bootstrap_ci_is_deterministic(self) -> None:
        values = [1.0, 0.0, 1.0, 0.5]
        first = bootstrap_mean_ci(values, n_bootstrap=200, seed=7)
        second = bootstrap_mean_ci(values, n_bootstrap=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "computed")
        self.assertLessEqual(first["ci95_low"], first["mean"])
        self.assertGreaterEqual(first["ci95_high"], first["mean"])


if __name__ == "__main__":
    unittest.main()
