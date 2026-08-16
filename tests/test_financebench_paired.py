from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.financebench.paired import (
    compare_paired_systems,
    mcnemar_exact,
    mcnemar_table,
    paired_bootstrap,
)


def _row(case_id: str, hit5: float, hit10: float, mrr: float, ndcg: float, rank: int = 0) -> dict:
    return {
        "case_id": case_id,
        "page": {
            "hit_at": {"5": hit5, "10": hit10},
            "mrr": mrr,
            "ndcg_at": {"10": ndcg},
            "first_relevant_rank": rank,
        },
        "failure_class": "hit" if hit5 else "miss_all",
        "latency_ms": 10.0,
        "rerank_fallback": False,
        "error_type": "",
    }


class FinanceBenchPairedStatsTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_keeps_query_pairs(self) -> None:
        baseline = [0.0, 0.0, 1.0, 1.0]
        candidate = [0.0, 0.0, 1.0, 1.0]
        first = paired_bootstrap(baseline, candidate, n_bootstrap=200, seed=7)
        second = paired_bootstrap(baseline, candidate, n_bootstrap=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["mean_delta"], 0.0)
        self.assertEqual(first["ci95_low"], 0.0)
        self.assertEqual(first["ci95_high"], 0.0)
        crossed = paired_bootstrap(baseline, [1.0, 1.0, 0.0, 0.0], n_bootstrap=200, seed=7)
        self.assertEqual(crossed["mean_delta"], 0.0)
        self.assertLess(crossed["ci95_low"], 0.0)
        self.assertGreater(crossed["ci95_high"], 0.0)

    def test_mcnemar_exact_known_sample(self) -> None:
        result = mcnemar_exact(4, 14)
        self.assertEqual(result["test_name"], "mcnemar_exact")
        self.assertEqual(result["discordant"], 18)
        self.assertAlmostEqual(float(result["p_value"]), 0.030884, places=5)

    def test_dense_qwen3_hit5_counts_are_recomputed(self) -> None:
        dense = (
            [_row(f"both-{i}", 1, 1, 1.0, 1.0, 1) for i in range(33)]
            + [_row(f"dense-{i}", 1, 1, 1.0, 1.0, 1) for i in range(4)]
            + [_row(f"qwen-{i}", 0, 0, 0.0, 0.0, 0) for i in range(14)]
            + [_row(f"none-{i}", 0, 0, 0.0, 0.0, 0) for i in range(49)]
        )
        qwen3 = (
            [_row(f"both-{i}", 1, 1, 1.0, 1.0, 1) for i in range(33)]
            + [_row(f"dense-{i}", 0, 0, 0.0, 0.0, 0) for i in range(4)]
            + [_row(f"qwen-{i}", 1, 1, 1.0, 1.0, 1) for i in range(14)]
            + [_row(f"none-{i}", 0, 0, 0.0, 0.0, 0) for i in range(49)]
        )
        table = mcnemar_table(
            [row["page"]["hit_at"]["5"] for row in dense],
            [row["page"]["hit_at"]["5"] for row in qwen3],
        )
        self.assertEqual(table["both_hit"], 33)
        self.assertEqual(table["baseline_only"], 4)
        self.assertEqual(table["candidate_only"], 14)
        self.assertEqual(table["neither_hit"], 49)
        self.assertEqual(table["n"], 100)
        comparison = compare_paired_systems(
            dense,
            qwen3,
            baseline_name="dense",
            candidate_name="hybrid-qwen3",
        )
        self.assertEqual(comparison["mcnemar"]["hit_at_5"]["both_hit"], 33)
        self.assertEqual(comparison["mcnemar"]["hit_at_5"]["baseline_only"], 4)
        self.assertEqual(comparison["mcnemar"]["hit_at_5"]["candidate_only"], 14)
        self.assertEqual(comparison["mcnemar"]["hit_at_5"]["neither_hit"], 49)
        self.assertIn("paired_bootstrap", comparison)
        self.assertIn("delta_hit_at_5", comparison["paired_bootstrap"])
        self.assertIn("delta_mrr", comparison["paired_bootstrap"])
        self.assertIn("delta_ndcg_at_10", comparison["paired_bootstrap"])


if __name__ == "__main__":
    unittest.main()
