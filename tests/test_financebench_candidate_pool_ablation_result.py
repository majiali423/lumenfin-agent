from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.financebench.frozen import PUBLISHED_CONFIG_HASH

RESULT_PATH = ROOT / "data" / "eval_rag" / "financebench" / "candidate_pool_ablation_result.json"
PUBLISHED_COMMIT = "4bbac4e6c7695b75a051ce6a51c6d40a1309153d"
PUBLISHED_CONFIG_HASH_ABLATION = (
    "d0370c073be46fc42c2c6dded458182e3332e644666388b7111265d9b40e7bbd"
)
REQUIRED_ARTIFACTS = (
    "environment.json",
    "manifest.json",
    "checkpoint.json",
    "summary.json",
    "paired.json",
    "results.md",
    "per_case.jsonl",
    "failures.jsonl",
)


class FinanceBenchCandidatePoolAblationResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_seal_flags_forbid_production_change_and_test100_retune(self) -> None:
        self.assertEqual(self.payload["status"], "RECORDED")
        self.assertIs(self.payload["exposed_test_post_hoc"], True)
        self.assertIs(self.payload["held_out"], False)
        self.assertIs(self.payload["product_accuracy_claim"], False)
        self.assertIs(self.payload["end_to_end_accuracy_claim"], False)
        self.assertIs(self.payload["production_change_authorized"], False)
        self.assertIs(self.payload["retuning_on_test100_forbidden"], True)
        self.assertEqual(self.payload["production_arm"], "A")
        self.assertEqual(self.payload["next_generation_candidate"]["arm"], "C")
        self.assertEqual(
            self.payload["next_generation_candidate"]["status"],
            "diagnostic_best_not_production_authorized",
        )
        self.assertTrue(self.payload["decision"]["keep_production_A"])
        self.assertFalse(self.payload["decision"]["adopt_B"])
        self.assertFalse(self.payload["decision"]["authorize_C_in_production"])

    def test_published_hashes_and_arm_metrics(self) -> None:
        self.assertEqual(self.payload["lumenfin_commit"], PUBLISHED_COMMIT)
        self.assertEqual(self.payload["tag"], "financebench-candidate-pool-ablation-v1")
        self.assertEqual(self.payload["config_hash"], PUBLISHED_CONFIG_HASH_ABLATION)
        self.assertNotEqual(self.payload["config_hash"], PUBLISHED_CONFIG_HASH)
        self.assertEqual(self.payload["cases"], 100)
        self.assertTrue(self.payload["primary_comparison_valid"])
        self.assertEqual(self.payload["arms"]["A"]["hit_at_5"], 0.48)
        self.assertEqual(self.payload["arms"]["A"]["hit_at_10"], 0.65)
        self.assertEqual(self.payload["arms"]["B"]["hit_at_10"], 0.63)
        self.assertEqual(self.payload["arms"]["C"]["hit_at_5"], 0.53)
        self.assertEqual(self.payload["arms"]["C"]["hit_at_10"], 0.67)
        self.assertEqual(self.payload["rank_11_30"]["a_miss_b_hit"], 0)
        self.assertEqual(self.payload["rank_11_30"]["a_miss_c_hit"], 5)
        self.assertFalse(self.payload["paired"]["C_vs_A"]["significant_overall_vs_A"])
        self.assertEqual(self.payload["arms"]["C"]["failure_class_counts"]["gold_in_pool_not_in_final_top10"], 11)

    def test_artifact_digests_are_sha256_hex_and_do_not_embed_raw_eval(self) -> None:
        artifacts = self.payload["artifact_sha256"]
        self.assertEqual(tuple(artifacts), REQUIRED_ARTIFACTS)
        for digest in artifacts.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        serialized = json.dumps(self.payload)
        self.assertNotIn("financebench_id_", serialized)
        self.assertNotIn("evidence_text", serialized)
        self.assertNotIn("DASHSCOPE", serialized)


if __name__ == "__main__":
    unittest.main()
