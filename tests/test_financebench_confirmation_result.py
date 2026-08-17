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

RESULT_PATH = ROOT / "data" / "eval_rag" / "financebench" / "confirmation_result.json"
PUBLISHED_COMMIT = "379a8b053256fd43260ecf031cdf675af7c3be4b"
REQUIRED_ARTIFACTS = (
    "environment.json",
    "manifest.json",
    "results.json",
    "results.md",
    "per_case.jsonl",
    "failures.jsonl",
)


class FinanceBenchConfirmationResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_aggregate_is_recorded_and_not_a_product_claim(self) -> None:
        self.assertEqual(self.payload["status"], "RECORDED")
        self.assertIs(self.payload["product_accuracy_claim"], False)
        self.assertIs(self.payload["end_to_end_accuracy_claim"], False)
        self.assertIs(self.payload["confirmation_consumed"], True)
        self.assertIs(self.payload["retuning_forbidden"], True)
        self.assertEqual(self.payload["failures"]["never_retrieved_across_modes"], "NOT_APPLICABLE")
        self.assertNotIn("never_retrieved", self.payload["failures"])

    def test_published_hashes_and_counts(self) -> None:
        self.assertEqual(self.payload["lumenfin_commit"], PUBLISHED_COMMIT)
        self.assertEqual(self.payload["tag"], "financebench-confirmation-v1")
        self.assertEqual(self.payload["frozen_config_hash"], PUBLISHED_CONFIG_HASH)
        self.assertTrue(self.payload["frozen_config_verified"])
        self.assertEqual(self.payload["cases"], 50)
        self.assertEqual(self.payload["page"]["hit_at_5"], 0.5)
        self.assertEqual(self.payload["page"]["hit_at_10"], 0.62)
        self.assertEqual(self.payload["page"]["mrr"], 0.2955)
        self.assertEqual(self.payload["page"]["ndcg_at_10"], 0.3461)
        self.assertEqual(self.payload["failures"]["hit_at_5"], "25/50")
        self.assertEqual(self.payload["failures"]["hit_at_10"], "31/50")
        self.assertEqual(self.payload["failures"]["top10_missed"], "19/50")
        self.assertEqual(self.payload["failures"]["miss_all"], 16)
        self.assertEqual(self.payload["failures"]["wrong_document"], 2)
        self.assertEqual(self.payload["failures"]["ingestion_failure"], 1)
        self.assertNotIn("hit_at_10_ci95", self.payload["page"])

    def test_artifact_digests_are_sha256_hex_and_do_not_embed_raw_eval(self) -> None:
        artifacts = self.payload["artifact_sha256"]
        self.assertEqual(tuple(artifacts), REQUIRED_ARTIFACTS)
        for digest in artifacts.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        serialized = json.dumps(self.payload)
        self.assertNotIn("financebench_id_03029", serialized)
        self.assertNotIn("evidence_text", serialized)


if __name__ == "__main__":
    unittest.main()
