from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "qwen3_rerank_live_gate_20260812.json"
DATASET = ROOT / "data" / "eval_rag" / "rerank_cases.json"


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        for item in value.values():
            keys.update(_walk_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in value:
            keys.update(_walk_keys(item))
        return keys
    return set()


class Qwen3EvidenceArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    def test_live_gate_metrics_and_telemetry_are_internally_consistent(self) -> None:
        telemetry = self.payload["telemetry"]

        self.assertTrue(self.payload["passed"])
        self.assertTrue(self.payload["telemetry_complete"])
        self.assertEqual(self.payload["fallbacks"], 0)
        self.assertEqual(len(telemetry), 10)
        self.assertEqual(sum(item["attempts"] for item in telemetry), 10)
        self.assertEqual(sum(item["tokens"] for item in telemetry), 3873)
        self.assertTrue(all(item["model"] == "qwen3-rerank" for item in telemetry))
        self.assertTrue(all(not item["fallback"] for item in telemetry))
        self.assertEqual(
            self.payload["summaries"]["qwen3"],
            {
                "cases": 10,
                "top1_accuracy": 1.0,
                "mean_mrr": 1.0,
                "mean_ndcg_at_k": 0.9711,
            },
        )

    def test_dataset_hash_and_traceability_boundary_are_explicit(self) -> None:
        boundary = self.payload["run_boundary"]
        dataset_hash = hashlib.sha256(DATASET.read_bytes()).hexdigest()

        self.assertEqual(boundary["dataset_sha256"], dataset_hash)
        self.assertEqual(boundary["dataset_case_count"], 10)
        self.assertIsNone(boundary["execution_commit"])
        self.assertIn("not capture", boundary["execution_commit_note"])
        self.assertRegex(boundary["source_artifact_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(boundary["implementation_commit"], r"^[0-9a-f]{40}$")

    def test_artifact_contains_no_sensitive_request_fields(self) -> None:
        forbidden_keys = {"query", "text", "endpoint", "base_url", "api_key", "request_id"}
        keys = _walk_keys(self.payload)
        serialized = json.dumps(self.payload, ensure_ascii=False)

        self.assertTrue(forbidden_keys.isdisjoint(keys))
        self.assertNotRegex(serialized, re.compile(r"https?://", re.IGNORECASE))
        self.assertNotRegex(serialized, re.compile(r"\bsk-[A-Za-z0-9_-]+"))
        self.assertFalse(self.payload["privacy_boundary"]["full_request_ids_persisted"])


if __name__ == "__main__":
    unittest.main()
