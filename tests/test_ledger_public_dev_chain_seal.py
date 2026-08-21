from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEAL_PATH = ROOT / "data" / "eval_rag" / "holdout" / "ledger_public_dev_chain_seal.json"
FORBIDDEN = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|DASHSCOPE_API_KEY|DEEPSEEK_API_KEY|dashscope\.aliyuncs|"
    r"api\.deepseek\.com|query_text|gold_answer|page_text|mmd_text|"
    r"BEGIN PRIVATE KEY)",
    re.IGNORECASE,
)
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _remote_calls(obj: dict) -> int:
    if "remote_calls" in obj:
        return int(obj["remote_calls"])
    accounting = obj.get("call_accounting") or {}
    if "remote_calls" in accounting:
        return int(accounting["remote_calls"])
    return int(obj.get("qwen3_calls") or 0) + int(obj.get("generate_calls") or 0)


def _query_count(obj: dict) -> int:
    if isinstance(obj.get("cases"), int):
        return obj["cases"]
    selection = obj.get("selection") or {}
    if isinstance(selection.get("cases"), int):
        return int(selection["cases"])
    splits = obj.get("splits") or {}
    public_dev = splits.get("public_dev") or {}
    if isinstance(public_dev.get("queries"), int):
        return int(public_dev["queries"])
    return 0


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


class LedgerPublicDevChainSealTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))

    def test_governance_flags_forbid_holdout_retune_and_production_switch(self) -> None:
        self.assertEqual(self.seal["schema_version"], "lumenfin_ledger_public_dev_chain_seal.v1")
        self.assertEqual(self.seal["split"], "public_dev")
        self.assertEqual(self.seal["public_dev_status"], "sealed_stopped")
        self.assertIs(self.seal["public_holdout_consumed"], False)
        self.assertIs(self.seal["product_accuracy_claim"], False)
        self.assertIs(self.seal["production_change_authorized"], False)
        self.assertEqual(self.seal["financebench_phase4"], "NOT_RUN")
        self.assertIs(self.seal["retuning_on_public_dev_forbidden"], True)
        self.assertIs(
            self.seal["public_holdout_scoring_requires_new_explicit_approval"], True
        )
        self.assertIs(self.seal["do_not_embed_page_parent_index"], True)
        self.assertEqual(self.seal["production_rag_arm"], "A")
        self.assertNotIn("sha256", self.seal)
        self.assertNotIn("self_sha256", self.seal)

    def test_listed_artifacts_exist_and_match_sha256(self) -> None:
        artifacts = self.seal["artifacts"]
        self.assertGreaterEqual(len(artifacts), 11)
        paths = [item["path"] for item in artifacts]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertNotIn("data/eval_rag/holdout/ledger_public_dev_chain_seal.json", paths)
        for item in artifacts:
            path = ROOT / item["path"]
            self.assertTrue(path.is_file(), item["path"])
            payload = json.loads(path.read_bytes().replace(b"\r\n", b"\n"))
            self.assertEqual(item["sha256"], _sha256_file(path), item["path"])
            self.assertEqual(item["schema_version"], payload.get("schema_version"))
            self.assertEqual(item["query_count"], _query_count(payload), item["path"])
            self.assertEqual(item["remote_calls"], _remote_calls(payload), item["path"])
            self.assertTrue(item["result_role"])

        manifest_path = ROOT / self.seal["split_manifest_path"]
        self.assertEqual(self.seal["split_manifest_sha256"], _sha256_file(manifest_path))
        manifest = json.loads(manifest_path.read_bytes().replace(b"\r\n", b"\n"))
        self.assertEqual(
            self.seal["dataset_revision"], manifest["dataset"]["source_revision"]
        )
        self.assertEqual(
            self.seal["dataset_snapshot_sha256"], manifest["dataset_snapshot_sha256"]
        )
        self.assertIs(manifest.get("scoring_enabled"), False)

    def test_seal_contains_no_questions_answers_page_text_or_secrets(self) -> None:
        blob = SEAL_PATH.read_text(encoding="utf-8")
        self.assertIsNone(FORBIDDEN.search(blob))
        for item in self.seal["artifacts"]:
            self.assertNotIn("question", json.dumps(item).lower())

    def test_recommended_tag_points_at_existing_data_seal_commit(self) -> None:
        self.assertEqual(self.seal["recommended_annotated_tag"], "ledger-public-dev-chain-v1")
        target = self.seal["recommended_tag_target_commit"]
        data_seal = self.seal["data_seal_commit"]
        readme = self.seal["readme_followup_commit"]
        self.assertRegex(target, COMMIT_RE)
        self.assertEqual(target, data_seal)
        self.assertNotEqual(target, readme)
        self.assertEqual(_git("cat-file", "-t", target), "commit")
        self.assertEqual(_git("merge-base", "--is-ancestor", target, "HEAD"), "")
        self.assertEqual(_git("rev-parse", target), target)
        self.assertEqual(_git("rev-parse", readme), readme)
        self.assertIn("README-only", self.seal["recommended_tag_target_reason"])


if __name__ == "__main__":
    unittest.main()
