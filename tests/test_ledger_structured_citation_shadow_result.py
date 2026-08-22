from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout.ledger import ledger_snapshot_sha256
from lumenfin.eval.ledger_structured_citation_shadow import (
    DEFAULT_FROZEN_CONFIG_PATH,
    DEFAULT_OFFICIAL_OUTPUT_DIR,
    DEFAULT_PREFLIGHT_OUTPUT_DIR,
    EVALUATION_MODE,
    GOLD_IDENTITY_SHA256,
    LEGACY_PREFLIGHT_OUTPUT_DIR,
    PROTOCOL_COMMIT,
    SEAL_TAG,
    SEAL_TARGET_COMMIT,
    SUPERSEDED_PREFLIGHT_OUTPUT_DIR,
    NetworkProbe,
    INCOMPLETE_V1_PREFLIGHT_SHA256,
    V2_PREFLIGHT_SHA256,
    load_frozen_config,
    public_dev_snapshot_relative,
    published_config_hash,
    sha256_normalized_file,
    sha256_raw_file,
)

RESULT_PATH = ROOT / "data" / "eval_rag" / "ledger_structured_citation_shadow_result.json"
OFFICIAL_DIR = ROOT / DEFAULT_OFFICIAL_OUTPUT_DIR
EXPECTED_CONFIG_HASH = "54f6e30074fa5ee9806216cb4c0320ba1a5a2e707d155d01fb0cf4b5fe9bac05"
EXPECTED_EXECUTION_COMMIT = "fc77288d39c349b182ce94c0540237ef9d172ec0"
EXPECTED_V3_PREFLIGHT = "b49a3e705b94b01fcf4dbe926d34642db18f8ad39c7adcf62d0f415bea5074eb"


def _load_result() -> dict:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_gate(payload: dict) -> bool:
    evidence = payload["completion_evidence"]
    calls = payload["calls"]
    return (
        evidence["cases_total"] == 50
        and evidence["cases_remaining"] == 0
        and calls["provider_errors"] == 0
    )


def _quality_gate(payload: dict) -> bool:
    citations = payload["citations"]
    claims = payload["claims"]
    return (
        citations["unknown_citations"] == 0
        and citations["citation_validation_failed"] == 0
        and claims["supported_claims"] > 0
        and claims["citation_support_rate"] > 0
    )


class LedgerStructuredCitationShadowResultTests(unittest.TestCase):
    def test_sealed_ledger_identity_and_claims_are_locked(self) -> None:
        payload = _load_result()
        self.assertEqual(payload["schema_version"], "lumenfin_ledger_structured_citation_shadow_result.v1")
        self.assertEqual(payload["kind"], "sealed_result_ledger")
        self.assertEqual(payload["seal_status"], "RECORDED_COMPLETE")
        self.assertEqual(payload["raw_result_status_field"], "MISSING")
        self.assertEqual(payload["evaluation_mode"], EVALUATION_MODE)
        self.assertIs(payload["exposed_public_dev_shadow"], True)
        self.assertIs(payload["held_out"], False)
        self.assertIs(payload["product_accuracy_claim"], False)
        self.assertIs(payload["benchmark_claim"], False)
        self.assertIs(payload["live_retrieval_claim"], False)
        self.assertIs(payload["suitable_for_model_selection"], False)
        self.assertIs(payload["retuning_allowed"], False)
        self.assertIs(payload["retuning_on_public_dev_forbidden"], True)
        self.assertIs(payload["reliable_citation_claim_authorized"], False)
        self.assertIs(payload["production_change_authorized"], False)
        provenance = payload["provenance"]
        self.assertEqual(provenance["execution_commit"], EXPECTED_EXECUTION_COMMIT)
        self.assertEqual(provenance["protocol_ancestor"], PROTOCOL_COMMIT)
        self.assertEqual(provenance["config_hash"], EXPECTED_CONFIG_HASH)
        self.assertEqual(provenance["config_hash"], published_config_hash())
        self.assertEqual(provenance["preflight_sha256"], EXPECTED_V3_PREFLIGHT)
        self.assertEqual(provenance["preflight_schema_version"], "1.1")
        self.assertIs(provenance["preflight_case_binding_verified"], True)
        self.assertEqual(provenance["seal_tag"], SEAL_TAG)
        self.assertEqual(provenance["seal_commit"], SEAL_TARGET_COMMIT)
        self.assertEqual(provenance["case_count"], 50)
        self.assertEqual(provenance["gold_identity_sha256"], GOLD_IDENTITY_SHA256)
        self.assertEqual(provenance["raw_executed_at_field"], "MISSING")
        self.assertTrue(provenance["worktree_clean"])
        evidence = payload["completion_evidence"]
        self.assertEqual(evidence["cli_exit_code"], 0)
        self.assertEqual(evidence["cases_total"], 50)
        self.assertEqual(evidence["cases_remaining"], 0)
        self.assertEqual(evidence["cases_succeeded"], 50)
        self.assertEqual(evidence["cases_failed"], 0)
        self.assertIs(evidence["manifest_checkpoint_per_case_ids_match"], True)
        calls = payload["calls"]
        self.assertEqual(calls["logical_generate_calls"], 50)
        self.assertEqual(calls["recorded_remote_calls"], 50)
        self.assertEqual(calls["recorded_generate_attempts"], 50)
        self.assertEqual(calls["provider_errors"], 0)
        self.assertEqual(calls["embedding_calls"], 0)
        self.assertEqual(calls["runtime_rerank_calls"], 0)
        self.assertEqual(calls["billing_semantics"], "at_least_once")
        self.assertIs(calls["exactly_once"], False)
        self.assertIs(calls["unobserved_inflight_remote_calls_possible"], True)
        citations = payload["citations"]
        self.assertEqual(citations["structured_answer_present"], 22)
        self.assertEqual(citations["structured_emission_rate"], 0.44)
        self.assertEqual(citations["citation_source"]["structured"], 22)
        self.assertEqual(citations["citation_source"]["unavailable"], 28)
        self.assertEqual(citations["answers_with_citations"], 22)
        self.assertEqual(citations["citations_total"], 29)
        self.assertEqual(citations["valid_citations"], 11)
        self.assertEqual(citations["unknown_citations"], 18)
        self.assertEqual(citations["citation_validation_failed"], 11)
        claims = payload["claims"]
        self.assertEqual(claims["claims_total"], 23)
        self.assertEqual(claims["supported_claims"], 0)
        self.assertEqual(claims["unsupported_claims"], 22)
        self.assertEqual(claims["citation_support_rate"], 0.0)
        self.assertEqual(claims["fully_supported_answers"], 0)
        self.assertEqual(claims["partially_supported_answers"], 0)
        self.assertEqual(claims["unsupported_answers"], 22)
        self.assertEqual(claims["unclassified_as_supported_or_unsupported"], 1)
        residual = claims["residual"]
        self.assertEqual(residual["count"], 1)
        self.assertEqual(residual["case_id"], "BWA_capex_2017")
        self.assertEqual(residual["claim_support"], "unavailable")
        self.assertEqual(residual["outcome"], "incomplete_data")
        self.assertEqual(residual["formal_class"], "claim_counted_without_citations")
        self.assertEqual(
            claims["supported_claims"]
            + claims["unsupported_claims"]
            + claims["unclassified_as_supported_or_unsupported"],
            claims["claims_total"],
        )
        outcomes = payload["outcomes"]
        self.assertEqual(sum(outcomes.values()), 50)
        self.assertEqual(outcomes["complete"], 0)
        self.assertEqual(outcomes["incomplete_data"], 28)
        self.assertEqual(outcomes["degraded"], 22)
        self.assertEqual(outcomes["failed"], 0)
        self.assertIs(payload["paired_vs_sealed"]["post_hoc_exposed_comparison"], True)
        self.assertIs(payload["paired_vs_sealed"]["retuning_allowed"], False)

    def test_quality_gate_fails_on_recorded_metrics(self) -> None:
        payload = _load_result()
        self.assertTrue(_execution_gate(payload))
        self.assertFalse(_quality_gate(payload))
        self.assertIs(payload["gates"]["execution_gate_passed"], True)
        self.assertIs(payload["gates"]["structured_citation_quality_gate_passed"], False)
        self.assertEqual(payload["gates"]["execution_gate_passed"], _execution_gate(payload))
        self.assertEqual(
            payload["gates"]["structured_citation_quality_gate_passed"],
            _quality_gate(payload),
        )
        self.assertIs(payload["gates"]["reliable_citation_claim_authorized"], False)
        self.assertIs(payload["gates"]["production_change_authorized"], False)
        self.assertIs(payload["gates"]["retuning_on_public_dev_forbidden"], True)

    def test_ledger_has_no_secrets_paths_queries_or_gold(self) -> None:
        blob = RESULT_PATH.read_text(encoding="utf-8")
        lowered = blob.casefold()
        self.assertNotIn("authorization", lowered)
        self.assertNotIn("bearer ", lowered)
        self.assertNotIn("sk-", blob)
        self.assertNotIn("https://", lowered)
        self.assertNotIn("http://", lowered)
        self.assertNotIn("c:\\", lowered)
        self.assertNotIn("c:/users/", lowered)
        self.assertNotIn("/users/", lowered)
        self.assertNotIn("/home/", lowered)
        self.assertNotIn("gold_value", lowered)
        self.assertNotIn("expected_answer", lowered)
        self.assertNotIn('"query_text"', lowered)
        self.assertNotIn("DEEPSEEK_API_KEY=", blob)
        self.assertIn("query_texts_sha256", lowered)

    def test_official_outputs_remain_gitignored_and_ledger_is_tracked(self) -> None:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(DEFAULT_OFFICIAL_OUTPUT_DIR / "summary.json")],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(ignored.returncode, 0)
        tracked = subprocess.run(
            ["git", "check-ignore", "-q", str(RESULT_PATH.relative_to(ROOT))],
            cwd=ROOT,
            check=False,
        )
        self.assertNotEqual(tracked.returncode, 0)

    def test_official_artifact_hashes_match_when_local_outputs_exist(self) -> None:
        payload = _load_result()
        if not OFFICIAL_DIR.is_dir():
            self.skipTest("official shadow output directory is not in this checkout")
        names = sorted(item.name for item in OFFICIAL_DIR.iterdir() if item.is_file())
        self.assertEqual(names, sorted(payload["official_artifacts"]))
        for name, expected in payload["official_artifacts"].items():
            path = OFFICIAL_DIR / name
            self.assertEqual(path.stat().st_size, expected["size"], name)
            self.assertEqual(_sha256(path), expected["sha256"], name)
        summary = json.loads((OFFICIAL_DIR / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((OFFICIAL_DIR / "manifest.json").read_text(encoding="utf-8"))
        checkpoint = json.loads((OFFICIAL_DIR / "checkpoint.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (OFFICIAL_DIR / "per_case.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertIsNone(summary.get("status"))
        self.assertIsNone(manifest.get("status"))
        self.assertEqual([row["case_id"] for row in rows], manifest["completed_case_ids"])
        self.assertEqual(manifest["completed_case_ids"], checkpoint["completed_case_ids"])
        self.assertEqual(len(rows), 50)
        self.assertEqual(summary["provider_errors"], 0)
        leftover = [
            row["case_id"]
            for row in rows
            if int(row.get("claims_total") or 0)
            and not row.get("supported_claim")
            and not row.get("unsupported_claim")
        ]
        self.assertEqual(leftover, ["BWA_capex_2017"])

    def test_readonly_input_hashes_match_when_local_artifacts_exist(self) -> None:
        payload = _load_result()
        provenance = payload["provenance"]
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        self.assertEqual(config.config_hash, provenance["config_hash"])
        v3 = ROOT / DEFAULT_PREFLIGHT_OUTPUT_DIR / "preflight.json"
        if v3.is_file():
            self.assertEqual(sha256_raw_file(v3), provenance["preflight_sha256"])
        v1 = ROOT / LEGACY_PREFLIGHT_OUTPUT_DIR / "preflight.json"
        if v1.is_file():
            self.assertEqual(sha256_raw_file(v1), INCOMPLETE_V1_PREFLIGHT_SHA256)
        v2 = ROOT / SUPERSEDED_PREFLIGHT_OUTPUT_DIR / "preflight.json"
        if v2.is_file():
            self.assertEqual(sha256_raw_file(v2), V2_PREFLIGHT_SHA256)
        cache = ROOT / "outputs" / "ledger_public_dev_qwen3_paired_5x50_v3" / "candidates.jsonl"
        if cache.is_file():
            self.assertEqual(sha256_raw_file(cache), provenance["cache_file_sha256"])
        manifest = ROOT / "data" / "eval_rag" / "structured_citation_shadow_cache_manifest.json"
        self.assertEqual(sha256_normalized_file(manifest), provenance["cache_manifest_sha256"])
        baseline = ROOT / str(config.field("sealed_baseline", "path"))
        self.assertEqual(sha256_normalized_file(baseline), provenance["sealed_baseline_sha256"])
        snapshot = ROOT / public_dev_snapshot_relative(config)
        if snapshot.exists():
            self.assertEqual(ledger_snapshot_sha256(snapshot), provenance["snapshot_source_artifact_sha256"])

    def test_hashing_official_artifacts_makes_no_remote_calls(self) -> None:
        probe = NetworkProbe()
        probe.install()
        try:
            _load_result()
            if OFFICIAL_DIR.is_dir():
                for path in OFFICIAL_DIR.iterdir():
                    if path.is_file():
                        path.read_bytes()
        finally:
            probe.remove()
        self.assertEqual(probe.remote_request_count, 0)


if __name__ == "__main__":
    unittest.main()
