from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.ledger_public_holdout_e2e import wilson_interval
from lumenfin.eval.ledger_public_holdout_e2e_v2 import (
    AUTHORIZATION_V2_PATH,
    CONTRACT_V2_PATH,
    EXPECTED_ABSTAIN,
    EXPECTED_ANSWER_CORRECT,
    EXPECTED_CASES_TOTAL,
    EXPECTED_CITATION_SUPPORTED,
    EXPECTED_CONFIG_HASH,
    EXPECTED_INDEX_HASH,
    EXPECTED_OFFICIAL_RAW_ARTIFACTS,
    EXPECTED_PROVIDER_SUCCESS,
    EXPECTED_RATE,
    EXPECTED_SELECTED_QUERY_IDS_SHA256,
    EXPECTED_SELECTION_FILE_SHA256,
    EXPECTED_STRUCTURED_CONTRACT_VALID,
    EXPECTED_SUCCESSES,
    EXPECTED_WILSON_DISPLAY,
    PRODUCT_COMMIT,
    PRODUCT_TAG,
    RESULT_KIND,
    RESULT_SCHEMA_VERSION,
    RESULT_V2_PATH,
    SELECTION_PATH,
    SHA256_HEX_RE,
    ids_sha256,
    inspect_official_raw_artifacts,
    load_authorization_v2,
    load_contract_v2,
    load_result_v2,
    load_selection,
    reverify_raw_artifacts_against_seal,
    scan_text_for_leaks,
    sha256_text,
    tally_official_per_case,
    wilson_display,
)

HARNESS_COMMIT = "4cb7a5ade811c9e481742c59f703c9e006199326"
TRACKED_V2 = (
    SELECTION_PATH,
    CONTRACT_V2_PATH,
    AUTHORIZATION_V2_PATH,
    RESULT_V2_PATH,
)


class HoldoutE2EV2ResultSealRequiredTests(unittest.TestCase):
    """CI-required layer. Must run even when gitignored raw outputs are absent."""

    def test_tracked_result_schema_identity_and_counts_without_raw_outputs(self) -> None:
        result = load_result_v2(repo_root=ROOT)
        self.assertEqual(result["result_kind"], RESULT_KIND)
        self.assertEqual(result["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(result["seal_status"], "SEALED")
        self.assertIs(result["dataset_specific"], True)
        self.assertIs(result["held_out"], True)
        self.assertIs(result["single_use"], True)
        self.assertIs(result["general_product_accuracy_claim"], False)
        self.assertEqual(result["product_tag"], PRODUCT_TAG)
        self.assertEqual(result["product_commit"], PRODUCT_COMMIT)
        self.assertEqual(result["evaluation_harness_commit"], HARNESS_COMMIT)
        self.assertEqual(result["execution_commit"], PRODUCT_COMMIT)
        self.assertEqual(result["config_hash"], EXPECTED_CONFIG_HASH)
        self.assertEqual(result["index_hash"], EXPECTED_INDEX_HASH)
        self.assertEqual(result["selection_hash"], EXPECTED_SELECTION_FILE_SHA256)
        self.assertEqual(result["cases_total"], EXPECTED_CASES_TOTAL)
        self.assertEqual(result["successes"], EXPECTED_SUCCESSES)
        self.assertEqual(result["rate"], EXPECTED_RATE)
        self.assertEqual(result["wilson_95"], list(EXPECTED_WILSON_DISPLAY))
        self.assertEqual(result["answer_correct"], EXPECTED_ANSWER_CORRECT)
        self.assertEqual(result["structured_contract_valid"], EXPECTED_STRUCTURED_CONTRACT_VALID)
        self.assertEqual(result["citation_supported"], EXPECTED_CITATION_SUPPORTED)
        self.assertEqual(result["provider_success"], EXPECTED_PROVIDER_SUCCESS)
        self.assertEqual(result["abstain"], EXPECTED_ABSTAIN)
        self.assertIs(result["holdout_consumed"], True)
        self.assertIs(result["dataset_consumed"], True)
        self.assertIs(result["retuning_forbidden"], True)
        self.assertIs(result["second_fresh_run_forbidden"], True)
        self.assertEqual(result["raw_status_field"], "MISSING")
        self.assertEqual(result["raw_executed_at_field"], "MISSING")
        self.assertIs(result["inferred_mtime"]["authoritative"], False)
        self.assertNotIn("executed_at", result)

        selection = load_selection(repo_root=ROOT, allow_consumed=True)
        contract = load_contract_v2(repo_root=ROOT)
        auth = load_authorization_v2(repo_root=ROOT)
        self.assertEqual(selection["selected_query_ids_sha256"], EXPECTED_SELECTED_QUERY_IDS_SHA256)
        self.assertEqual(sha256_text((ROOT / SELECTION_PATH).read_text(encoding="utf-8")), result["selection_hash"])
        self.assertEqual(contract["config_hash"], result["config_hash"])
        record = auth["records"][contract["config_hash"]]
        self.assertEqual(record["identity_status"], "CONSUMED")
        self.assertIs(record["holdout_consumed"], True)
        self.assertIs(auth["default_deny"], True)

        stored = result["strict_verified_e2e_success"]
        recomputed = wilson_interval(EXPECTED_SUCCESSES, EXPECTED_CASES_TOTAL)
        self.assertEqual(stored["n"], EXPECTED_CASES_TOTAL)
        self.assertEqual(stored["successes"], EXPECTED_SUCCESSES)
        self.assertEqual(stored["rate"], EXPECTED_RATE)
        self.assertAlmostEqual(stored["low"], recomputed["low"])
        self.assertAlmostEqual(stored["high"], recomputed["high"])
        self.assertEqual(wilson_display(stored["low"], stored["high"]), list(EXPECTED_WILSON_DISPLAY))

        artifacts = result["official_raw_artifacts"]
        self.assertEqual(len(artifacts), 2)
        for item, expected in zip(artifacts, EXPECTED_OFFICIAL_RAW_ARTIFACTS, strict=True):
            self.assertEqual(item["relative_path"], expected["relative_path"])
            self.assertEqual(item["size_bytes"], expected["size_bytes"])
            self.assertRegex(item["sha256"], SHA256_HEX_RE.pattern)
            self.assertEqual(item["sha256"], expected["sha256"])

        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "outputs/ledger_public_holdout_e2e_v2/aggregate.json"],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(ignored.returncode, 0)

        inspection = inspect_official_raw_artifacts(repo_root=ROOT)
        if inspection["raw_artifacts_status"] == "NOT_PRESENT":
            self.assertIs(inspection["raw_bytes_reverified"], False)
            self.assertEqual(inspection["raw_artifacts_status"], "NOT_PRESENT")
        else:
            self.assertEqual(inspection["raw_artifacts_status"], "PRESENT")

    def test_tracked_v2_files_have_no_secret_or_holdout_body_leak(self) -> None:
        for rel in TRACKED_V2:
            text = (ROOT / rel).read_text(encoding="utf-8")
            leaks = scan_text_for_leaks(text)
            self.assertEqual(leaks, [], msg=str(rel))
            self.assertNotRegex(text, r"[A-Za-z]:\\Users\\")
            self.assertNotIn("sk-", text)

    def test_result_file_is_tracked(self) -> None:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", RESULT_V2_PATH.as_posix()],
            cwd=ROOT,
            check=False,
        )
        self.assertNotEqual(ignored.returncode, 0)


class HoldoutE2EV2ResultSealOptionalTests(unittest.TestCase):
    """Local-only byte reverification when gitignored raw outputs exist."""

    def test_optional_raw_reverification_or_explicit_not_present(self) -> None:
        result = load_result_v2(repo_root=ROOT)
        report = reverify_raw_artifacts_against_seal(repo_root=ROOT, result=result)
        if report["raw_artifacts_status"] == "NOT_PRESENT":
            self.assertIs(report["raw_bytes_reverified"], False)
            return
        self.assertEqual(report["raw_artifacts_status"], "PRESENT")
        self.assertIs(report["raw_bytes_reverified"], True)
        self.assertEqual(report.get("mismatches") or [], [])
        self.assertEqual(report.get("extra_files") or [], [])

        tally = tally_official_per_case(repo_root=ROOT)
        selection = load_selection(repo_root=ROOT, allow_consumed=True)
        self.assertEqual(tally["rows"], EXPECTED_CASES_TOTAL)
        self.assertEqual(tally["unique_query_ids"], EXPECTED_CASES_TOTAL)
        self.assertEqual(tally["remaining"], 0)
        self.assertEqual(tally["verified_e2e_success"], EXPECTED_SUCCESSES)
        self.assertEqual(tally["answer_correct"], EXPECTED_ANSWER_CORRECT)
        self.assertEqual(tally["structured_contract_valid"], EXPECTED_STRUCTURED_CONTRACT_VALID)
        self.assertEqual(tally["citation_supported"], EXPECTED_CITATION_SUPPORTED)
        self.assertEqual(tally["provider_success"], EXPECTED_PROVIDER_SUCCESS)
        self.assertEqual(tally["abstain"], EXPECTED_ABSTAIN)
        self.assertEqual(tally["provider_errors"], 0)
        self.assertEqual(set(tally["query_ids"]), set(selection["selected_query_ids"]))
        self.assertEqual(ids_sha256(tally["query_ids"]), EXPECTED_SELECTED_QUERY_IDS_SHA256)

        aggregate = json.loads(
            (ROOT / "outputs" / "ledger_public_holdout_e2e_v2" / "aggregate.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("status", aggregate)
        self.assertNotIn("executed_at", aggregate)
        self.assertEqual(aggregate["cases"], EXPECTED_CASES_TOTAL)
        self.assertEqual(result["raw_status_field"], "MISSING")

        raw_text = (
            (ROOT / "outputs" / "ledger_public_holdout_e2e_v2" / "aggregate.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(scan_text_for_leaks(raw_text), [])
        self.assertNotRegex(raw_text, r"[A-Za-z]:\\Users\\")


if __name__ == "__main__":
    unittest.main()
