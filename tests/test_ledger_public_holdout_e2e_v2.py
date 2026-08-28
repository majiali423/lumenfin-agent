from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.ledger_public_holdout_e2e_v2 import (
    AUTHORIZATION_V2_PATH,
    CLAIM,
    CONTRACT_V2_PATH,
    EXPECTED_CASES_TOTAL,
    EXPECTED_CONFIG_HASH,
    EXPECTED_INDEX_HASH,
    EXPECTED_SELECTED_QUERY_IDS_SHA256,
    EXPECTED_SELECTION_FILE_SHA256,
    FORBIDDEN_CLI,
    PRODUCT_COMMIT,
    PRODUCT_TAG,
    SELECTION_PATH,
    HoldoutE2EV2Error,
    compute_config_hash,
    holdout_consumed_v2,
    inspect_official_raw_artifacts,
    load_authorization_v2,
    load_contract_v2,
    load_selection,
    refuse_unauthorized_v2,
    reverify_raw_artifacts_against_seal,
    sha256_text,
    write_contract_and_auth,
)

CLI = ROOT / "scripts" / "run_ledger_public_holdout_e2e_v2.py"
FREEZE_CLI = ROOT / "scripts" / "freeze_ledger_public_holdout_e2e_selection_v2.py"


def _copy_v2_ledgers(dest_root: Path, *, include_selection: bool = True) -> None:
    folder = dest_root / "data" / "eval_rag"
    folder.mkdir(parents=True)
    names = [
        CONTRACT_V2_PATH.name,
        AUTHORIZATION_V2_PATH.name,
    ]
    if include_selection:
        names.append(SELECTION_PATH.name)
    for name in names:
        shutil.copy(ROOT / "data" / "eval_rag" / name, folder / name)


class HoldoutE2EV2HarnessTests(unittest.TestCase):
    def test_selection_contract_auth_are_locked(self) -> None:
        selection = load_selection(repo_root=ROOT, allow_consumed=True)
        self.assertEqual(selection["selected_cases"], EXPECTED_CASES_TOTAL)
        self.assertEqual(selection["selected_companies"], 26)
        self.assertEqual(selection["product_commit"], PRODUCT_COMMIT)
        self.assertEqual(selection["product_tag"], PRODUCT_TAG)
        self.assertEqual(
            selection["selected_query_ids_sha256"],
            EXPECTED_SELECTED_QUERY_IDS_SHA256,
        )
        self.assertIs(selection["holdout_consumed"], True)
        self.assertIs(selection["holdout_query_text_accessed"], False)
        self.assertIs(selection["holdout_gold_accessed"], False)
        self.assertIs(selection["holdout_qrels_accessed"], False)
        selection_hash = sha256_text((ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        self.assertEqual(selection_hash, EXPECTED_SELECTION_FILE_SHA256)

        contract = load_contract_v2(repo_root=ROOT)
        self.assertEqual(contract["claim"], CLAIM)
        self.assertEqual(contract["product"]["product_commit"], PRODUCT_COMMIT)
        self.assertEqual(contract["product"]["release_tag"], PRODUCT_TAG)
        self.assertIs(contract["dataset_specific"], True)
        self.assertIs(contract["held_out"], True)
        self.assertIs(contract["single_use"], True)
        self.assertIs(contract["general_product_accuracy_claim"], False)
        self.assertIs(contract["index_gate"]["compatible_prebuilt_index_present"], True)
        self.assertEqual(contract["call_budget"]["document_reembedding_calls"], 0)
        self.assertEqual(contract["config_hash"], EXPECTED_CONFIG_HASH)
        self.assertEqual(contract["retrieval"]["index_milvus_db_sha256"], EXPECTED_INDEX_HASH)
        self.assertEqual(compute_config_hash(contract), contract["config_hash"])

        auth = load_authorization_v2(repo_root=ROOT)
        self.assertIs(auth["default_deny"], True)
        record = auth["records"][contract["config_hash"]]
        self.assertEqual(record["identity_status"], "CONSUMED")
        self.assertIs(record["holdout_consumed"], True)
        self.assertIs(record["dataset_consumed"], True)
        self.assertIs(record["execution_authorized"], False)
        self.assertIs(record["remote_run_authorized"], False)
        self.assertEqual(record["official_remote_executions"], 1)
        self.assertTrue(holdout_consumed_v2(repo_root=ROOT, contract=contract))

        for rel in (SELECTION_PATH, CONTRACT_V2_PATH, AUTHORIZATION_V2_PATH):
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", rel.as_posix()],
                cwd=ROOT,
                check=False,
            )
            self.assertNotEqual(ignored.returncode, 0, msg=rel.as_posix())

    def test_consumed_deny_happens_before_index_or_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _copy_v2_ledgers(repo)
            with self.assertRaisesRegex(HoldoutE2EV2Error, "already consumed"):
                refuse_unauthorized_v2(
                    repo_root=repo,
                    argv=[
                        "--confirm-public-holdout-consumption",
                        "--allow-remote",
                    ],
                )
            with self.assertRaisesRegex(HoldoutE2EV2Error, "already consumed"):
                refuse_unauthorized_v2(
                    repo_root=repo,
                    argv=[
                        "--confirm-public-holdout-consumption",
                        "--allow-remote",
                        "--resume",
                    ],
                )
            with self.assertRaisesRegex(HoldoutE2EV2Error, "already consumed"):
                refuse_unauthorized_v2(repo_root=repo, argv=["--preflight-only"])
            with self.assertRaisesRegex(HoldoutE2EV2Error, "already consumed"):
                write_contract_and_auth(repo_root=repo)
            self.assertFalse((repo / "outputs").exists())

    def test_forbidden_cli_and_env_cannot_bypass(self) -> None:
        with self.assertRaisesRegex(HoldoutE2EV2Error, "runtime overrides"):
            refuse_unauthorized_v2(
                repo_root=ROOT,
                argv=["--split", "public_dev", "--confirm-public-holdout-consumption"],
            )
        for flag in ("--output-dir", "--model", "--dataset-path", "--cases-path"):
            self.assertIn(flag, FORBIDDEN_CLI)
            with self.assertRaisesRegex(HoldoutE2EV2Error, "runtime overrides"):
                refuse_unauthorized_v2(repo_root=ROOT, argv=[flag, "hijack"])
        with patch.dict(os.environ, {"LEDGER_PUBLIC_HOLDOUT_E2E_FORCE": "1"}):
            with self.assertRaisesRegex(HoldoutE2EV2Error, "environment cannot authorize"):
                refuse_unauthorized_v2(
                    repo_root=ROOT,
                    argv=[
                        "--confirm-public-holdout-consumption",
                        "--allow-remote",
                    ],
                )

    def test_official_cli_default_deny_does_not_touch_raw_outputs(self) -> None:
        official = ROOT / "outputs" / "ledger_public_holdout_e2e_v2"
        before = inspect_official_raw_artifacts(repo_root=ROOT)
        commands = [
            ["--confirm-public-holdout-consumption", "--allow-remote"],
            ["--confirm-public-holdout-consumption", "--allow-remote", "--resume"],
            ["--preflight-only"],
            ["--write-contract"],
            ["--output-dir", "tmp/holdout-hijack"],
            ["--split", "public_holdout"],
        ]
        env = dict(os.environ)
        env.pop("LEDGER_PUBLIC_HOLDOUT_E2E_FORCE", None)
        env.pop("MAS_ALLOW_HOLDOUT_RERUN", None)
        for extra in commands:
            completed = subprocess.run(
                [sys.executable, str(CLI), *extra],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(completed.returncode, 2, msg=extra)
            printed = json.loads(completed.stdout)
            self.assertEqual(printed["status"], "FAILED")
            self.assertNotIn("query_text", completed.stdout)
            self.assertNotRegex(completed.stdout.casefold(), r'"gold_value"')
        freeze = subprocess.run(
            [sys.executable, str(FREEZE_CLI), "--write"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(freeze.returncode, 2)
        freeze_printed = json.loads(freeze.stdout)
        self.assertIn("consumed", freeze_printed["error"].casefold())
        after = inspect_official_raw_artifacts(repo_root=ROOT)
        self.assertEqual(before, after)
        if official.exists():
            self.assertEqual(before["raw_artifacts_status"], "PRESENT")


class HoldoutE2EV2RawFixtureTests(unittest.TestCase):
    def test_raw_absent_fixture_reports_not_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_official_raw_artifacts(repo_root=Path(tmp))
            self.assertEqual(report["raw_artifacts_status"], "NOT_PRESENT")
            self.assertIs(report["raw_bytes_reverified"], False)
            verified = reverify_raw_artifacts_against_seal(repo_root=Path(tmp), result={})
            self.assertEqual(verified["raw_artifacts_status"], "NOT_PRESENT")
            self.assertIs(verified["raw_bytes_reverified"], False)

    def test_raw_present_fixture_recomputes_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            dest = repo / "outputs" / "ledger_public_holdout_e2e_v2"
            dest.mkdir(parents=True)
            aggregate = dest / "aggregate.json"
            per_case = dest / "per_case.jsonl"
            aggregate.write_bytes(b'{"cases":100}\n')
            per_case.write_bytes(b'{"query_id":"q1"}\n')
            (dest / "extra.json").write_text("{}", encoding="utf-8")
            inspection = inspect_official_raw_artifacts(repo_root=repo)
            self.assertEqual(inspection["raw_artifacts_status"], "PRESENT")
            self.assertEqual(inspection["extra_files"], ["outputs/ledger_public_holdout_e2e_v2/extra.json"])
            artifacts = {item["relative_path"]: item for item in inspection["artifacts"]}
            agg_rel = "outputs/ledger_public_holdout_e2e_v2/aggregate.json"
            self.assertEqual(artifacts[agg_rel]["size_bytes"], aggregate.stat().st_size)
            self.assertEqual(len(artifacts[agg_rel]["sha256"]), 64)
            result = {
                "official_raw_artifacts": [
                    {
                        "relative_path": "outputs/ledger_public_holdout_e2e_v2/aggregate.json",
                        "size_bytes": aggregate.stat().st_size,
                        "sha256": artifacts["outputs/ledger_public_holdout_e2e_v2/aggregate.json"]["sha256"],
                    },
                    {
                        "relative_path": "outputs/ledger_public_holdout_e2e_v2/per_case.jsonl",
                        "size_bytes": per_case.stat().st_size,
                        "sha256": artifacts["outputs/ledger_public_holdout_e2e_v2/per_case.jsonl"]["sha256"],
                    },
                ]
            }
            verified = reverify_raw_artifacts_against_seal(repo_root=repo, result=result)
            self.assertEqual(verified["raw_artifacts_status"], "PRESENT")
            self.assertIs(verified["raw_bytes_reverified"], False)
            self.assertIn("extra_official_files", verified["mismatches"])


if __name__ == "__main__":
    unittest.main()
