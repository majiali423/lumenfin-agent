from __future__ import annotations

import json
import os
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

from lumenfin.eval.ledger_structured_citation_shadow import (
    CONSUMED_CACHE_FILE_SHA256,
    CONSUMED_CACHE_MANIFEST_SHA256,
    CONSUMED_PUBLIC_DEV_QUERY_IDS_SHA256,
    CONTRACT_IMPLEMENTATION_HASH,
    DEFAULT_EXECUTION_LEDGER_PATH,
    DEFAULT_FROZEN_CONFIG_PATH,
    DEFAULT_OFFICIAL_OUTPUT_DIR,
    DEFAULT_PREFLIGHT_OUTPUT_DIR,
    EXECUTION_LEDGER_KIND,
    EXECUTION_LEDGER_SCHEMA_VERSION,
    FrozenShadowConfig,
    GOAL_A_CONFIG_HASH,
    GOAL_A_IDENTITY_COMMIT,
    GOLD_IDENTITY_SHA256,
    NetworkProbe,
    UNAUTHORIZED_SHADOW_EXECUTION,
    assert_preflight_authorizes_shadow,
    compute_config_hash,
    compute_execution_ledger_hash,
    config_matches_consumed_public_dev,
    consumed_public_dev_dataset_identity,
    execution_authorized,
    load_execution_ledger,
    load_frozen_config,
    published_config_hash,
    refuse_unauthorized_shadow_execution,
    run_preflight,
    run_shadow,
    shadow_execution_record,
)

from tests.test_ledger_structured_citation_shadow import _load_cli, _mini_world


UNAUTHORIZED_RE = "not authorized: consumed exposed public/dev"


class GoalAExecutionAuthorizationTests(unittest.TestCase):
    def test_goal_a_hash_is_contract_identity_and_not_executable(self) -> None:
        path = ROOT / DEFAULT_FROZEN_CONFIG_PATH
        loaded = load_frozen_config(path, require_published=True)
        record = shadow_execution_record(loaded.config_hash)
        self.assertEqual(loaded.config_hash, GOAL_A_CONFIG_HASH)
        self.assertEqual(loaded.config_hash, CONTRACT_IMPLEMENTATION_HASH)
        self.assertEqual(loaded.config_hash, published_config_hash())
        self.assertEqual(loaded.config_hash, compute_config_hash(loaded.payload))
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["identity_status"], "CONTRACT_IMPLEMENTATION_IDENTITY")
        self.assertEqual(record["status"], "NOT_AUTHORIZED_FOR_EXECUTION")
        self.assertEqual(record["reason"], "consumed_exposed_public_dev")
        self.assertIs(record["preflight_authorized"], False)
        self.assertIs(record["shadow_authorized"], False)
        self.assertIs(record["execution_authorized"], False)
        self.assertEqual(record["preflight_executions"], 0)
        self.assertEqual(record["shadow_executions"], 0)
        self.assertEqual(record["superseded_at_commit"], GOAL_A_IDENTITY_COMMIT)
        self.assertIs(execution_authorized(loaded), False)
        self.assertTrue(config_matches_consumed_public_dev(loaded))
        self.assertEqual(
            record["dataset_identity"]["query_ids_sha256"],
            CONSUMED_PUBLIC_DEV_QUERY_IDS_SHA256,
        )
        self.assertEqual(record["dataset_identity"]["gold_identity_sha256"], GOLD_IDENTITY_SHA256)
        self.assertEqual(
            record["dataset_identity"]["cache_file_sha256"],
            CONSUMED_CACHE_FILE_SHA256,
        )
        self.assertEqual(
            record["dataset_identity"]["cache_manifest_sha256"],
            CONSUMED_CACHE_MANIFEST_SHA256,
        )
        self.assertNotIn(loaded.config_hash, {"unseen", "new_candidate"})
        self.assertEqual(
            UNAUTHORIZED_SHADOW_EXECUTION,
            "config is not authorized: consumed exposed public/dev dataset",
        )

    def test_tracked_json_ledger_is_authoritative(self) -> None:
        path = ROOT / DEFAULT_EXECUTION_LEDGER_PATH
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded = load_execution_ledger()
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(payload["schema_version"], EXECUTION_LEDGER_SCHEMA_VERSION)
        self.assertEqual(payload["kind"], EXECUTION_LEDGER_KIND)
        self.assertEqual(payload["ledger_sha256"], compute_execution_ledger_hash(payload))
        self.assertEqual(payload["ledger_sha256"], loaded["ledger_sha256"])
        recorded = payload["records"][GOAL_A_CONFIG_HASH]
        self.assertEqual(recorded["dataset_identity"], consumed_public_dev_dataset_identity())
        self.assertEqual(loaded["records"][GOAL_A_CONFIG_HASH]["identity_status"], recorded["identity_status"])
        self.assertIs(recorded["execution_authorized"], False)
        self.assertIs(recorded["preflight_authorized"], False)
        self.assertIs(recorded["shadow_authorized"], False)
        blob = path.read_text(encoding="utf-8")
        self.assertNotIn("sk-", blob)
        self.assertNotIn("https://", blob)
        self.assertNotIn("C:\\", blob)
        self.assertNotIn("gold_value", blob)
        self.assertNotIn("query_text", blob)
        tracked = subprocess.run(
            ["git", "check-ignore", "-q", str(path.relative_to(ROOT))],
            cwd=ROOT,
            check=False,
        )
        self.assertNotEqual(tracked.returncode, 0)

    def test_cli_preflight_only_is_rejected(self) -> None:
        cli = _load_cli()
        with patch.object(cli, "run_shadow", side_effect=AssertionError("run_shadow")) as mocked:
            code = cli.main(
                [
                    "--split",
                    "public-dev",
                    "--frozen-config",
                    str(ROOT / DEFAULT_FROZEN_CONFIG_PATH),
                    "--confirm-exposed-shadow",
                    "--preflight-only",
                ]
            )
        self.assertEqual(code, 2)
        mocked.assert_not_called()

    def test_cli_allow_remote_is_rejected(self) -> None:
        cli = _load_cli()
        with patch.object(cli, "run_shadow", side_effect=AssertionError("run_shadow")) as mocked:
            code = cli.main(
                [
                    "--split",
                    "public-dev",
                    "--frozen-config",
                    str(ROOT / DEFAULT_FROZEN_CONFIG_PATH),
                    "--confirm-exposed-shadow",
                    "--allow-remote",
                ]
            )
        self.assertEqual(code, 2)
        mocked.assert_not_called()

    def test_cli_resume_is_rejected(self) -> None:
        cli = _load_cli()
        with patch.object(cli, "run_shadow", side_effect=AssertionError("run_shadow")) as mocked:
            code = cli.main(
                [
                    "--split",
                    "public-dev",
                    "--frozen-config",
                    str(ROOT / DEFAULT_FROZEN_CONFIG_PATH),
                    "--confirm-exposed-shadow",
                    "--allow-remote",
                    "--resume",
                ]
            )
        self.assertEqual(code, 2)
        mocked.assert_not_called()

    def test_direct_run_preflight_and_run_shadow_are_rejected(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                run_preflight(
                    repo_root=root,
                    frozen_config=config,
                    official_output_dir=root / "official",
                    preflight_output_dir=root / "preflight",
                    split="public-dev",
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                    require_chat_key=False,
                )
            with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                run_shadow(
                    repo_root=root,
                    frozen_config=config,
                    split="public-dev",
                    confirm_exposed_shadow=True,
                    output_dir=root / "official",
                    preflight_output_dir=root / "preflight",
                    preflight_only=True,
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                    require_chat_key=False,
                )
            with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                run_shadow(
                    repo_root=root,
                    frozen_config=config,
                    split="public-dev",
                    confirm_exposed_shadow=True,
                    output_dir=root / "official",
                    preflight_output_dir=root / "preflight",
                    allow_remote=True,
                    live_generate=True,
                    resume=True,
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                    require_chat_key=False,
                )
            self.assertFalse((root / "official").exists())
            self.assertFalse((root / "preflight").exists())

    def test_empty_temp_output_dir_is_still_rejected(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty-out"
            empty.mkdir()
            with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                run_shadow(
                    repo_root=root,
                    frozen_config=config,
                    split="public-dev",
                    confirm_exposed_shadow=True,
                    output_dir=empty,
                    preflight_output_dir=root / "empty-preflight",
                    preflight_only=True,
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                    require_chat_key=False,
                )
            self.assertEqual(list(empty.iterdir()), [])
            self.assertFalse((root / "empty-preflight").exists())

    def test_missing_official_directory_is_still_rejected(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official = root / DEFAULT_OFFICIAL_OUTPUT_DIR
            self.assertFalse(official.exists())
            with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                run_shadow(
                    repo_root=root,
                    frozen_config=config,
                    split="public-dev",
                    confirm_exposed_shadow=True,
                    output_dir=official,
                    preflight_output_dir=root / DEFAULT_PREFLIGHT_OUTPUT_DIR,
                    allow_remote=True,
                    live_generate=True,
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                    require_chat_key=False,
                )
            self.assertFalse(official.exists())
            self.assertFalse((root / DEFAULT_PREFLIGHT_OUTPUT_DIR).exists())

    def test_copied_same_content_config_is_rejected_by_hash(self) -> None:
        source = (ROOT / DEFAULT_FROZEN_CONFIG_PATH).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "elsewhere" / "frozen.json"
            copied.parent.mkdir(parents=True)
            copied.write_bytes(source)
            loaded = load_frozen_config(copied)
            self.assertEqual(loaded.config_hash, GOAL_A_CONFIG_HASH)
            with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                refuse_unauthorized_shadow_execution(
                    loaded,
                    output_dir=Path(tmp) / "out",
                    preflight_output_dir=Path(tmp) / "preflight",
                )

    def test_env_and_allow_remote_cannot_bypass(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        env = {
            "LUMENFIN_SHADOW_ALLOW_REMOTE": "1",
            "LUMENFIN_SHADOW_FORCE_EXECUTE": "1",
            "LUMENFIN_SHADOW_AUTHORIZE": "1",
            "DEEPSEEK_API_KEY": "sk-test-not-for-disk",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, env, clear=False):
                with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                    run_shadow(
                        repo_root=root,
                        frozen_config=config,
                        split="public-dev",
                        confirm_exposed_shadow=True,
                        output_dir=root / "out",
                        preflight_output_dir=root / "preflight",
                        allow_remote=True,
                        live_generate=True,
                        verify_tag=False,
                        require_clean=False,
                        verify_runtime=False,
                    )
            self.assertFalse((root / "out").exists())

    def test_refusal_happens_before_credential_data_provider_and_output(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        blocked = AssertionError("side effect reached")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = "lumenfin.eval.ledger_structured_citation_shadow"
            with patch(f"{target}.require_chat_credential", side_effect=blocked):
                with patch(f"{target}.credential_presence", side_effect=blocked):
                    with patch(f"{target}.bind_chain_seal", side_effect=blocked):
                        with patch(f"{target}.verify_candidate_cache", side_effect=blocked):
                            with patch(f"{target}.bind_cases_from_verified_cache", side_effect=blocked):
                                with patch(f"{target}.build_live_generate", side_effect=blocked):
                                    with patch(f"{target}.atomic_write_json", side_effect=blocked):
                                        with patch(f"{target}.git_snapshot", side_effect=blocked):
                                            with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                                                run_preflight(
                                                    repo_root=root,
                                                    frozen_config=config,
                                                    official_output_dir=root / "official",
                                                    preflight_output_dir=root / "preflight",
                                                    split="public-dev",
                                                    verify_tag=False,
                                                    require_clean=False,
                                                    verify_runtime=False,
                                                )
                                            with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                                                run_shadow(
                                                    repo_root=root,
                                                    frozen_config=config,
                                                    split="public-dev",
                                                    confirm_exposed_shadow=True,
                                                    output_dir=root / "official",
                                                    preflight_output_dir=root / "preflight",
                                                    allow_remote=True,
                                                    live_generate=True,
                                                    resume=True,
                                                    verify_tag=False,
                                                    require_clean=False,
                                                    verify_runtime=False,
                                                )
            self.assertFalse((root / "official").exists())
            self.assertFalse((root / "preflight").exists())

    def test_reserved_output_protection_survives_auth_patch(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official = root / DEFAULT_OFFICIAL_OUTPUT_DIR
            preflight = root / DEFAULT_PREFLIGHT_OUTPUT_DIR
            similar = root / "outputs" / f"{DEFAULT_OFFICIAL_OUTPUT_DIR.name}_retry"
            with patch(
                "lumenfin.eval.ledger_structured_citation_shadow.execution_authorized",
                return_value=True,
            ):
                with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                    run_shadow(
                        repo_root=root,
                        frozen_config=config,
                        split="public-dev",
                        confirm_exposed_shadow=True,
                        output_dir=official,
                        preflight_output_dir=preflight,
                        preflight_only=True,
                        verify_tag=False,
                        require_clean=False,
                        verify_runtime=False,
                        require_chat_key=False,
                    )
                with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                    run_shadow(
                        repo_root=root,
                        frozen_config=config,
                        split="public-dev",
                        confirm_exposed_shadow=True,
                        output_dir=similar,
                        preflight_output_dir=root / "other",
                        allow_remote=True,
                        resume=True,
                        live_generate=True,
                        verify_tag=False,
                        require_clean=False,
                        verify_runtime=False,
                        require_chat_key=False,
                    )
            self.assertFalse(official.exists())
            self.assertFalse(preflight.exists())
            self.assertFalse(similar.exists())

    def test_planted_v5_preflight_cannot_authorize_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planted = root / DEFAULT_PREFLIGHT_OUTPUT_DIR
            planted.mkdir(parents=True)
            (planted / "preflight.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "v5 preflight is not authorized"):
                assert_preflight_authorizes_shadow(
                    repo_root=root,
                    execution_commit="deadbeef" * 5,
                )

    def test_refusal_makes_no_remote_calls(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        probe = NetworkProbe()
        probe.install()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                    run_shadow(
                        repo_root=root,
                        frozen_config=config,
                        split="public-dev",
                        confirm_exposed_shadow=True,
                        output_dir=root / "out",
                        preflight_output_dir=root / "preflight",
                        allow_remote=True,
                        live_generate=True,
                        verify_tag=False,
                        require_clean=False,
                        verify_runtime=False,
                    )
        finally:
            probe.remove()
        self.assertEqual(probe.remote_request_count, 0)

    def test_unlisted_hash_is_denied_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"])
            self.assertNotEqual(config.config_hash, GOAL_A_CONFIG_HASH)
            self.assertFalse(config_matches_consumed_public_dev(config))
            self.assertFalse(execution_authorized(config, repo_root=root))
            with self.assertRaisesRegex(Exception, "not authorized"):
                refuse_unauthorized_shadow_execution(
                    config,
                    repo_root=root,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                )
            refuse_unauthorized_shadow_execution(
                config,
                repo_root=root,
                output_dir=root / "out",
                preflight_output_dir=root / "preflight",
                synthetic_unlisted_ok=True,
            )

    def test_same_fingerprint_new_hash_is_denied(self) -> None:
        source = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        payload = json.loads(json.dumps(source.payload))
        payload["prompts"]["version"] = "ledger_structured_citation_shadow_v1_renamed"
        payload["config_hash"] = compute_config_hash(payload)
        mutated = FrozenShadowConfig(source.path, payload)
        self.assertNotEqual(mutated.config_hash, GOAL_A_CONFIG_HASH)
        self.assertTrue(config_matches_consumed_public_dev(mutated))
        self.assertFalse(execution_authorized(mutated))
        with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
            refuse_unauthorized_shadow_execution(mutated)

    def test_string_false_is_not_authorized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"])
            payload = {
                "schema_version": EXECUTION_LEDGER_SCHEMA_VERSION,
                "kind": EXECUTION_LEDGER_KIND,
                "records": {
                    config.config_hash: {
                        "config_hash": config.config_hash,
                        "identity_status": "SYNTHETIC",
                        "dataset_identity": {
                            "split": "public_dev",
                            "query_count": 1,
                            "query_ids_sha256": "a" * 64,
                            "gold_identity_sha256": GOLD_IDENTITY_SHA256,
                            "dataset_snapshot_sha256": "b" * 64,
                            "source_artifact_sha256": "c" * 64,
                            "cache_file_sha256": "d" * 64,
                            "cache_manifest_sha256": "e" * 64,
                        },
                        "execution_authorized": "false",
                        "preflight_authorized": "false",
                        "shadow_authorized": "false",
                        "reason": "test",
                        "preflight_executions": 0,
                        "shadow_executions": 0,
                        "superseded_at_commit": GOAL_A_IDENTITY_COMMIT,
                    }
                },
            }
            payload["ledger_sha256"] = compute_execution_ledger_hash(payload)
            ledger_path = root / DEFAULT_EXECUTION_LEDGER_PATH
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(load_execution_ledger(repo_root=root))
            self.assertFalse(execution_authorized(config, repo_root=root))

    def test_missing_or_malformed_ledger_denies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"])
            self.assertIsNone(load_execution_ledger(repo_root=root))
            self.assertFalse(execution_authorized(config, repo_root=root))
            (root / DEFAULT_EXECUTION_LEDGER_PATH).parent.mkdir(parents=True, exist_ok=True)
            (root / DEFAULT_EXECUTION_LEDGER_PATH).write_text("{", encoding="utf-8")
            self.assertIsNone(load_execution_ledger(repo_root=root))
            self.assertFalse(execution_authorized(config, repo_root=root))

    def test_conflicting_dataset_authorization_denies_all(self) -> None:
        payload = json.loads((ROOT / DEFAULT_EXECUTION_LEDGER_PATH).read_text(encoding="utf-8"))
        other_hash = "ab" * 32
        payload["records"][other_hash] = {
            **payload["records"][GOAL_A_CONFIG_HASH],
            "config_hash": other_hash,
            "execution_authorized": True,
            "preflight_authorized": True,
            "shadow_authorized": True,
        }
        payload["ledger_sha256"] = compute_execution_ledger_hash(payload)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / DEFAULT_EXECUTION_LEDGER_PATH
            dest.parent.mkdir(parents=True)
            dest.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(load_execution_ledger(repo_root=root))
            config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
            self.assertFalse(execution_authorized(config, repo_root=root))

    def test_clean_checkout_without_cache_outputs_still_denies_first(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        blocked = AssertionError("missing local data reached")
        target = "lumenfin.eval.ledger_structured_citation_shadow"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(f"{target}.require_chat_credential", side_effect=blocked):
                with patch(f"{target}.bind_chain_seal", side_effect=blocked):
                    with patch(f"{target}.verify_candidate_cache", side_effect=blocked):
                        with patch(f"{target}.git_snapshot", side_effect=blocked):
                            with self.assertRaisesRegex(Exception, UNAUTHORIZED_RE):
                                run_shadow(
                                    repo_root=root,
                                    frozen_config=config,
                                    split="public-dev",
                                    confirm_exposed_shadow=True,
                                    output_dir=root / "out",
                                    preflight_output_dir=root / "preflight",
                                    preflight_only=True,
                                    allow_remote=False,
                                    verify_tag=False,
                                    require_clean=False,
                                    verify_runtime=False,
                                )
            self.assertFalse((root / "out").exists())
            self.assertFalse((root / "preflight").exists())
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
