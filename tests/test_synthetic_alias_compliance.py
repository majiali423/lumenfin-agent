from __future__ import annotations

import importlib.util
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

from lumenfin.citation_alias import (
    CITATION_ALIAS_PROTOCOL_VERSION,
    LOCKED_FINAL_K,
    allowlist_from_window,
    build_citation_alias_map,
    build_final_evidence_window,
    official_ranked_hits,
    parse_and_map_citation_aliases,
    render_prompt_evidence,
)
from lumenfin.eval.ledger_structured_citation_shadow import (
    DEFAULT_FROZEN_CONFIG_PATH as SHADOW_CONFIG_PATH,
    GOAL_A_CONFIG_HASH,
    execution_authorized as shadow_execution_authorized,
    load_frozen_config as load_shadow_config,
    refuse_unauthorized_shadow_execution,
)
from lumenfin.eval.synthetic_alias_compliance import (
    DEFAULT_AUTHORIZATION_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_OFFICIAL_OUTPUT_DIR,
    DEFAULT_PREFLIGHT_OUTPUT_DIR,
    EVAL_GOLD_SENTINEL,
    EXPECTED_CITATION_CASES,
    FORBIDDEN_PATH_TOKENS,
    RETIRED_LEDGER_V5_HASH,
    SUITE,
    CanaryError,
    NetworkProbe,
    assert_output_not_overwritten,
    assert_resume_compatible,
    bind_case,
    build_preflight_payload,
    dataset_identity,
    empty_metrics,
    evaluate_protocol_gate,
    execution_authorized,
    load_authorization,
    load_cases,
    load_dataset_payload,
    load_frozen_config,
    parse_cli_guard,
    published_config_hash,
    refuse_unauthorized,
    resume_identity,
    run_canary,
    run_preflight,
    score_bound_output,
)
from lumenfin.structured_answer import (
    public_structured_answer_fields,
    validate_structured_answer,
)


def _load_cli():
    path = ROOT / "scripts" / "run_synthetic_alias_compliance_canary.py"
    spec = importlib.util.spec_from_file_location("run_synthetic_alias_compliance_canary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compliant(case) -> str:
    return json.dumps(
        {
            "answer": case.gold_answer or "not present",
            "citations": list(case.expected_aliases),
            "structured_answer_schema_version": "1.0",
            "abstain": case.incomplete,
        }
    )


class SyntheticAliasDatasetTests(unittest.TestCase):
    def test_eight_cases_and_hashes_are_reproducible(self) -> None:
        payload = load_dataset_payload(repo_root=ROOT)
        first = dataset_identity(payload)
        second = dataset_identity(load_dataset_payload(repo_root=ROOT))
        self.assertEqual(first, second)
        self.assertEqual(first["case_count"], 8)
        self.assertEqual(first["dataset_schema_version"], "synthetic_alias_compliance_dataset.v1")
        self.assertEqual(
            first["case_ids_sha256"],
            "1133f1e943221bfc6b78ff6c2196ee7ea198d099a3c47f2bf83505f923329d2a",
        )
        self.assertEqual(
            first["dataset_sha256"],
            "0d9822409b77707a0e5a54426c5f0ec592fef26553b238e9b0a7010045961b7a",
        )
        self.assertEqual(payload["identity"], first)

    def test_dataset_is_synthetic_and_path_isolated(self) -> None:
        path = ROOT / DEFAULT_DATASET_PATH
        blob = path.read_text(encoding="utf-8").casefold()
        for token in FORBIDDEN_PATH_TOKENS:
            self.assertNotIn(token, path.as_posix().casefold())
        for token in ("apple", "nvidia", "financebench", "public_holdout", "10-k"):
            self.assertNotIn(token, blob)
        cases = load_cases(repo_root=ROOT)
        self.assertEqual(len(cases), 8)
        for case in cases:
            self.assertTrue(case.case_id.startswith("SAC-"))
            self.assertEqual(len(case.passages), 10)

    def test_gold_alias_positions_dual_and_incomplete(self) -> None:
        cases = {case.case_id: case for case in load_cases(repo_root=ROOT)}
        self.assertEqual(cases["SAC-01-harborwick-crate-count"].expected_aliases, ("E01",))
        self.assertEqual(cases["SAC-02-wick-lot-label"].expected_aliases, ("E03",))
        self.assertEqual(cases["SAC-03-ribbon-spool-color"].expected_aliases, ("E07",))
        self.assertEqual(cases["SAC-04-shelf-hook-count"].expected_aliases, ("E10",))
        self.assertEqual(cases["SAC-05-dual-tag-and-bin"].expected_aliases, ("E02", "E09"))
        self.assertEqual(cases["SAC-06-combine-hours-and-lane"].expected_aliases, ("E04", "E06"))
        self.assertTrue(cases["SAC-07-no-answer-glint-count"].incomplete)
        self.assertEqual(cases["SAC-07-no-answer-glint-count"].expected_aliases, ())
        self.assertEqual(cases["SAC-08-middle-evidence-distractors"].expected_aliases, ("E05",))


class SyntheticAliasContractTests(unittest.TestCase):
    def test_production_alias_functions_are_called(self) -> None:
        case = load_cases(repo_root=ROOT)[0]
        bound = bind_case(case)
        hits = official_ranked_hits(
            [{"chunk_id": item["chunk_id"], "text": item["text"]} for item in case.passages],
            origin="test_fixture",
        )
        window = build_final_evidence_window(hits, final_k=LOCKED_FINAL_K)
        alias_map = build_citation_alias_map(window, case_id=case.case_id)
        rendered = render_prompt_evidence(window, alias_map, query_text=case.query_text)
        mapped = parse_and_map_citation_aliases(["E01"], alias_map)
        self.assertEqual(bound.alias_map.aliases, alias_map.aliases)
        self.assertEqual(bound.prompt_evidence, rendered)
        self.assertEqual(mapped, [case.chunk_ids[0]])
        self.assertEqual(CITATION_ALIAS_PROTOCOL_VERSION, "citation_alias_protocol.v1")

    def test_stable_ids_and_gold_sentinels_stay_out_of_prompt(self) -> None:
        for case in load_cases(repo_root=ROOT):
            bound = bind_case(case)
            request = json.dumps(bound.provider_request)
            for chunk_id in case.chunk_ids:
                self.assertNotIn(chunk_id, request)
                self.assertNotIn(chunk_id, bound.prompt_evidence)
            self.assertNotIn(EVAL_GOLD_SENTINEL, request)
            self.assertNotIn("qrels", request)
            self.assertNotIn("gold_answer", request)
            self.assertNotIn("expected_aliases", request)
            self.assertIn(case.query_text, bound.prompt_evidence)
            self.assertIn("[E01]", bound.prompt_evidence)
            self.assertIn("[E10]", bound.prompt_evidence)

    def test_raw_unknown_mixed_fail_closed_and_api_finrun_atomicity(self) -> None:
        case = load_cases(repo_root=ROOT)[0]
        bound = bind_case(case)
        ok = score_bound_output(bound, _compliant(case))
        self.assertTrue(ok["metrics"]["alias_mapping_success"])
        self.assertIsNotNone(ok["public_fields"])
        self.assertEqual(ok["api"]["citations"], list(ok["public_fields"]["citations"]))
        self.assertTrue(all(item in case.chunk_ids for item in ok["api"]["citations"]))
        self.assertEqual(ok["metrics"]["finrun_validation"], "passed")
        self.assertEqual(
            (ok["finrun"].get("metadata") or {}).get("citation_validation"),
            "passed",
        )
        validate_structured_answer(
            ok["structured_answer"],
            allowed=allowlist_from_window(bound.window, verified_ids=case.chunk_ids),
        )

        raw = score_bound_output(
            bound,
            json.dumps({"answer": "x", "citations": [case.chunk_ids[0]], "structured_answer_schema_version": "1.0"}),
        )
        unknown = score_bound_output(
            bound,
            json.dumps({"answer": "x", "citations": ["E99"], "structured_answer_schema_version": "1.0"}),
        )
        mixed = score_bound_output(
            bound,
            json.dumps({"answer": "x", "citations": ["E01", "E99"], "structured_answer_schema_version": "1.0"}),
        )
        self.assertTrue(raw["metrics"]["raw_chunk_id_leakage"])
        self.assertTrue(unknown["metrics"]["unknown_aliases"])
        self.assertTrue(mixed["metrics"]["mixed_valid_invalid"])
        for failed in (raw, unknown, mixed):
            self.assertTrue(failed["metrics"]["citation_validation_failed"])
            self.assertIsNone(failed["public_fields"])
            self.assertIsNone(failed["api"]["answer"])
            self.assertEqual(failed["api"]["citations"], [])
            self.assertIsNone(failed["api"]["structured_answer_schema_version"])
            self.assertEqual(failed["metrics"]["finrun_validation"], "failed")
            self.assertIsNone(public_structured_answer_fields({"structured_answer": failed["structured_answer"]}))

    def test_compliant_suite_meets_frozen_gates(self) -> None:
        from lumenfin.eval.synthetic_alias_compliance import apply_case_metrics

        totals = empty_metrics()
        for case in load_cases(repo_root=ROOT):
            scored = score_bound_output(bind_case(case), _compliant(case))
            apply_case_metrics(totals, scored["metrics"])
        self.assertEqual(totals["json_parse_success"], 8)
        self.assertEqual(totals["alias_mapping_success"], EXPECTED_CITATION_CASES)
        self.assertIs(totals["incomplete_case_handled"], True)
        self.assertTrue(evaluate_protocol_gate(totals))


class SyntheticAliasAuthorizationTests(unittest.TestCase):
    def test_published_record_is_default_deny(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_CONFIG_PATH, repo_root=ROOT, require_published=True)
        record = load_authorization(repo_root=ROOT)["records"][config.config_hash]
        self.assertEqual(config.config_hash, published_config_hash(repo_root=ROOT))
        self.assertEqual(
            config.config_hash,
            "51331a4f059d02905f8c2dd61abfe9c180919140f8973303570e6095c529f793",
        )
        self.assertEqual(config.payload["suite"], SUITE)
        self.assertIs(record["preflight_authorized"], False)
        self.assertIs(record["remote_run_authorized"], False)
        self.assertIs(record["execution_authorized"], False)
        self.assertEqual(record["official_preflight_executions"], 0)
        self.assertEqual(record["official_remote_executions"], 0)
        self.assertFalse(execution_authorized(config, repo_root=ROOT, want="preflight"))
        self.assertFalse(execution_authorized(config, repo_root=ROOT, want="remote"))
        with self.assertRaisesRegex(CanaryError, "not authorized"):
            refuse_unauthorized(config, repo_root=ROOT, want="preflight")

    def test_unknown_copy_dataset_change_dir_and_env_force_are_denied(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_CONFIG_PATH, repo_root=ROOT, require_published=True)
        self.assertFalse(execution_authorized(config, repo_root=ROOT, want="execution"))
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "copy.json"
            copied.write_text((ROOT / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(CanaryError, "copied or relocated"):
                load_frozen_config(copied, repo_root=ROOT, require_published=True)
            mutated = json.loads((ROOT / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
            mutated["dataset"]["dataset_sha256"] = "0" * 64
            mutated_path = Path(tmp) / "mutated.json"
            mutated_path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaisesRegex(CanaryError, "config_hash does not match|dataset dataset_sha256"):
                load_frozen_config(mutated_path, repo_root=ROOT)
        with self.assertRaisesRegex(CanaryError, "output directory swap"):
            refuse_unauthorized(
                config,
                repo_root=ROOT,
                output_dir=Path("outputs") / "other_dir",
                want="remote",
            )
        with patch.dict(os.environ, {"SYNTHETIC_ALIAS_COMPLIANCE_FORCE": "true"}):
            self.assertFalse(execution_authorized(config, repo_root=ROOT, want="preflight"))

    def test_official_preflight_and_remote_stay_zero(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_CONFIG_PATH, repo_root=ROOT, require_published=True)
        payload = build_preflight_payload(config, repo_root=ROOT)
        self.assertEqual(payload["cases_executed"], 0)
        self.assertEqual(payload["remote_request_count"], 0)
        with NetworkProbe() as probe:
            with self.assertRaisesRegex(CanaryError, "not authorized"):
                run_preflight(repo_root=ROOT, official=True)
            with self.assertRaisesRegex(CanaryError, "not authorized"):
                run_canary(
                    repo_root=ROOT,
                    confirm_synthetic_alias_compliance=True,
                    allow_remote=True,
                    official=True,
                )
            self.assertEqual(probe.remote_request_count, 0)
        self.assertFalse((ROOT / DEFAULT_PREFLIGHT_OUTPUT_DIR).exists())
        self.assertFalse((ROOT / DEFAULT_OFFICIAL_OUTPUT_DIR).exists())

    def test_cli_requires_dual_keys_and_still_denies_official_run(self) -> None:
        cli = _load_cli()
        with NetworkProbe() as probe:
            self.assertEqual(cli.main(["--preflight-only"]), 2)
            self.assertEqual(cli.main(["--allow-remote"]), 2)
            self.assertEqual(cli.main(["--confirm-synthetic-alias-compliance"]), 2)
            self.assertEqual(
                cli.main(["--confirm-synthetic-alias-compliance", "--allow-remote"]),
                2,
            )
            self.assertEqual(probe.remote_request_count, 0)
        with self.assertRaisesRegex(CanaryError, "refuses runtime overrides"):
            parse_cli_guard(["--limit", "2"])
        with self.assertRaisesRegex(CanaryError, "reserved eval"):
            parse_cli_guard(["public_holdout"])

    def test_resume_identity_and_output_overwrite(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_CONFIG_PATH, repo_root=ROOT, require_published=True)
        identity = resume_identity(config, repo_root=ROOT, output_dir=DEFAULT_OFFICIAL_OUTPUT_DIR)
        assert_resume_compatible(identity, identity)
        other = dict(identity)
        other["config_hash"] = "0" * 64
        with self.assertRaisesRegex(CanaryError, "resume identity"):
            assert_resume_compatible(identity, other)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "out"
            dest.mkdir()
            (dest / "existing.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(CanaryError, "overwrite"):
                assert_output_not_overwritten(dest)

    def test_tracked_files_have_no_credentials_or_absolute_paths(self) -> None:
        for rel in (DEFAULT_DATASET_PATH, DEFAULT_CONFIG_PATH, DEFAULT_AUTHORIZATION_PATH):
            blob = (ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("sk-", blob)
            self.assertNotIn("DEEPSEEK_API_KEY", blob)
            self.assertNotIn("https://", blob)
            self.assertNotIn("C:\\", blob)
            self.assertNotIn("/Users/", blob)
            tracked = subprocess.run(
                ["git", "check-ignore", "-q", str(rel.as_posix())],
                cwd=ROOT,
                check=False,
            )
            self.assertNotEqual(tracked.returncode, 0)

    def test_v5_shadow_remains_permanently_refused(self) -> None:
        self.assertEqual(RETIRED_LEDGER_V5_HASH, GOAL_A_CONFIG_HASH)
        shadow = load_shadow_config(ROOT / SHADOW_CONFIG_PATH, require_published=True)
        self.assertEqual(shadow.config_hash, GOAL_A_CONFIG_HASH)
        self.assertFalse(shadow_execution_authorized(shadow))
        with self.assertRaisesRegex(Exception, "not authorized"):
            refuse_unauthorized_shadow_execution(shadow, repo_root=ROOT)
        config = load_frozen_config(ROOT / DEFAULT_CONFIG_PATH, repo_root=ROOT, require_published=True)
        self.assertNotEqual(config.config_hash, GOAL_A_CONFIG_HASH)


if __name__ == "__main__":
    unittest.main()
