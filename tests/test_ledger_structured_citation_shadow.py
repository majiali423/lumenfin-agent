from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from lumenfin.eval.holdout.ledger_e2e import build_generation_prompt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.ledger_structured_citation_shadow import (
    CANONICAL_SPLIT,
    CLI_SPLIT,
    DEFAULT_FROZEN_CONFIG_PATH,
    DEFAULT_PREFLIGHT_OUTPUT_DIR,
    EVALUATION_MODE,
    GOLD_IDENTITY_SHA256,
    INCOMPLETE_AUDIT_CONFIG_HASH,
    INCOMPLETE_V1_PREFLIGHT_SHA256,
    LEGACY_PREFLIGHT_OUTPUT_DIR,
    PREFLIGHT_OK,
    PREFLIGHT_REQUIRED_FIELDS,
    PREFLIGHT_SCHEMA_VERSION,
    PREVIOUS_UNUSED_CONFIG_HASH,
    PROTOCOL_COMMIT,
    RETIRED_BEFORE_PREFLIGHT_HASH,
    RETIRED_CONFIG_HASHES,
    SUPERSEDED_PREFLIGHT_OUTPUT_DIR,
    SUPERSEDED_V2_CONFIG_HASH,
    V2_PREFLIGHT_EXECUTION_COMMIT,
    V2_PREFLIGHT_SHA256,
    forbid_runtime_embedding_call,
    forbid_runtime_reranker_call,
    SEAL_TAG,
    SEAL_TARGET_COMMIT,
    SUITE,
    FrozenShadowConfig,
    ShadowError,
    assert_case_ids,
    assert_safe_input_path,
    bind_chain_seal,
    canonical_split,
    chunk_ids_sha256,
    compute_config_hash,
    ids_sha256,
    assert_preflight_authorizes_shadow,
    bind_cases_from_verified_cache,
    build_live_generate,
    generation_case_view,
    load_case_fixture,
    load_frozen_config,
    public_dev_snapshot_relative,
    paired_descriptive_comparison,
    parse_cli_guard,
    peel_seal_tag,
    published_config_hash,
    published_frozen_config_fields,
    read_sealed_baseline_readonly,
    run_shadow,
    sha256_normalized_file,
    sha256_raw_file,
    summarize_rows,
    verify_candidate_cache,
)
from lumenfin.structured_answer import STRUCTURED_ANSWER_SCHEMA_VERSION


def _load_cli():
    path = ROOT / "scripts" / "run_ledger_structured_citation_shadow.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_structured_citation_shadow",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load structured citation shadow CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _hit(index: int, *, verified: bool = True, tenant: str = "default") -> dict:
    return {
        "chunk_id": f"doc-{index}:p1:c0",
        "document_id": f"doc-{index}",
        "text": f"The KPI equals {index * 10}.",
        "tenant_id": tenant,
        "session_id": "shadow",
        "unverified": not verified,
        "page": 1,
    }


def _case(case_id: str, index: int, **kwargs: object) -> dict:
    row = {
        "case_id": case_id,
        "query_text": f"What is KPI {index}?",
        "gold_value": float(index * 10),
        "hits": [_hit(index), _hit(index + 10)],
        "qrels": {f"doc-{index}": 1, f"doc-{index + 10}": 0},
        "tenant_id": "default",
        "session_id": "shadow",
    }
    row.update(kwargs)
    return row


def _structured_payload(index: int, *, source: str = "structured") -> str:
    chunk = f"doc-{index}:p1:c0"
    if source == "unavailable":
        return json.dumps(
            {
                "answer": "unknown",
                "citations": [],
                "value": None,
                "abstain": True,
            }
        )
    if source == "legacy":
        return json.dumps(
            {
                "value": index * 10,
                "cited_chunk_ids": [chunk],
                "abstain": False,
            }
        )
    if source == "unknown":
        return json.dumps(
            {
                "answer": "n",
                "citations": ["missing:p1:c0"],
                "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                "value": index * 10,
                "abstain": False,
            }
        )
    if source == "unverified":
        return json.dumps(
            {
                "answer": "n",
                "citations": [f"doc-{index}:p1:c0"],
                "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                "value": index * 10,
                "abstain": False,
            }
        )
    if source == "cross_scope":
        return json.dumps(
            {
                "answer": "n",
                "citations": [f"doc-{index}:p1:c0"],
                "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                "value": index * 10,
                "abstain": False,
                "tenant_id": "other",
            }
        )
    return json.dumps(
        {
            "answer": f"KPI is {index * 10}",
            "citations": [chunk],
            "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
            "value": index * 10,
            "abstain": False,
        }
    )


def _mini_world(
    tmp: Path,
    case_ids: list[str],
    *,
    cache_query_text: bool = True,
) -> tuple[FrozenShadowConfig, list[dict], Path]:
    seal_dir = tmp / "data" / "eval_rag" / "holdout"
    seal = {
        "schema_version": "lumenfin_ledger_public_dev_chain_seal.v1",
        "split": "public_dev",
        "public_holdout_consumed": False,
        "recommended_annotated_tag": SEAL_TAG,
        "recommended_tag_target_commit": SEAL_TARGET_COMMIT,
    }
    manifest = {"schema_version": "lumenfin_public_benchmark_manifest.v1", "split": "public_dev"}
    baseline = {
        "schema_version": "lumenfin_ledger_e2e_canary.v1",
        "cases": len(case_ids),
        "selection": {"query_ids_sha256": ids_sha256(case_ids)},
        "comparison": {
            "lexical": {"citation_support_rate": 0.0, "abstain_rate": 1.0, "cases": len(case_ids)}
        },
    }
    seal_path = seal_dir / "ledger_public_dev_chain_seal.json"
    manifest_path = seal_dir / "ledger_public_manifest.json"
    baseline_path = seal_dir / "ledger_public_dev_e2e_canary_5x10.json"
    _write_json(seal_path, seal)
    _write_json(manifest_path, manifest)
    _write_json(baseline_path, baseline)
    cases = [_case(case_id, index + 1) for index, case_id in enumerate(case_ids)]
    cache_rows = []
    for case in cases:
        row = {
            "query_id": case["case_id"],
            "company_key_sha256": "company-a",
            "hits": [
                {
                    "chunk_id": hit["chunk_id"],
                    "document_id": hit["document_id"],
                    "text": hit["text"],
                }
                for hit in case["hits"]
            ],
        }
        if cache_query_text:
            row["query_text"] = case["query_text"]
            row["query_text_sha256"] = hashlib.sha256(
                str(case["query_text"]).encode("utf-8")
            ).hexdigest()
            row["gold_value"] = case["gold_value"]
            row["qrels"] = case["qrels"]
        cache_rows.append(row)
    cache_rel = Path("outputs") / "cache" / "candidates.jsonl"
    cache_path = tmp / cache_rel
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cache_rows),
        encoding="utf-8",
    )
    cache_manifest = {
        "schema_version": "lumenfin_ledger_structured_citation_shadow_cache.v1",
        "cache_kind": "frozen_hybrid_candidate_replay",
        "not_live_production_retrieval": True,
        "rebuild_forbidden": True,
        "embedding_fallback_forbidden": True,
        "readonly": True,
        "source_path_identity": cache_rel.as_posix(),
        "source_local_manifest_identity": "",
        "source_schema_version": "lumenfin_ledger_hybrid_candidates.v1",
        "source_commit": SEAL_TARGET_COMMIT,
        "source_worktree_dirty": None,
        "source_worktree_recorded": False,
        "parent_case_count": len(case_ids),
        "parent_query_ids_sha256": ids_sha256(case_ids),
        "case_count": len(case_ids),
        "case_ids_sha256": ids_sha256(case_ids),
        "candidate_records_count": len(cache_rows),
        "hits_per_case": len(cases[0]["hits"]),
        "candidate_ordering": "cache_file_order_company_blocks_then_frozen_5x50_company_prefix_v1",
        "chunk_ids_sha256": chunk_ids_sha256(cache_rows),
        "candidate_set_identity_sha256": "test",
        "cache_file_sha256": sha256_raw_file(cache_path),
        "local_candidate_manifest_sha256": "",
        "rag_config": {
            "arm": "A",
            "ranking_arm": "A_prod",
            "pool_strategy": "ranked_chunks",
            "source_k": 20,
            "rerank_k": 20,
            "final_k": 10,
        },
        "embedding_identity": {
            "provider": "dashscope",
            "model": "text-embedding-v4",
            "dimension": 1024,
            "used_in_this_suite": False,
        },
        "rerank_identity": {"provider": "lexical", "model": "lexical", "used_in_this_suite": True},
    }
    cache_manifest_path = tmp / "data" / "eval_rag" / "structured_citation_shadow_cache_manifest.json"
    _write_json(cache_manifest_path, cache_manifest)
    fields = published_frozen_config_fields()
    fields["case_selection"]["query_count"] = len(case_ids)
    fields["case_selection"]["query_ids_sha256"] = ids_sha256(case_ids)
    fields["call_budget"]["cases_total"] = len(case_ids)
    fields["call_budget"]["generate_logical_calls_expected"] = len(case_ids)
    fields["call_budget"]["remote_calls_expected"] = len(case_ids)
    fields["split_manifest"]["sha256"] = sha256_normalized_file(manifest_path)
    fields["sealed_baseline"]["sha256"] = sha256_normalized_file(baseline_path)
    fields["candidate_cache"]["manifest_sha256"] = sha256_normalized_file(cache_manifest_path)
    fields["config_hash"] = compute_config_hash(fields)
    config_path = tmp / "frozen.json"
    _write_json(config_path, fields)
    return load_frozen_config(config_path), cases, tmp


def _authorized_run(generate, **kwargs):
    kwargs.setdefault("confirm_exposed_shadow", True)
    kwargs.setdefault("allow_remote", True)
    kwargs.setdefault("allow_injected_generate", True)
    kwargs.setdefault("verify_tag", False)
    kwargs.setdefault("require_clean", False)
    kwargs.setdefault("verify_runtime", False)
    kwargs["generate_fn"] = generate
    return run_shadow(**kwargs)


class LedgerStructuredCitationShadowTests(unittest.TestCase):
    def test_published_config_hash_is_reproducible_and_locked(self) -> None:
        path = ROOT / DEFAULT_FROZEN_CONFIG_PATH
        loaded = load_frozen_config(path, require_published=True)
        self.assertEqual(loaded.config_hash, published_config_hash())
        self.assertEqual(loaded.config_hash, compute_config_hash(loaded.payload))
        self.assertEqual(loaded.payload["suite"], SUITE)
        self.assertEqual(loaded.payload["split"], CANONICAL_SPLIT)
        self.assertIs(loaded.payload["held_out"], False)
        self.assertIs(loaded.payload["product_accuracy_claim"], False)
        self.assertIs(loaded.payload["benchmark_claim"], False)
        self.assertTrue(loaded.payload["candidate_cache"]["not_live_production_retrieval"])
        self.assertEqual(loaded.payload["evaluation_mode"], EVALUATION_MODE)
        self.assertIs(loaded.payload["runtime_embedding_enabled"], False)
        self.assertIs(loaded.payload["runtime_reranker_enabled"], False)
        self.assertEqual(
            loaded.payload["candidate_cache_generation"]["reranker"]["provider"],
            "lexical",
        )
        self.assertEqual(loaded.payload["preflight_schema_version"], PREFLIGHT_SCHEMA_VERSION)
        self.assertEqual(loaded.payload["output"]["preflight_dirname"], DEFAULT_PREFLIGHT_OUTPUT_DIR.name)
        self.assertEqual(loaded.payload["output"]["legacy_preflight_dirname"], LEGACY_PREFLIGHT_OUTPUT_DIR.name)
        self.assertEqual(loaded.payload["preflight_required_fields"], list(PREFLIGHT_REQUIRED_FIELDS))
        self.assertEqual(loaded.payload["chat"]["model"], "deepseek-v4-flash")
        self.assertEqual(loaded.payload["call_budget"]["remote_calls_expected"], 50)
        self.assertEqual(
            loaded.payload["candidate_cache"]["manifest_sha256"],
            "2550d0310caaa68f13107e8c0f870d891bda3797908b5a888e30b49048b9db90",
        )
        self.assertEqual(
            loaded.payload["predecessor_config"]["config_hash"],
            SUPERSEDED_V2_CONFIG_HASH,
        )
        self.assertEqual(loaded.payload["predecessor_config"]["preflight_executions"], 1)
        self.assertEqual(loaded.payload["predecessor_config"]["accepted_preflights"], 1)
        self.assertEqual(
            loaded.payload["predecessor_config"]["grant_status"],
            "SUPERSEDED_BEFORE_SHADOW",
        )
        self.assertEqual(
            loaded.payload["predecessor_config"]["accepted_at_execution_commit"],
            V2_PREFLIGHT_EXECUTION_COMMIT,
        )
        self.assertEqual(loaded.payload["predecessor_config"]["artifact_sha256"], V2_PREFLIGHT_SHA256)
        self.assertIs(loaded.payload["predecessor_config"]["accepted_for_shadow_execution"], False)
        self.assertEqual(RETIRED_CONFIG_HASHES[SUPERSEDED_V2_CONFIG_HASH]["shadow_executions"], 0)
        self.assertEqual(RETIRED_CONFIG_HASHES[INCOMPLETE_AUDIT_CONFIG_HASH]["accepted_preflights"], 0)
        self.assertEqual(
            loaded.payload["output"]["superseded_preflight_dirname"],
            SUPERSEDED_PREFLIGHT_OUTPUT_DIR.name,
        )
        self.assertEqual(loaded.payload["case_selection"]["gold_identity_sha256"], GOLD_IDENTITY_SHA256)
        self.assertNotEqual(loaded.config_hash, PREVIOUS_UNUSED_CONFIG_HASH)
        self.assertNotEqual(loaded.config_hash, RETIRED_BEFORE_PREFLIGHT_HASH)
        self.assertNotEqual(loaded.config_hash, INCOMPLETE_AUDIT_CONFIG_HASH)
        self.assertNotEqual(loaded.config_hash, SUPERSEDED_V2_CONFIG_HASH)
        blob = path.read_text(encoding="utf-8")
        self.assertNotIn("sk-", blob)
        self.assertNotIn("Authorization", blob)
        self.assertNotIn("https://", blob)
        self.assertNotIn("C:\\", blob)
        self.assertNotIn("C:/", blob)

    def test_endpoint_is_stored_as_sha256_only(self) -> None:
        fields = published_frozen_config_fields()
        self.assertEqual(
            fields["chat"]["base_url_sha256"],
            "a34e2a4708ed1c61008a151688838dcf1c44d4e7f08054633e72ba7c0b16cfc1",
        )
        self.assertNotIn("base_url", fields["chat"])
        self.assertNotIn("https://api.deepseek.com", json.dumps(fields))

    def test_seal_tag_peels_to_expected_commit(self) -> None:
        self.assertEqual(peel_seal_tag(ROOT, SEAL_TAG), SEAL_TARGET_COMMIT)
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        bind = bind_chain_seal(repo_root=ROOT, config=config, verify_tag=True)
        self.assertEqual(bind["seal_commit"], SEAL_TARGET_COMMIT)
        self.assertEqual(bind["query_ids_sha256"], config.field("case_selection", "query_ids_sha256"))
        self.assertEqual(config.field("lumenfin_protocol_commit"), PROTOCOL_COMMIT)

    def test_sealed_baseline_is_read_only(self) -> None:
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        path = ROOT / config.field("sealed_baseline", "path")
        before = sha256_normalized_file(path)
        payload = read_sealed_baseline_readonly(repo_root=ROOT, config=config)
        after = sha256_normalized_file(path)
        self.assertEqual(before, after)
        self.assertEqual(before, config.field("sealed_baseline", "sha256"))
        self.assertIs(payload["readonly"], True)
        self.assertEqual(payload["structured_answer_present"], 0)
        self.assertEqual(
            payload["selection"]["gold_identity_sha256"],
            "990a7ff71234a0a9b3e2c021b972fbb2e93c71da6e747e6f647e25b8c51238a2",
        )

    def test_split_and_holdout_path_rejected_before_read(self) -> None:
        with self.assertRaisesRegex(ShadowError, "split"):
            canonical_split("public_holdout")
        with self.assertRaisesRegex(ShadowError, "split"):
            canonical_split("holdout")
        with self.assertRaisesRegex(ShadowError, "split"):
            canonical_split("confirmation")
        with self.assertRaisesRegex(ShadowError, "split"):
            canonical_split("public_dev")
        self.assertEqual(canonical_split(CLI_SPLIT), CANONICAL_SPLIT)
        with tempfile.TemporaryDirectory() as tmp:
            forbidden = Path(tmp) / "public_holdout" / "cases.json"
            forbidden.parent.mkdir()
            forbidden.write_text('{"secret": "do-not-read"}', encoding="utf-8")
            original = Path.read_text

            def guarded(self: Path, *args: object, **kwargs: object) -> str:
                if "public_holdout" in str(self).replace("\\", "/"):
                    raise AssertionError("holdout path was read")
                return original(self, *args, **kwargs)

            with patch.object(Path, "read_text", guarded):
                with self.assertRaisesRegex(ShadowError, "forbidden"):
                    assert_safe_input_path(forbidden, field="cases")
                with self.assertRaisesRegex(ShadowError, "forbidden"):
                    load_case_fixture(forbidden, allowlist=["a"], expected_hash="dead")
            with self.assertRaisesRegex(ShadowError, "forbidden") as raised:
                load_case_fixture(forbidden, allowlist=["a"], expected_hash="dead")
            self.assertNotIn("do-not-read", str(raised.exception))
            self.assertNotIn("secret", str(raised.exception).casefold())

    def test_holdout_case_id_rejected_before_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1", "pd-2"])
            cases[1]["case_id"] = "not-in-allowlist"
            generate_calls = {"n": 0}

            def generate(_case: dict) -> str:
                generate_calls["n"] += 1
                raise AssertionError("generate must not run")

            with patch(
                "lumenfin.eval.holdout.ledger.iter_ledger_parquet_rows",
                side_effect=AssertionError("parquet must not be read"),
            ):
                with self.assertRaisesRegex(ShadowError, "allowlist"):
                    _authorized_run(
                        generate,
                        repo_root=root,
                        frozen_config=config,
                        split=CLI_SPLIT,
                        output_dir=root / "out",
                        preflight_output_dir=root / "preflight",
                        cases=cases,
                        allowlist=["pd-1", "pd-2"],
                    )
            self.assertEqual(generate_calls["n"], 0)
            self.assertFalse((root / "out").exists())

    def test_missing_extra_duplicate_and_order_fail_closed(self) -> None:
        expected = ["a", "b", "c"]
        digest = ids_sha256(expected)
        with self.assertRaisesRegex(ShadowError, "duplicate"):
            assert_case_ids(["a", "a"], expected_ids=["a"], expected_hash=ids_sha256(["a"]))
        with self.assertRaisesRegex(ShadowError, "allowlist"):
            assert_case_ids(["a", "b"], expected_ids=expected, expected_hash=digest)
        with self.assertRaisesRegex(ShadowError, "allowlist"):
            assert_case_ids(["a", "b", "c", "d"], expected_ids=expected, expected_hash=digest)
        with self.assertRaisesRegex(ShadowError, "order"):
            assert_case_ids(["a", "c", "b"], expected_ids=expected, expected_hash=digest)
        assert_case_ids(expected, expected_ids=expected, expected_hash=digest)

    def test_runtime_mismatch_fails_before_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1"])
            with patch.dict(os.environ, {"DEEPSEEK_MODEL": "other-model"}):
                with self.assertRaisesRegex(ShadowError, "chat model"):
                    _authorized_run(
                        lambda _case: _structured_payload(1),
                        repo_root=root,
                        frozen_config=config,
                        split=CLI_SPLIT,
                        output_dir=root / "out",
                        preflight_output_dir=root / "preflight",
                        cases=cases,
                        verify_runtime=True,
                        allowlist=["pd-1"],
                    )
            self.assertFalse((root / "out").exists())

    def test_preflight_writes_only_preflight_and_makes_no_remote_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1", "pd-2"])
            official = root / "official"
            preflight_dir = root / "preflight"
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-not-for-disk"}, clear=False):
                report = run_shadow(
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    confirm_exposed_shadow=True,
                    output_dir=official,
                    preflight_output_dir=preflight_dir,
                    preflight_only=True,
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                )
            self.assertEqual(report["remote_request_count"], 0)
            self.assertEqual(report["cases_executed"], 0)
            self.assertIs(report["case_binding_verified"], True)
            self.assertEqual(report["case_count"], 2)
            self.assertTrue(report["query_ids_sha256"])
            self.assertTrue(report["query_texts_sha256"])
            self.assertIs(report["gold_not_exposed_to_generator"], True)
            self.assertEqual(report["protocol_ancestor"], PROTOCOL_COMMIT)
            self.assertTrue(report["not_live_production_retrieval"])
            self.assertTrue(report["candidate_cache"]["not_live_production_retrieval"])
            self.assertEqual(report["evaluation_mode"], EVALUATION_MODE)
            self.assertEqual(report["status"], PREFLIGHT_OK)
            self.assertEqual(report["preflight_schema_version"], PREFLIGHT_SCHEMA_VERSION)
            self.assertEqual(report["exit_code"], 0)
            self.assertIs(report["public_holdout_used"], False)
            self.assertIs(report["sealed_aggregate_modified"], False)
            self.assertIs(report["candidate_cache_modified"], False)
            parsed_at = datetime.fromisoformat(str(report["executed_at"]).replace("Z", "+00:00"))
            self.assertIsNotNone(parsed_at.tzinfo)
            self.assertEqual(parsed_at.utcoffset().total_seconds(), 0)
            self.assertFalse(report["runtime_components"]["reranker"]["enabled"])
            self.assertFalse(report["runtime_components"]["embedding"]["enabled"])
            self.assertEqual(report["remote_calls_expected"], 2)
            self.assertEqual(
                sorted(item.name for item in preflight_dir.iterdir()),
                ["preflight.json"],
            )
            self.assertFalse(official.exists())
            blob = (preflight_dir / "preflight.json").read_text(encoding="utf-8")
            self.assertNotIn("sk-test-not-for-disk", blob)
            self.assertNotIn("Authorization", blob)
            self.assertNotIn("https://", blob)
            self.assertNotIn("C:\\\\", blob)
            creds = {item["key"]: item for item in report["credentials"]}
            self.assertEqual(list(creds), ["DEEPSEEK_API_KEY"])
            self.assertTrue(creds["DEEPSEEK_API_KEY"]["present"])
            self.assertNotIn("length", creds["DEEPSEEK_API_KEY"])
            self.assertNotIn("value", creds["DEEPSEEK_API_KEY"])
            self.assertNotIn("DASHSCOPE_API_KEY", blob)

    def test_resume_skips_complete_cases_and_counts_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_ids = ["pd-1", "pd-2", "pd-3"]
            config, cases, _ = _mini_world(root, case_ids)
            out = root / "out"
            calls: list[str] = []

            def generate(case: dict) -> str:
                calls.append(str(case["case_id"]))
                return _structured_payload(int(str(case["case_id"]).split("-")[1]))

            first = _authorized_run(
                generate,
                repo_root=root,
                frozen_config=config,
                split=CLI_SPLIT,
                output_dir=out,
                preflight_output_dir=root / "preflight",
                cases=cases,
                allowlist=case_ids,
            )
            self.assertEqual(calls, case_ids)
            self.assertEqual(first["identity"]["calls_this_invocation"], 3)
            self.assertEqual(first["identity"]["calls_total"], 3)
            calls.clear()
            second = _authorized_run(
                generate,
                repo_root=root,
                frozen_config=config,
                split=CLI_SPLIT,
                output_dir=out,
                preflight_output_dir=root / "preflight",
                cases=cases,
                resume=True,
                allowlist=case_ids,
            )
            self.assertEqual(calls, [])
            self.assertEqual(second["identity"]["calls_this_invocation"], 0)
            self.assertEqual(second["identity"]["calls_total"], 3)
            self.assertEqual(second["identity"]["cases_remaining"], 0)
            self.assertIs(second["identity"]["exactly_once"], False)
            self.assertEqual(second["identity"]["billing_semantics"], "at_least_once")
            markdown = (out / "results.md").read_text(encoding="utf-8")
            self.assertIn("at_least_once", markdown)
            self.assertNotIn("exactly-once", markdown)

    def test_resume_partial_then_continue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_ids = ["pd-1", "pd-2"]
            config, cases, _ = _mini_world(root, case_ids)
            out = root / "out"
            calls: list[str] = []

            def generate_interrupt(case: dict) -> str:
                calls.append(str(case["case_id"]))
                if case["case_id"] == "pd-2":
                    raise ShadowError("injected interrupt after first complete case")
                return _structured_payload(1)

            with self.assertRaisesRegex(ShadowError, "injected interrupt"):
                _authorized_run(
                    generate_interrupt,
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=out,
                    preflight_output_dir=root / "preflight",
                    cases=cases,
                    allowlist=case_ids,
                )
            self.assertEqual(calls, ["pd-1", "pd-2"])
            calls.clear()

            def generate_resume(case: dict) -> str:
                calls.append(str(case["case_id"]))
                return _structured_payload(int(str(case["case_id"]).split("-")[1]))

            resumed = _authorized_run(
                generate_resume,
                repo_root=root,
                frozen_config=config,
                split=CLI_SPLIT,
                output_dir=out,
                preflight_output_dir=root / "preflight",
                cases=cases,
                resume=True,
                allowlist=case_ids,
            )
            self.assertEqual(calls, ["pd-2"])
            self.assertEqual(resumed["identity"]["completed_cases"], 2)
            self.assertEqual(resumed["identity"]["calls_this_invocation"], 1)
            self.assertEqual(resumed["identity"]["calls_total"], 2)

    def test_resume_rejects_three_file_fork_and_identity_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_ids = ["pd-1", "pd-2"]
            config, cases, _ = _mini_world(root, case_ids)
            out = root / "out"

            def generate(case: dict) -> str:
                return _structured_payload(int(str(case["case_id"]).split("-")[1]))

            _authorized_run(
                generate,
                repo_root=root,
                frozen_config=config,
                split=CLI_SPLIT,
                output_dir=out,
                preflight_output_dir=root / "preflight",
                cases=cases,
                allowlist=case_ids,
            )
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            checkpoint = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
            forked_manifest = dict(manifest)
            forked_manifest["completed_case_ids"] = ["pd-1"]
            forked_manifest["completed_cases"] = 1
            forked_checkpoint = dict(checkpoint)
            forked_checkpoint["completed_case_ids"] = ["pd-2"]
            forked_checkpoint["completed_cases"] = 1
            (out / "manifest.json").write_text(
                json.dumps(forked_manifest, indent=2) + "\n", encoding="utf-8"
            )
            (out / "checkpoint.json").write_text(
                json.dumps(forked_checkpoint, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ShadowError, "diverge|missing from per_case"):
                _authorized_run(
                    generate,
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=out,
                    preflight_output_dir=root / "preflight",
                    cases=cases,
                    resume=True,
                    allowlist=case_ids,
                )
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            changed = dict(manifest)
            changed["config_hash"] = "0" * 64
            (out / "manifest.json").write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
            (out / "checkpoint.json").write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ShadowError, "config_hash"):
                _authorized_run(
                    generate,
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=out,
                    preflight_output_dir=root / "preflight",
                    cases=cases,
                    resume=True,
                    allowlist=case_ids,
                )

    def test_metrics_and_paired_descriptive_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_ids = ["s1", "u1", "v1", "l1"]
            config, cases, _ = _mini_world(root, case_ids)
            cases[2]["hits"] = [_hit(3, verified=False)]
            payloads = {
                "s1": _structured_payload(1),
                "u1": _structured_payload(2, source="unavailable"),
                "v1": _structured_payload(3, source="unknown"),
                "l1": _structured_payload(4, source="legacy"),
            }

            def generate(case: dict) -> str:
                return payloads[str(case["case_id"])]

            result = _authorized_run(
                generate,
                repo_root=root,
                frozen_config=config,
                split=CLI_SPLIT,
                output_dir=root / "out",
                preflight_output_dir=root / "preflight",
                cases=cases,
                allowlist=case_ids,
            )
            summary = result["summary"]
            self.assertGreaterEqual(summary["structured_answer_present"], 1)
            self.assertIn("unavailable", summary["citation_source_distribution"])
            self.assertFalse(summary["product_accuracy_claim"])
            self.assertFalse(summary["held_out"])
            self.assertFalse(summary["benchmark_claim"])
            paired = summary["paired_vs_sealed"]
            self.assertEqual(paired["comparison_kind"], "post-hoc exposed comparison")
            self.assertFalse(paired["suitable_for_model_selection"])
            self.assertFalse(paired["held_out"])
            self.assertFalse(paired["writes_sealed_aggregate"])
            self.assertIn("structured_present_delta", paired)
            sealed_path = root / "data" / "eval_rag" / "holdout" / "ledger_public_dev_e2e_canary_5x10.json"
            self.assertEqual(
                sha256_normalized_file(sealed_path),
                config.field("sealed_baseline", "sha256"),
            )

    def test_provider_errors_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1"])

            def generate(_case: dict) -> str:
                raise RuntimeError(
                    "Authorization: Bearer sk-secret123 https://api.deepseek.com C:\\Users\\lili\\key"
                )

            result = _authorized_run(
                generate,
                repo_root=root,
                frozen_config=config,
                split=CLI_SPLIT,
                output_dir=root / "out",
                preflight_output_dir=root / "preflight",
                cases=cases,
                allowlist=["pd-1"],
            )
            blob = (root / "out" / "failures.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("sk-secret123", blob)
            self.assertNotIn("Authorization", blob)
            self.assertNotIn("https://api.deepseek.com", blob)
            self.assertNotIn("C:\\Users\\lili", blob)
            row = json.loads(blob.strip())
            self.assertIn("error_type", row)
            self.assertIn("error_category", row)
            self.assertNotIn("detail", row)
            self.assertTrue(result["summary"]["cases_failed"])
            self.assertIs(result["summary"]["exactly_once"], False)

    def test_cli_requires_fixed_flags_and_refuses_overrides(self) -> None:
        with self.assertRaisesRegex(ShadowError, "runtime parameter"):
            parse_cli_guard(["--model", "x"])
        with self.assertRaisesRegex(ShadowError, "runtime parameter"):
            parse_cli_guard(["--top-k", "20"])
        cli = _load_cli()
        self.assertEqual(
            cli.main(
                [
                    "--split",
                    "public_holdout",
                    "--frozen-config",
                    str(ROOT / DEFAULT_FROZEN_CONFIG_PATH),
                    "--confirm-exposed-shadow",
                    "--preflight-only",
                ]
            ),
            2,
        )
        self.assertEqual(
            cli.main(
                [
                    "--split",
                    "public-dev",
                    "--frozen-config",
                    str(ROOT / DEFAULT_FROZEN_CONFIG_PATH),
                    "--confirm-exposed-shadow",
                ]
            ),
            2,
        )
        self.assertEqual(
            cli.main(
                [
                    "--split",
                    "public-dev",
                    "--frozen-config",
                    str(ROOT / DEFAULT_FROZEN_CONFIG_PATH),
                    "--confirm-exposed-shadow",
                    "--preflight-only",
                    "--allow-remote",
                ]
            ),
            2,
        )
        with self.assertRaisesRegex(ShadowError, "runtime parameter"):
            parse_cli_guard(["--cases-path", "cases.json"])

    def test_dual_key_and_env_cannot_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1"])
            generate_calls = {"n": 0}

            def generate(_case: dict) -> str:
                generate_calls["n"] += 1
                return _structured_payload(1)

            with self.assertRaisesRegex(ShadowError, "confirm-exposed-shadow"):
                run_shadow(
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    confirm_exposed_shadow=False,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                    cases=cases,
                    allow_remote=True,
                    allow_injected_generate=True,
                    generate_fn=generate,
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                    allowlist=["pd-1"],
                )
            with self.assertRaisesRegex(ShadowError, "allow-remote"):
                run_shadow(
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    confirm_exposed_shadow=True,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                    cases=cases,
                    allow_remote=False,
                    allow_injected_generate=True,
                    generate_fn=generate,
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                    allowlist=["pd-1"],
                )
            with self.assertRaisesRegex(ShadowError, "CLI authorization path"):
                run_shadow(
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    confirm_exposed_shadow=True,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                    cases=cases,
                    allow_remote=True,
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                    allowlist=["pd-1"],
                )
            with patch.dict(os.environ, {"LUMENFIN_SHADOW_ALLOW_REMOTE": "1"}):
                with self.assertRaisesRegex(ShadowError, "environment variables cannot authorize"):
                    run_shadow(
                        repo_root=root,
                        frozen_config=config,
                        split=CLI_SPLIT,
                        confirm_exposed_shadow=True,
                        output_dir=root / "out",
                        preflight_output_dir=root / "preflight",
                        cases=cases,
                        allow_remote=False,
                        allow_injected_generate=True,
                        generate_fn=generate,
                        verify_tag=False,
                        require_clean=False,
                        verify_runtime=False,
                        allowlist=["pd-1"],
                    )
            with self.assertRaisesRegex(ShadowError, "allow-remote with --preflight-only"):
                run_shadow(
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    confirm_exposed_shadow=True,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                    preflight_only=True,
                    allow_remote=True,
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                )
            result = _authorized_run(
                generate,
                repo_root=root,
                frozen_config=config,
                split=CLI_SPLIT,
                output_dir=root / "out",
                preflight_output_dir=root / "preflight",
                cases=cases,
                allowlist=["pd-1"],
            )
            self.assertEqual(generate_calls["n"], 1)
            self.assertEqual(result["remote_request_count"], 0)
            self.assertEqual(result["identity"]["protocol_ancestor"], PROTOCOL_COMMIT)
            self.assertNotEqual(
                result["identity"]["execution_commit"],
                result["identity"]["protocol_ancestor"],
            )
            self.assertTrue(result["identity"]["not_live_production_retrieval"])

    def test_candidate_cache_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1", "pd-2"])
            cache_path = root / "outputs" / "cache" / "candidates.jsonl"
            original = cache_path.read_bytes()
            manifest_path = root / "data" / "eval_rag" / "structured_citation_shadow_cache_manifest.json"
            original_manifest = manifest_path.read_bytes()
            generate_calls = {"n": 0}

            def generate(_case: dict) -> str:
                generate_calls["n"] += 1
                raise AssertionError("generate must not run")

            missing_root = root / "missing"
            missing_config, missing_cases, _ = _mini_world(missing_root, ["pd-1", "pd-2"])
            (missing_root / "outputs" / "cache" / "candidates.jsonl").unlink()
            with self.assertRaisesRegex(ShadowError, "missing"):
                _authorized_run(
                    generate,
                    repo_root=missing_root,
                    frozen_config=missing_config,
                    split=CLI_SPLIT,
                    output_dir=missing_root / "out",
                    preflight_output_dir=missing_root / "preflight",
                    cases=missing_cases,
                    allowlist=["pd-1", "pd-2"],
                )
            cache_path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ShadowError, "file hash"):
                _authorized_run(
                    generate,
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                    cases=cases,
                    allowlist=["pd-1", "pd-2"],
                )
            cache_path.write_bytes(original)
            mutated_rows = json.loads(json.dumps([
                json.loads(line) for line in original.decode("utf-8").splitlines() if line.strip()
            ]))
            mutated_rows[0]["hits"][0]["chunk_id"] = "mutated-chunk"
            mutated_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mutated_rows).encode("utf-8")
            cache_path.write_bytes(mutated_bytes)
            manifest_path = root / "data" / "eval_rag" / "structured_citation_shadow_cache_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cache_file_sha256"] = sha256_raw_file(cache_path)
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            fields = json.loads((root / "frozen.json").read_text(encoding="utf-8"))
            fields["candidate_cache"]["manifest_sha256"] = sha256_normalized_file(manifest_path)
            fields["config_hash"] = compute_config_hash(fields)
            (root / "frozen.json").write_text(json.dumps(fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            mutated_config = load_frozen_config(root / "frozen.json")
            with self.assertRaisesRegex(ShadowError, "chunk id hash"):
                _authorized_run(
                    generate,
                    repo_root=root,
                    frozen_config=mutated_config,
                    split=CLI_SPLIT,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                    cases=cases,
                    allowlist=["pd-1", "pd-2"],
                )
            cache_path.write_bytes(original)
            manifest_path.write_bytes(original_manifest)
            report = verify_candidate_cache(repo_root=root, config=config)
            self.assertTrue(report["verified"])
            self.assertEqual(sha256_raw_file(cache_path), json.loads(
                (root / "data" / "eval_rag" / "structured_citation_shadow_cache_manifest.json").read_text(
                    encoding="utf-8"
                )
            )["cache_file_sha256"])
            self.assertEqual(generate_calls["n"], 0)
            self.assertFalse((root / "out").exists())

    def test_live_path_binds_cases_from_verified_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1", "pd-2"], cache_query_text=True)
            seen: list[str] = []

            def generate(case: dict) -> str:
                seen.append(str(case["case_id"]))
                self.assertTrue(str(case.get("query_text") or "").strip())
                self.assertTrue(case["hits"][0]["chunk_id"])
                return _structured_payload(int(str(case["case_id"]).split("-")[1]))

            with patch(
                "lumenfin.eval.holdout.ledger.iter_ledger_parquet_rows",
                side_effect=AssertionError("parquet must not be read"),
            ):
                result = _authorized_run(
                    generate,
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                    allowlist=["pd-1", "pd-2"],
                )
            self.assertEqual(seen, ["pd-1", "pd-2"])
            self.assertEqual(result["identity"]["cases_total"], 2)
            bound = bind_cases_from_verified_cache(
                repo_root=root,
                config=config,
                cache_report=verify_candidate_cache(repo_root=root, config=config),
                allowlist=["pd-1", "pd-2"],
            )
            self.assertEqual([item["case_id"] for item in bound], ["pd-1", "pd-2"])
            self.assertEqual(bound[0]["query_text"], cases[0]["query_text"])
            self.assertEqual(bound[0]["hits"][0]["chunk_id"], cases[0]["hits"][0]["chunk_id"])

    def test_live_bind_without_query_text_or_snapshot_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"], cache_query_text=False)
            generate_calls = {"n": 0}

            def generate(_case: dict) -> str:
                generate_calls["n"] += 1
                raise AssertionError("generate must not run")

            with self.assertRaisesRegex(ShadowError, "not auto-fetched"):
                _authorized_run(
                    generate,
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                    allowlist=["pd-1"],
                )
            self.assertEqual(generate_calls["n"], 0)
            self.assertFalse((root / "out").exists())

    def test_live_bind_missing_cache_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"], cache_query_text=True)
            (root / "outputs" / "cache" / "candidates.jsonl").unlink()
            with self.assertRaisesRegex(ShadowError, "missing"):
                _authorized_run(
                    lambda _case: (_ for _ in ()).throw(AssertionError("generate must not run")),
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                    allowlist=["pd-1"],
                )
            self.assertFalse((root / "out").exists())

    def test_live_bind_query_text_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"], cache_query_text=True)
            cache_path = root / "outputs" / "cache" / "candidates.jsonl"
            row = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0])
            row["query_text"] = "mutated query"
            cache_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest_path = root / "data" / "eval_rag" / "structured_citation_shadow_cache_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cache_file_sha256"] = sha256_raw_file(cache_path)
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            fields = json.loads((root / "frozen.json").read_text(encoding="utf-8"))
            fields["candidate_cache"]["manifest_sha256"] = sha256_normalized_file(manifest_path)
            fields["config_hash"] = compute_config_hash(fields)
            (root / "frozen.json").write_text(json.dumps(fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            mutated = load_frozen_config(root / "frozen.json")
            with self.assertRaisesRegex(ShadowError, "query text"):
                bind_cases_from_verified_cache(
                    repo_root=root,
                    config=mutated,
                    cache_report=verify_candidate_cache(repo_root=root, config=mutated),
                    allowlist=["pd-1"],
                )
            self.assertFalse((root / "out").exists())

    def test_live_bind_joins_allowlisted_public_dev_snapshot(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from lumenfin.eval.holdout.ledger import ledger_snapshot_sha256

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1"], cache_query_text=False)
            snapshot = root / public_dev_snapshot_relative(config)
            snapshot.mkdir(parents=True)
            table = pa.table(
                {
                    "query_id": ["pd-1", "holdout-secret"],
                    "query_text": [cases[0]["query_text"], "DO-NOT-BIND"],
                    "value": [cases[0]["gold_value"], 99.0],
                    "mmd_text": ["unused-dev", "unused-holdout"],
                }
            )
            pq.write_table(table, snapshot / "0000.parquet")
            cache_path = root / "outputs" / "cache" / "candidates.jsonl"
            row = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0])
            row["query_text_sha256"] = hashlib.sha256(
                str(cases[0]["query_text"]).encode("utf-8")
            ).hexdigest()
            cache_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            manifest_path = root / "data" / "eval_rag" / "structured_citation_shadow_cache_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cache_file_sha256"] = sha256_raw_file(cache_path)
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            fields = json.loads((root / "frozen.json").read_text(encoding="utf-8"))
            fields["dataset"]["source_artifact_sha256"] = ledger_snapshot_sha256(snapshot)
            fields["candidate_cache"]["manifest_sha256"] = sha256_normalized_file(manifest_path)
            fields["config_hash"] = compute_config_hash(fields)
            (root / "frozen.json").write_text(json.dumps(fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            joined_config = load_frozen_config(root / "frozen.json")
            bound = bind_cases_from_verified_cache(
                repo_root=root,
                config=joined_config,
                cache_report=verify_candidate_cache(repo_root=root, config=joined_config),
                allowlist=["pd-1"],
            )
            self.assertEqual(bound[0]["case_id"], "pd-1")
            self.assertEqual(bound[0]["query_text"], cases[0]["query_text"])
            self.assertEqual(bound[0]["gold_value"], cases[0]["gold_value"])
            self.assertNotIn("DO-NOT-BIND", json.dumps(bound))

    def test_live_bind_refuses_holdout_snapshot_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"], cache_query_text=False)
            forbidden = root / "data" / "public_holdout" / "eval-test"
            forbidden.mkdir(parents=True)
            with patch(
                "lumenfin.eval.ledger_structured_citation_shadow.public_dev_snapshot_relative",
                return_value=Path("data") / "public_holdout" / "eval-test",
            ):
                with self.assertRaisesRegex(ShadowError, "forbidden"):
                    bind_cases_from_verified_cache(
                        repo_root=root,
                        config=config,
                        cache_report=verify_candidate_cache(repo_root=root, config=config),
                        allowlist=["pd-1"],
                    )
            self.assertFalse((root / "out").exists())

    def test_strict_output_path_rejects_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1"])
            with self.assertRaisesRegex(ShadowError, "exact frozen output directory"):
                _authorized_run(
                    lambda _case: _structured_payload(1),
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=root / "outputs" / "ledger_structured_citation_shadow_v1" / "nested",
                    preflight_output_dir=root / "outputs" / "ledger_structured_citation_shadow_preflight_v2",
                    cases=cases,
                    strict_paths=True,
                    allowlist=["pd-1"],
                )

    def test_runtime_qwen3_env_does_not_break_replay_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1"])
            with patch.dict(
                os.environ,
                {
                    "MAS_RAG_RERANK_PROVIDER": "qwen3",
                    "DEEPSEEK_API_KEY": "sk-test-not-for-disk",
                },
            ):
                result = _authorized_run(
                    lambda _case: _structured_payload(1),
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=root / "out",
                    preflight_output_dir=root / "preflight",
                    cases=cases,
                    verify_runtime=True,
                    allowlist=["pd-1"],
                )
            self.assertEqual(result["identity"]["evaluation_mode"], EVALUATION_MODE)
            self.assertTrue(result["identity"]["not_live_production_retrieval"])
            blob = (root / "out" / "environment.json").read_text(encoding="utf-8")
            self.assertNotIn("sk-test-not-for-disk", blob)
            self.assertNotIn("Authorization", blob)
            env = json.loads(blob)
            self.assertEqual(env["runtime"]["runtime_components"]["reranker"]["status"], "not_applicable")
            self.assertEqual(
                env["runtime"]["runtime_components"]["reranker"]["observed_env_provider"],
                "qwen3",
            )

    def test_missing_chat_key_fails_preflight_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"])

            class _Source:
                def __init__(self, key: str) -> None:
                    self.key = key
                    self.source = "unset"
                    self.length = 0

            with patch(
                "lumenfin.eval.ledger_structured_citation_shadow.describe_credential_sources",
                return_value=[_Source("DEEPSEEK_API_KEY"), _Source("DASHSCOPE_API_KEY")],
            ):
                with self.assertRaisesRegex(ShadowError, "DEEPSEEK_API_KEY"):
                    run_shadow(
                        repo_root=root,
                        frozen_config=config,
                        split=CLI_SPLIT,
                        confirm_exposed_shadow=True,
                        output_dir=root / "out",
                        preflight_output_dir=root / "preflight",
                        preflight_only=True,
                        verify_tag=False,
                        require_clean=False,
                        verify_runtime=False,
                    )
            self.assertFalse((root / "preflight").exists())
            self.assertFalse((root / "out").exists())

    def test_runtime_retrieval_calls_fail_closed(self) -> None:
        with self.assertRaisesRegex(ShadowError, "embedding calls"):
            forbid_runtime_embedding_call()
        with self.assertRaisesRegex(ShadowError, "runtime reranker calls"):
            forbid_runtime_reranker_call()

    def test_retired_config_hash_is_not_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"])
            payload = json.loads((root / "frozen.json").read_text(encoding="utf-8"))
            for retired_hash in (
                RETIRED_BEFORE_PREFLIGHT_HASH,
                INCOMPLETE_AUDIT_CONFIG_HASH,
                SUPERSEDED_V2_CONFIG_HASH,
            ):
                payload["config_hash"] = retired_hash
                retired = root / "retired.json"
                retired.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ShadowError, "retired|does not match canonical"):
                    load_frozen_config(retired)
        self.assertEqual(
            RETIRED_CONFIG_HASHES[INCOMPLETE_AUDIT_CONFIG_HASH]["retired_reason"],
            "incomplete_preflight_audit_schema",
        )

    def test_paired_helper_does_not_claim_accuracy(self) -> None:
        shadow = summarize_rows(
            [
                {
                    "failed": False,
                    "structured_answer_present": True,
                    "citation_source": "structured",
                    "citations": ["a"],
                    "citations_total": 1,
                    "valid_citations": 1,
                    "unknown_citations": 0,
                    "unverified_citations": 0,
                    "cross_scope_citations": 0,
                    "stale_citations": 0,
                    "citation_validation_failed": False,
                    "supported_claim": True,
                    "unsupported_claim": False,
                    "claims_total": 1,
                    "outcome": "complete",
                    "latency_ms": 10,
                }
            ],
            cases_total=1,
            remote_request_count=0,
            sealed={
                "structured_answer_present": 0,
                "valid_citations": 0,
                "supported_claims": 0,
                "incomplete_or_degraded": 1,
            },
        )
        self.assertFalse(shadow["product_accuracy_claim"])
        paired = paired_descriptive_comparison(shadow, {"structured_answer_present": 0, "valid_citations": 0, "supported_claims": 0, "incomplete_or_degraded": 1})
        self.assertEqual(paired["structured_present_delta"], 1)
        self.assertFalse(paired["suitable_for_model_selection"])

    def test_preflight_integrity_failures_do_not_write_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"])
            official = root / "official"
            preflight_dir = root / "preflight"
            cache_path = root / "outputs" / "cache" / "candidates.jsonl"
            baseline_path = root / config.field("sealed_baseline", "path")
            original_cache = cache_path.read_bytes()
            original_baseline = baseline_path.read_bytes()

            real_verify = verify_candidate_cache

            def mutate_cache(**kwargs):
                report = real_verify(**kwargs)
                cache_path.write_bytes(cache_path.read_bytes() + b" ")
                return report

            with patch(
                "lumenfin.eval.ledger_structured_citation_shadow.verify_candidate_cache",
                side_effect=mutate_cache,
            ):
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-not-for-disk"}):
                    with self.assertRaisesRegex(ShadowError, "hash changed|changed after verification"):
                        run_shadow(
                            repo_root=root,
                            frozen_config=config,
                            split=CLI_SPLIT,
                            confirm_exposed_shadow=True,
                            output_dir=official,
                            preflight_output_dir=preflight_dir,
                            preflight_only=True,
                            verify_tag=False,
                            require_clean=False,
                            verify_runtime=False,
                        )
            self.assertFalse((preflight_dir / "preflight.json").exists())
            cache_path.write_bytes(original_cache)

            real_baseline = read_sealed_baseline_readonly

            def mutate_baseline(**kwargs):
                report = real_baseline(**kwargs)
                baseline_path.write_bytes(baseline_path.read_bytes() + b" ")
                return report

            with patch(
                "lumenfin.eval.ledger_structured_citation_shadow.read_sealed_baseline_readonly",
                side_effect=mutate_baseline,
            ):
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-not-for-disk"}):
                    with self.assertRaisesRegex(ShadowError, "hash changed"):
                        run_shadow(
                            repo_root=root,
                            frozen_config=config,
                            split=CLI_SPLIT,
                            confirm_exposed_shadow=True,
                            output_dir=official,
                            preflight_output_dir=root / "preflight-baseline",
                            preflight_only=True,
                            verify_tag=False,
                            require_clean=False,
                            verify_runtime=False,
                        )
            self.assertFalse((root / "preflight-baseline" / "preflight.json").exists())
            baseline_path.write_bytes(original_baseline)

            from lumenfin.eval.ledger_structured_citation_shadow import InputAccessAudit

            original_prove = InputAccessAudit.prove_holdout_unused

            def mark_holdout(self, **kwargs):
                self.mark_holdout_loader()
                return original_prove(self, **kwargs)

            with patch.object(InputAccessAudit, "prove_holdout_unused", mark_holdout):
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-not-for-disk"}):
                    with self.assertRaisesRegex(ShadowError, "public holdout access"):
                        run_shadow(
                            repo_root=root,
                            frozen_config=config,
                            split=CLI_SPLIT,
                            confirm_exposed_shadow=True,
                            output_dir=official,
                            preflight_output_dir=root / "preflight-holdout",
                            preflight_only=True,
                            verify_tag=False,
                            require_clean=False,
                            verify_runtime=False,
                        )
            self.assertFalse((root / "preflight-holdout" / "preflight.json").exists())

    def test_preflight_atomic_write_failure_leaves_no_success_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"])
            preflight_dir = root / "preflight"

            def boom(*_args, **_kwargs):
                raise OSError("disk full")

            with patch("lumenfin.eval.ledger_structured_citation_shadow.os.replace", boom):
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-not-for-disk"}):
                    with self.assertRaises(OSError):
                        run_shadow(
                            repo_root=root,
                            frozen_config=config,
                            split=CLI_SPLIT,
                            confirm_exposed_shadow=True,
                            output_dir=root / "official",
                            preflight_output_dir=preflight_dir,
                            preflight_only=True,
                            verify_tag=False,
                            require_clean=False,
                            verify_runtime=False,
                        )
            self.assertFalse((preflight_dir / "preflight.json").exists())
            self.assertFalse(any(preflight_dir.glob("*.tmp")) if preflight_dir.exists() else True)

    def test_preflight_directory_isolation_and_legacy_v1_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1"])
            with self.assertRaisesRegex(ShadowError, "exact frozen output directory"):
                _authorized_run(
                    lambda _case: _structured_payload(1),
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=root / DEFAULT_PREFLIGHT_OUTPUT_DIR,
                    preflight_output_dir=root / LEGACY_PREFLIGHT_OUTPUT_DIR,
                    cases=cases,
                    strict_paths=True,
                    allowlist=["pd-1"],
                )
            with self.assertRaisesRegex(ShadowError, "exact frozen output directory"):
                _authorized_run(
                    lambda _case: _structured_payload(1),
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    output_dir=root / "outputs" / "ledger_structured_citation_shadow_v1",
                    preflight_output_dir=root
                    / "outputs"
                    / "ledger_structured_citation_shadow_preflight_v2-extra",
                    cases=cases,
                    strict_paths=True,
                    allowlist=["pd-1"],
                )
            self.assertFalse((root / "outputs" / "ledger_structured_citation_shadow_v1").exists())
            self.assertFalse((root / DEFAULT_PREFLIGHT_OUTPUT_DIR).exists())
        ledger = RETIRED_CONFIG_HASHES[INCOMPLETE_AUDIT_CONFIG_HASH]
        self.assertEqual(ledger["artifact_sha256"], INCOMPLETE_V1_PREFLIGHT_SHA256)
        self.assertEqual(ledger["artifact_status"], "INCOMPLETE_PREFLIGHT_AUDIT_SCHEMA")
        self.assertEqual(ledger["preflight_executions"], 1)
        self.assertEqual(ledger["accepted_preflights"], 0)
        self.assertIs(ledger["accepted_for_shadow_execution"], False)
        fields = published_frozen_config_fields()
        self.assertEqual(fields["predecessor_config"]["artifact_sha256"], V2_PREFLIGHT_SHA256)
        self.assertEqual(fields["output"]["preflight_dirname"], DEFAULT_PREFLIGHT_OUTPUT_DIR.name)
        self.assertEqual(fields["output"]["legacy_preflight_dirname"], LEGACY_PREFLIGHT_OUTPUT_DIR.name)
        self.assertEqual(fields["output"]["superseded_preflight_dirname"], SUPERSEDED_PREFLIGHT_OUTPUT_DIR.name)
        v1 = ROOT / LEGACY_PREFLIGHT_OUTPUT_DIR / "preflight.json"
        if v1.is_file():
            digest = hashlib.sha256(v1.read_bytes()).hexdigest()
            self.assertEqual(digest, INCOMPLETE_V1_PREFLIGHT_SHA256)
            self.assertEqual(v1.stat().st_size, 3920)

    def test_cli_returns_zero_after_successful_preflight_write(self) -> None:
        cli = _load_cli()
        with patch.object(cli, "run_shadow", return_value={"status": PREFLIGHT_OK, "exit_code": 0}):
            self.assertEqual(
                cli.main(
                    [
                        "--split",
                        "public-dev",
                        "--frozen-config",
                        str(DEFAULT_FROZEN_CONFIG_PATH),
                        "--confirm-exposed-shadow",
                        "--preflight-only",
                    ]
                ),
                0,
            )

    def test_official_cli_has_no_cases_path_flag(self) -> None:
        cli = _load_cli()
        flags = [action.option_strings for action in cli.build_parser()._actions]
        flat = {item for group in flags for item in group}
        self.assertNotIn("--cases-path", flat)

    def test_gold_sentinel_never_enters_provider_request(self) -> None:
        sentinel = 555666777.888
        label = "SENTINEL_GOLD_LABEL_ZX9"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1", "pd-2"])
            cases[0]["gold_value"] = sentinel
            cases[0]["gold_label"] = label
            cases[1]["gold_value"] = sentinel + 1
            captured: list[str] = []

            def generate(case: dict) -> str:
                prompt = build_generation_prompt(
                    query_text=str(case.get("query_text") or ""),
                    hits=list(case.get("hits") or []),
                    max_document_chars=4000,
                )
                request = json.dumps(
                    {
                        "system": "shadow",
                        "user": prompt,
                        "case": case,
                    },
                    ensure_ascii=False,
                )
                captured.append(request)
                self.assertNotIn(str(sentinel), request)
                self.assertNotIn(label, request)
                self.assertNotIn("gold_value", request.casefold())
                self.assertNotIn("expected_answer", request.casefold())
                generation_case_view(case)
                return _structured_payload(int(str(case["case_id"]).split("-")[1]))

            result = _authorized_run(
                generate,
                repo_root=root,
                frozen_config=config,
                split=CLI_SPLIT,
                output_dir=root / "out",
                preflight_output_dir=root / "preflight",
                cases=cases,
                allowlist=["pd-1", "pd-2"],
            )
            self.assertEqual(len(captured), 2)
            self.assertNotIn(str(sentinel), "".join(captured))
            self.assertGreaterEqual(result["summary"]["cases_succeeded"], 0)
            params = inspect.signature(build_generation_prompt).parameters
            self.assertNotIn("gold_value", params)
            self.assertNotIn("gold", params)
            self.assertNotIn("expected_answer", params)

    def test_snapshot_duplicate_allowlist_id_fails_closed(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from lumenfin.eval.holdout.ledger import ledger_snapshot_sha256

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1"], cache_query_text=False)
            snapshot = root / public_dev_snapshot_relative(config)
            snapshot.mkdir(parents=True)
            table = pa.table(
                {
                    "query_id": ["pd-1", "pd-1"],
                    "query_text": [cases[0]["query_text"], cases[0]["query_text"]],
                    "value": [cases[0]["gold_value"], cases[0]["gold_value"]],
                }
            )
            pq.write_table(table, snapshot / "0000.parquet")
            fields = json.loads((root / "frozen.json").read_text(encoding="utf-8"))
            fields["dataset"]["source_artifact_sha256"] = ledger_snapshot_sha256(snapshot)
            fields["config_hash"] = compute_config_hash(fields)
            (root / "frozen.json").write_text(json.dumps(fields, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            mutated = load_frozen_config(root / "frozen.json")
            with self.assertRaisesRegex(ShadowError, "duplicate"):
                bind_cases_from_verified_cache(
                    repo_root=root,
                    config=mutated,
                    cache_report=verify_candidate_cache(repo_root=root, config=mutated),
                    allowlist=["pd-1"],
                )
            self.assertFalse((root / "out").exists())

    def test_gold_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1"])
            sealed = {
                "selection": {"gold_identity_sha256": "0" * 64},
            }
            with self.assertRaisesRegex(ShadowError, "gold identity"):
                bind_cases_from_verified_cache(
                    repo_root=root,
                    config=config,
                    cache_report=verify_candidate_cache(repo_root=root, config=config),
                    allowlist=["pd-1"],
                    sealed=sealed,
                )

    def test_v2_preflight_cannot_authorize_new_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v2 = root / SUPERSEDED_PREFLIGHT_OUTPUT_DIR
            v2.mkdir(parents=True)
            (v2 / "preflight.json").write_text(
                json.dumps(
                    {
                        "kind": "preflight",
                        "status": PREFLIGHT_OK,
                        "execution_commit": V2_PREFLIGHT_EXECUTION_COMMIT,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ShadowError, "v2 preflight cannot authorize"):
                assert_preflight_authorizes_shadow(
                    repo_root=root,
                    execution_commit="deadbeef" * 5,
                )
            self.assertFalse((root / DEFAULT_PREFLIGHT_OUTPUT_DIR).exists())
            self.assertFalse((root / "outputs" / "ledger_structured_citation_shadow_v1").exists())

    def test_v3_directory_is_fixed_and_preflight_binds_without_remote(self) -> None:
        self.assertEqual(
            DEFAULT_PREFLIGHT_OUTPUT_DIR.name,
            "ledger_structured_citation_shadow_preflight_v3",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1", "pd-2"])
            official = root / "outputs" / "ledger_structured_citation_shadow_v1"
            preflight_dir = root / DEFAULT_PREFLIGHT_OUTPUT_DIR
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-not-for-disk"}, clear=False):
                report = run_shadow(
                    repo_root=root,
                    frozen_config=config,
                    split=CLI_SPLIT,
                    confirm_exposed_shadow=True,
                    output_dir=official,
                    preflight_output_dir=preflight_dir,
                    preflight_only=True,
                    verify_tag=False,
                    require_clean=False,
                    verify_runtime=False,
                    strict_paths=True,
                )
            self.assertEqual(report["cases_executed"], 0)
            self.assertEqual(report["remote_request_count"], 0)
            self.assertIs(report["case_binding_verified"], True)
            self.assertEqual(report["case_count"], 2)
            self.assertIs(report["gold_not_exposed_to_generator"], True)
            self.assertFalse(official.exists())
            self.assertEqual(sorted(item.name for item in preflight_dir.iterdir()), ["preflight.json"])

    def test_official_prefix_binds_fifty_when_local_artifacts_present(self) -> None:
        cache = ROOT / "outputs" / "ledger_public_dev_qwen3_paired_5x50_v3" / "candidates.jsonl"
        snapshot = ROOT / public_dev_snapshot_relative(
            load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        )
        if not cache.is_file() or not snapshot.exists():
            self.skipTest("local sealed cache or public/dev snapshot is not in this checkout")
        config = load_frozen_config(ROOT / DEFAULT_FROZEN_CONFIG_PATH, require_published=True)
        sealed = read_sealed_baseline_readonly(repo_root=ROOT, config=config)
        bound = bind_cases_from_verified_cache(
            repo_root=ROOT,
            config=config,
            cache_report=verify_candidate_cache(repo_root=ROOT, config=config),
            allowlist=None,
            sealed=sealed,
        )
        self.assertEqual(len(bound), 50)
        self.assertEqual(ids_sha256([item["case_id"] for item in bound]), config.field("case_selection", "query_ids_sha256"))
        views = [generation_case_view(item) for item in bound]
        blob = json.dumps(views, ensure_ascii=False)
        self.assertNotIn("gold_value", blob)
        self.assertNotIn("qrels", blob)


if __name__ == "__main__":
    unittest.main()
