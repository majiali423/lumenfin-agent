from __future__ import annotations

import importlib.util
import json
import os
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
    CANONICAL_SPLIT,
    CLI_SPLIT,
    DEFAULT_FROZEN_CONFIG_PATH,
    PREVIOUS_UNUSED_CONFIG_HASH,
    PROTOCOL_COMMIT,
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
    load_case_fixture,
    load_frozen_config,
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


def _mini_world(tmp: Path, case_ids: list[str]) -> tuple[FrozenShadowConfig, list[dict], Path]:
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
        cache_rows.append(
            {
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
        )
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
        self.assertNotEqual(
            loaded.config_hash,
            PREVIOUS_UNUSED_CONFIG_HASH,
        )
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
            self.assertEqual(report["protocol_ancestor"], PROTOCOL_COMMIT)
            self.assertTrue(report["not_live_production_retrieval"])
            self.assertTrue(report["candidate_cache"]["not_live_production_retrieval"])
            self.assertEqual(report["remote_calls_expected"], 2)
            self.assertEqual(
                sorted(item.name for item in preflight_dir.iterdir()),
                ["preflight.json"],
            )
            self.assertFalse(official.exists())
            blob = (preflight_dir / "preflight.json").read_text(encoding="utf-8")
            self.assertNotIn("sk-test-not-for-disk", blob)
            self.assertNotIn("Authorization", blob)
            creds = {item["key"]: item for item in report["credentials"]}
            self.assertTrue(creds["DEEPSEEK_API_KEY"]["present"])
            self.assertNotIn("length", creds["DEEPSEEK_API_KEY"])
            self.assertNotIn("value", creds["DEEPSEEK_API_KEY"])

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
                    preflight_output_dir=root / "outputs" / "ledger_structured_citation_shadow_preflight_v1",
                    cases=cases,
                    strict_paths=True,
                    allowlist=["pd-1"],
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


if __name__ == "__main__":
    unittest.main()
