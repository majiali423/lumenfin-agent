from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.structured_citation_canary import (
    CASE_IDS,
    SUITE,
    CanaryError,
    NetworkProbe,
    canonical_config,
    case_manifest_hash,
    config_hash,
    gates_passed,
    parse_cli_guard,
    run_canary,
    run_cases,
    sha256_file,
)
from lumenfin.rag.chunking import chunk_document
from lumenfin.structured_answer import STRUCTURED_ANSWER_SCHEMA_VERSION


def _load_cli():
    path = ROOT / "scripts" / "run_structured_citation_canary.py"
    spec = importlib.util.spec_from_file_location("run_structured_citation_canary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load structured citation canary CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StructuredCitationCanaryTests(unittest.TestCase):
    def test_config_hash_is_reproducible_and_locked(self) -> None:
        first = config_hash()
        second = config_hash(canonical_config())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(canonical_config()["suite"], SUITE)
        self.assertEqual(canonical_config()["network_allowed"], False)
        self.assertEqual(
            canonical_config()["structured_answer_schema_version"],
            STRUCTURED_ANSWER_SCHEMA_VERSION,
        )
        self.assertNotEqual(first, config_hash({**canonical_config(), "network_allowed": True}))

    def test_cli_refuses_remote_and_public_holdout(self) -> None:
        with self.assertRaisesRegex(CanaryError, "allow-remote"):
            parse_cli_guard(["--allow-remote"])
        with self.assertRaisesRegex(CanaryError, "public_holdout"):
            parse_cli_guard(["--output-dir", "outputs/public_holdout"])
        with self.assertRaisesRegex(CanaryError, "FinanceBench"):
            parse_cli_guard(["--seal-path", "data/eval_rag/financebench/confirmation_result.json"])
        cli = _load_cli()
        self.assertEqual(cli.main(["--allow-remote", "--allow-dirty"]), 2)

    def test_refuses_nonempty_output_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            output_dir.mkdir()
            (output_dir / "prior.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(CanaryError, "overwrite"):
                run_canary(
                    output_dir=output_dir,
                    repo_root=ROOT,
                    require_clean_worktree=False,
                )
            self.assertEqual((output_dir / "prior.json").read_text(encoding="utf-8"), "{}")

    def test_each_synthetic_case_passes_and_mutations_are_caught(self) -> None:
        probe = NetworkProbe()
        with probe:
            results, trace = run_cases(probe=probe)
        by_id = {item.case_id: item for item in results}
        self.assertEqual(list(by_id), list(CASE_IDS))
        failed = [item.case_id for item in results if not item.passed]
        self.assertEqual(failed, [], [item.detail for item in results if not item.passed])
        self.assertEqual(probe.remote_request_count, 0)
        self.assertGreaterEqual(trace.counts["chunk_document"], 1)
        self.assertGreaterEqual(trace.counts["claim_binder"], 1)
        self.assertGreaterEqual(trace.counts["synthesizer"], 1)
        self.assertGreaterEqual(trace.counts["export_finrun_state"], 1)
        self.assertGreaterEqual(trace.counts["parse_answer_payload"], 1)
        self.assertGreaterEqual(by_id["C_repair_stale_evidence"].invalid_citations_detected, 1)
        self.assertEqual(by_id["C_repair_stale_evidence"].invalid_citations_missed, 0)
        self.assertGreaterEqual(by_id["E_cross_tenant"].invalid_citations_detected, 1)
        self.assertGreaterEqual(by_id["F_cross_session"].invalid_citations_detected, 1)
        self.assertGreaterEqual(by_id["G_unknown_chunk"].invalid_citations_detected, 1)
        self.assertGreaterEqual(by_id["H_unverified_evidence"].invalid_citations_detected, 1)
        self.assertGreaterEqual(by_id["I_conflicting_metadata"].invalid_citations_detected, 1)
        self.assertEqual(by_id["K_unsupported_claim"].unsupported_claims_detected, 1)
        self.assertTrue(by_id["L_api_atomicity"].api_atomicity_passed)
        self.assertEqual(by_id["J_legacy_structured"].citation_source, "legacy_structured")

    def test_production_chunk_document_is_required(self) -> None:
        with patch(
            "lumenfin.eval.structured_citation_canary.chunk_document",
            side_effect=AssertionError("parallel fake chunker"),
        ):
            results, _trace = run_cases()
        self.assertFalse(results[0].passed)
        self.assertIn("parallel fake chunker", results[0].detail)

    def test_finrun_and_ledger_roundtrip_and_api_atomicity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "canary"
            with patch(
                "lumenfin.eval.structured_citation_canary.git_snapshot",
                return_value={
                    "lumenfin_commit": "deadbeef",
                    "worktree_dirty": True,
                    "worktree_status": "dirty",
                },
            ):
                result = run_canary(
                    output_dir=output_dir,
                    repo_root=ROOT,
                    require_clean_worktree=False,
                )
            metrics = result["metrics"]
            self.assertTrue(gates_passed(metrics))
            self.assertTrue(metrics["api_atomicity_passed"])
            self.assertTrue(metrics["finrun_roundtrip_passed"])
            self.assertTrue(metrics["ledger_roundtrip_passed"])
            self.assertEqual(metrics["remote_request_count"], 0)
            self.assertEqual(metrics["cases_failed"], 0)
            self.assertFalse(result["product_accuracy_claim"])
            self.assertFalse(result["public_holdout_used"])
            self.assertFalse(result["sealed_aggregate_modified"])
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["config_hash"], config_hash())
            self.assertEqual(summary["case_manifest_hash"], case_manifest_hash())
            for artifact in summary["raw_artifacts"]:
                path = output_dir / artifact["name"]
                self.assertTrue(path.exists(), artifact["name"])
                self.assertEqual(sha256_file(path), artifact["sha256"], artifact["name"])

    def test_network_probe_counts_blocked_connect(self) -> None:
        probe = NetworkProbe()
        with probe:
            with self.assertRaises(OSError):
                import socket

                socket.create_connection(("example.invalid", 443), timeout=0.1)
        self.assertGreaterEqual(probe.remote_request_count, 1)

    def test_official_run_requires_clean_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "lumenfin.eval.structured_citation_canary.git_snapshot",
                return_value={
                    "lumenfin_commit": "deadbeef",
                    "worktree_dirty": True,
                    "worktree_status": "dirty",
                },
            ):
                with self.assertRaisesRegex(CanaryError, "clean git worktree"):
                    run_canary(
                        output_dir=Path(tmp) / "blocked",
                        repo_root=ROOT,
                        require_clean_worktree=True,
                    )


class StructuredCitationCanaryProductionPathTests(unittest.TestCase):
    def test_success_path_calls_real_chunk_document(self) -> None:
        original = chunk_document
        calls = {"n": 0}

        def wrapped(document, **kwargs):
            calls["n"] += 1
            return original(document, **kwargs)

        with patch("lumenfin.eval.structured_citation_canary.chunk_document", wrapped):
            results, _trace = run_cases()
        self.assertTrue(all(item.passed for item in results))
        self.assertGreaterEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
