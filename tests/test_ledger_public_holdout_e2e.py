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

from lumenfin.eval.ledger_public_holdout_e2e import (
    AUTHORIZATION_PATH,
    BLOCK_REASON,
    CLAIM,
    CONTRACT_PATH,
    PRODUCT_COMMIT,
    PRODUCT_TAG,
    RESULT_PATH,
    HoldoutE2EError,
    apply_alias_and_validate,
    blocked_preflight_payload,
    evaluate_verified_e2e_case,
    execution_authorized,
    fixture_hits,
    load_authorization,
    load_contract,
    refuse_unauthorized,
    summarize_cases,
    wilson_interval,
)


class ContractAndBlockTests(unittest.TestCase):
    def test_contract_is_locked_and_blocked(self) -> None:
        contract = load_contract(repo_root=ROOT)
        self.assertEqual(contract["claim"], CLAIM)
        self.assertEqual(contract["product"]["product_commit"], PRODUCT_COMMIT)
        self.assertEqual(contract["product"]["release_tag"], PRODUCT_TAG)
        self.assertIs(contract["llm_judge_as_primary"], False)
        self.assertIs(contract["index_gate"]["compatible_prebuilt_index_present"], False)
        self.assertEqual(contract["call_budget"]["document_reembedding_calls"], 0)
        self.assertEqual(contract["retrieval"]["final_k"], 10)
        auth = load_authorization(repo_root=ROOT)
        record = auth["records"][contract["config_hash"]]
        self.assertEqual(record["reason"], BLOCK_REASON)
        self.assertIs(record["holdout_consumed"], False)
        self.assertIs(record["dataset_consumed"], False)
        self.assertFalse(execution_authorized(contract, repo_root=ROOT, want="preflight"))
        self.assertFalse(execution_authorized(contract, repo_root=ROOT, want="remote"))

    def test_cli_and_preflight_stay_blocked_without_holdout_io(self) -> None:
        contract = load_contract(repo_root=ROOT)
        with self.assertRaisesRegex(HoldoutE2EError, "no compatible prebuilt index"):
            refuse_unauthorized(repo_root=ROOT, argv=["--preflight-only"])
        with self.assertRaisesRegex(HoldoutE2EError, "runtime overrides"):
            refuse_unauthorized(repo_root=ROOT, argv=["--limit", "5"])
        payload = blocked_preflight_payload(contract)
        self.assertEqual(payload["status"], "PREFLIGHT_BLOCKED")
        self.assertEqual(payload["cases_executed"], 0)
        self.assertEqual(payload["remote_request_count"], 0)
        self.assertIs(payload["holdout_content_logged"], False)
        self.assertIs(payload["holdout_consumed"], False)
        cli = ROOT / "scripts" / "run_ledger_public_holdout_e2e.py"
        completed = subprocess.run(
            [sys.executable, str(cli), "--preflight-only"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        printed = json.loads(completed.stdout)
        self.assertEqual(printed["status"], "PREFLIGHT_BLOCKED")
        self.assertEqual(printed["remote_request_count"], 0)
        self.assertNotIn("query_text", completed.stdout)
        self.assertNotIn("gold", completed.stdout.casefold())

    def test_tracked_result_is_blocked_and_readable_without_raw_outputs(self) -> None:
        payload = json.loads((ROOT / RESULT_PATH).read_text(encoding="utf-8"))
        self.assertEqual(payload["seal_status"], "BLOCKED_BEFORE_PREFLIGHT")
        self.assertEqual(payload["block_reason"], BLOCK_REASON)
        self.assertIs(payload["holdout_consumed"], False)
        self.assertIs(payload["general_product_accuracy_claim"], False)
        self.assertEqual(payload["official_preflight_executions"], 0)
        self.assertEqual(payload["official_remote_executions"], 0)
        for rel in (CONTRACT_PATH, AUTHORIZATION_PATH, RESULT_PATH):
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", rel.as_posix()],
                cwd=ROOT,
                check=False,
            )
            self.assertNotEqual(ignored.returncode, 0)


class MetricMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hits = fixture_hits()
        self.gold_doc = self.hits[0]["document_id"]
        self.qrels = {self.gold_doc: 2, self.hits[1]["document_id"]: 0}

    def _alias_ok(self) -> dict:
        return apply_alias_and_validate(["E01"], self.hits)

    def test_numeric_unit_period_direction_and_incomplete(self) -> None:
        mapped = self._alias_ok()
        numeric_wrong = evaluate_verified_e2e_case(
            predicted=9.0,
            gold=3.0,
            abstain=False,
            citations=mapped["citations"],
            hits=mapped["hits"],
            qrels=self.qrels,
            provider_success=True,
            structured_contract_valid=True,
        )
        self.assertFalse(numeric_wrong["verified_e2e_success"])
        self.assertFalse(numeric_wrong["answer_correct"])

        unit_wrong = evaluate_verified_e2e_case(
            predicted=3.0,
            gold=7.0,
            abstain=False,
            citations=mapped["citations"],
            hits=mapped["hits"],
            qrels=self.qrels,
            provider_success=True,
            structured_contract_valid=True,
        )
        self.assertFalse(unit_wrong["answer_correct"])

        period_wrong = evaluate_verified_e2e_case(
            predicted=3.0,
            gold=3.0,
            abstain=False,
            citations=mapped["citations"],
            hits=mapped["hits"],
            qrels={self.hits[8]["document_id"]: 2},
            provider_success=True,
            structured_contract_valid=True,
        )
        self.assertTrue(period_wrong["answer_correct"])
        self.assertFalse(period_wrong["citation_supported"])
        self.assertFalse(period_wrong["verified_e2e_success"])

        direction_wrong = evaluate_verified_e2e_case(
            predicted=-3.0,
            gold=3.0,
            abstain=False,
            citations=mapped["citations"],
            hits=mapped["hits"],
            qrels=self.qrels,
            provider_success=True,
            structured_contract_valid=True,
        )
        self.assertFalse(direction_wrong["answer_correct"])

        incomplete = evaluate_verified_e2e_case(
            predicted=None,
            gold=3.0,
            abstain=True,
            citations=[],
            hits=mapped["hits"],
            qrels=self.qrels,
            provider_success=True,
            structured_contract_valid=True,
            incomplete_data=True,
        )
        self.assertEqual(incomplete["answer_metric_status"], "INCOMPLETE_DATA")
        self.assertFalse(incomplete["verified_e2e_success"])

    def test_citation_mutations_are_caught(self) -> None:
        deleted = apply_alias_and_validate([], self.hits)
        self.assertFalse(deleted["structured_contract_valid"])
        unknown = apply_alias_and_validate(["E99"], self.hits)
        self.assertGreater(unknown["unknown_aliases"], 0)
        raw = apply_alias_and_validate([self.hits[0]["chunk_id"]], self.hits)
        self.assertGreater(raw["raw_chunk_id_leakage"], 0)
        wrong_doc = apply_alias_and_validate(["E02"], self.hits)
        scored = evaluate_verified_e2e_case(
            predicted=3.0,
            gold=3.0,
            abstain=False,
            citations=wrong_doc["citations"],
            hits=wrong_doc["hits"],
            qrels=self.qrels,
            provider_success=True,
            structured_contract_valid=True,
        )
        self.assertFalse(scored["citation_supported"])
        missing_qrels = evaluate_verified_e2e_case(
            predicted=3.0,
            gold=3.0,
            abstain=False,
            citations=self._alias_ok()["citations"],
            hits=self.hits,
            qrels=None,
            provider_success=True,
            structured_contract_valid=True,
        )
        self.assertEqual(missing_qrels["qrels_status"], "NOT_EVALUABLE")
        self.assertFalse(missing_qrels["verified_e2e_success"])

    def test_cross_tenant_stale_and_unsupported_correct_answer(self) -> None:
        mapped = self._alias_ok()
        mixed = [dict(hit) for hit in self.hits]
        mixed[0]["tenant_id"] = "other"
        with self.assertRaises(Exception):
            apply_alias_and_validate(["E01"], mixed, tenant_id="t1")
        stale_hits = [dict(hit) for hit in self.hits]
        stale_hits[0]["stale_repair_attempt"] = True
        stale = apply_alias_and_validate(["E01"], stale_hits)
        self.assertTrue(stale["citation_validation_failed"] or not stale["structured_contract_valid"])
        unsupported = evaluate_verified_e2e_case(
            predicted=3.0,
            gold=3.0,
            abstain=False,
            citations=mapped["citations"],
            hits=mapped["hits"],
            qrels={self.hits[9]["document_id"]: 2},
            provider_success=True,
            structured_contract_valid=True,
        )
        self.assertTrue(unsupported["answer_correct"])
        self.assertFalse(unsupported["citation_supported"])
        self.assertFalse(unsupported["verified_e2e_success"])

    def test_wilson_and_strict_denominator(self) -> None:
        interval = wilson_interval(1, 2)
        self.assertEqual(interval["n"], 2)
        self.assertGreater(interval["high"], interval["low"])
        rows = [
            {"verified_e2e_success": True, "answer_metric_status": "EVALUABLE", "qrels_status": "BOUND", "provider_success": True},
            {"verified_e2e_success": False, "answer_metric_status": "NOT_EVALUABLE", "qrels_status": "NOT_EVALUABLE", "provider_success": True},
        ]
        summary = summarize_cases(rows)
        self.assertEqual(summary["strict_verified_e2e_success"]["successes"], 1)
        self.assertEqual(summary["strict_verified_e2e_success"]["n"], 2)
        self.assertEqual(summary["evaluable_cases"], 1)
        self.assertIs(summary["general_product_accuracy_claim"], False)


if __name__ == "__main__":
    unittest.main()
