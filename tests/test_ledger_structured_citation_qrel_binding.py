from __future__ import annotations

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

from lumenfin.eval.holdout.ledger_e2e import citation_supported
from lumenfin.eval.ledger_structured_citation_shadow import (
    CLAIM_SUPPORT_NOT_EVALUABLE,
    SUPPORT_INVALID_QRELS_NOT_BOUND,
    SUPPORT_METRIC_CONTRACT_VERSION,
    ShadowError,
    attach_evaluator_qrels,
    bind_cases_from_verified_cache,
    bind_citation_contract,
    citation_support_metric,
    evaluation_case_view,
    generation_case_view,
    normalize_evaluator_qrels,
    qrels_identity_sha256,
    score_case,
    summarize_rows,
    verify_candidate_cache,
)
from lumenfin.finrun import export_finrun_state
from lumenfin.structured_answer import STRUCTURED_ANSWER_SCHEMA_VERSION

from tests.test_ledger_structured_citation_shadow import (
    _authorized_run,
    _case,
    _mini_world,
    _structured_payload,
)


class LedgerStructuredCitationQrelBindingTests(unittest.TestCase):
    def test_normalize_snapshot_list_and_reject_empty(self) -> None:
        qrels = normalize_evaluator_qrels(
            [{"doc_id": "NYSE_MLR_2017/page_0004", "relevance": 1}, {"doc_id": "other", "relevance": 0}],
            case_id="x",
        )
        self.assertEqual(qrels["NYSE_MLR_2017/page_0004"], 1)
        with self.assertRaisesRegex(ShadowError, "empty"):
            normalize_evaluator_qrels([], case_id="x")
        with self.assertRaisesRegex(ShadowError, "missing"):
            normalize_evaluator_qrels(None, case_id="x")
        with self.assertRaisesRegex(ShadowError, "no positive"):
            normalize_evaluator_qrels([{"doc_id": "doc", "relevance": 0}], case_id="x")
        with self.assertRaisesRegex(ShadowError, "invalid"):
            normalize_evaluator_qrels("bad", case_id="x")

    def test_qrels_identity_hash_is_stable(self) -> None:
        first = qrels_identity_sha256({"b": {"d2": 0, "d1": 1}, "a": {"d0": 2}})
        second = qrels_identity_sha256({"a": {"d0": 2}, "b": {"d1": 1, "d2": 0}})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_citation_supported_membership_unchanged(self) -> None:
        hits = [{"chunk_id": "c1", "document_id": "gold-doc"}]
        self.assertTrue(citation_supported(["c1"], hits, {"gold-doc": 1}))
        self.assertFalse(citation_supported(["c1"], hits, {"other": 1}))

    def test_score_supported_and_unsupported_with_bound_qrels(self) -> None:
        gold = _case("pd-1", 1)
        gold = attach_evaluator_qrels(gold, gold["qrels"])
        contract = bind_citation_contract(gold)
        alias = contract["alias_map"].alias_for(gold["hits"][0]["chunk_id"])
        supported = score_case(
            gold,
            raw=_structured_payload(1, citations=[alias]),
            latency_ms=1,
            generate_attempts=1,
            remote_calls=1,
            contract=contract,
        )
        self.assertTrue(supported["support_metric_valid"])
        self.assertTrue(supported["supported_claim"])
        self.assertFalse(supported["unsupported_claim"])
        self.assertEqual(supported["claim_support"], "supported")

        wrong = attach_evaluator_qrels(_case("pd-1", 1), {"other-doc": 1})
        wrong_contract = bind_citation_contract(wrong)
        wrong_alias = wrong_contract["alias_map"].alias_for(wrong["hits"][0]["chunk_id"])
        unsupported = score_case(
            wrong,
            raw=_structured_payload(1, citations=[wrong_alias]),
            latency_ms=1,
            generate_attempts=1,
            remote_calls=1,
            contract=wrong_contract,
        )
        self.assertTrue(unsupported["support_metric_valid"])
        self.assertFalse(unsupported["supported_claim"])
        self.assertTrue(unsupported["unsupported_claim"])
        self.assertEqual(unsupported["claim_support"], "unsupported")

    def test_missing_or_empty_qrels_are_not_evaluable(self) -> None:
        missing = _case("pd-1", 1)
        missing["qrels"] = {}
        missing["qrels_bound"] = False
        row = score_case(
            missing,
            raw=_structured_payload(1),
            latency_ms=1,
            generate_attempts=1,
            remote_calls=1,
        )
        self.assertFalse(row["support_metric_valid"])
        self.assertEqual(row["support_metric_invalid_reason"], SUPPORT_INVALID_QRELS_NOT_BOUND)
        self.assertFalse(row["supported_claim"])
        self.assertFalse(row["unsupported_claim"])
        self.assertEqual(row["claim_support"], CLAIM_SUPPORT_NOT_EVALUABLE)
        summary = summarize_rows([row], cases_total=1, remote_request_count=0)
        self.assertFalse(summary["support_metric_valid"])
        self.assertIsNone(summary["citation_support_rate"])
        self.assertEqual(summary["unsupported_claims"], 0)

        unbound = _case("pd-1", 1)
        unbound["qrels_bound"] = False
        unbound_row = score_case(
            unbound,
            raw=_structured_payload(1),
            latency_ms=1,
            generate_attempts=1,
            remote_calls=1,
        )
        self.assertFalse(unbound_row["support_metric_valid"])
        self.assertFalse(unbound_row["unsupported_claim"])
        self.assertEqual(unbound_row["claim_support"], CLAIM_SUPPORT_NOT_EVALUABLE)

    def test_cache_qrels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"])
            cache_path = root / "outputs" / "cache" / "candidates.jsonl"
            row = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0])
            row["qrels"] = {"forged": 1}
            cache_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest_path = root / "data" / "eval_rag" / "structured_citation_shadow_cache_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            from lumenfin.eval.ledger_structured_citation_shadow import (
                compute_config_hash,
                load_frozen_config,
                sha256_normalized_file,
                sha256_raw_file,
            )
            manifest["cache_file_sha256"] = sha256_raw_file(cache_path)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            fields = json.loads((root / "frozen.json").read_text(encoding="utf-8"))
            fields["candidate_cache"]["manifest_sha256"] = sha256_normalized_file(manifest_path)
            fields["config_hash"] = compute_config_hash(fields)
            (root / "frozen.json").write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")
            mutated = load_frozen_config(root / "frozen.json")
            with self.assertRaisesRegex(ShadowError, "cache must not supply qrels"):
                bind_cases_from_verified_cache(
                    repo_root=root,
                    config=mutated,
                    cache_report=verify_candidate_cache(repo_root=root, config=mutated),
                    allowlist=["pd-1"],
                )

    def test_case_id_mismatch_and_gold_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(root, ["pd-1"])
            with self.assertRaisesRegex(ShadowError, "gold identity"):
                bind_cases_from_verified_cache(
                    repo_root=root,
                    config=config,
                    cache_report=verify_candidate_cache(repo_root=root, config=config),
                    allowlist=["pd-1"],
                    sealed={"selection": {"gold_identity_sha256": "0" * 64}},
                )

    def test_no_network_fallback_for_missing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, _cases, _ = _mini_world(
                root, ["pd-1"], cache_query_text=True, write_snapshot=False
            )
            with self.assertRaisesRegex(ShadowError, "not auto-fetched"):
                bind_cases_from_verified_cache(
                    repo_root=root,
                    config=config,
                    cache_report=verify_candidate_cache(repo_root=root, config=config),
                    allowlist=["pd-1"],
                )

    def test_generation_and_finrun_hide_qrel_sentinels(self) -> None:
        sentinel_doc = "SENTINEL_QREL_DOC_ZX9"
        sentinel_value = 555666777.888
        expected = "SENTINEL_EXPECTED_ANSWER_ZX9"
        case = attach_evaluator_qrels(
            _case("pd-1", 1, gold_value=sentinel_value, expected_answer=expected),
            {sentinel_doc: 1},
        )
        view = generation_case_view(case)
        blob = json.dumps(view, ensure_ascii=False)
        self.assertNotIn(sentinel_doc, blob)
        self.assertNotIn(str(sentinel_value), blob)
        self.assertNotIn(expected, blob)
        self.assertNotIn('"qrels":', blob)
        ev = evaluation_case_view(case)
        self.assertIn(sentinel_doc, ev["qrels"])
        self.assertEqual(citation_support_metric(case)["qrels"][sentinel_doc], 1)

        captured: list[str] = []

        def generate(payload: dict) -> str:
            text = json.dumps(payload, ensure_ascii=False)
            captured.append(text)
            self.assertNotIn(sentinel_doc, text)
            self.assertNotIn(str(sentinel_value), text)
            self.assertNotIn(expected, text)
            return _structured_payload(1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, cases, _ = _mini_world(root, ["pd-1"])
            cases[0]["gold_value"] = sentinel_value
            cases[0]["expected_answer"] = expected
            cases[0] = attach_evaluator_qrels(cases[0], {sentinel_doc: 1})
            result = _authorized_run(
                generate,
                repo_root=root,
                frozen_config=config,
                split="public-dev",
                output_dir=root / "out",
                preflight_output_dir=root / "preflight",
                cases=cases,
                allowlist=["pd-1"],
            )
        self.assertTrue(captured)
        self.assertNotIn(sentinel_doc, "".join(captured))
        cases_blob = json.dumps(result["cases"], ensure_ascii=False)
        self.assertNotIn(sentinel_doc, cases_blob)
        self.assertNotIn('"qrels":', cases_blob)

        finrun = export_finrun_state(
            {
                "final_report": "ok",
                "companies": ["Acme"],
                "structured_answer": {
                    "answer": "ok",
                    "citations": [],
                    "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                    "qrels": {sentinel_doc: 1},
                    "gold_value": sentinel_value,
                    "expected_answer": expected,
                },
            }
        )
        dumped = json.dumps(finrun, ensure_ascii=False)
        self.assertNotIn(sentinel_doc, dumped)
        self.assertNotIn(expected, dumped)
        self.assertNotIn('"qrels":', dumped.casefold())


if __name__ == "__main__":
    unittest.main()
