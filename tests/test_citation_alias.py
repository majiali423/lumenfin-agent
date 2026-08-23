from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from concurrent.futures import ThreadPoolExecutor

from lumenfin.citation_alias import (
    CITATION_ALIAS_PROTOCOL_VERSION,
    LOCKED_FINAL_K,
    CitationAliasError,
    allowlist_from_window,
    build_citation_alias_map,
    build_final_evidence_window,
    flatten_rag_evidence,
    official_ranked_hits,
    parse_and_map_citation_aliases,
    prompt_hits_for_generator,
    render_prompt_evidence,
    replace_ephemeral_aliases_in_user_text,
    window_identity_report,
)
from lumenfin.eval.ledger_structured_citation_shadow import (
    NetworkProbe,
    bind_citation_contract,
    published_config_hash,
    score_case,
)
from lumenfin.finrun import export_finrun_state
from lumenfin.structured_answer import (
    STRUCTURED_ANSWER_SCHEMA_VERSION,
    allowed_evidence_from_state,
    public_structured_answer_fields,
)

from tests.test_ledger_structured_citation_shadow import _case, _hit, _structured_payload


def _official(hits: list[dict], *, origin: str = "test_fixture"):
    return official_ranked_hits(hits, origin=origin)


def _window(n: int = 10, *, start: int = 1, extra: int = 0) -> list[dict]:
    hits = [_hit(start + index) for index in range(n + extra)]
    return build_final_evidence_window(_official(hits))


class CitationAliasContractTests(unittest.TestCase):
    def test_e01_maps_to_top1_and_order_is_stable(self) -> None:
        window = _window(3)
        alias_map = build_citation_alias_map(window, case_id="c1", attempt_id="a1")
        self.assertEqual(alias_map.aliases, ("E01", "E02", "E03"))
        self.assertEqual(parse_and_map_citation_aliases(["E01"], alias_map), [window[0]["chunk_id"]])
        self.assertEqual(
            parse_and_map_citation_aliases(["E03", "E01", "E02"], alias_map),
            [window[2]["chunk_id"], window[0]["chunk_id"], window[1]["chunk_id"]],
        )
        self.assertEqual(
            parse_and_map_citation_aliases(["E01", "E01", "[E02]"], alias_map),
            [window[0]["chunk_id"], window[1]["chunk_id"]],
        )

    def test_unknown_and_mixed_lists_fail_closed(self) -> None:
        alias_map = build_citation_alias_map(_window(2), case_id="c1")
        with self.assertRaisesRegex(CitationAliasError, "not in the current evidence map"):
            parse_and_map_citation_aliases(["E99"], alias_map)
        with self.assertRaisesRegex(CitationAliasError, "not in the current evidence map"):
            parse_and_map_citation_aliases(["E01", "E99"], alias_map)

    def test_raw_ids_digits_and_filenames_are_rejected(self) -> None:
        pool = [_hit(index) for index in range(1, 13)]
        window = build_final_evidence_window(_official(pool))
        alias_map = build_citation_alias_map(window, case_id="c1")
        raw_top10 = window[0]["chunk_id"]
        raw_outside = pool[10]["chunk_id"]
        with self.assertRaisesRegex(CitationAliasError, "raw chunk ids are not allowed"):
            parse_and_map_citation_aliases([raw_top10], alias_map)
        with self.assertRaisesRegex(CitationAliasError, "not in the current evidence map"):
            parse_and_map_citation_aliases([raw_outside], alias_map)
        with self.assertRaisesRegex(CitationAliasError, "not allowed"):
            parse_and_map_citation_aliases(["1"], alias_map)
        with self.assertRaisesRegex(CitationAliasError, "not allowed"):
            parse_and_map_citation_aliases(["filing.pdf#p4"], alias_map)
        with self.assertRaisesRegex(CitationAliasError, "must be a string"):
            parse_and_map_citation_aliases([None], alias_map)
        with self.assertRaisesRegex(CitationAliasError, "must be a string"):
            parse_and_map_citation_aliases([1], alias_map)
        with self.assertRaisesRegex(CitationAliasError, "must be a string"):
            parse_and_map_citation_aliases([{"alias": "E01"}], alias_map)
        with self.assertRaisesRegex(CitationAliasError, "must be a string"):
            parse_and_map_citation_aliases([""], alias_map)

    def test_same_window_identity_and_prompt_hides_ids(self) -> None:
        pool = [_hit(index) for index in range(1, 21)]
        pool[0]["text"] = "VISIBLE_TOP1_BODY"
        pool[10]["text"] = "HIDDEN_TOP11_BODY"
        pool[10]["chunk_id"] = "SENTINEL_TOP11_CHUNK"
        window = build_final_evidence_window(_official(pool))
        self.assertEqual(len(window), 10)
        self.assertEqual(LOCKED_FINAL_K, 10)
        alias_map = build_citation_alias_map(window, case_id="c1", attempt_id="a1")
        allowlist = allowlist_from_window(window)
        report = window_identity_report(window, alias_map, allowlist)
        self.assertTrue(report["prompt_window_equals_alias_window"])
        self.assertTrue(report["alias_window_equals_validator_window"])
        prompt = render_prompt_evidence(window, alias_map, query_text="q")
        self.assertIn("[E01]", prompt)
        self.assertIn("VISIBLE_TOP1_BODY", prompt)
        self.assertNotIn("HIDDEN_TOP11_BODY", prompt)
        self.assertNotIn("SENTINEL_TOP11_CHUNK", prompt)
        self.assertNotIn("chunk_id=", prompt)
        self.assertNotIn(window[0]["chunk_id"], prompt)
        view_hits = prompt_hits_for_generator(window, alias_map)
        self.assertEqual([hit["alias"] for hit in view_hits], list(alias_map.aliases))
        self.assertTrue(all("chunk_id" not in hit for hit in view_hits))

    def test_ranking_runs_once_for_unranked_pool(self) -> None:
        calls: list[int] = []

        def rank(hits, *, final_k):
            calls.append(final_k)
            return list(reversed(hits))[:final_k]

        pool = [_hit(index) for index in range(1, 6)]
        window = build_final_evidence_window(pool, rank=rank)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], 10)
        self.assertEqual(window[0]["chunk_id"], pool[-1]["chunk_id"])
        with self.assertRaisesRegex(CitationAliasError, "final_k must stay 10"):
            build_final_evidence_window(_official(pool), final_k=20)
        with self.assertRaisesRegex(CitationAliasError, "already_ranked requires official_ranked_hits"):
            build_final_evidence_window(pool, already_ranked=True)

    def test_cross_case_and_stale_repair_fail(self) -> None:
        first = build_citation_alias_map(
            _window(2, start=1),
            case_id="case-a",
            attempt_id="1",
            tenant_id="tenant-a",
            session_id="session-a",
        )
        second = build_citation_alias_map(
            _window(2, start=3),
            case_id="case-b",
            attempt_id="2",
            tenant_id="tenant-b",
            session_id="session-b",
        )
        self.assertEqual(first.aliases[0], second.aliases[0])
        self.assertNotEqual(first.chunk_ids[0], second.chunk_ids[0])
        with self.assertRaisesRegex(CitationAliasError, "not bound to this case"):
            parse_and_map_citation_aliases(["E01"], first, expected_case_id="case-b")
        with self.assertRaisesRegex(CitationAliasError, "not bound to this repair attempt"):
            parse_and_map_citation_aliases(["E01"], first, expected_attempt_id="2")
        with self.assertRaisesRegex(CitationAliasError, "not bound to this tenant"):
            parse_and_map_citation_aliases(["E01"], first, expected_tenant_id="tenant-b")
        with self.assertRaisesRegex(CitationAliasError, "not bound to this session"):
            parse_and_map_citation_aliases(["E01"], first, expected_session_id="session-b")
        with self.assertRaisesRegex(CitationAliasError, "raw chunk ids are not allowed"):
            parse_and_map_citation_aliases([second.chunk_ids[0]], second)
        with self.assertRaisesRegex(CitationAliasError, "not in the current evidence map"):
            parse_and_map_citation_aliases([first.chunk_ids[0]], second)
        stale = _window(1)
        stale[0]["stale_repair_attempt"] = True
        allowlist = allowlist_from_window(stale)
        self.assertTrue(allowlist[0].stale)
        conflict = [_hit(1), dict(_hit(1), tenant_id="other")]
        with self.assertRaisesRegex(CitationAliasError, "conflicting tenant or session"):
            build_final_evidence_window(_official(conflict))
        with self.assertRaisesRegex(CitationAliasError, "conflicting metadata"):
            allowlist_from_window(conflict)
        duplicate = [_hit(1), dict(_hit(1))]
        with self.assertRaisesRegex(CitationAliasError, "duplicate chunk_id"):
            build_final_evidence_window(_official(duplicate))

    def test_score_case_maps_alias_and_rejects_raw_id(self) -> None:
        case = _case("pd-1", 1)
        contract = bind_citation_contract(case)
        alias = contract["alias_map"].alias_for(case["hits"][0]["chunk_id"])
        supported = score_case(
            case,
            raw=_structured_payload(1, citations=[alias]),
            latency_ms=1,
            generate_attempts=1,
            remote_calls=1,
            contract=contract,
        )
        self.assertEqual(supported["citations"], [case["hits"][0]["chunk_id"]])
        self.assertTrue(supported["supported_claim"])
        self.assertFalse(supported["citation_validation_failed"])

        rejected = score_case(
            case,
            raw=_structured_payload(1, citations=[case["hits"][0]["chunk_id"]]),
            latency_ms=1,
            generate_attempts=1,
            remote_calls=1,
            contract=contract,
        )
        self.assertEqual(rejected["citations"], [])
        self.assertTrue(rejected["citation_validation_failed"])
        self.assertFalse(rejected["structured_answer_present"])
        self.assertNotIn(case["hits"][0]["chunk_id"], json.dumps(rejected))

    def test_finrun_and_api_only_expose_mapped_ids(self) -> None:
        window = _window(1)
        window[0]["tenant_id"] = "tenant-a"
        window[0]["session_id"] = "session-a"
        alias_map = build_citation_alias_map(window, case_id="c1")
        mapped = parse_and_map_citation_aliases(["[E01]"], alias_map)
        finrun = export_finrun_state(
            {
                "final_report": "ok",
                "companies": ["Acme"],
                "thread_id": "session-a",
                "rag_tenant_id": "tenant-a",
                "rag_evidence": {"Acme": window},
                "claims": [
                    {
                        "claim_id": "c1",
                        "entity": "Acme",
                        "claim_type": "numeric",
                        "statement": "ok",
                        "verification": "verified",
                        "evidence_refs": [
                            {
                                "evidence_id": "e1",
                                "entity": "Acme",
                                "citation": "filing.pdf#p1",
                                "source_type": "rag",
                                "text": "ok",
                                "chunk_id": window[0]["chunk_id"],
                                "tenant_id": "tenant-a",
                                "session_id": "session-a",
                            }
                        ],
                    }
                ],
                "structured_answer": {
                    "answer": "ok",
                    "citations": mapped,
                    "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                },
            }
        )
        dumped = json.dumps(finrun, ensure_ascii=False)
        self.assertIn(window[0]["chunk_id"], dumped)
        self.assertNotIn("E01", dumped)
        api = public_structured_answer_fields(
            {
                "final_report": "ok",
                "structured_answer": finrun["structured_answer"],
            }
        )
        self.assertEqual(api["citations"], [window[0]["chunk_id"]])
        failed = public_structured_answer_fields(
            {
                "final_report": "ok",
                "structured_answer": {
                    "answer": "ok",
                    "citations": [],
                    "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                    "citation_validation": "failed",
                },
            }
        )
        self.assertIsNone(failed)

    def test_protocol_version_is_not_finrun_schema(self) -> None:
        self.assertEqual(CITATION_ALIAS_PROTOCOL_VERSION, "citation_alias_protocol.v1")
        self.assertNotEqual(CITATION_ALIAS_PROTOCOL_VERSION, "1.0")
        self.assertNotEqual(CITATION_ALIAS_PROTOCOL_VERSION, STRUCTURED_ANSWER_SCHEMA_VERSION)

    def test_generation_view_hides_gold_and_stable_ids(self) -> None:
        case = _case("pd-1", 1, gold_label="SENTINEL_GOLD_LABEL_ZX9")
        case["qrels"] = {"secret-gold-doc": 1}
        contract = bind_citation_contract(case)
        view = contract["generation_view"]
        blob = json.dumps(view, ensure_ascii=False)
        self.assertNotIn("gold_value", blob)
        self.assertNotIn("gold_label", blob)
        self.assertNotIn("qrels", blob)
        self.assertNotIn(case["hits"][0]["chunk_id"], blob)
        self.assertNotIn("tenant_id", blob)
        self.assertNotIn("session_id", blob)
        self.assertTrue(all(hit["alias"].startswith("E") for hit in view["hits"]))

    def test_provider_error_does_not_leak_mapping(self) -> None:
        from lumenfin.eval.ledger_structured_citation_shadow import failed_case

        case = _case("pd-1", 1)
        contract = bind_citation_contract(case)
        dumped = json.dumps(
            failed_case(
                case,
                RuntimeError(f"mapped {contract['alias_map'].alias_to_chunk} gold={case['gold_value']}"),
                remote_calls=1,
            ),
            ensure_ascii=False,
        )
        self.assertNotIn("E01", dumped)
        self.assertNotIn(case["hits"][0]["chunk_id"], dumped)
        self.assertNotIn(str(case["gold_value"]), dumped)
        self.assertNotIn("alias_to_chunk", dumped)

    def test_bind_and_hash_make_no_remote_calls(self) -> None:
        probe = NetworkProbe()
        probe.install()
        try:
            contract = bind_citation_contract(_case("pd-1", 1))
            digest = published_config_hash()
        finally:
            probe.remove()
        self.assertEqual(probe.remote_request_count, 0)
        self.assertEqual(len(contract["window"]), min(2, LOCKED_FINAL_K))
        self.assertEqual(
            digest,
            "7db4156491fbd0cb500ae71772002a494a3cc37b751eb5e55b707307fd02b91b",
        )

    def test_parallel_cases_do_not_share_e01_mapping(self) -> None:
        first = _case("case-a", 1, tenant_id="tenant-a", session_id="session-a")
        second = _case("case-b", 2, tenant_id="tenant-b", session_id="session-b")

        def _score(case: dict) -> dict:
            contract = bind_citation_contract(case)
            alias = contract["alias_map"].alias_for(case["hits"][0]["chunk_id"])
            self.assertEqual(alias, "E01")
            row = score_case(
                case,
                raw=_structured_payload(1, citations=["E01"]),
                latency_ms=1,
                generate_attempts=1,
                remote_calls=1,
                contract=contract,
            )
            return row

        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(_score, [first, second]))
        self.assertEqual(rows[0]["citations"], [first["hits"][0]["chunk_id"]])
        self.assertEqual(rows[1]["citations"], [second["hits"][0]["chunk_id"]])
        self.assertNotEqual(rows[0]["citations"], rows[1]["citations"])

    def test_flatten_keeps_company_order_and_identity_fields(self) -> None:
        zebra = dict(_hit(1), tenant_id="t1", session_id="s1", company="Zebra")
        apple = dict(_hit(2), tenant_id="t1", session_id="s1", company="Apple")
        flat = flatten_rag_evidence(
            {"Zebra": [zebra], "Apple": [apple]},
            company_order=["Zebra", "Apple"],
        )
        self.assertEqual([hit["company"] for hit in flat], ["Zebra", "Apple"])
        self.assertEqual(flat[0]["tenant_id"], "t1")
        self.assertEqual(flat[0]["session_id"], "s1")
        self.assertEqual(flat[0]["chunk_id"], zebra["chunk_id"])
        window = build_final_evidence_window(
            _official(flat, origin="production_rag"),
        )
        self.assertEqual(window[0]["company"], "Zebra")
        allowlist = allowed_evidence_from_state(
            {
                "companies": ["Zebra", "Apple"],
                "rag_tenant_id": "t1",
                "thread_id": "s1",
                "rag_evidence": {"Zebra": [zebra], "Apple": [apple]},
            }
        )
        self.assertEqual([item.chunk_id for item in allowlist], [zebra["chunk_id"], apple["chunk_id"]])

    def test_user_text_aliases_map_to_page_anchors_not_chunk_ids(self) -> None:
        window = _window(1)
        window[0]["citation"] = "filing.pdf#p4"
        alias_map = build_citation_alias_map(window, case_id="c1")
        text = replace_ephemeral_aliases_in_user_text("See [E01] for revenue.", window, alias_map)
        self.assertEqual(text, "See filing.pdf#p4 for revenue.")
        self.assertNotIn("E01", text)
        self.assertNotIn(window[0]["chunk_id"], text)

    def test_alias_failure_is_not_a_provider_error(self) -> None:
        case = _case("pd-1", 1)
        contract = bind_citation_contract(case)
        rejected = score_case(
            case,
            raw=_structured_payload(1, citations=[case["hits"][0]["chunk_id"]]),
            latency_ms=1,
            generate_attempts=1,
            remote_calls=1,
            contract=contract,
        )
        self.assertFalse(rejected.get("failed"))
        self.assertFalse(rejected.get("provider_error"))
        self.assertTrue(rejected["citation_validation_failed"])
        self.assertEqual(rejected["citation_source"], "unavailable")
        self.assertEqual(rejected["citations"], [])
        self.assertEqual(rejected["outcome"], "degraded")

    def test_production_page_anchors_do_not_emit_aliases(self) -> None:
        from lumenfin.reporting import format_rag_citation_section

        hit = dict(_hit(1), filename="filing.pdf", page=4, citation="filing.pdf#p4")
        blob = "\n".join(format_rag_citation_section({"Acme": [hit]}))
        self.assertIn("filing.pdf p.4", blob)
        self.assertNotIn("E01", blob)
        self.assertNotIn("[E01]", blob)
        self.assertNotIn(hit["chunk_id"], blob)


if __name__ == "__main__":
    unittest.main()
