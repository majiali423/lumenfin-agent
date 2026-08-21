from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_e2e import (
    account_ledger_citations,
    build_generation_prompt,
    citation_supported,
    load_ledger_gold_values,
    numeric_match,
    parse_answer_payload,
    score_generated_answer,
)
from lumenfin.rag.rerank import FallbackReranker, LexicalReranker


def _load_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_e2e_canary.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_e2e_canary",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load e2e canary CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hit(index: int, *, page: int | None = None) -> dict:
    number = index if page is None else page
    return {
        "chunk_id": f"chunk-{index}",
        "document_id": f"doc-{number}",
        "text": f"The KPI equals {index * 1000}.",
        "companies": ["company-a"],
        "page": number,
        "retrieval_method": "hybrid_dense_bm25_rrf",
        "fusion_score": 1.0 / (index + 1),
    }


class _FakeLLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.last_attempts = 1
        self.calls = 0

    def mark_usage_start(self) -> None:
        return None

    def usage_since_mark(self) -> dict[str, int]:
        return {"prompt_tokens": 10, "completion_tokens": 5}

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int = 200) -> str:
        self.calls += 1
        if "invent" in system_prompt and temperature != 0.0:
            raise AssertionError("generation must be locked to temperature 0")
        return self.payload


class _FailingReranker:
    provider_name = "dashscope"
    model_name = "qwen3-rerank"

    def rerank(self, query: str, hits: list[dict], *, top_k: int):
        raise RuntimeError("transient provider error")


class LedgerE2ECanaryTests(unittest.TestCase):
    def test_numeric_match_accepts_million_scale(self) -> None:
        matched = numeric_match(2104.6, 2_104_600_000.0)
        self.assertTrue(matched["matched"])
        self.assertEqual(matched["scale_factor"], 1e6)

    def test_parse_and_citation_fail_closed(self) -> None:
        parsed = parse_answer_payload(
            'prefix {"value": 12.5, "cited_chunk_ids": ["chunk-1"], "abstain": false}'
        )
        self.assertEqual(parsed["value"], 12.5)
        hits = [_hit(1), _hit(2)]
        self.assertTrue(
            citation_supported(["chunk-1"], hits, {"doc-1": 1, "doc-2": 0})
        )
        self.assertFalse(
            citation_supported(["chunk-2"], hits, {"doc-1": 1, "doc-2": 0})
        )
        with self.assertRaisesRegex(HoldoutError, "abstain"):
            parse_answer_payload(
                '{"value": 1, "cited_chunk_ids": ["chunk-1"], "abstain": true}'
            )

    def test_qwen3_failure_falls_back_to_lexical_then_generates(self) -> None:
        cli = _load_cli()
        hits = [_hit(index, page=(index + 1) // 2) for index in range(1, 21)]
        query = {
            "query_text": "What is revenue?",
            "qrels": [{"doc_id": "doc-1", "relevance": 1}],
        }
        reranker = FallbackReranker(_FailingReranker(), fallback=LexicalReranker())
        llm = _FakeLLM(
            '{"value": 1000, "cited_chunk_ids": ["chunk-1"], "abstain": false}'
        )
        result = cli._run_arm(
            arm="qwen3",
            query=query,
            hits=hits,
            gold_value=1000.0,
            reranker=reranker,
            llm=llm,
            max_document_chars=4000,
        )
        self.assertEqual(llm.calls, 1)
        self.assertTrue(result["rerank_fallback"])
        self.assertTrue(result["numeric_match"])
        self.assertGreater(result["total_latency_ms"], 0)

    def test_prefix_selection_is_nested_in_frozen_company_order(self) -> None:
        cli = _load_cli()
        rows = [
            {"query_id": f"c0-q{index}", "hits": []}
            for index in range(50)
        ] + [
            {"query_id": f"c1-q{index}", "hits": []}
            for index in range(50)
        ]
        plans = [
            {"query_ids": tuple(f"c0-q{index}" for index in range(50))},
            {"query_ids": tuple(f"c1-q{index}" for index in range(50))},
        ]
        selected = cli._prefix_candidate_rows(
            rows,
            plans,
            cases_per_company=10,
        )
        self.assertEqual(
            [row["query_id"] for row in selected],
            [f"c0-q{index}" for index in range(10)]
            + [f"c1-q{index}" for index in range(10)],
        )

    def test_plan_phase_rejects_allow_remote(self) -> None:
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            code = cli.main(
                [
                    "--phase",
                    "plan",
                    "--parquet-path",
                    str(Path(tmp) / "missing.parquet"),
                    "--manifest",
                    str(Path(tmp) / "manifest.json"),
                    "--split-salt",
                    "salt",
                    "--candidate-dir",
                    tmp,
                    "--output-dir",
                    str(Path(tmp) / "out"),
                    "--baseline-aggregate",
                    str(Path(tmp) / "base.json"),
                    "--baseline-per-case",
                    str(Path(tmp) / "base.jsonl"),
                    "--prerank-aggregate",
                    str(Path(tmp) / "pre.json"),
                    "--allow-remote",
                ]
            )
        self.assertEqual(code, 2)

    def test_prompt_does_not_include_gold_value(self) -> None:
        prompt = build_generation_prompt(
            query_text="What is revenue?",
            hits=[_hit(1)],
            max_document_chars=4000,
        )
        self.assertNotIn("2104600000", prompt)
        self.assertIn("chunk-1", prompt)

    def test_score_marks_abstain_separately_from_miss(self) -> None:
        scored = score_generated_answer(
            gold_value=10.0,
            parsed={"value": None, "cited_chunk_ids": [], "abstain": True},
            hits=[_hit(1)],
            qrels={"doc-1": 1},
        )
        self.assertFalse(scored["numeric_match"])
        self.assertEqual(scored["outcome"], "abstain")

    def test_gold_values_fail_closed_when_incomplete(self) -> None:
        with (
            patch(
                "lumenfin.eval.holdout.ledger_e2e.iter_ledger_parquet_rows",
                return_value=[{"query_id": "q1", "value": 12.5}],
            ),
            self.assertRaisesRegex(HoldoutError, "incomplete"),
        ):
            load_ledger_gold_values("unused.parquet", query_ids=["q1", "q2"])

    def test_completed_row_hash_mismatch_fails_closed(self) -> None:
        cli = _load_cli()
        row = {
            "query_id": "q1",
            "run_identity_sha256": "run",
            "shared_candidate_identity_sha256": "cand",
            "arms": {"lexical": {}, "qwen3": {}},
            "row_sha256": "wrong",
        }
        with self.assertRaisesRegex(HoldoutError, "identity mismatch"):
            cli._validate_completed_rows(
                [row],
                candidate_by_id={"q1": {"candidate_identity_sha256": "cand"}},
                run_identity_sha256="run",
            )

    def test_structured_citations_are_preferred_and_prose_is_not_guessed(self) -> None:
        parsed = parse_answer_payload(
            '{"answer": "12.5", "value": 12.5, "citations": ["chunk-1"], '
            '"structured_answer_schema_version": "1.0", "abstain": false}'
        )
        self.assertEqual(parsed["citations"], ["chunk-1"])
        self.assertEqual(parsed["citation_source"], "structured")
        with self.assertRaisesRegex(HoldoutError, "did not return JSON"):
            parse_answer_payload("The answer is 12.5 from chunk-1 without JSON")

    def test_citation_mutations_are_detected(self) -> None:
        hits = [
            {**_hit(1), "tenant_id": "t1", "session_id": "s1"},
            {**_hit(2), "tenant_id": "t1", "session_id": "s1"},
            {**_hit(3), "tenant_id": "t2", "session_id": "s1"},
            {**_hit(4), "tenant_id": "t1", "session_id": "s1", "unverified": True},
        ]
        qrels = {"doc-1": 1}
        empty = account_ledger_citations(
            cited_chunk_ids=[],
            hits=hits,
            qrels=qrels,
            citation_source="structured",
            tenant_id="t1",
            session_id="s1",
        )
        self.assertTrue(empty["no_citation"])
        unknown = account_ledger_citations(
            cited_chunk_ids=["does-not-exist"],
            hits=hits,
            qrels=qrels,
            citation_source="structured",
            tenant_id="t1",
            session_id="s1",
        )
        self.assertGreaterEqual(unknown["unknown_citation"], 1)
        other_question = account_ledger_citations(
            cited_chunk_ids=["chunk-2"],
            hits=hits,
            qrels=qrels,
            citation_source="structured",
            tenant_id="t1",
            session_id="s1",
        )
        self.assertTrue(other_question["unsupported_claim"])
        cross = account_ledger_citations(
            cited_chunk_ids=["chunk-3"],
            hits=hits,
            qrels=qrels,
            citation_source="structured",
            tenant_id="t1",
            session_id="s1",
        )
        self.assertGreaterEqual(cross["cross_run_or_tenant_citation"], 1)
        unverified = account_ledger_citations(
            cited_chunk_ids=["chunk-4"],
            hits=hits,
            qrels=qrels,
            citation_source="structured",
            tenant_id="t1",
            session_id="s1",
        )
        self.assertGreaterEqual(unverified["unverified_citation"], 1)
        mismatch = score_generated_answer(
            gold_value=1000.0,
            parsed={"value": 1000.0, "citations": ["chunk-2"], "citation_source": "structured"},
            hits=hits,
            qrels=qrels,
        )
        self.assertTrue(mismatch["numeric_match"])
        self.assertFalse(mismatch["citation_supported"])
        self.assertTrue(mismatch["citation_accounting"]["unsupported_claim"])


if __name__ == "__main__":
    unittest.main()
