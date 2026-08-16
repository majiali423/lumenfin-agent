from __future__ import annotations

import unittest
from typing import Any

from lumenfin.eval.financebench.constants import EVAL_COMPANY_TAG, EVAL_SESSION_ID
from lumenfin.eval.financebench.retrieval import RemoteEvalBlocked, retrieve_for_mode
from lumenfin.eval.financebench.taxonomy import classify_case, classify_failure
from tests.financebench_fixtures import parsed_question


class FakeRAGStore:
    bm25_enabled = True

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks

    def iter_eval_chunks(self) -> list[dict[str, Any]]:
        return list(self.chunks)

    def _rank(self, query: str, *, reverse: bool = False) -> list[dict[str, Any]]:
        tokens = {token.lower() for token in query.split() if len(token) > 3}
        scored = []
        for chunk in self.chunks:
            text = str(chunk.get("text") or "").lower()
            score = sum(1.0 for token in tokens if token in text)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=not reverse)
        hits = []
        for score, chunk in scored:
            hit = dict(chunk)
            hit["score"] = score
            hit["retrieval_method"] = "fake"
            hit["companies"] = [EVAL_COMPANY_TAG]
            hits.append(hit)
        return hits

    def bm25_search(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
        return self._rank(query, reverse=False)

    def vector_search(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
        return self._rank(query, reverse=True)


class FinanceBenchRetrievalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FakeRAGStore(
            [
                {
                    "chunk_id": "gold:p1:c0",
                    "document_id": "GOLD_10K",
                    "filename": "GOLD_10K.pdf",
                    "page": 1,
                    "text": "FY2018 capital expenditure amount purchases of property plant equipment 1577",
                    "companies": [EVAL_COMPANY_TAG],
                },
                {
                    "chunk_id": "noise:p2:c0",
                    "document_id": "NOISE_10K",
                    "filename": "NOISE_10K.pdf",
                    "page": 2,
                    "text": "unrelated marketing commentary and brand strategy",
                    "companies": [EVAL_COMPANY_TAG],
                },
            ]
        )

    def test_bm25_and_dense_do_not_require_remote(self) -> None:
        query = "What is the FY2018 capital expenditure amount"
        bm25, meta_b = retrieve_for_mode(
            mode="bm25",
            store=self.store,
            query=query,
            top_k=2,
            allow_remote=False,
            embedding_provider="deterministic",
        )
        dense, meta_d = retrieve_for_mode(
            mode="dense",
            store=self.store,
            query=query,
            top_k=2,
            allow_remote=False,
            embedding_provider="deterministic",
        )
        self.assertEqual(bm25[0]["chunk_id"], "gold:p1:c0")
        self.assertEqual(dense[0]["chunk_id"], "noise:p2:c0")
        self.assertFalse(meta_b["degraded"])
        self.assertFalse(meta_d["degraded"])

    def test_hybrid_uses_retriever_without_document_body(self) -> None:
        hits, meta = retrieve_for_mode(
            mode="hybrid",
            store=self.store,
            query="FY2018 capital expenditure amount",
            top_k=2,
            allow_remote=False,
            embedding_provider="deterministic",
        )
        self.assertTrue(hits)
        self.assertNotIn("pages", str(meta))

    def test_qwen3_mode_blocked_without_allow_remote(self) -> None:
        with self.assertRaises(RemoteEvalBlocked):
            retrieve_for_mode(
                mode="hybrid-qwen3",
                store=self.store,
                query="capex",
                top_k=2,
                allow_remote=False,
                embedding_provider="deterministic",
            )

    def test_dashscope_embeddings_blocked_without_allow_remote(self) -> None:
        with self.assertRaises(RemoteEvalBlocked):
            retrieve_for_mode(
                mode="dense",
                store=self.store,
                query="capex",
                top_k=2,
                allow_remote=False,
                embedding_provider="dashscope",
            )

    def test_session_defaults_are_eval_scoped(self) -> None:
        self.assertEqual(EVAL_SESSION_ID, "financebench-eval-v1")
        self.assertEqual(EVAL_COMPANY_TAG, "FinanceBenchEval")


class FinanceBenchTaxonomyTestCase(unittest.TestCase):
    def test_classify_case_and_failures(self) -> None:
        question = parsed_question(
            financebench_id="financebench_id_00001",
            company="3M",
            doc_name="3M_2018_10K",
            question="How much was FY2018 capex versus FY2017?",
            question_type="metrics-generated",
            question_reasoning="Numerical reasoning",
        )
        labels = classify_case(question)
        self.assertFalse(labels["cross_document"])
        self.assertEqual(labels["evidence_pages"], "single_page")
        self.assertEqual(
            classify_failure(
                retrieved_pages=[],
                gold_pages={("3M_2018_10K", 60)},
                top_k=5,
                empty=True,
            ),
            "empty_retrieval",
        )
        self.assertEqual(
            classify_failure(
                retrieved_pages=[("OTHER", 1)],
                gold_pages={("3M_2018_10K", 60)},
                top_k=5,
                empty=False,
            ),
            "wrong_document",
        )
        self.assertEqual(
            classify_failure(
                retrieved_pages=[("3M_2018_10K", 60)],
                gold_pages={("3M_2018_10K", 60)},
                top_k=5,
                empty=False,
            ),
            "hit",
        )


if __name__ == "__main__":
    unittest.main()
