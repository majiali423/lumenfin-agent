from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout import (
    HoldoutError,
    LedgerPublicDevDataset,
    score_ledger_public_dev,
)


def _hit(
    chunk_id: str,
    doc_id: str,
    *,
    company: str = "nyse:dev",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "page": 1,
        "companies": [company],
        "text": f"secret text for {chunk_id}",
    }


def _dataset() -> tuple[LedgerPublicDevDataset, list[dict]]:
    report = "NYSE_DEV_2025"
    doc_ids = [f"{report}/page_{index:04d}" for index in range(12)]
    page_documents = tuple(
        {
            "ledger_doc_id": doc_id,
            "document_id": doc_id,
            "pages": [f"page {index}"],
        }
        for index, doc_id in enumerate(doc_ids)
    )
    query = {
        "query_id": "dev-query-1",
        "query_text": "SECRET QUERY TEXT",
        "company_key": "nyse:dev",
        "qrels": (
            {"doc_id": doc_ids[1], "relevance": 2},
            {"doc_id": doc_ids[2], "relevance": 1},
            {"doc_id": doc_ids[0], "relevance": 0},
        ),
    }
    hits = [_hit(f"duplicate-{index}", doc_ids[0]) for index in range(10)]
    hits.append(_hit("primary-gold", doc_ids[1]))
    hits.extend(
        _hit(f"other-{index}", doc_ids[index])
        for index in range(2, 11)
    )
    return (
        LedgerPublicDevDataset(
            queries=(query,),
            page_documents=page_documents,
            companies=("nyse:dev",),
            reports=1,
        ),
        hits,
    )


class LedgerPublicDevScoringTests(unittest.TestCase):
    def test_offline_arms_share_top20_and_page_arm_improves_diversity(self) -> None:
        dataset, hits = _dataset()
        calls: list[tuple[str, str, int]] = []

        def retrieve(query: str, company: str, top_k: int):
            calls.append((query, company, top_k))
            return list(hits), {"mode": "bm25", "remote_calls": 0}

        result = score_ledger_public_dev(
            dataset,
            retrieve_candidates=retrieve,
            retrieval_requires_remote=False,
        )
        self.assertEqual(calls, [("SECRET QUERY TEXT", "nyse:dev", 20)])
        self.assertEqual(result["call_accounting"]["retrieval_calls"], 1)
        self.assertEqual(result["call_accounting"]["remote_calls"], 0)
        self.assertFalse(result["primary_comparison_valid"])
        self.assertEqual(result["arms"]["A_prod"]["page_hit_at_10"], 0.0)
        self.assertEqual(result["arms"]["R_page"]["page_hit_at_10"], 1.0)
        self.assertGreater(
            result["arms"]["R_page"]["mean_unique_pages_top10"],
            result["arms"]["A_prod"]["mean_unique_pages_top10"],
        )
        serialized = json.dumps(result)
        self.assertNotIn("SECRET QUERY TEXT", serialized)
        self.assertNotIn("secret text", serialized)

    def test_rerank_requires_explicit_remote_and_counts_each_arm(self) -> None:
        dataset, hits = _dataset()

        def retrieve(_query: str, _company: str, _top_k: int):
            return list(hits), {"remote_calls": 0}

        def rerank(_query: str, pool: list[dict], top_k: int, _arm: str):
            return list(pool[:top_k]), {
                "rerank_fallback": False,
                "rerank_attempts": 1,
            }

        with self.assertRaisesRegex(HoldoutError, "allow_remote"):
            score_ledger_public_dev(
                dataset,
                retrieve_candidates=retrieve,
                retrieval_requires_remote=False,
                rerank=rerank,
            )
        result = score_ledger_public_dev(
            dataset,
            retrieve_candidates=retrieve,
            retrieval_requires_remote=False,
            rerank=rerank,
            allow_remote=True,
        )
        self.assertEqual(result["call_accounting"]["rerank_calls"], 2)
        self.assertEqual(result["call_accounting"]["remote_calls"], 2)
        self.assertTrue(result["primary_comparison_valid"])

    def test_rerank_fallback_invalidates_primary_comparison(self) -> None:
        dataset, hits = _dataset()

        def retrieve(_query: str, _company: str, _top_k: int):
            return list(hits), {"remote_calls": 0}

        def rerank(_query: str, pool: list[dict], top_k: int, _arm: str):
            return list(pool[:top_k]), {
                "rerank_fallback": True,
                "rerank_attempts": 2,
            }

        result = score_ledger_public_dev(
            dataset,
            retrieve_candidates=retrieve,
            retrieval_requires_remote=False,
            rerank=rerank,
            allow_remote=True,
        )
        self.assertEqual(result["call_accounting"]["rerank_fallbacks"], 2)
        self.assertEqual(result["call_accounting"]["rerank_attempts"], 4)
        self.assertEqual(result["call_accounting"]["remote_calls"], 4)
        self.assertFalse(result["primary_comparison_valid"])

    def test_cross_company_and_unknown_pages_fail_closed(self) -> None:
        dataset, hits = _dataset()
        cases = (
            [dict(hits[0], companies=["nyse:other"]), *hits[1:]],
            [dict(hits[0], document_id="NYSE_OTHER_2025/page_0000"), *hits[1:]],
        )
        for bad_hits in cases:
            with self.subTest(first=bad_hits[0]):
                with self.assertRaises(HoldoutError):
                    score_ledger_public_dev(
                        dataset,
                        retrieve_candidates=lambda _q, _c, _k: (
                            list(bad_hits),
                            {"remote_calls": 0},
                        ),
                        retrieval_requires_remote=False,
                    )

    def test_reranker_cannot_fabricate_candidates(self) -> None:
        dataset, hits = _dataset()

        def rerank(_query: str, pool: list[dict], _top_k: int, _arm: str):
            return [
                dict(
                    pool[0],
                    chunk_id="fabricated",
                )
            ], {"rerank_fallback": False, "rerank_attempts": 1}

        with self.assertRaisesRegex(HoldoutError, "outside"):
            score_ledger_public_dev(
                dataset,
                retrieve_candidates=lambda _q, _c, _k: (
                    list(hits),
                    {"remote_calls": 0},
                ),
                retrieval_requires_remote=False,
                rerank=rerank,
                allow_remote=True,
            )

    def test_empty_reranker_output_fails_closed(self) -> None:
        dataset, hits = _dataset()

        def rerank(_query: str, _pool: list[dict], _top_k: int, _arm: str):
            return [], {"rerank_fallback": False, "rerank_attempts": 1}

        with self.assertRaisesRegex(HoldoutError, "returned no hits"):
            score_ledger_public_dev(
                dataset,
                retrieve_candidates=lambda _q, _c, _k: (
                    list(hits),
                    {"remote_calls": 0},
                ),
                retrieval_requires_remote=False,
                rerank=rerank,
                allow_remote=True,
            )

    def test_duplicate_chunks_cannot_make_graded_ndcg_exceed_one(self) -> None:
        dataset, hits = _dataset()
        gold_doc = dataset.queries[0]["qrels"][0]["doc_id"]
        ranked = [
            _hit("gold-first", gold_doc),
            _hit("gold-duplicate", gold_doc),
            *hits[:18],
        ]
        result = score_ledger_public_dev(
            dataset,
            retrieve_candidates=lambda _q, _c, _k: (
                ranked,
                {"remote_calls": 0},
            ),
            retrieval_requires_remote=False,
        )
        ndcg = result["per_case"]["A_prod"][0]["ndcg_at_10"]
        self.assertGreater(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_retrieval_remote_calls_are_required_and_counted(self) -> None:
        dataset, hits = _dataset()
        with self.assertRaisesRegex(HoldoutError, "remote_calls"):
            score_ledger_public_dev(
                dataset,
                retrieve_candidates=lambda _q, _c, _k: (list(hits), {}),
                retrieval_requires_remote=False,
            )
        remote_calls = 0

        def remote_retrieve(_query: str, _company: str, _top_k: int):
            nonlocal remote_calls
            remote_calls += 1
            return list(hits), {"remote_calls": 2}

        with self.assertRaisesRegex(TypeError, "retrieval_requires_remote"):
            score_ledger_public_dev(
                dataset,
                retrieve_candidates=remote_retrieve,
            )
        self.assertEqual(remote_calls, 0)
        with self.assertRaisesRegex(HoldoutError, "explicit allow_remote"):
            score_ledger_public_dev(
                dataset,
                retrieve_candidates=remote_retrieve,
                retrieval_requires_remote=True,
            )
        self.assertEqual(remote_calls, 0)
        with self.assertRaisesRegex(HoldoutError, "undeclared remote"):
            score_ledger_public_dev(
                dataset,
                retrieve_candidates=remote_retrieve,
                retrieval_requires_remote=False,
                allow_remote=True,
            )
        result = score_ledger_public_dev(
            dataset,
            retrieve_candidates=remote_retrieve,
            retrieval_requires_remote=True,
            allow_remote=True,
        )
        self.assertEqual(result["call_accounting"]["retrieval_remote_calls"], 2)
        self.assertEqual(result["call_accounting"]["remote_calls"], 2)


if __name__ == "__main__":
    unittest.main()
