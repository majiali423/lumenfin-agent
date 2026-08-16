from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.financebench.loader import parse_question
from lumenfin.eval.financebench.qrels import (
    map_chunks_to_qrels,
    page_qrel_records,
    retrieved_page_keys,
)


def _question(*, pages: list[tuple[str, int, str]], doc_name: str = "ACME_2022_10K"):
    return parse_question(
        {
            "financebench_id": "financebench_id_00009",
            "question": "What is capex?",
            "answer": "1",
            "company": "Acme",
            "doc_name": doc_name,
            "question_type": "metrics-generated",
            "evidence": [
                {
                    "evidence_doc_name": doc,
                    "evidence_page_num": page_zero,
                    "evidence_text": text,
                }
                for doc, page_zero, text in pages
            ],
        }
    )


class FinanceBenchQrelsTests(unittest.TestCase):
    def test_page_qrels_keep_zero_and_one_indexed_pages(self) -> None:
        question = _question(pages=[("ACME_2022_10K", 1, "capex 1577 million")])
        records = page_qrel_records([question])
        self.assertEqual(records[0]["evidence_page_num_zero"], 1)
        self.assertEqual(records[0]["evidence_page_num_one"], 2)
        self.assertEqual(records[0]["evidence_doc_name"], "ACME_2022_10K")

    def test_page_cover_and_span_overlap_are_auditable(self) -> None:
        question = _question(pages=[("ACME_2022_10K", 1, "capital expenditures were 1577 million")])
        chunks = [
            {
                "chunk_id": "ACME_2022_10K:p2:c0",
                "document_id": "ACME_2022_10K",
                "filename": "ACME_2022_10K.pdf",
                "page": 2,
                "text": "unrelated paragraph on the gold page",
            },
            {
                "chunk_id": "ACME_2022_10K:p3:c0",
                "document_id": "ACME_2022_10K",
                "filename": "ACME_2022_10K.pdf",
                "page": 3,
                "text": "capital expenditures were 1577 million in the footnotes",
            },
            {
                "chunk_id": "OTHER:p2:c0",
                "document_id": "OTHER_DOC",
                "filename": "OTHER_DOC.pdf",
                "page": 2,
                "text": "capital expenditures were 1577 million",
            },
        ]
        qrels = map_chunks_to_qrels(question, chunks)
        reasons = {item.chunk_id: item.match_reason for item in qrels.chunk_qrels}
        self.assertEqual(reasons["ACME_2022_10K:p2:c0"], "page_cover")
        self.assertEqual(reasons["ACME_2022_10K:p3:c0"], "span_overlap")
        self.assertNotIn("OTHER:p2:c0", reasons)
        self.assertTrue(qrels.page_provenance_ok)

    def test_missing_page_metadata_is_a_provenance_gap(self) -> None:
        question = _question(pages=[("ACME_2022_10K", 0, "cover page text here")])
        qrels = map_chunks_to_qrels(
            question,
            [
                {
                    "chunk_id": "ACME_2022_10K:c0",
                    "document_id": "ACME_2022_10K",
                    "filename": "ACME_2022_10K.pdf",
                    "text": "cover page text here",
                }
            ],
        )
        self.assertFalse(qrels.page_provenance_ok)
        self.assertEqual(qrels.gold_chunk_ids, ())
        self.assertIn("page_provenance_gap", qrels.notes)

    def test_page_derived_and_span_metrics_are_separate(self) -> None:
        question = _question(pages=[("ACME_2022_10K", 1, "capital expenditures were 1577 million")])
        chunks = [
            {
                "chunk_id": "ACME_2022_10K:p2:c0",
                "document_id": "ACME_2022_10K",
                "filename": "ACME_2022_10K.pdf",
                "page": 2,
                "text": "unrelated paragraph on the gold page",
            },
            {
                "chunk_id": "ACME_2022_10K:p2:c1",
                "document_id": "ACME_2022_10K",
                "filename": "ACME_2022_10K.pdf",
                "page": 2,
                "text": "capital expenditures were 1577 million in the footnotes",
            },
            {
                "chunk_id": "ACME_2022_10K:p3:c0",
                "document_id": "ACME_2022_10K",
                "filename": "ACME_2022_10K.pdf",
                "page": 3,
                "text": "capital expenditures were 1577 million continued on the next page",
            },
        ]
        qrels = map_chunks_to_qrels(question, chunks)
        self.assertEqual(
            set(qrels.page_chunk_ids),
            {"ACME_2022_10K:p2:c0", "ACME_2022_10K:p2:c1"},
        )
        self.assertEqual(
            set(qrels.span_chunk_ids),
            {"ACME_2022_10K:p2:c1", "ACME_2022_10K:p3:c0"},
        )
        self.assertEqual(qrels.span_mapped_count, 1)
        self.assertEqual(qrels.span_unmapped_count, 0)

    def test_short_span_and_unmapped_span_are_recorded(self) -> None:
        question = _question(pages=[("ACME_2022_10K", 1, "n/a")])
        qrels = map_chunks_to_qrels(
            question,
            [
                {
                    "chunk_id": "ACME_2022_10K:p2:c0",
                    "document_id": "ACME_2022_10K",
                    "filename": "ACME_2022_10K.pdf",
                    "page": 2,
                    "text": "this page has no overlap with the tiny gold span",
                }
            ],
        )
        self.assertEqual(qrels.page_chunk_ids, ("ACME_2022_10K:p2:c0",))
        self.assertEqual(qrels.span_chunk_ids, ())
        self.assertEqual(qrels.span_unmapped_count, 1)
        self.assertIn("span_qrel_unmapped", qrels.notes)

    def test_no_span_overlap_on_other_page_is_not_page_derived(self) -> None:
        question = _question(pages=[("ACME_2022_10K", 1, "capital expenditures were 1577 million")])
        qrels = map_chunks_to_qrels(
            question,
            [
                {
                    "chunk_id": "ACME_2022_10K:p9:c0",
                    "document_id": "ACME_2022_10K",
                    "filename": "ACME_2022_10K.pdf",
                    "page": 9,
                    "text": "inventory discussion without the evidence span",
                }
            ],
        )
        self.assertEqual(qrels.page_chunk_ids, ())
        self.assertEqual(qrels.span_chunk_ids, ())
        self.assertEqual(qrels.span_unmapped_count, 1)

    def test_multi_gold_pages_and_empty_hits(self) -> None:
        question = _question(
            pages=[
                ("ACME_2022_10K", 1, "capex 1577 million"),
                ("ACME_2022_10Q", 1, "net sales 900 million"),
            ]
        )
        chunks = [
            {
                "chunk_id": "ACME_2022_10K:p2:c0",
                "document_id": "ACME_2022_10K",
                "filename": "ACME_2022_10K.pdf",
                "page": 2,
                "text": "capex 1577 million",
            },
            {
                "chunk_id": "ACME_2022_10Q:p2:c0",
                "document_id": "ACME_2022_10Q",
                "filename": "ACME_2022_10Q.pdf",
                "page": 2,
                "text": "net sales 900 million",
            },
        ]
        qrels = map_chunks_to_qrels(question, chunks)
        self.assertEqual(len(qrels.gold_pages), 2)
        self.assertFalse(qrels.single_gold_page)
        self.assertEqual(retrieved_page_keys([]), [])
        self.assertEqual(
            retrieved_page_keys(
                [
                    {"document_id": "ACME_2022_10K", "page": 2, "chunk_id": "a"},
                    {"document_id": "ACME_2022_10K", "page": 2, "chunk_id": "b"},
                    {"filename": "ACME_2022_10Q.pdf", "page": 2, "chunk_id": "c"},
                ]
            ),
            [("ACME_2022_10K", 2), ("ACME_2022_10Q", 2)],
        )


if __name__ == "__main__":
    unittest.main()
