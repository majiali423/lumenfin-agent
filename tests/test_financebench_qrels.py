from __future__ import annotations

import unittest

from lumenfin.eval.financebench.qrels import (
    auditable_span_overlap,
    gold_pages_for,
    map_chunks_to_qrels,
)
from tests.financebench_fixtures import parsed_question


class FinanceBenchQrelsTestCase(unittest.TestCase):
    def test_gold_pages_are_one_indexed(self) -> None:
        question = parsed_question(
            financebench_id="financebench_id_00001",
            company="3M",
            doc_name="3M_2018_10K",
            question="capex?",
            page_zero=59,
            evidence_text="Purchases of property, plant and equipment (PP&E) (1,577)",
        )
        pages = gold_pages_for(question)
        self.assertEqual(pages[0].page_zero, 59)
        self.assertEqual(pages[0].page_one, 60)

    def test_page_cover_marks_same_doc_same_page(self) -> None:
        question = parsed_question(
            financebench_id="financebench_id_00001",
            company="3M",
            doc_name="3M_2018_10K",
            question="capex?",
            page_zero=59,
            evidence_text="Purchases of property, plant and equipment (PP&E) (1,577)",
        )
        chunks = [
            {
                "chunk_id": "3M_2018_10K:p60:c0",
                "document_id": "3M_2018_10K",
                "filename": "3M_2018_10K.pdf",
                "page": 60,
                "text": "unrelated table header",
            }
        ]
        qrels = map_chunks_to_qrels(question, chunks)
        self.assertEqual(qrels.gold_chunk_ids, ("3M_2018_10K:p60:c0",))
        self.assertEqual(qrels.chunk_qrels[0].match_reason, "page_cover")

    def test_span_overlap_when_page_differs(self) -> None:
        question = parsed_question(
            financebench_id="financebench_id_00001",
            company="3M",
            doc_name="3M_2018_10K",
            question="capex?",
            page_zero=59,
            evidence_text="Purchases of property, plant and equipment (PP&E) (1,577)",
        )
        chunks = [
            {
                "chunk_id": "3M_2018_10K:p12:c0",
                "document_id": "3M_2018_10K",
                "page": 12,
                "text": "Purchases of property, plant and equipment (PP&E) (1,577) continued",
            }
        ]
        qrels = map_chunks_to_qrels(question, chunks)
        self.assertEqual(qrels.chunk_qrels[0].match_reason, "span_overlap")

    def test_missing_page_metadata_is_not_relevant(self) -> None:
        question = parsed_question(
            financebench_id="financebench_id_00001",
            company="3M",
            doc_name="3M_2018_10K",
            question="capex?",
            page_zero=59,
            evidence_text="Purchases of property, plant and equipment (PP&E) (1,577)",
        )
        chunks = [
            {
                "chunk_id": "3M_2018_10K:unknown",
                "document_id": "3M_2018_10K",
                "text": "Purchases of property, plant and equipment (PP&E) (1,577)",
            }
        ]
        qrels = map_chunks_to_qrels(question, chunks)
        self.assertEqual(qrels.gold_chunk_ids, ())
        self.assertFalse(qrels.page_provenance_ok)
        self.assertIn("page_provenance_gap", qrels.notes)

    def test_wrong_document_is_not_relevant(self) -> None:
        question = parsed_question(
            financebench_id="financebench_id_00001",
            company="3M",
            doc_name="3M_2018_10K",
            question="capex?",
            page_zero=59,
            evidence_text="Purchases of property, plant and equipment (PP&E) (1,577)",
        )
        chunks = [
            {
                "chunk_id": "OTHER:p60:c0",
                "document_id": "OTHER_10K",
                "page": 60,
                "text": "Purchases of property, plant and equipment (PP&E) (1,577)",
            }
        ]
        qrels = map_chunks_to_qrels(question, chunks)
        self.assertEqual(qrels.gold_chunk_ids, ())

    def test_short_span_does_not_fuzzy_match(self) -> None:
        self.assertFalse(auditable_span_overlap("the company reported results", "FY"))
        self.assertTrue(
            auditable_span_overlap(
                "Purchases of property, plant and equipment (PP&E) (1,577) here",
                "Purchases of property, plant and equipment (PP&E) (1,577)",
            )
        )


if __name__ == "__main__":
    unittest.main()
