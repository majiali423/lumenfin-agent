from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.financebench.ingestion import (
    build_ingestion_report,
    cohort_case_ids,
    question_uses_zero_chunk,
)
from lumenfin.eval.financebench.loader import parse_question
from lumenfin.eval.financebench.taxonomy import classify_failure


class FinanceBenchIngestionTests(unittest.TestCase):
    def test_zero_chunk_is_ingestion_failure_not_retrieval_miss(self) -> None:
        question = parse_question(
            {
                "financebench_id": "financebench_id_00009",
                "question": "What is capex?",
                "answer": "1",
                "company": "JnJ",
                "doc_name": "JOHNSON_JOHNSON_2022Q4_EARNINGS",
                "question_type": "novel-generated",
                "evidence": [
                    {
                        "evidence_doc_name": "JOHNSON_JOHNSON_2022Q4_EARNINGS",
                        "evidence_page_num": 0,
                        "evidence_text": "adjusted EPS",
                    }
                ],
            }
        )
        report = build_ingestion_report(
            questions=[question],
            parsed_documents={
                "JOHNSON_JOHNSON_2022Q4_EARNINGS": {"pages": ["", ""]},
                "ACME_2022_10K": {"pages": ["cover", "capex"]},
            },
            chunks_by_doc={
                "JOHNSON_JOHNSON_2022Q4_EARNINGS": [],
                "ACME_2022_10K": [{"chunk_id": "a"}],
            },
            fallback_records=[{"doc_name": "JOHNSON_JOHNSON_2022Q4_EARNINGS"}],
        )
        self.assertEqual(report["zero_chunk_documents"], ["JOHNSON_JOHNSON_2022Q4_EARNINGS"])
        self.assertEqual(
            report["zero_chunk_status"]["JOHNSON_JOHNSON_2022Q4_EARNINGS"],
            "ingestion_failure",
        )
        self.assertTrue(
            question_uses_zero_chunk(question, set(report["zero_chunk_documents"]))
        )
        self.assertEqual(
            classify_failure(
                retrieved_pages=[],
                gold_pages={("JOHNSON_JOHNSON_2022Q4_EARNINGS", 1)},
                top_k=10,
                empty=True,
                ingestion_failure=True,
            ),
            "ingestion_failure",
        )
        self.assertEqual(
            classify_failure(
                retrieved_pages=[],
                gold_pages={("ACME_2022_10K", 2)},
                top_k=10,
                empty=True,
            ),
            "empty_retrieval",
        )

    def test_fallback_and_real_pdf_cohorts(self) -> None:
        real = parse_question(
            {
                "financebench_id": "financebench_id_00001",
                "question": "real?",
                "answer": "1",
                "company": "Acme",
                "doc_name": "ACME_2022_10K",
                "question_type": "metrics-generated",
                "evidence": [
                    {
                        "evidence_doc_name": "ACME_2022_10K",
                        "evidence_page_num": 1,
                        "evidence_text": "capex 1577 million",
                    }
                ],
            }
        )
        fallback = parse_question(
            {
                "financebench_id": "financebench_id_00002",
                "question": "fallback?",
                "answer": "1",
                "company": "JnJ",
                "doc_name": "JOHNSON_JOHNSON_2022_10K",
                "question_type": "domain-relevant",
                "evidence": [
                    {
                        "evidence_doc_name": "JOHNSON_JOHNSON_2022_10K",
                        "evidence_page_num": 1,
                        "evidence_text": "gross margin",
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pdf_fallback.json").write_text(
                '[{"doc_name": "JOHNSON_JOHNSON_2022_10K"}]\n',
                encoding="utf-8",
            )
            from lumenfin.eval.financebench.ingestion import load_fallback_documents

            names = {item["doc_name"] for item in load_fallback_documents(root)}
            self.assertEqual(
                cohort_case_ids([real, fallback], fallback_names=names, cohort="real_pdf"),
                {real.case_id},
            )
            self.assertEqual(
                cohort_case_ids([real, fallback], fallback_names=names, cohort="fallback"),
                {fallback.case_id},
            )
            self.assertEqual(
                len(cohort_case_ids([real, fallback], fallback_names=names, cohort="all")),
                2,
            )


if __name__ == "__main__":
    unittest.main()
