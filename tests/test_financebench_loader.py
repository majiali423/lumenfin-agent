from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from lumenfin.eval.financebench.loader import (
    FinanceBenchLoadError,
    case_id_for,
    discover_financebench_paths,
    load_financebench_dataset,
    parse_evidence,
    parse_question,
    zero_to_one_page,
)
from tests.financebench_fixtures import question_row, write_jsonl, write_tiny_dataset


class FinanceBenchLoaderTestCase(unittest.TestCase):
    def test_zero_indexed_evidence_maps_to_one_indexed_page(self) -> None:
        self.assertEqual(zero_to_one_page(0), 1)
        self.assertEqual(zero_to_one_page(59), 60)
        spans = parse_evidence(
            [{"doc_name": "3M_2018_10K", "evidence_page_num": 59, "evidence_text": "capex 1577"}],
            question_doc_name="3M_2018_10K",
            case_id="fb-x",
        )
        self.assertEqual(spans[0].evidence_page_num_zero, 59)
        self.assertEqual(spans[0].evidence_page_num_one, 60)

    def test_case_id_prefixes_financebench_id(self) -> None:
        self.assertEqual(case_id_for("financebench_id_03029"), "fb-financebench_id_03029")
        self.assertEqual(case_id_for("fb-already"), "fb-already")

    def test_evidence_accepts_doc_name_or_evidence_doc_name(self) -> None:
        a = parse_evidence(
            [{"evidence_doc_name": "A_10K", "evidence_page_num": 0, "evidence_text": "hello world"}],
            question_doc_name="A_10K",
            case_id="fb-a",
        )
        b = parse_evidence(
            [{"doc_name": "A_10K", "evidence_page_num": 0, "evidence_text": "hello world"}],
            question_doc_name="A_10K",
            case_id="fb-b",
        )
        self.assertEqual(a[0].evidence_doc_name, "A_10K")
        self.assertEqual(b[0].evidence_doc_name, "A_10K")

    def test_missing_evidence_text_fails_closed(self) -> None:
        with self.assertRaises(FinanceBenchLoadError):
            parse_question(
                {
                    "financebench_id": "x",
                    "company": "3M",
                    "doc_name": "3M_2018_10K",
                    "question": "What is capex?",
                    "evidence": [{"doc_name": "3M_2018_10K", "evidence_page_num": 0}],
                }
            )

    def test_official_layout_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tiny_dataset(Path(tmp), n_questions=3)
            questions, documents, paths = load_financebench_dataset(
                root, expected_questions=3, require_pdfs=False
            )
            self.assertEqual(len(questions), 3)
            self.assertEqual(len(documents), 3)
            self.assertFalse(paths.merged)
            self.assertEqual(questions[0].case_id, "fb-financebench_id_00000")

    def test_merged_layout_synthesizes_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                question_row(
                    financebench_id="financebench_id_00001",
                    company="3M",
                    doc_name="3M_2018_10K",
                    question="What is capex?",
                )
            ]
            write_jsonl(root / "data" / "financebench_merged.jsonl", rows)
            questions, documents, paths = load_financebench_dataset(
                root, expected_questions=1, require_pdfs=False
            )
            self.assertTrue(paths.merged)
            self.assertIn("3M_2018_10K", documents)
            self.assertEqual(questions[0].document.doc_link, "https://example.invalid/filing.pdf")

    def test_corrupt_jsonl_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "financebench_open_source.jsonl").write_text("{not json\n", encoding="utf-8")
            (root / "financebench_document_information.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FinanceBenchLoadError):
                discover_financebench_paths(root)
                load_financebench_dataset(root, expected_questions=None)

    def test_real_merged_file_has_150_when_present(self) -> None:
        candidate = ROOT / "data" / "external" / "financebench-src"
        if not (candidate / "data" / "financebench_merged.jsonl").is_file():
            self.skipTest("local FinanceBench merged JSONL not present")
        questions, documents, paths = load_financebench_dataset(candidate, expected_questions=150)
        self.assertEqual(len(questions), 150)
        self.assertGreaterEqual(len(documents), 1)
        self.assertTrue(paths.merged)
        sample = json.loads(
            (candidate / "data" / "financebench_merged.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertIn("financebench_id", sample)


if __name__ == "__main__":
    unittest.main()
