from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.financebench_fixtures import make_financebench_tree, write_jsonl

from lumenfin.eval.financebench.loader import (
    FinanceBenchLoadError,
    case_id_for,
    load_financebench_dataset,
    normalize_doc_name,
    parse_question,
    zero_to_one_page,
)
from lumenfin.eval.financebench.prepare import prepare_financebench_eval


class FinanceBenchLoaderTests(unittest.TestCase):
    def test_loads_jsonl_and_maps_zero_indexed_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_financebench_tree(Path(tmp))
            questions, documents, paths = load_financebench_dataset(
                root, expected_questions=4, require_pdfs=True
            )
            self.assertEqual(len(questions), 4)
            self.assertEqual(len(documents), 2)
            self.assertTrue(paths.pdf_dir.is_dir())
            first = questions[0]
            self.assertEqual(first.case_id, "fb-financebench_id_00001")
            self.assertEqual(first.evidence[0].evidence_page_num_zero, 1)
            self.assertEqual(first.evidence[0].evidence_page_num_one, 2)
            self.assertEqual(zero_to_one_page(0), 1)
            self.assertEqual(normalize_doc_name("ACME_2022_10K.pdf"), "ACME_2022_10K")
            # Evidence may use doc_name instead of evidence_doc_name.
            self.assertEqual(questions[1].evidence[0].evidence_doc_name, "ACME_2022_10K")
            self.assertEqual(questions[1].evidence[0].evidence_page_num_one, 1)

    def test_missing_and_corrupt_rows_fail_closed(self) -> None:
        with self.assertRaises(FinanceBenchLoadError):
            parse_question({"financebench_id": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "financebench_open_source.jsonl", [{"financebench_id": "x"}])
            write_jsonl(root / "financebench_document_information.jsonl", [{"doc_name": "DOC"}])
            with self.assertRaises(FinanceBenchLoadError):
                load_financebench_dataset(root, expected_questions=None)
            (root / "financebench_open_source.jsonl").write_text("{not json\n", encoding="utf-8")
            with self.assertRaises(FinanceBenchLoadError):
                load_financebench_dataset(root, expected_questions=None)

    def test_count_mismatch_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = make_financebench_tree(Path(tmp))
            with self.assertRaises(FinanceBenchLoadError):
                load_financebench_dataset(root, expected_questions=150)

    def test_prepare_writes_page_qrels_and_stable_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = make_financebench_tree(Path(tmp) / "src")
            out = Path(tmp) / "prepared"
            prepared = prepare_financebench_eval(
                source_dir=source,
                output_dir=out,
                expected_questions=4,
                require_pdfs=True,
            )
            self.assertTrue((out / "manifest.json").is_file())
            self.assertTrue((out / "qrels_page.jsonl").is_file())
            self.assertTrue((out / "split_manifest.json").is_file())
            qrels = [
                json.loads(line)
                for line in (out / "qrels_page.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(qrels), 4)
            self.assertEqual(qrels[0]["query_id"], case_id_for("financebench_id_00001"))
            self.assertIn("evidence_page_num_one", qrels[0])
            self.assertTrue(prepared["manifest"]["page_provenance_ok"])


if __name__ == "__main__":
    unittest.main()
