from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lumenfin.eval.financebench.split import (
    SplitError,
    assign_splits,
    forbid_test_split_tuning,
    questions_for_split,
    split_manifest,
)
from lumenfin.eval.financebench.loader import load_financebench_dataset
from tests.financebench_fixtures import parsed_question, write_tiny_dataset


class FinanceBenchSplitTestCase(unittest.TestCase):
    def test_order_independent_hash_split(self) -> None:
        questions = [
            parsed_question(
                financebench_id=f"financebench_id_{index:05d}",
                company="TestCo",
                doc_name=f"DOC_{index}",
                question=f"q{index}",
            )
            for index in range(6)
        ]
        forward = assign_splits(questions)
        backward = assign_splits(list(reversed(questions)))
        self.assertEqual(forward, backward)
        self.assertEqual(sum(1 for split in forward.values() if split == "dev"), 2)
        self.assertEqual(sum(1 for split in forward.values() if split == "test"), 4)

    def test_canonical_150_is_50_100(self) -> None:
        questions = [
            parsed_question(
                financebench_id=f"financebench_id_{index:05d}",
                company="TestCo",
                doc_name=f"DOC_{index}",
                question=f"q{index}",
            )
            for index in range(150)
        ]
        assignment = assign_splits(questions)
        self.assertEqual(sum(1 for split in assignment.values() if split == "dev"), 50)
        self.assertEqual(sum(1 for split in assignment.values() if split == "test"), 100)

    def test_forbid_test_split_tuning(self) -> None:
        forbid_test_split_tuning("dev", tuning=True)
        with self.assertRaises(SplitError):
            forbid_test_split_tuning("test", tuning=True)
        with self.assertRaises(SplitError):
            forbid_test_split_tuning("all", tuning=True)
        forbid_test_split_tuning("test", tuning=False)

    def test_questions_for_split_filters(self) -> None:
        questions = [
            parsed_question(
                financebench_id=f"financebench_id_{index:05d}",
                company="TestCo",
                doc_name=f"DOC_{index}",
                question=f"q{index}",
            )
            for index in range(6)
        ]
        assignment = assign_splits(questions)
        dev = questions_for_split(questions, assignment, "dev")
        test = questions_for_split(questions, assignment, "test")
        self.assertEqual(len(dev) + len(test), 6)
        self.assertTrue(all(assignment[item.financebench_id] == "dev" for item in dev))

    def test_manifest_includes_salt(self) -> None:
        questions = [
            parsed_question(
                financebench_id="financebench_id_00001",
                company="TestCo",
                doc_name="DOC",
                question="q",
            )
        ]
        assignment = assign_splits(questions)
        manifest = split_manifest(questions, assignment)
        self.assertEqual(manifest["n_dev"] + manifest["n_test"], 1)
        self.assertIn("lumenfin-financebench-split-v1", str(manifest["salt"]))

    def test_tiny_dataset_split_via_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_tiny_dataset(Path(tmp), n_questions=6)
            questions, _documents, _paths = load_financebench_dataset(root, expected_questions=6)
            assignment = assign_splits(questions)
            self.assertEqual(len(assignment), 6)


if __name__ == "__main__":
    unittest.main()
