from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.financebench.loader import parse_question
from lumenfin.eval.financebench.split import (
    SplitError,
    assign_splits,
    forbid_test_split_tuning,
    questions_for_split,
)


def _question(index: int):
    return parse_question(
        {
            "financebench_id": f"financebench_id_{index:05d}",
            "question": f"Question {index}?",
            "answer": "1",
            "company": "Acme",
            "doc_name": "ACME_2022_10K",
            "question_type": "metrics-generated",
            "evidence": [
                {
                    "evidence_doc_name": "ACME_2022_10K",
                    "evidence_page_num": 0,
                    "evidence_text": "Revenue was 10 million.",
                }
            ],
        }
    )


class FinanceBenchSplitTests(unittest.TestCase):
    def test_split_is_order_independent_and_repeatable(self) -> None:
        forward = [_question(index) for index in range(1, 151)]
        reverse = list(reversed(forward))
        first = assign_splits(forward)
        second = assign_splits(reverse)
        self.assertEqual(first, second)
        self.assertEqual(sum(1 for split in first.values() if split == "dev"), 50)
        self.assertEqual(sum(1 for split in first.values() if split == "test"), 100)
        again = assign_splits(forward)
        self.assertEqual(first, again)

    def test_held_out_test_cannot_be_used_for_tuning(self) -> None:
        with self.assertRaises(SplitError):
            forbid_test_split_tuning("dev", tuning=True)
        with self.assertRaises(SplitError):
            forbid_test_split_tuning("confirmation", tuning=True)
        with self.assertRaises(SplitError):
            forbid_test_split_tuning("test", tuning=True)
        with self.assertRaises(SplitError):
            forbid_test_split_tuning("all", tuning=True)
        forbid_test_split_tuning("test", tuning=False)

    def test_confirmation_alias_uses_dev_ids_and_governance_marks_exposed_test(self) -> None:
        questions = [_question(index) for index in range(1, 151)]
        assignment = assign_splits(questions)
        confirmation = questions_for_split(questions, assignment, "confirmation")
        dev = questions_for_split(questions, assignment, "dev")
        self.assertEqual([item.financebench_id for item in confirmation], [item.financebench_id for item in dev])
        from lumenfin.eval.financebench.split import experiment_governance

        test_role = experiment_governance("test", "corpus")
        self.assertEqual(test_role["split_status"], "exposed_test")
        self.assertEqual(test_role["experiment_role"], "exploratory_baseline")
        self.assertFalse(test_role["held_out"])
        company_role = experiment_governance("test", "company")
        self.assertEqual(company_role["experiment_role"], "post_hoc_paired_diagnostic")
        self.assertEqual(experiment_governance("confirmation", "company")["split_status"], "confirmation")

    def test_questions_for_split_filters_without_reordering_ids(self) -> None:
        questions = [_question(index) for index in range(1, 151)]
        assignment = assign_splits(questions)
        dev = questions_for_split(questions, assignment, "dev")
        test = questions_for_split(questions, assignment, "test")
        self.assertEqual(len(dev), 50)
        self.assertEqual(len(test), 100)
        self.assertTrue(set(item.financebench_id for item in dev).isdisjoint(
            item.financebench_id for item in test
        ))


if __name__ == "__main__":
    unittest.main()
