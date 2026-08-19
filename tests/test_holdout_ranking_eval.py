from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout import (
    ARM_SPECS,
    SECTION_METADATA_UNAVAILABLE,
    HoldoutError,
    attach_section_metadata,
    collapse_to_unique_pages,
    duplicate_page_occupancy,
    evaluate_ranking_case,
    holdout_file_sha256,
    load_holdout_questions,
    page_identity_coverage_top_k,
    prepare_rerank_pool,
    resolve_holdout_questions_path,
    section_metadata_for,
    summarize_ranking_cases,
    unique_pages_top_k,
    validate_holdout_request,
)
from lumenfin.eval.holdout.governance import validate_holdout_split
from lumenfin.eval.holdout.page_collapse import page_key

SCHEMA_EXAMPLE = ROOT / "data" / "eval_rag" / "holdout" / "schema_example.jsonl"


def _load_cli():
    path = ROOT / "scripts" / "run_holdout_ranking_eval.py"
    spec = importlib.util.spec_from_file_location("run_holdout_ranking_eval_cli", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load holdout CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(*, case_id: str = "holdout-0001") -> dict:
    return {
        "case_id": case_id,
        "company": "ExampleCo",
        "doc_name": "EXAMPLECO_2024_10K",
        "question": "What was capital expenditure?",
        "evidence": [
            {
                "evidence_doc_name": "EXAMPLECO_2024_10K",
                "evidence_page_num_one": 12,
            }
        ],
    }


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class HoldoutGovernanceTests(unittest.TestCase):
    def test_only_holdout_split_is_allowed(self) -> None:
        self.assertEqual(validate_holdout_split("holdout"), "holdout")
        for split in ("test", "dev", "confirmation", "all", "unknown", ""):
            with self.subTest(split=split):
                with self.assertRaises(HoldoutError):
                    validate_holdout_split(split)

    def test_financebench_paths_and_remote_calls_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            holdout = root / "holdout"
            holdout.mkdir()
            financebench = root / "financebench-copy"
            financebench.mkdir()
            with self.assertRaisesRegex(HoldoutError, "FinanceBench"):
                validate_holdout_request(
                    split="holdout",
                    dataset_dir=financebench,
                    repo_root=root,
                )
            with self.assertRaisesRegex(HoldoutError, "remote"):
                validate_holdout_request(
                    split="holdout",
                    dataset_dir=holdout,
                    repo_root=root,
                    allow_remote=True,
                )

    def test_questions_path_cannot_escape_validated_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "holdout"
            dataset.mkdir()
            outside = root / "outside.jsonl"
            outside.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(HoldoutError, "inside"):
                resolve_holdout_questions_path(dataset, outside)


class HoldoutDatasetTests(unittest.TestCase):
    def test_tracked_schema_example_loads_and_hashes(self) -> None:
        rows = load_holdout_questions(SCHEMA_EXAMPLE)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "holdout-example-0001")
        self.assertRegex(holdout_file_sha256(SCHEMA_EXAMPLE), r"^[0-9a-f]{64}$")

    def test_duplicate_case_ids_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.jsonl"
            _write_jsonl(path, [_row(), _row()])
            with self.assertRaisesRegex(HoldoutError, "duplicate"):
                load_holdout_questions(path)

    def test_financebench_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "questions.jsonl"
            row = _row(case_id="fb-financebench_id_00001")
            row["financebench_id"] = "financebench_id_00001"
            _write_jsonl(path, [row])
            with self.assertRaisesRegex(HoldoutError, "FinanceBench"):
                load_holdout_questions(path)

    def test_invalid_evidence_pages_fail_closed(self) -> None:
        for page in (None, 0, -1, True, "1.0", "page-1"):
            with self.subTest(page=page):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "questions.jsonl"
                    row = _row()
                    row["evidence"][0]["evidence_page_num_one"] = page
                    _write_jsonl(path, [row])
                    with self.assertRaisesRegex(HoldoutError, "integer >= 1"):
                        load_holdout_questions(path)


class HoldoutPageDiversityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hits = [
            {"chunk_id": "a1", "filename": "ACME.pdf", "page": 1},
            {"chunk_id": "a1-duplicate", "filename": "acme", "page": "1"},
            {"chunk_id": "a2", "filename": "ACME.pdf", "page": 2},
            {"chunk_id": "unknown", "filename": "ACME.pdf"},
            {"chunk_id": "b1", "filename": "BETA.pdf", "page": 1},
        ]

    def test_collapse_keeps_first_per_page_and_preserves_unknown_hits(self) -> None:
        collapsed = collapse_to_unique_pages(self.hits)
        self.assertEqual(
            [hit["chunk_id"] for hit in collapsed],
            ["a1", "a2", "unknown", "b1"],
        )

    def test_collapse_backfills_after_removing_duplicate_pages(self) -> None:
        collapsed = collapse_to_unique_pages(self.hits, k=2)
        self.assertEqual([hit["chunk_id"] for hit in collapsed], ["a1", "a2"])

    def test_page_diversity_metrics_are_explicit_about_missing_identity(self) -> None:
        self.assertEqual(unique_pages_top_k(self.hits, k=5), 3)
        self.assertEqual(page_identity_coverage_top_k(self.hits, k=5), 0.8)
        self.assertEqual(duplicate_page_occupancy(self.hits, k=5), 0.2)
        self.assertIsNone(page_key(self.hits[3]))

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            collapse_to_unique_pages(self.hits, k=-1)


class HoldoutOfflineRankingTests(unittest.TestCase):
    @staticmethod
    def _hit(chunk_id: str, page: int | None) -> dict:
        hit = {"chunk_id": chunk_id, "filename": "EXAMPLECO_2024_10K.pdf"}
        if page is not None:
            hit["page"] = page
        return hit

    @staticmethod
    def _question(case_id: str, page: int) -> dict:
        row = _row(case_id=case_id)
        row["evidence"][0]["evidence_page_num_one"] = page
        return row

    def test_arm_specs_keep_candidate_budget_equal(self) -> None:
        self.assertEqual(ARM_SPECS["A_prod"].source_k, 20)
        self.assertEqual(ARM_SPECS["R_page"].source_k, 20)
        self.assertEqual(ARM_SPECS["A_prod"].rerank_k, 20)
        self.assertEqual(ARM_SPECS["R_page"].rerank_k, 20)
        self.assertEqual(ARM_SPECS["A_prod"].final_k, 10)
        self.assertEqual(ARM_SPECS["R_page"].final_k, 10)

    def test_r_page_does_not_take_candidates_beyond_prod_top20(self) -> None:
        hits: list[dict] = []
        for page in range(1, 11):
            hits.append(self._hit(f"page-{page}-a", page))
            hits.append(self._hit(f"page-{page}-b", page))
        hits.append(self._hit("gold-page-12", 12))
        hits.extend(self._hit(f"tail-{page}", page) for page in range(13, 30))

        prod_pool = prepare_rerank_pool(hits, arm="A_prod")
        page_pool = prepare_rerank_pool(hits, arm="R_page")
        self.assertEqual(len(prod_pool), 20)
        self.assertEqual(len(page_pool), 10)
        self.assertNotIn("gold-page-12", {hit["chunk_id"] for hit in prod_pool})
        self.assertNotIn("gold-page-12", {hit["chunk_id"] for hit in page_pool})
        self.assertEqual(unique_pages_top_k(page_pool, k=20), 10)

    def test_case_metrics_preserve_unknown_rank_slots(self) -> None:
        question = self._question("holdout-ranking-1", 12)
        pool = [self._hit("gold", 12), self._hit("other", 13)]
        final = [self._hit("unknown", None), self._hit("gold", 12)]
        row = evaluate_ranking_case(
            question,
            rerank_pool=pool,
            final_hits=final,
        )
        self.assertTrue(row["pool_hit"])
        self.assertEqual(row["hit_at_5"], 1.0)
        self.assertEqual(row["mrr"], 0.5)
        self.assertEqual(row["page_identity_coverage_top10"], 0.5)
        self.assertEqual(row["failure_class"], "hit_at_10")

    def test_failure_classes_and_summary_separate_pool_from_ranking(self) -> None:
        hit = evaluate_ranking_case(
            self._question("holdout-hit", 1),
            rerank_pool=[self._hit("gold-1", 1)],
            final_hits=[self._hit("gold-1", 1)],
        )
        ranking_miss = evaluate_ranking_case(
            self._question("holdout-ranking-miss", 2),
            rerank_pool=[self._hit("gold-2", 2)],
            final_hits=[self._hit("wrong-3", 3)],
        )
        pool_miss = evaluate_ranking_case(
            self._question("holdout-pool-miss", 4),
            rerank_pool=[self._hit("wrong-3", 3)],
            final_hits=[self._hit("wrong-3", 3)],
        )
        summary = summarize_ranking_cases(
            [hit, ranking_miss, pool_miss],
            arm="R_page",
        )
        self.assertEqual(summary["cases"], 3)
        self.assertEqual(summary["pool_hit_rate"], 0.6667)
        self.assertEqual(summary["page_hit_at_10"], 0.3333)
        self.assertEqual(
            summary["failure_class_counts"],
            {
                "hit_at_10": 1,
                "gold_in_pool_not_in_final_top10": 1,
                "gold_not_in_rerank_pool": 1,
            },
        )
        self.assertEqual(summary["scoring_status"], "synthetic_offline_only")
        self.assertEqual(summary["remote_calls"], 0)

    def test_empty_summary_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize_ranking_cases([], arm="A_prod")


class HoldoutSectionSchemaTests(unittest.TestCase):
    def test_missing_metadata_stays_unavailable(self) -> None:
        metadata = section_metadata_for({"text": "Item 7 Management Discussion"})
        self.assertEqual(
            metadata,
            {
                "section_id": SECTION_METADATA_UNAVAILABLE,
                "section_title": SECTION_METADATA_UNAVAILABLE,
                "parent_chunk_id": SECTION_METADATA_UNAVAILABLE,
            },
        )

    def test_explicit_metadata_round_trips_without_mutating_input(self) -> None:
        chunk = {
            "chunk_id": "child-1",
            "section_id": "item-7",
            "section_title": "Management Discussion",
            "parent_chunk_id": "parent-1",
        }
        attached = attach_section_metadata(chunk)
        self.assertEqual(attached["section_id"], "item-7")
        self.assertEqual(attached["section_title"], "Management Discussion")
        self.assertEqual(attached["parent_chunk_id"], "parent-1")
        self.assertIsNot(attached, chunk)


class HoldoutCliTests(unittest.TestCase):
    def test_validate_only_cli_reports_zero_remote_calls(self) -> None:
        cli = _load_cli()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main(
                [
                    "--dataset-dir",
                    str(SCHEMA_EXAMPLE.parent),
                    "--questions-path",
                    str(SCHEMA_EXAMPLE),
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn("VALIDATE_OK", output.getvalue())
        self.assertIn("scoring=NOT_ENABLED", output.getvalue())
        self.assertIn("remote_calls=0", output.getvalue())

    def test_cli_refuses_remote_and_consumed_splits(self) -> None:
        cli = _load_cli()
        for args in (
            ["--dataset-dir", str(SCHEMA_EXAMPLE.parent), "--allow-remote"],
            ["--dataset-dir", str(SCHEMA_EXAMPLE.parent), "--split", "test"],
            ["--dataset-dir", str(SCHEMA_EXAMPLE.parent), "--split", "confirmation"],
        ):
            with self.subTest(args=args):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(args), 2)


if __name__ == "__main__":
    unittest.main()
