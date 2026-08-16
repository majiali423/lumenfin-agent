"""Prepare FinanceBench split, page qrels, and optional chunk qrels."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lumenfin.eval.financebench.constants import SPLIT_SALT
from lumenfin.eval.financebench.loader import (
    load_financebench_dataset,
    resolve_financebench_source,
    resolve_pdf_path,
)
from lumenfin.eval.financebench.qrels import map_chunks_to_qrels, page_qrel_records
from lumenfin.eval.financebench.reporting import write_json, write_jsonl
from lumenfin.eval.financebench.schema import FinanceBenchQuestion
from lumenfin.eval.financebench.split import assign_splits, split_manifest

log = logging.getLogger("lumenfin.eval.financebench.prepare")


def prepare_artifacts(
    *,
    dataset_root: Path | None,
    out_dir: Path,
    expected_questions: int | None = 150,
    require_pdfs: bool = False,
) -> dict[str, Any]:
    """Write split + page-qrel artifacts. Does not index embeddings."""
    source = resolve_financebench_source(dataset_root)
    questions, documents, paths = load_financebench_dataset(
        source,
        expected_questions=expected_questions,
        require_pdfs=require_pdfs,
    )
    assignment = assign_splits(questions)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "split_manifest.json", split_manifest(questions, assignment))
    write_jsonl(out_dir / "page_qrels.jsonl", page_qrel_records(questions))
    write_json(
        out_dir / "documents.json",
        {name: info.to_public_dict() for name, info in documents.items()},
    )

    pdf_present = 0
    pdf_missing: list[str] = []
    for name, info in documents.items():
        if resolve_pdf_path(paths.pdf_dir, name, info) is not None:
            pdf_present += 1
        else:
            pdf_missing.append(name)

    n_dev = sum(1 for split in assignment.values() if split == "dev")
    n_test = sum(1 for split in assignment.values() if split == "test")
    summary = {
        "questions": len(questions),
        "documents": len(documents),
        "split_salt": SPLIT_SALT,
        "n_dev": n_dev,
        "n_test": n_test,
        "merged": paths.merged,
        "pdf_dir": str(paths.pdf_dir),
        "pdf_present": pdf_present,
        "pdf_missing": pdf_missing,
        "questions_path": str(paths.questions_path),
        "documents_path": str(paths.documents_path) if paths.documents_path else None,
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "prepare_summary.json", summary)
    log.info(
        "prepared financebench artifacts questions=%s docs=%s pdfs=%s/%s",
        len(questions),
        len(documents),
        pdf_present,
        len(documents),
    )
    return summary


def write_chunk_qrels(
    out_dir: Path,
    questions: Sequence[FinanceBenchQuestion],
    chunks: Sequence[dict[str, Any]],
) -> Path:
    payload: dict[str, Any] = {}
    for question in questions:
        payload[question.case_id] = map_chunks_to_qrels(question, chunks).to_dict()
    path = out_dir / "chunk_qrels.json"
    write_json(path, payload)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare FinanceBench split and page qrels")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/financebench_eval/prepared"))
    parser.add_argument("--expected-questions", type=int, default=150)
    parser.add_argument("--require-pdfs", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = prepare_artifacts(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        expected_questions=args.expected_questions or None,
        require_pdfs=bool(args.require_pdfs),
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "pdf_missing"}))
    if summary["pdf_missing"]:
        print(json.dumps({"pdf_missing_count": len(summary["pdf_missing"])}))
    return 0
