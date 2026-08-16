#!/usr/bin/env python3
"""Prepare a gitignored FinanceBench evaluation workspace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.financebench.constants import EXPECTED_OPEN_SOURCE_QUESTIONS
from lumenfin.eval.financebench.prepare import prepare_financebench_eval
from lumenfin.stdio import configure_stdio_utf8


def main() -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        description="Load FinanceBench JSONL/PDFs and write split + page-level qrels."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="FinanceBench checkout or download directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "external" / "financebench"),
        help="Gitignored prepared dataset directory",
    )
    parser.add_argument(
        "--expected-questions",
        type=int,
        default=EXPECTED_OPEN_SOURCE_QUESTIONS,
        help="Set 0 to skip the open-source 150-question count check",
    )
    parser.add_argument("--require-pdfs", action="store_true")
    parser.add_argument("--skip-pdf-parse", action="store_true")
    args = parser.parse_args()

    prepared = prepare_financebench_eval(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        expected_questions=args.expected_questions or None,
        require_pdfs=args.require_pdfs,
        parse_pdfs=not args.skip_pdf_parse,
    )
    manifest = prepared["manifest"]
    print(f"questions={manifest['question_count']} documents={manifest['document_count']}")
    print(f"missing_pdfs={len(manifest['missing_pdfs'])} page_qrels={manifest['page_qrels']}")
    print(f"page_provenance_ok={manifest['page_provenance_ok']}")
    print(f"wrote {prepared['output_dir'] / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
