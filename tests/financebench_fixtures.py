#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import fitz


def write_pdf(path: Path, pages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=612, height=792)
        page.insert_text((48, 72), text, fontsize=11, fontname="helv")
    doc.save(str(path))
    doc.close()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def make_financebench_tree(root: Path) -> Path:
    """Create a tiny FinanceBench-shaped tree with two PDFs and four questions."""
    pdf_dir = root / "pdfs"
    write_pdf(
        pdf_dir / "ACME_2022_10K.pdf",
        [
            "ACME Corporation cover page and table of contents.",
            "ACME FY2022 capital expenditures were 1577 million USD on the cash flow statement.",
        ],
    )
    write_pdf(
        pdf_dir / "ACME_2022_10Q.pdf",
        [
            "ACME quarterly report. Net sales were not disclosed on this page.",
            "ACME FY2022 Q2 net sales were 900 million USD.",
        ],
    )
    questions = [
        {
            "financebench_id": "financebench_id_00001",
            "company": "Acme",
            "doc_name": "ACME_2022_10K",
            "question_type": "metrics-generated",
            "question_reasoning": "Information extraction",
            "question": "What is the FY2022 capital expenditure amount for Acme?",
            "answer": "$1577.00",
            "justification": "Direct extraction from cash flow statement.",
            "dataset_subset_label": "OPEN_SOURCE",
            "evidence": [
                {
                    "evidence_doc_name": "ACME_2022_10K",
                    "evidence_page_num": 1,
                    "evidence_text": "ACME FY2022 capital expenditures were 1577 million USD on the cash flow statement.",
                }
            ],
        },
        {
            "financebench_id": "financebench_id_00002",
            "company": "Acme",
            "doc_name": "ACME_2022_10K",
            "question_type": "domain-relevant",
            "question_reasoning": "Logical reasoning (based on numerical reasoning)",
            "question": "Did Acme report capex on the cover page?",
            "answer": "No",
            "justification": "Cover page has no capex.",
            "dataset_subset_label": "OPEN_SOURCE",
            "evidence": [
                {
                    "doc_name": "ACME_2022_10K",
                    "evidence_page_num": 0,
                    "evidence_text": "ACME Corporation cover page and table of contents.",
                }
            ],
        },
        {
            "financebench_id": "financebench_id_00003",
            "company": "Acme",
            "doc_name": "ACME_2022_10Q",
            "question_type": "novel-generated",
            "question_reasoning": None,
            "question": "What were Acme net sales in Q2 FY2022?",
            "answer": "$900 million",
            "justification": None,
            "dataset_subset_label": "OPEN_SOURCE",
            "evidence": [
                {
                    "evidence_doc_name": "ACME_2022_10Q",
                    "evidence_page_num": 1,
                    "evidence_text": "ACME FY2022 Q2 net sales were 900 million USD.",
                }
            ],
        },
        {
            "financebench_id": "financebench_id_00004",
            "company": "Acme",
            "doc_name": "ACME_2022_10K",
            "question_type": "metrics-generated",
            "question_reasoning": "Numerical reasoning",
            "question": "Using both the 10-K capex page and the 10-Q sales page, what is capex?",
            "answer": "$1577.00",
            "justification": "Cross-document evidence.",
            "dataset_subset_label": "OPEN_SOURCE",
            "evidence": [
                {
                    "evidence_doc_name": "ACME_2022_10K",
                    "evidence_page_num": 1,
                    "evidence_text": "ACME FY2022 capital expenditures were 1577 million USD on the cash flow statement.",
                },
                {
                    "evidence_doc_name": "ACME_2022_10Q",
                    "evidence_page_num": 1,
                    "evidence_text": "ACME FY2022 Q2 net sales were 900 million USD.",
                },
            ],
        },
    ]
    documents = [
        {
            "doc_name": "ACME_2022_10K",
            "company": "Acme",
            "doc_type": "10K",
            "period": "FY2022",
            "gics_sector": "Industrials",
            "ticker": "ACME",
        },
        {
            "doc_name": "ACME_2022_10Q",
            "company": "Acme",
            "doc_type": "10Q",
            "period": "Q2 FY2022",
            "gics_sector": "Industrials",
            "ticker": "ACME",
        },
    ]
    write_jsonl(root / "data" / "financebench_open_source.jsonl", questions)
    write_jsonl(root / "data" / "financebench_document_information.jsonl", documents)
    return root
