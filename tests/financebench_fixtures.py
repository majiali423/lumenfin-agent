"""Tiny FinanceBench-shaped fixtures for unit tests (not the 150-question dataset)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lumenfin.eval.financebench.schema import EvidenceSpan, FinanceBenchQuestion


def evidence(
    *,
    doc_name: str,
    page_zero: int,
    text: str,
) -> dict[str, Any]:
    return {
        "evidence_doc_name": doc_name,
        "evidence_page_num": page_zero,
        "evidence_text": text,
    }


def question_row(
    *,
    financebench_id: str,
    company: str,
    doc_name: str,
    question: str,
    question_type: str = "metrics-generated",
    question_reasoning: str | None = "Information extraction",
    page_zero: int = 0,
    evidence_text: str = "Revenue was 100 million in FY2018.",
) -> dict[str, Any]:
    return {
        "financebench_id": financebench_id,
        "company": company,
        "doc_name": doc_name,
        "question": question,
        "answer": "100",
        "justification": "extracted",
        "question_type": question_type,
        "question_reasoning": question_reasoning,
        "dataset_subset_label": "OPEN_SOURCE",
        "doc_type": "10k",
        "doc_link": "https://example.invalid/filing.pdf",
        "evidence": [evidence(doc_name=doc_name, page_zero=page_zero, text=evidence_text)],
    }


def document_row(doc_name: str, company: str) -> dict[str, Any]:
    return {
        "doc_name": doc_name,
        "company": company,
        "doc_type": "10k",
        "gics_sector": "Information Technology",
        "ticker": "TEST",
        "pdf_filename": f"{doc_name}.pdf",
        "doc_link": "https://example.invalid/filing.pdf",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def write_tiny_dataset(root: Path, *, n_questions: int = 3) -> Path:
    questions = []
    documents = {}
    for index in range(n_questions):
        doc_name = f"TESTCO_201{index}_10K"
        company = "TestCo"
        documents[doc_name] = document_row(doc_name, company)
        questions.append(
            question_row(
                financebench_id=f"financebench_id_{index:05d}",
                company=company,
                doc_name=doc_name,
                question=f"What is FY201{index} revenue for TestCo?",
                page_zero=index,
            )
        )
    write_jsonl(root / "financebench_open_source.jsonl", questions)
    write_jsonl(root / "financebench_document_information.jsonl", list(documents.values()))
    (root / "pdfs").mkdir(exist_ok=True)
    return root


def parsed_question(**kwargs: Any) -> FinanceBenchQuestion:
    row = question_row(**kwargs)
    return FinanceBenchQuestion(
        financebench_id=row["financebench_id"],
        case_id=f"fb-{row['financebench_id']}",
        question=row["question"],
        answer=row["answer"],
        justification=row["justification"],
        question_type=row["question_type"],
        question_reasoning=row["question_reasoning"] or "",
        domain_question_num="",
        company=row["company"],
        doc_name=row["doc_name"],
        dataset_subset_label="OPEN_SOURCE",
        evidence=(
            EvidenceSpan(
                evidence_doc_name=row["doc_name"],
                evidence_page_num_zero=row["evidence"][0]["evidence_page_num"],
                evidence_page_num_one=row["evidence"][0]["evidence_page_num"] + 1,
                evidence_text=row["evidence"][0]["evidence_text"],
            ),
        ),
    )
