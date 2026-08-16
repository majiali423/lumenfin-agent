from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...documents import parse_pdf_document
from ...rag.chunking import chunk_document
from .constants import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    EXPECTED_OPEN_SOURCE_QUESTIONS,
    SCHEMA_VERSION,
    SPLIT_VERSION,
)
from .loader import (
    DatasetPaths,
    FinanceBenchLoadError,
    load_financebench_dataset,
    referenced_doc_names,
    resolve_pdf_path,
)
from .qrels import map_chunks_to_qrels, page_qrel_records
from .reporting import sha256_bytes, sha256_file, write_json, write_jsonl
from .schema import DocumentInfo, FinanceBenchQuestion
from .split import assign_splits, split_manifest
from .taxonomy import classify_case


def stamp_eval_document(
    parsed: dict[str, Any],
    *,
    doc_name: str,
    company: str,
    pdf_path: Path,
) -> dict[str, Any]:
    """Keep parser output, but pin FinanceBench identity for retrieval filters."""
    stamped = dict(parsed)
    stamped["document_id"] = doc_name
    stamped["source_document_id"] = doc_name
    stamped["filename"] = f"{doc_name}.pdf"
    stamped["path"] = str(pdf_path)
    stamped["issuer_companies"] = [company] if company else list(stamped.get("issuer_companies") or [])
    stamped["detected_companies"] = list(stamped["issuer_companies"])
    stamped["mentioned_companies"] = list(
        dict.fromkeys(list(stamped.get("mentioned_companies") or []) + stamped["issuer_companies"])
    )
    return stamped


def load_indexed_documents(
    *,
    questions: list[FinanceBenchQuestion],
    documents: dict[str, DocumentInfo],
    paths: DatasetPaths,
    max_chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], list[str]]:
    parsed_docs: dict[str, dict[str, Any]] = {}
    chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for doc_name in referenced_doc_names(questions):
        info = documents.get(doc_name)
        pdf_path = resolve_pdf_path(paths.pdf_dir, doc_name, info)
        if pdf_path is None:
            missing.append(doc_name)
            continue
        company = (info.company if info and info.company else "")
        if not company:
            for question in questions:
                if question.doc_name == doc_name or any(
                    span.evidence_doc_name == doc_name for span in question.evidence
                ):
                    company = question.company
                    break
        parsed = stamp_eval_document(
            parse_pdf_document(pdf_path),
            doc_name=doc_name,
            company=company,
            pdf_path=pdf_path,
        )
        parsed_docs[doc_name] = parsed
        chunks_by_doc[doc_name] = chunk_document(
            parsed,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )
    return parsed_docs, chunks_by_doc, missing


def prepare_financebench_eval(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    expected_questions: int | None = EXPECTED_OPEN_SOURCE_QUESTIONS,
    require_pdfs: bool = False,
    parse_pdfs: bool = True,
    max_chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    questions, documents, paths = load_financebench_dataset(
        source_dir,
        expected_questions=expected_questions,
        require_pdfs=require_pdfs,
    )
    assignment = assign_splits(questions)
    manifest_split = split_manifest(questions, assignment)
    parsed_docs: dict[str, dict[str, Any]] = {}
    chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
    missing_pdfs: list[str] = []
    case_qrels = []
    if parse_pdfs:
        parsed_docs, chunks_by_doc, missing_pdfs = load_indexed_documents(
            questions=questions,
            documents=documents,
            paths=paths,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )
        if require_pdfs and missing_pdfs:
            raise FinanceBenchLoadError(f"missing PDFs: {missing_pdfs}")
        for question in questions:
            related_chunks: list[dict[str, Any]] = []
            for span in question.evidence:
                related_chunks.extend(chunks_by_doc.get(span.evidence_doc_name) or [])
            if question.doc_name not in {span.evidence_doc_name for span in question.evidence}:
                related_chunks.extend(chunks_by_doc.get(question.doc_name) or [])
            case_qrels.append(map_chunks_to_qrels(question, related_chunks).to_dict())

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "questions.jsonl", (item.to_public_dict() for item in questions))
    write_jsonl(out / "documents.jsonl", (item.to_public_dict() for item in documents.values()))
    write_jsonl(out / "qrels_page.jsonl", page_qrel_records(questions))
    if case_qrels:
        write_jsonl(out / "qrels_chunk.jsonl", case_qrels)
    write_json(out / "split_manifest.json", manifest_split)
    dataset_hash = sha256_file(paths.questions_path)
    split_hash = sha256_bytes(
        json.dumps(manifest_split, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    page_provenance_ok = all(item.get("page_provenance_ok", True) for item in case_qrels) if case_qrels else True
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split_version": SPLIT_VERSION,
        "source_root": str(paths.root),
        "questions_path": str(paths.questions_path),
        "documents_path": str(paths.documents_path),
        "pdf_dir": str(paths.pdf_dir),
        "question_count": len(questions),
        "document_count": len(documents),
        "missing_pdfs": missing_pdfs,
        "dataset_hash": dataset_hash,
        "split_manifest_hash": split_hash,
        "page_qrels": len(page_qrel_records(questions)),
        "chunk_qrels_cases": len(case_qrels),
        "page_provenance_ok": page_provenance_ok,
        "page_index": {
            "financebench": "zero-indexed evidence_page_num",
            "lumenfin": "1-indexed chunk.page",
            "mapping": "lumenfin_page = evidence_page_num + 1",
        },
        "gold_relevance": {
            "page": "query_id -> evidence_doc_name + 1-indexed page",
            "page_chunk": "same document AND same gold page",
            "span_chunk": "same document AND auditable evidence-span overlap",
            "chunk_deprecated": "union of page_chunk and span_chunk ids",
        },
        "labels": [classify_case(item) | {"case_id": item.case_id} for item in questions],
        "pdfs_parsed": bool(parsed_docs),
        "chunk_size": max_chunk_chars,
        "chunk_overlap": overlap_chars,
    }
    write_json(out / "manifest.json", manifest)
    return {
        "manifest": manifest,
        "questions": questions,
        "documents": documents,
        "paths": paths,
        "assignment": assignment,
        "parsed_documents": parsed_docs,
        "chunks_by_doc": chunks_by_doc,
        "output_dir": out,
    }
