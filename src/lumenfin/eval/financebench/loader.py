from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import EXPECTED_OPEN_SOURCE_QUESTIONS
from .schema import DocumentInfo, EvidenceSpan, FinanceBenchQuestion


class FinanceBenchLoadError(ValueError):
    """Raised when FinanceBench JSONL is missing, corrupt, or incomplete."""


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    questions_path: Path
    documents_path: Path | None
    pdf_dir: Path
    merged: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FinanceBenchLoadError(f"missing JSONL file: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FinanceBenchLoadError(f"cannot read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise FinanceBenchLoadError(
                f"corrupt JSON on line {line_no} of {path.name}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise FinanceBenchLoadError(f"expected object on line {line_no} of {path.name}")
        rows.append(payload)
    return rows


def default_source_candidates(repo_root: Path | None = None) -> list[Path]:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    env = (os.getenv("FINANCEBENCH_DIR") or "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            root / "data" / "external" / "financebench-src",
            root / "data" / "external" / "financebench",
            Path.home() / "financebench",
        ]
    )
    return candidates


def resolve_financebench_source(source_dir: str | Path | None = None) -> Path:
    if source_dir is not None:
        path = Path(source_dir).expanduser().resolve()
        if not path.is_dir():
            raise FinanceBenchLoadError(f"FinanceBench source directory not found: {path}")
        return path
    for candidate in default_source_candidates():
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        try:
            discover_financebench_paths(resolved)
        except FinanceBenchLoadError:
            continue
        return resolved
    raise FinanceBenchLoadError(
        "FinanceBench source not found. Clone PatronusAI/financebench or place "
        "financebench_merged.jsonl under data/external/financebench-src/data/, "
        "or set FINANCEBENCH_DIR."
    )


def discover_financebench_paths(source_dir: str | Path) -> DatasetPaths:
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise FinanceBenchLoadError(f"FinanceBench source directory not found: {root}")

    official = [
        root / "financebench_open_source.jsonl",
        root / "data" / "financebench_open_source.jsonl",
    ]
    questions_path = next((path for path in official if path.is_file()), None)
    merged = False
    if questions_path is None:
        merged_candidates = [
            root / "financebench_merged.jsonl",
            root / "data" / "financebench_merged.jsonl",
        ]
        questions_path = next((path for path in merged_candidates if path.is_file()), None)
        merged = questions_path is not None
    if questions_path is None:
        raise FinanceBenchLoadError(
            "financebench_open_source.jsonl or financebench_merged.jsonl not found"
        )

    documents_path = None
    if not merged:
        doc_candidates = [
            questions_path.parent / "financebench_document_information.jsonl",
            root / "financebench_document_information.jsonl",
            root / "data" / "financebench_document_information.jsonl",
        ]
        documents_path = next((path for path in doc_candidates if path.is_file()), None)
        if documents_path is None:
            raise FinanceBenchLoadError(
                "financebench_document_information.jsonl not found next to questions file"
            )

    pdf_candidates = [root / "pdfs", questions_path.parent.parent / "pdfs", root / "data" / "pdfs"]
    pdf_dir = next((path for path in pdf_candidates if path.is_dir()), pdf_candidates[0])
    return DatasetPaths(
        root=root,
        questions_path=questions_path,
        documents_path=documents_path,
        pdf_dir=pdf_dir,
        merged=merged,
    )


def normalize_doc_name(value: str | None) -> str:
    text = str(value or "").strip()
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    return text


def case_id_for(financebench_id: str) -> str:
    token = str(financebench_id or "").strip()
    if not token:
        raise FinanceBenchLoadError("financebench_id is required")
    if token.startswith("fb-"):
        return token
    return f"fb-{token}"


def zero_to_one_page(page_zero: int) -> int:
    return int(page_zero) + 1


def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def _parse_page_num(value: Any, *, field: str, case_id: str) -> int:
    if value is None or value == "":
        raise FinanceBenchLoadError(f"{case_id}: missing {field}")
    try:
        page = int(value)
    except (TypeError, ValueError) as exc:
        raise FinanceBenchLoadError(f"{case_id}: {field} is not an integer: {value!r}") from exc
    if page < 0:
        raise FinanceBenchLoadError(f"{case_id}: {field} must be >= 0, got {page}")
    return page


def parse_evidence(raw: Any, *, question_doc_name: str, case_id: str) -> tuple[EvidenceSpan, ...]:
    if not isinstance(raw, list) or not raw:
        raise FinanceBenchLoadError(f"{case_id}: evidence must be a non-empty list")
    spans: list[EvidenceSpan] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise FinanceBenchLoadError(f"{case_id}: evidence[{index}] must be an object")
        doc_name = normalize_doc_name(
            item.get("evidence_doc_name") or item.get("doc_name") or question_doc_name
        )
        if not doc_name:
            raise FinanceBenchLoadError(f"{case_id}: evidence[{index}] missing document name")
        page_zero = _parse_page_num(
            item.get("evidence_page_num"),
            field=f"evidence[{index}].evidence_page_num",
            case_id=case_id,
        )
        evidence_text = str(item.get("evidence_text") or "").strip()
        if not evidence_text:
            raise FinanceBenchLoadError(f"{case_id}: evidence[{index}] missing evidence_text")
        spans.append(
            EvidenceSpan(
                evidence_doc_name=doc_name,
                evidence_page_num_zero=page_zero,
                evidence_page_num_one=zero_to_one_page(page_zero),
                evidence_text=evidence_text,
                evidence_text_full_page=str(item.get("evidence_text_full_page") or ""),
            )
        )
    return tuple(spans)


def parse_document_info(raw: dict[str, Any]) -> DocumentInfo:
    doc_name = normalize_doc_name(raw.get("doc_name") or raw.get("document_name"))
    if not doc_name:
        raise FinanceBenchLoadError("document information row missing doc_name")
    return DocumentInfo(
        doc_name=doc_name,
        company=_optional_str(raw.get("company")),
        doc_type=_optional_str(raw.get("doc_type") or raw.get("document_type")),
        period=_optional_str(raw.get("period") or raw.get("doc_period")),
        gics_sector=_optional_str(raw.get("gics_sector")),
        ticker=_optional_str(raw.get("ticker")),
        pdf_filename=_optional_str(raw.get("pdf_filename") or raw.get("filename"))
        or f"{doc_name}.pdf",
        doc_link=_optional_str(raw.get("doc_link") or raw.get("url")),
    )


def parse_question(
    raw: dict[str, Any],
    *,
    documents: dict[str, DocumentInfo] | None = None,
) -> FinanceBenchQuestion:
    financebench_id = _optional_str(raw.get("financebench_id"))
    if not financebench_id:
        raise FinanceBenchLoadError("question missing financebench_id")
    case_id = case_id_for(financebench_id)
    question = str(raw.get("question") or "").strip()
    if not question:
        raise FinanceBenchLoadError(f"{case_id}: missing question text")
    doc_name = normalize_doc_name(raw.get("doc_name"))
    if not doc_name:
        raise FinanceBenchLoadError(f"{case_id}: missing doc_name")
    company = _optional_str(raw.get("company"))
    if not company:
        raise FinanceBenchLoadError(f"{case_id}: missing company")
    evidence = parse_evidence(raw.get("evidence"), question_doc_name=doc_name, case_id=case_id)
    document = (documents or {}).get(doc_name)
    return FinanceBenchQuestion(
        financebench_id=financebench_id,
        case_id=case_id,
        question=question,
        answer=_optional_str(raw.get("answer")),
        justification=_optional_str(raw.get("justification")),
        question_type=_optional_str(raw.get("question_type")),
        question_reasoning=_optional_str(raw.get("question_reasoning")),
        domain_question_num=_optional_str(raw.get("domain_question_num")),
        company=company,
        doc_name=doc_name,
        dataset_subset_label=_optional_str(raw.get("dataset_subset_label")) or "OPEN_SOURCE",
        evidence=evidence,
        document=document,
    )


def documents_from_merged_rows(rows: list[dict[str, Any]]) -> dict[str, DocumentInfo]:
    documents: dict[str, DocumentInfo] = {}
    for raw in rows:
        info = parse_document_info(raw)
        documents[info.doc_name] = info
    return documents


def load_document_information(path: Path) -> dict[str, DocumentInfo]:
    return {info.doc_name: info for info in (parse_document_info(raw) for raw in _read_jsonl(path))}


def load_questions(
    path: Path,
    *,
    documents: dict[str, DocumentInfo] | None = None,
) -> list[FinanceBenchQuestion]:
    questions = [parse_question(raw, documents=documents) for raw in _read_jsonl(path)]
    ids = [item.financebench_id for item in questions]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise FinanceBenchLoadError(f"duplicate financebench_id values: {duplicates}")
    return questions


def validate_open_source_count(
    questions: Iterable[FinanceBenchQuestion],
    *,
    expected: int = EXPECTED_OPEN_SOURCE_QUESTIONS,
) -> None:
    count = sum(1 for _ in questions)
    if expected and count != expected:
        raise FinanceBenchLoadError(f"expected {expected} open-source questions, found {count}")


def referenced_doc_names(questions: Iterable[FinanceBenchQuestion]) -> list[str]:
    names: set[str] = set()
    for question in questions:
        names.add(question.doc_name)
        for span in question.evidence:
            names.add(span.evidence_doc_name)
    return sorted(names)


def resolve_pdf_path(pdf_dir: Path, doc_name: str, info: DocumentInfo | None = None) -> Path | None:
    filename = (info.pdf_filename if info and info.pdf_filename else f"{doc_name}.pdf")
    for path in (pdf_dir / filename, pdf_dir / f"{doc_name}.pdf", pdf_dir / f"{doc_name}.PDF"):
        if path.is_file() and path.stat().st_size > 100:
            return path
    return None


def load_financebench_dataset(
    source_dir: str | Path,
    *,
    expected_questions: int | None = EXPECTED_OPEN_SOURCE_QUESTIONS,
    require_pdfs: bool = False,
) -> tuple[list[FinanceBenchQuestion], dict[str, DocumentInfo], DatasetPaths]:
    paths = discover_financebench_paths(source_dir)
    rows = _read_jsonl(paths.questions_path)
    if paths.merged:
        documents = documents_from_merged_rows(rows)
    else:
        documents = load_document_information(paths.documents_path) if paths.documents_path else {}
    questions = [parse_question(raw, documents=documents) for raw in rows]
    ids = [item.financebench_id for item in questions]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise FinanceBenchLoadError(f"duplicate financebench_id values: {duplicates}")
    if expected_questions:
        validate_open_source_count(questions, expected=expected_questions)
    missing_docs = [name for name in referenced_doc_names(questions) if name not in documents]
    if missing_docs:
        raise FinanceBenchLoadError(
            f"questions reference documents missing from document information: {missing_docs}"
        )
    if require_pdfs:
        missing_pdfs = [
            name
            for name in referenced_doc_names(questions)
            if resolve_pdf_path(paths.pdf_dir, name, documents.get(name)) is None
        ]
        if missing_pdfs:
            raise FinanceBenchLoadError(f"missing PDFs for documents: {missing_pdfs}")
    return questions, documents, paths
