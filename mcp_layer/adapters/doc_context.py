"""Build document_contexts from bundled MCP research notes."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

DOCS_DIR = Path(__file__).resolve().parents[1] / "data" / "docs"

_FILENAME_COMPANY = {
    "apple": "Apple",
    "microsoft": "Microsoft",
    "nvidia": "NVIDIA",
}


def _guess_company(filename: str) -> list[str]:
    lowered = filename.lower()
    for token, company in _FILENAME_COMPANY.items():
        if token in lowered:
            return [company]
    return []


def load_research_document_contexts(docs_dir: Path | None = None) -> list[dict[str, Any]]:
    root = docs_dir or DOCS_DIR
    if not root.exists():
        return []

    contexts: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        contexts.append(
            {
                "document_id": f"mcp_doc_{path.stem}",
                "filename": path.name,
                "detected_companies": _guess_company(path.name),
                "text": text,
                "pages": [text],
                "excerpt": text[:500],
                "source_type": "mcp_research_note",
            }
        )
    return contexts


def resolve_company_hint(query: str, explicit: str | None = None) -> str | None:
    if explicit:
        for token, company in _FILENAME_COMPANY.items():
            if explicit.lower() in {token, company.lower()}:
                return company
        return explicit.strip().title()

    lowered = query.lower()
    for token, company in _FILENAME_COMPANY.items():
        if token in lowered or company.lower() in lowered:
            return company
    return None
