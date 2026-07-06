"""One-off rebrand helper: LumenFin naming + import paths."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".venv", "outputs", "test_artifacts", "data", "uploads", "__pycache__", ".git"}
SKIP_SUFFIX = ".egg-info"

REPLACEMENTS = [
    ("LumenFinAgentSystem", "LumenFinAgentSystem"),
    ("LumenFinAnalysisService", "LumenFinAnalysisService"),
    ("LumenFin", "LumenFin"),
    ("lumenfin-worker", "lumenfin-worker"),
    ("lumenfin-api", "lumenfin-api"),
    ("lumenfin-agent", "lumenfin-agent"),
    ("LumenFin", "LumenFin"),
    ("Projects\lumenfin-agent", r"Projects\lumenfin-agent"),
    ("LumenFin Intelligence Report", "LumenFin Intelligence Report"),
    ("LumenFin Agent v1.0", "LumenFin Agent v1.0"),
    ("LumenFin Agent", "LumenFin Agent"),
    ("lumenfin_chunks", "lumenfin_chunks"),
    (
        "LumenFin: evidence-grounded multi-agent financial diligence built with LangGraph.",
        "LumenFin: evidence-grounded multi-agent financial diligence built with LangGraph.",
    ),
    ("Run the LumenFin agent demo.", "Run the LumenFin agent demo."),
    ("LumenFin agent platform.", "LumenFin agent platform."),
    ("API package for the LumenFin service.", "API package for the LumenFin service."),
    ("lumenfin", "lumenfin"),
]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".html",
    ".json",
    ".toml",
    ".yml",
    ".txt",
    ".ps1",
    ".example",
    ".env",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".dockerignore",
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts) or any(
        part.endswith(SKIP_SUFFIX) for part in path.parts
    )


def main() -> int:
    updated = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".env", ".env.example"}:
            continue
        original = path.read_text(encoding="utf-8")
        content = original
        for old, new in REPLACEMENTS:
            content = content.replace(old, new)
        if content != original:
            path.write_text(content, encoding="utf-8", newline="\n")
            updated += 1
    db_old = ROOT / "data" / "lumenfin.db"
    db_new = ROOT / "data" / "lumenfin.db"
    if db_old.exists() and not db_new.exists():
        db_old.rename(db_new)
    print(f"Updated {updated} files under {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
