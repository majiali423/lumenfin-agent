"""Offline markdown link checker for release docs (relative links only)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "README.md",
    "README.zh-CN.md",
    "CHANGELOG.md",
    "Release_Checklist.md",
    "docs/README.md",
    "docs/ARCHITECTURE.md",
    "docs/PRODUCTION_LIMITATIONS.md",
    "docs/MULTI_TENANCY_BOUNDARY.md",
    "docs/PORTFOLIO_RELEASE_REPORT.md",
    "docs/DEMO_GUIDE.md",
    "docs/CONFIGURATION.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ARCHITECTURE_INDEX.md",
    "docs/STRUCTURED_ANSWER.md",
]
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def main() -> int:
    broken: list[str] = []
    local_only: list[str] = []
    for rel in TARGETS:
        path = ROOT / rel
        if not path.exists():
            broken.append(f"{rel}: missing document")
            continue
        text = path.read_text(encoding="utf-8")
        if "../../finagentbench" in text:
            local_only.append(rel)
        for _, target in LINK.findall(text):
            link = target.split("#")[0].strip()
            if not link or link.startswith(("http://", "https://", "mailto:")):
                continue
            if not (path.parent / link).exists():
                broken.append(f"{rel}: {link}")
    for item in broken:
        print(f"BROKEN {item}")
    for item in local_only:
        print(f"LOCAL-ONLY-SIBLING-PATH {item}")
    if broken or local_only:
        return 1
    print(f"OK: {len(TARGETS)} documents, all relative links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
