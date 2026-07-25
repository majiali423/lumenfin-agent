"""ARCHIVED AUDIT SCRIPT.

Historical purpose: convert entire downloaded SEC HTML text into paginated PDFs.
Replacement: scripts/fetch_sec_fixtures.py and manifested minimal extracts.
Last compatible schema: historical e2e_real fixture layout.
Not part of the supported release interface; do not run on production fixtures.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "fixtures" / "e2e_real"


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", html)
    html = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p>", "\n\n", html)
    html = re.sub(r"(?i)</div>", "\n", html)
    html = re.sub(r"(?i)</tr>", "\n", html)
    html = re.sub(r"(?i)</(h[1-6]|li|table)>", "\n\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_to_pdf(text: str, out: Path, *, max_pages: int | None = None, chars_per_page: int = 2800) -> int:
    doc = fitz.open()
    pages = 0
    for start in range(0, len(text), chars_per_page):
        if max_pages is not None and pages >= max_pages:
            break
        chunk = text[start : start + chars_per_page]
        page = doc.new_page(width=612, height=792)
        rect = fitz.Rect(36, 36, 576, 756)
        page.insert_textbox(rect, chunk, fontsize=9, fontname="helv", align=0)
        pages += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    doc.close()
    return pages


def main() -> None:
    jobs = [
        ("aapl-20240928.htm", "aapl_fy2024_10k_sec.pdf", 60),
        ("nvda-20250126.htm", "nvda_fy2025_10k_sec.pdf", 60),
        ("msft-20240630.htm", "msft_fy2024_10k_sec.pdf", 40),
        ("tsla-20241231.htm", "tsla_fy2024_10k_sec.pdf", 60),
        ("aapl-20240928.htm", "aapl_fy2024_10k_sec_long.pdf", 220),
        ("msft-20240630.htm", "msft_fy2024_10k_sec_long.pdf", 220),
    ]
    for src_name, pdf_name, max_pages in jobs:
        src = SRC / src_name
        text = html_to_text(src.read_text(encoding="utf-8", errors="ignore"))
        header = (
            "SOURCE: SEC EDGAR filing converted to PDF for LumenFin E2E audit.\n"
            f"Original file: {src_name}\n"
            "Converter: pymupdf text pagination (content from HTML extract, not synthetic numbers).\n\n"
        )
        out = SRC / pdf_name
        n = text_to_pdf(header + text, out, max_pages=max_pages)
        print(f"{pdf_name}: pages={n} chars_in={len(text)} size={out.stat().st_size}")


if __name__ == "__main__":
    main()
