"""Download FinanceBench PDFs from HuggingFace merged ``doc_link`` URLs.

PDFs are never committed. SEC.gov requires a descriptive User-Agent.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from lumenfin.eval.financebench.loader import (
    load_financebench_dataset,
    resolve_financebench_source,
)

log = logging.getLogger("lumenfin.eval.financebench.fetch_pdfs")

_ACCESSION_RE = re.compile(r"(\d{10}-\d{2}-\d{6})")
_DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_SEC_UA = "LumenFinAgent/0.1 (financebench-eval; research@example.com)"


def _headers(url: str) -> dict[str, str]:
    host = urlparse(url).netloc.lower()
    if "sec.gov" in host:
        ua = (os.getenv("SEC_USER_AGENT") or "").strip() or _SEC_UA
    else:
        ua = _DEFAULT_UA
    return {
        "User-Agent": ua,
        "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    }


def _adobe_pdf_url(url: str) -> str | None:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    target = qs.get("pdfTarget", [None])[0]
    if not target:
        return None
    try:
        decoded = base64.b64decode(unquote(target)).decode("utf-8")
    except Exception:
        return None
    if decoded.startswith("http"):
        return decoded
    return None


def _sec_pdf_urls(url: str) -> list[str]:
    match = _ACCESSION_RE.search(url)
    if not match:
        return []
    accession = match.group(1)
    compact = accession.replace("-", "")
    cik = str(int(accession.split("-")[0]))
    return [
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{compact}/{accession}.pdf",
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{compact}/{accession}-index.html",
    ]


def candidate_urls(doc_link: str) -> list[str]:
    urls: list[str] = []
    adobe = _adobe_pdf_url(doc_link)
    if adobe:
        urls.append(adobe)
    urls.append(doc_link)
    if "/sec-filings/filter/annual-filings/content/" in doc_link:
        urls.append(doc_link.replace("/sec-filings/filter/annual-filings/content/", "/financial-information/sec-filings/content/"))
    urls.extend(_sec_pdf_urls(doc_link))
    if adobe:
        urls.extend(_sec_pdf_urls(adobe))
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _looks_like_pdf(data: bytes) -> bool:
    return len(data) >= 1024 and data[:5].startswith(b"%PDF")


def _download_one(url: str, dest: Path, timeout: int = 90) -> bool:
    import urllib.error
    import urllib.request

    if url.lower().endswith("-index.html") or url.lower().endswith(".htm"):
        return False
    req = urllib.request.Request(url, headers=_headers(url))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            content_type = str(resp.headers.get("Content-Type") or "")
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False
    if "html" in content_type.lower() and not _looks_like_pdf(data):
        return False
    if not _looks_like_pdf(data):
        return False
    dest.write_bytes(data)
    return True


def fetch_pdfs(
    *,
    dataset_root: Path | None = None,
    pdf_dir: Path | None = None,
    sleep_s: float = 0.2,
    expected_questions: int | None = 150,
) -> dict[str, Any]:
    """Download missing PDFs named ``{doc_name}.pdf``. Returns a summary."""
    source = resolve_financebench_source(dataset_root)
    questions, documents, paths = load_financebench_dataset(
        source,
        expected_questions=expected_questions,
        require_pdfs=False,
    )
    target_dir = pdf_dir or paths.pdf_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    unique: dict[str, str] = {}
    for question in questions:
        if question.doc_name not in unique and question.doc_link:
            unique[question.doc_name] = question.doc_link
        elif question.doc_name not in unique:
            info = documents.get(question.doc_name)
            if info and info.doc_link:
                unique[question.doc_name] = info.doc_link
    ok = 0
    skipped = 0
    missing: list[str] = []
    for doc_name, link in sorted(unique.items()):
        dest = target_dir / f"{doc_name}.pdf"
        if dest.is_file() and dest.stat().st_size > 1024:
            skipped += 1
            ok += 1
            continue
        downloaded = False
        for url in candidate_urls(link):
            if _download_one(url, dest):
                downloaded = True
                log.info("downloaded %s", doc_name)
                break
            time.sleep(sleep_s)
        if downloaded:
            ok += 1
        else:
            missing.append(doc_name)
            log.warning("missing pdf %s", doc_name)
        time.sleep(sleep_s)
    summary = {
        "unique_docs": len(unique),
        "present": ok,
        "skipped_existing": skipped,
        "missing": missing,
        "pdf_dir": str(target_dir),
    }
    (target_dir.parent / "pdf_fetch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download FinanceBench PDFs (gitignored)")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--pdf-dir", type=Path, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    summary = fetch_pdfs(dataset_root=args.dataset_root, pdf_dir=args.pdf_dir, sleep_s=args.sleep)
    printable = dict(summary)
    printable["missing_count"] = len(summary["missing"])
    print(json.dumps({k: v for k, v in printable.items() if k != "missing"}))
    return 0 if not summary["missing"] else 2
