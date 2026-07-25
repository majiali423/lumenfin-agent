#!/usr/bin/env python3
"""Fetch, build and verify minimized SEC-derived test fixtures.

Full EDGAR filings are downloaded only into ``.local-fixtures/sec/downloads``,
which is ignored. Committed fixtures are minimized extracts or derived PDFs
described by ``tests/fixtures/sec/manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import fitz
import httpx

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "sec"
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
DOWNLOAD_ROOT = ROOT / ".local-fixtures" / "sec" / "downloads"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def fetch_sources() -> int:
    user_agent = (os.getenv("SEC_USER_AGENT") or "").strip()
    if not user_agent:
        raise RuntimeError("SEC_USER_AGENT is required for SEC fair-access downloads")
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    entries = load_manifest()["fixtures"]
    with httpx.Client(headers={"User-Agent": user_agent}, timeout=60, follow_redirects=True) as client:
        for entry in entries:
            url = entry.get("source_url")
            expected = entry.get("source_sha256")
            if not url:
                continue
            target = DOWNLOAD_ROOT / f"{entry['id']}.source.html"
            response = client.get(url)
            response.raise_for_status()
            target.write_bytes(response.content)
            actual = sha256_file(target)
            if expected and actual != expected:
                raise RuntimeError(
                    f"Checksum mismatch for {entry['id']}: expected={expected} actual={actual}"
                )
            print(f"fetched {entry['id']} sha256={actual} path={target}")
            time.sleep(0.2)
    return 0


def _write_pdf(source: Path, target: Path, *, pages: int) -> None:
    text = source.read_text(encoding="utf-8")
    doc = fitz.open()
    for page_number in range(1, pages + 1):
        page = doc.new_page(width=612, height=792)
        rect = fitz.Rect(40, 40, 572, 752)
        body = (
            f"DERIVED TEST FIXTURE — page {page_number}/{pages}\n"
            "Not an official SEC PDF. Selected/paraphrased content only.\n\n"
            f"{text}\n\nStress-page marker: {page_number}."
        )
        page.insert_textbox(rect, body, fontsize=8.5, fontname="helv")
    doc.set_metadata(
        {
            "title": target.stem,
            "author": "LumenFin fixture builder",
            "subject": "Minimized SEC-derived test fixture",
            "creator": "scripts/fetch_sec_fixtures.py",
            "producer": "PyMuPDF",
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    doc.save(target, garbage=4, deflate=True)
    doc.close()


def build_derived() -> int:
    for entry in load_manifest()["fixtures"]:
        derived = entry.get("derived_pdf")
        if not derived:
            continue
        _write_pdf(
            ROOT / entry["source_extract"],
            ROOT / derived["path"],
            pages=int(derived["pages"]),
        )
        path = ROOT / derived["path"]
        print(f"built {path} sha256={sha256_file(path)}")
    return 0


def verify() -> int:
    failures = 0
    for entry in load_manifest()["fixtures"]:
        for path_key, checksum_key in (
            ("fixture_path", "fixture_sha256"),
            ("source_extract", "source_extract_sha256"),
        ):
            raw_path = entry.get(path_key)
            expected = entry.get(checksum_key)
            if not raw_path:
                continue
            path = ROOT / raw_path
            actual = sha256_file(path) if path.exists() else "missing"
            ok = bool(expected) and actual == expected
            print(f"{'PASS' if ok else 'FAIL'} {raw_path} sha256={actual}")
            failures += 0 if ok else 1
    return 1 if failures else 0


def inventory(paths: list[str]) -> int:
    for pattern in paths:
        matches = list(ROOT.glob(pattern))
        for path in matches:
            if path.is_file():
                print(f"{path.relative_to(ROOT)} {sha256_file(path)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch", "build", "verify", "inventory"))
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if args.command == "fetch":
        return fetch_sources()
    if args.command == "build":
        return build_derived()
    if args.command == "verify":
        return verify()
    return inventory(args.paths)


if __name__ == "__main__":
    raise SystemExit(main())
