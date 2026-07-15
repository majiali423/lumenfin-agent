from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz


COMPANY_HINTS = {
    "apple": "Apple",
    "microsoft": "Microsoft",
    "tesla": "Tesla",
    "amazon": "Amazon",
    "google": "Alphabet",
    "alphabet": "Alphabet",
    "meta": "Meta",
    "meta platforms": "Meta",
    "facebook": "Meta",
    "nvidia": "NVIDIA",
    "nvda": "NVIDIA",
    "amd": "AMD",
    "byd": "BYD",
    "比亚迪": "BYD",
    "tencent": "Tencent",
    "腾讯": "Tencent",
    "toyota": "Toyota",
    "samsung": "Samsung",
    "tsmc": "TSMC",
    "taiwan semiconductor": "TSMC",
    "broadcom": "Broadcom",
    "avgo": "Broadcom",
    "alibaba": "Alibaba",
    "阿里巴巴": "Alibaba",
    "oracle": "Oracle",
    "shopify": "Shopify",
    "block": "Block",
    "square": "Block",
    "openai": "OpenAI",
    "openai inc": "OpenAI",
}


def detect_companies_from_text(text: str, filename: str = "") -> list[str]:
    lowered = text.lower()
    name_lower = filename.lower()
    found = {
        name
        for key, name in COMPANY_HINTS.items()
        if key in lowered or key in name_lower
    }
    return sorted(found)


def parse_pdf_document(file_path: Path) -> dict[str, Any]:
    doc = fitz.open(file_path)
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    full_text = "\n".join(pages).strip()
    detected_companies = detect_companies_from_text(full_text, file_path.name)
    # Document-level hints remain a fallback for single-entity filings.
    metric_hints = _extract_metric_hints(full_text)
    per_company_hints = {
        company: extract_metric_hints_for_company(full_text, company)
        for company in detected_companies
    }
    return {
        "document_id": file_path.stem,
        "filename": file_path.name,
        "path": str(file_path),
        "page_count": len(pages),
        "pages": pages,
        "text": full_text,
        "excerpt": full_text[:4000],
        "detected_companies": detected_companies,
        "metric_hints": metric_hints,
        "per_company_metric_hints": per_company_hints,
        "source_type": "pdf",
    }


def _extract_metric_hints(text: str) -> dict[str, float]:
    hints: dict[str, float] = {}
    lowered = text.lower()

    for metric, keywords in [
        ("revenue", [r"revenue", r"revenues", r"收入", r"营收"]),
        ("ebitda", [r"ebitda"]),
        ("r_and_d", [r"r\s*[&]\s*d\b", r"r\s+&\s+d\b", r"research\s+(?:and|&)\s+development", r"研发"]),
    ]:
        for kw in keywords:
            kw_match = re.search(kw, lowered, flags=re.IGNORECASE)
            if not kw_match:
                continue
            context = lowered[kw_match.end() : kw_match.end() + 200]
            value = _first_metric_number(context)
            if value is not None:
                hints[metric] = value
                break
    return hints


def extract_metric_hints_for_company(text: str, company: str) -> dict[str, float]:
    """Extract metrics from text windows near a company mention (multi-issuer PDFs)."""
    aliases = {company.lower()}
    for key, canonical in COMPANY_HINTS.items():
        if canonical == company:
            aliases.add(key.lower())
    lowered = text.lower()
    windows: list[str] = []
    for alias in sorted(aliases, key=len, reverse=True):
        start = 0
        while True:
            idx = lowered.find(alias, start)
            if idx < 0:
                break
            # Do not look backwards — peer PDFs put another issuer's metrics on the prior line.
            windows.append(lowered[idx : idx + 360])
            start = idx + max(len(alias), 1)
    if not windows:
        return {}
    merged: dict[str, float] = {}
    for window in windows:
        for metric, value in _extract_metric_hints(window).items():
            merged.setdefault(metric, value)
    return merged


def _first_metric_number(context: str) -> float | None:
    for num_match in re.finditer(r"\$?\s*([0-9][0-9,\.]+)", context):
        raw = num_match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if 2020 <= value <= 2035 and value == int(value):
            continue
        after = context[num_match.end() : num_match.end() + 20].strip().lower()
        if any(u in after for u in ["million", "万"]) and not any(
            u in after for u in ["billion", "亿", "万亿"]
        ):
            value /= 1000
        return round(value, 1)
    return None
