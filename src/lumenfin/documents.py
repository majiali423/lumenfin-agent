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
    "nvidia": "NVIDIA",
    "byd": "BYD",
    "比亚迪": "BYD",
    "tencent": "Tencent",
    "腾讯": "Tencent",
    "toyota": "Toyota",
    "samsung": "Samsung",
    "tsmc": "TSMC",
}


def parse_pdf_document(file_path: Path) -> dict[str, Any]:
    doc = fitz.open(file_path)
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    full_text = "\n".join(pages).strip()
    lowered = full_text.lower()
    detected_companies = sorted({name for key, name in COMPANY_HINTS.items() if key in lowered or key in file_path.name.lower()})
    metric_hints = _extract_metric_hints(full_text)
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
            context = lowered[kw_match.end():kw_match.end() + 200]
            # Find the first valid number that's not a year
            for num_match in re.finditer(r"\$?\s*([0-9][0-9,\.]+)", context):
                raw = num_match.group(1).replace(",", "")
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if 2020 <= value <= 2035 and value == int(value):
                    continue
                # Scale: millions → billions
                after = context[num_match.end():num_match.end() + 20].strip().lower()
                if any(u in after for u in ["billion", "亿", "万亿"]):
                    pass
                elif any(u in after for u in ["million", "万"]):
                    value /= 1000
                hints[metric] = round(value, 1)
                break  # Use first valid number closest to keyword
            if metric in hints:
                break
    return hints
