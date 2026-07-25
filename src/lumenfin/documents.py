from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz


COMPANY_HINTS = {
    "apple": "Apple",
    "苹果": "Apple",
    "microsoft": "Microsoft",
    "微软": "Microsoft",
    "tesla": "Tesla",
    "特斯拉": "Tesla",
    "amazon": "Amazon",
    "亚马逊": "Amazon",
    "google": "Alphabet",
    "alphabet": "Alphabet",
    "谷歌": "Alphabet",
    "meta": "Meta",
    "meta platforms": "Meta",
    "facebook": "Meta",
    "英伟达": "NVIDIA",
    "nvidia": "NVIDIA",
    "nvda": "NVIDIA",
    "amd": "AMD",
    "byd": "BYD",
    "比亚迪": "BYD",
    "tencent": "Tencent",
    "腾讯": "Tencent",
    "toyota": "Toyota",
    "丰田": "Toyota",
    "samsung": "Samsung",
    "三星": "Samsung",
    "tsmc": "TSMC",
    "台积电": "TSMC",
    "taiwan semiconductor": "TSMC",
    "broadcom": "Broadcom",
    "avgo": "Broadcom",
    "alibaba": "Alibaba",
    "阿里巴巴": "Alibaba",
    "oracle": "Oracle",
    "甲骨文": "Oracle",
    "shopify": "Shopify",
    "block": "Block",
    "square": "Block",
    "openai": "OpenAI",
    "openai inc": "OpenAI",
    "softbank": "SoftBank",
    "softbank group": "SoftBank",
    "软银": "SoftBank",
}


def detect_companies_from_text(text: str, filename: str = "") -> list[str]:
    lowered = text.lower()
    name_lower = filename.lower()
    found = set()
    for key, name in COMPANY_HINTS.items():
        token = key.lower()
        if re.search(r"[\u4e00-\u9fff]", token):
            hit = token in text or token in lowered or token in name_lower
        elif " " in token or len(token) >= 6:
            hit = token in lowered or token in name_lower
        else:
            hit = bool(
                re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered)
                or re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", name_lower)
            )
        if hit:
            found.add(name)
    return sorted(found)


def parse_pdf_document(file_path: Path) -> dict[str, Any]:
    from .document_entity import resolve_document_entities

    doc = fitz.open(file_path)
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    full_text = "\n".join(pages).strip()
    entity = resolve_document_entities(text=full_text, pages=pages, filename=file_path.name)
    detected_companies = list(entity.get("detected_companies") or [])
    mentioned = list(entity.get("mentioned_companies") or detected_companies)
    # Document-level hints remain a fallback for single-entity filings.
    metric_hints = _extract_metric_hints(full_text)
    hint_scope = detected_companies or mentioned
    per_company_hints = merge_per_company_metric_hints(full_text, hint_scope)
    if len(detected_companies) == 1 and detected_companies[0] in per_company_hints:
        metric_hints = {**metric_hints, **per_company_hints[detected_companies[0]]}
    return {
        "document_id": file_path.stem,
        "filename": file_path.name,
        "path": str(file_path),
        "page_count": len(pages),
        "pages": pages,
        "text": full_text,
        "excerpt": full_text[:4000],
        "detected_companies": detected_companies,
        "issuer_companies": list(entity.get("issuer_companies") or detected_companies),
        "mentioned_companies": mentioned,
        "primary_company": entity.get("primary_company"),
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
        ("operating_income", [r"operating\s+income", r"营业利润", r"经营利润"]),
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


_COLUMNAR_METRIC_LABELS: list[tuple[str, re.Pattern[str]]] = [
    ("revenue", re.compile(r"^(?:total\s+)?revenue(?:\s*\(.*\))?$|^营收$|^收入$|^营业收入$", re.I)),
    ("ebitda", re.compile(r"^ebitda(?:\s*\(.*\))?$|^息税折旧摊销前利润$", re.I)),
    ("operating_income", re.compile(r"^operating\s+income(?:\s*\(.*\))?$|^营业利润$|^经营利润$", re.I)),
    (
        "r_and_d",
        re.compile(
            r"^(?:r\s*&\s*d|r\s+&\s+d|research\s+(?:and|&)\s+development)(?:\s+expense)?(?:\s*\(.*\))?$"
            r"|^研发(?:费用|支出|投入)?$",
            re.I,
        ),
    ),
]


def _resolve_company_line(line: str) -> str | None:
    cleaned = (line or "").strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    for key, canonical in COMPANY_HINTS.items():
        token = key.lower()
        if re.search(r"[\u4e00-\u9fff]", token):
            if token == cleaned or token in cleaned:
                return canonical
        elif lowered == token or lowered == canonical.lower():
            return canonical
    return None


def extract_columnar_peer_metrics(text: str, companies: list[str]) -> dict[str, dict[str, float]]:
    """Parse vertical peer tables where company headers are followed by metric/value rows.

    Handles PDF text extraction shaped like::

        Metric
        Apple
        Microsoft
        Revenue
        383.3
        245.1
    """
    if len(companies) < 2:
        return {}
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < len(companies) + 2:
        return {}

    target = set(companies)
    header_at = -1
    header_order: list[str] = []
    for i in range(len(lines) - len(companies) + 1):
        resolved: list[str] = []
        for j in range(len(companies)):
            name = _resolve_company_line(lines[i + j])
            if name is None or name in resolved:
                resolved = []
                break
            resolved.append(name)
        if resolved and set(resolved) == target:
            header_at = i
            header_order = resolved
            break
    if header_at < 0:
        return {}

    out: dict[str, dict[str, float]] = {c: {} for c in header_order}
    n = len(header_order)
    i = header_at + n
    while i < len(lines):
        label = lines[i]
        metric_key = None
        for key, pattern in _COLUMNAR_METRIC_LABELS:
            if pattern.match(label):
                metric_key = key
                break
        if metric_key is None:
            i += 1
            continue
        values: list[float] = []
        j = i + 1
        while j < len(lines) and len(values) < n:
            raw = lines[j].replace(",", "").replace("$", "").strip()
            try:
                num = float(raw)
            except ValueError:
                break
            if 2020 <= num <= 2035 and num == int(num):
                break
            values.append(round(num, 1))
            j += 1
        if len(values) == n:
            for company, value in zip(header_order, values):
                out[company][metric_key] = value
            i = j
        else:
            i += 1
    return {c: hints for c, hints in out.items() if hints}


def _window_looks_sentence_metric(window: str) -> bool:
    return bool(
        re.search(
            r"(revenue|ebitda|r\s*&\s*d|research\s+and\s+development)\s+(was|were|of|reached|totaled)\b",
            window,
            flags=re.I,
        )
        or re.search(
            r"\b(revenue|ebitda)\s+\$?\s*[0-9]",
            window,
            flags=re.I,
        )
    )


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
    sentence_hits: dict[str, float] = {}
    loose_hits: dict[str, float] = {}
    for window in windows:
        hints = _extract_metric_hints(window)
        bucket = sentence_hits if _window_looks_sentence_metric(window) else loose_hits
        for metric, value in hints.items():
            bucket.setdefault(metric, value)
    # Sentence-style evidence wins over first-number-after-header-window noise.
    return {**loose_hits, **sentence_hits}


def merge_per_company_metric_hints(
    text: str,
    companies: list[str],
    *,
    base: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Combine window extraction with columnar peer-table parsing."""
    merged: dict[str, dict[str, float]] = {
        company: dict((base or {}).get(company) or {}) for company in companies
    }
    for company in companies:
        if not merged.get(company):
            merged[company] = extract_metric_hints_for_company(text, company)
    columnar = extract_columnar_peer_metrics(text, companies)
    for company, hints in columnar.items():
        slot = merged.setdefault(company, {})
        slot.update(hints)
    return merged


def _first_metric_number(context: str) -> float | None:
    for num_match in re.finditer(r"\$?\s*([0-9][0-9,\.]+)", context):
        raw = num_match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        suffix = context[num_match.end() : num_match.end() + 4].lstrip()
        if suffix.startswith("%"):
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
