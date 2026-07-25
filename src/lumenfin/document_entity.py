"""Document primary-entity resolution for filings (issuer vs peer mentions).

SEC 10-Ks mention competitors/customers freely. Live fundamentals and report
scope must bind to the **issuer** (or explicit peer-table columns / user compare
intent), not every COMPANY_HINTS hit in the body text.
"""

from __future__ import annotations

import re
from typing import Any

from .documents import COMPANY_HINTS, detect_companies_from_text

# Filename / stem hints → canonical company.
_FILENAME_HINTS: dict[str, str] = {
    "aapl": "Apple",
    "apple": "Apple",
    "msft": "Microsoft",
    "microsoft": "Microsoft",
    "nvda": "NVIDIA",
    "nvidia": "NVIDIA",
    "tsla": "Tesla",
    "tesla": "Tesla",
    "amd": "AMD",
    "orcl": "Oracle",
    "oracle": "Oracle",
    "tsmc": "TSMC",
    "meta": "Meta",
    "goog": "Alphabet",
    "googl": "Alphabet",
    "amzn": "Amazon",
    "amazon": "Amazon",
}

_FORM_ISSUER_RE = re.compile(
    r"(?:form\s*10-?k|annual\s+report(?:\s+on\s+form\s*10-?k)?)"
    r".{0,120}?(?:for|of)\s+([A-Z][A-Za-z0-9&.,' \-]{2,60}?(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|Co\.?|Company|N\.?V\.?))",
    re.IGNORECASE | re.DOTALL,
)
_INC_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.,' \-]{1,50}?(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|Company|N\.?V\.?))\b"
)


def _canonical_from_raw(name: str) -> str | None:
    cleaned = (name or "").strip().rstrip(".,")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    for key, canonical in COMPANY_HINTS.items():
        if key in lowered or lowered in key:
            return canonical
    # Strip legal suffix and retry.
    stem = re.sub(
        r"\b(inc\.?|corp\.?|corporation|ltd\.?|limited|company|co\.?|n\.?v\.?)\b",
        "",
        lowered,
        flags=re.IGNORECASE,
    ).strip(" ,.")
    for key, canonical in COMPANY_HINTS.items():
        if stem and (stem == key or stem in key or key in stem):
            return canonical
    return None


def _filename_primary(filename: str) -> tuple[str | None, float]:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", (filename or "").lower())
    for key, canonical in _FILENAME_HINTS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", stem):
            return canonical, 0.92
    # Long names in filename
    for key, canonical in COMPANY_HINTS.items():
        if len(key) >= 5 and key in stem:
            return canonical, 0.85
    return None, 0.0


def _cover_primary(pages: list[str], mentioned: list[str]) -> tuple[str | None, float, str]:
    cover = "\n".join((pages or [])[:3])
    if not cover.strip():
        return None, 0.0, ""
    # Form 10-K issuer phrasing
    match = _FORM_ISSUER_RE.search(cover)
    if match:
        canonical = _canonical_from_raw(match.group(1))
        if canonical:
            return canonical, 0.9, "form_10k_header"
    # Count legal-name hits on cover among mentioned companies
    scores: dict[str, int] = {name: 0 for name in mentioned}
    for raw in _INC_NAME_RE.findall(cover):
        canonical = _canonical_from_raw(raw)
        if canonical in scores:
            scores[canonical] += 3
    cover_lower = cover.lower()
    for name in mentioned:
        aliases = [name.lower()] + [
            alias for alias, canonical in COMPANY_HINTS.items() if canonical == name
        ]
        for alias in aliases:
            scores[name] = scores.get(name, 0) + cover_lower.count(alias)
    if not scores:
        return None, 0.0, ""
    best = max(scores.items(), key=lambda item: item[1])
    if best[1] <= 0:
        return None, 0.0, ""
    confidence = min(0.88, 0.55 + 0.05 * best[1])
    return best[0], confidence, "cover_frequency"


def _peer_table_issuers(pages: list[str], mentioned: list[str]) -> list[str]:
    """Two+ issuers on a short metric header line → multi-issuer table pack."""
    if len(mentioned) < 2:
        return []
    for page in (pages or [])[:4]:
        for line in page.splitlines()[:40]:
            stripped = line.strip()
            if len(stripped) > 120:
                continue
            present = [c for c in mentioned if c.lower() in stripped.lower()]
            if len(present) >= 2 and any(
                token in stripped.lower() for token in ("metric", "revenue", "ebitda", "指标")
            ):
                return present
            # Header like "Metric Apple Microsoft"
            if len(present) >= 2 and re.search(r"\bmetric\b", stripped, re.I):
                return present
    return []


def resolve_document_entities(
    *,
    text: str,
    pages: list[str] | None = None,
    filename: str = "",
) -> dict[str, Any]:
    """Resolve issuer vs body mentions for a filing/upload.

    Returns:
      primary_company: {name, ticker?, cik?, confidence, method}
      issuer_companies: companies allowed for live lookup / report scope
      mentioned_companies: all hint hits (competitors etc.; RAG narrative only)
      detected_companies: alias of issuer_companies (planner/supervisor contract)
    """
    page_list = list(pages or [])
    if not page_list and text:
        page_list = [text]
    mentioned = detect_companies_from_text(text or "", filename)
    file_name, file_conf = _filename_primary(filename)
    cover_name, cover_conf, cover_method = _cover_primary(page_list, mentioned)
    peer_issuers = _peer_table_issuers(page_list, mentioned)

    primary_name: str | None = None
    confidence = 0.0
    method = "none"
    if file_name and file_conf >= cover_conf:
        primary_name, confidence, method = file_name, file_conf, "filename"
    elif cover_name:
        primary_name, confidence, method = cover_name, cover_conf, cover_method
    elif file_name:
        primary_name, confidence, method = file_name, file_conf, "filename"
    elif mentioned:
        # Short docs / notes: first detected is acceptable with low confidence.
        primary_name, confidence, method = mentioned[0], 0.4, "first_mention"

    if peer_issuers:
        issuers = list(peer_issuers)
        # Keep primary as first peer-table column when possible.
        if primary_name and primary_name in issuers:
            issuers = [primary_name] + [c for c in issuers if c != primary_name]
        method = "peer_table_header"
        confidence = max(confidence, 0.8)
        if not primary_name:
            primary_name = issuers[0]
    elif primary_name:
        issuers = [primary_name]
    else:
        issuers = []

    # Ensure issuers ⊆ mentioned when mentions exist; still allow filename-only issuer.
    if mentioned:
        issuers = [c for c in issuers if c in mentioned] or (
            [primary_name] if primary_name else []
        )

    primary = None
    if primary_name:
        primary = {
            "name": primary_name,
            "ticker": None,
            "cik": None,
            "confidence": round(float(confidence), 3),
            "method": method,
        }

    return {
        "primary_company": primary,
        "issuer_companies": issuers,
        "mentioned_companies": mentioned,
        "detected_companies": issuers,
    }
