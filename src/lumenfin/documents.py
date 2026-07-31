from __future__ import annotations

import re
from dataclasses import asdict, dataclass
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
    extracted = extract_metric_amounts(full_text)
    metric_hints = _compatibility_hints(extracted)
    metric_hint_meta = {key: amount_to_meta(amount) for key, amount in extracted.items()}
    hint_scope = detected_companies or mentioned
    per_company_hints = merge_per_company_metric_hints(full_text, hint_scope)
    per_company_meta = merge_per_company_metric_hint_meta(full_text, hint_scope)
    if len(detected_companies) == 1 and detected_companies[0] in per_company_hints:
        metric_hints = {**metric_hints, **per_company_hints[detected_companies[0]]}
        if detected_companies[0] in per_company_meta:
            metric_hint_meta = {**metric_hint_meta, **per_company_meta[detected_companies[0]]}
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
        "metric_hint_meta": metric_hint_meta,
        "per_company_metric_hints": per_company_hints,
        "per_company_metric_hint_meta": per_company_meta,
        "source_type": "pdf",
    }


# Absolute metrics that live on LumenFin's billion-USD scale after normalization.
_ABS_BILLION_METRICS = frozenset({"revenue", "ebitda", "operating_income", "r_and_d"})
# Unitless magnitudes at/above this may be inferred as statement millions (low confidence).
_UNITLESS_MILLION_FLOOR = 1000.0

_METRIC_KEYWORDS: list[tuple[str, list[str]]] = [
    ("revenue", [r"revenue", r"revenues", r"收入", r"营收"]),
    ("ebitda", [r"ebitda"]),
    ("r_and_d", [r"r\s*[&]\s*d\b", r"r\s+&\s+d\b", r"research\s+(?:and|&)\s+development", r"研发"]),
    ("operating_income", [r"operating\s+income", r"营业利润", r"经营利润"]),
]


@dataclass(frozen=True)
class ExtractedAmount:
    raw_value: float
    raw_scale: str | None
    currency: str | None
    normalized_value: float | None
    normalized_unit: str | None
    normalization_source: str
    confidence: str
    period_hint: str | None = None
    is_normalized: bool = False


_TRUSTED_NORM_SOURCES = frozenset(
    {"table_caption", "inline_unit", "structured_table", "provider_metadata"}
)

_PERIOD_TYPE_VALUES = frozenset({"annual", "quarter", "ttm", "latest"})


def amount_to_meta(amount: ExtractedAmount) -> dict[str, Any]:
    return asdict(amount)


def is_trusted_ast_amount(meta: dict[str, Any] | ExtractedAmount | None) -> bool:
    """Whether an extracted amount may enter AST/market_data fundamentals."""
    if meta is None:
        return False
    payload = asdict(meta) if isinstance(meta, ExtractedAmount) else dict(meta)
    if payload.get("confidence") != "high":
        return False
    if payload.get("normalized_value") is None:
        return False
    if payload.get("normalized_unit") != "billion_usd":
        return False
    currency = payload.get("currency")
    if currency != "USD":
        return False
    period_type = payload.get("period_type") or payload.get("period_hint")
    if period_type == "quarter":
        return False
    source = str(payload.get("normalization_source") or "")
    if source not in _TRUSTED_NORM_SOURCES:
        return False
    if source == "provider_metadata":
        return trusted_provider_amount(payload)
    return True


def trusted_provider_amount(meta: dict[str, Any] | ExtractedAmount | None) -> bool:
    """Provider-supplied normalized amounts require explicit provenance fields."""
    if meta is None:
        return False
    payload = asdict(meta) if isinstance(meta, ExtractedAmount) else dict(meta)
    if payload.get("confidence") != "high":
        return False
    if payload.get("normalized_value") is None:
        return False
    if payload.get("normalized_unit") != "billion_usd":
        return False
    if payload.get("currency") != "USD":
        return False
    if not str(payload.get("provider") or "").strip():
        return False
    period = str(payload.get("period") or "").strip()
    if not period or period.lower() in _PERIOD_TYPE_VALUES:
        return False
    if str(payload.get("normalization_source") or "") != "provider_metadata":
        return False
    return bool(payload.get("is_normalized", True))


def detect_statement_scale(text: str) -> str | None:
    """Detect filing table/statement unit from captions like '(In millions)'.

    Returns ``million``, ``thousand``, ``billion``, or ``None`` when undeclared.
    When multiple distinct statement scales appear in one blob, returns ``None`` so
    callers do not silently apply one caption to another table (low-confidence path).
    """
    lowered = (text or "").lower()
    if not lowered:
        return None
    found: list[str] = []
    if re.search(
        r"\(\s*in\s+millions(?:\s+of\s+(?:u\.?s\.?\s+)?(?:dollars|usd|eur|euros))?\s*\)"
        r"|\bin\s+millions\s+of\s+(?:u\.?s\.?\s+)?(?:dollars|usd|eur|euros)\b"
        r"|\b\(millions\)\b"
        r"|单位[：:]\s*百万",
        lowered,
    ):
        found.append("million")
    if re.search(
        r"\(\s*in\s+thousands(?:\s+of\s+(?:u\.?s\.?\s+)?(?:dollars|usd|eur|euros))?\s*\)"
        r"|\bin\s+thousands\s+of\s+(?:u\.?s\.?\s+)?(?:dollars|usd|eur|euros)\b"
        r"|\b\(thousands\)\b"
        r"|单位[：:]\s*千",
        lowered,
    ):
        found.append("thousand")
    if re.search(
        r"\(\s*in\s+billions(?:\s+of\s+(?:u\.?s\.?\s+)?(?:dollars|usd|eur|euros))?\s*\)"
        r"|\bin\s+billions\s+of\s+(?:u\.?s\.?\s+)?(?:dollars|usd|eur|euros)\b"
        r"|\b\(billions\)\b"
        r"|单位[：:]\s*十亿",
        lowered,
    ):
        found.append("billion")
    unique = list(dict.fromkeys(found))
    if len(unique) == 1:
        return unique[0]
    return None


def detect_statement_currency(text: str) -> str | None:
    lowered = (text or "").lower()
    if not lowered:
        return None
    if re.search(r"\b(?:eur|euro|euros)\b|单位[：:]\s*欧元", lowered):
        return "EUR"
    if re.search(r"\b(?:u\.?s\.?\s*)?dollars?\b|\busd\b|\$", lowered):
        return "USD"
    return None


def detect_period_hint(text: str) -> str | None:
    lowered = (text or "").lower()
    if re.search(r"\bthree months ended\b|\bquarter(?:ly)?\b|\bq[1-4]\b", lowered):
        return "quarter"
    if re.search(r"\bfiscal year\b|\bfy\s*20\d{2}\b|\byear ended\b", lowered):
        return "annual"
    return None


def _looks_already_normalized(value: float, scale: str | None) -> bool:
    """Deprecated: do not use shape heuristics for normalization state."""
    del value, scale
    return False


def normalize_extracted_amount(
    raw_value: float,
    *,
    raw_scale: str | None,
    currency: str | None,
    normalization_source: str,
    period_hint: str | None = None,
    already_normalized: bool = False,
) -> ExtractedAmount:
    """Project one raw statement amount exactly once.

    Explicit statement scales always convert unless ``already_normalized`` is set via
    metadata. Never infer normalization state from integer/fractional shape.
    """
    value = float(raw_value)
    scale = raw_scale
    confidence = "high"
    source = normalization_source
    unit: str | None = None
    normalized: float | None = value

    if currency not in (None, "USD"):
        return ExtractedAmount(
            raw_value=value,
            raw_scale=scale,
            currency=currency,
            normalized_value=None,
            normalized_unit=None,
            normalization_source=source or "non_usd",
            confidence="low",
            period_hint=period_hint,
            is_normalized=False,
        )

    if already_normalized and scale in {"million", "thousand", "billion", None}:
        return ExtractedAmount(
            raw_value=value,
            raw_scale=scale,
            currency=currency or "USD",
            normalized_value=round(value, 8),
            normalized_unit="billion_usd",
            normalization_source=source or "pre_normalized",
            confidence="high" if source in _TRUSTED_NORM_SOURCES else "low",
            period_hint=period_hint,
            is_normalized=True,
        )

    if scale == "million":
        normalized = value / 1000.0
        unit = "billion_usd"
        confidence = "high" if source in _TRUSTED_NORM_SOURCES else "low"
    elif scale == "thousand":
        normalized = value / 1_000_000.0
        unit = "billion_usd"
        confidence = "high" if source in _TRUSTED_NORM_SOURCES else "low"
    elif scale == "billion":
        normalized = value
        unit = "billion_usd"
        confidence = "high" if source in _TRUSTED_NORM_SOURCES else "low"
    elif abs(value) >= _UNITLESS_MILLION_FLOOR:
        normalized = value / 1000.0
        scale = "million"
        unit = "billion_usd"
        source = "inferred_million"
        confidence = "low"
    else:
        unit = "billion_usd"
        source = source or "unitless"
        confidence = "low"

    if period_hint == "quarter" and confidence == "high":
        confidence = "low"

    if normalized is not None:
        # Keep enough precision for thousand-scale fractions (e.g. 500.25 / 1e6).
        normalized = round(float(normalized), 8)
    return ExtractedAmount(
        raw_value=value,
        raw_scale=scale,
        currency=currency or "USD",
        normalized_value=normalized,
        normalized_unit=unit,
        normalization_source=source,
        confidence=confidence,
        period_hint=period_hint,
        is_normalized=True,
    )


def normalize_metric_hints_to_billion_usd(
    hints: dict[str, float],
    *,
    text: str = "",
    hint_meta: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Normalize absolute document hints onto the shared billion-USD scale (compat path).

    Scaling happens once per value. Unitless large magnitudes may still be projected for
    compatibility, but accompanying ``hint_meta`` (when built via ``extract_metric_amounts``)
    marks them ``confidence=low``. Pre-normalized metadata (``is_normalized`` /
    ``normalized_unit=billion_usd``) is never re-scaled by caption heuristics.
    """
    if not hints:
        return {}
    from .fundamentals import is_plausible_revenue_billion_usd

    scale = detect_statement_scale(text)
    currency = detect_statement_currency(text)
    period_hint = detect_period_hint(text)
    out: dict[str, float] = {}
    for key, raw in hints.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        base_key = key[:-5] if key.endswith("_2025") else key
        if base_key not in _ABS_BILLION_METRICS:
            out[key] = value
            continue
        meta = (hint_meta or {}).get(key) or (hint_meta or {}).get(base_key) or {}
        if meta.get("is_normalized") or (
            meta.get("normalized_value") is not None
            and meta.get("normalized_unit") == "billion_usd"
            and meta.get("confidence") == "high"
            and str(meta.get("normalization_source") or "") in _TRUSTED_NORM_SOURCES
            and abs(float(meta["normalized_value"]) - value) <= max(0.01, abs(value) * 0.001)
        ):
            out[key] = float(meta.get("normalized_value", value))
            continue
        if meta.get("normalized_value") is not None and meta.get("normalized_unit") == "billion_usd":
            # Prefer explicit metadata projection even when low-confidence (compat float only).
            if meta.get("is_normalized"):
                out[key] = float(meta["normalized_value"])
                continue
        amount = normalize_extracted_amount(
            value,
            raw_scale=str(meta.get("raw_scale") or scale) if meta.get("raw_scale") or scale else scale,
            currency=meta.get("currency") or currency,
            normalization_source=str(meta.get("normalization_source") or ("table_caption" if scale else "unitless")),
            period_hint=meta.get("period_hint") or period_hint,
            already_normalized=bool(meta.get("is_normalized")),
        )
        if amount.normalized_unit == "billion_usd" and amount.normalized_value is not None:
            out[key] = float(amount.normalized_value)

    revenue = out.get("revenue")
    if revenue is not None and not is_plausible_revenue_billion_usd(revenue):
        if abs(float(revenue)) >= _UNITLESS_MILLION_FLOOR:
            rescued = round(float(revenue) / 1000.0, 4)
            if is_plausible_revenue_billion_usd(rescued):
                out["revenue"] = rescued
                for key in list(out):
                    base_key = key[:-5] if key.endswith("_2025") else key
                    if (
                        base_key in _ABS_BILLION_METRICS - {"revenue"}
                        and abs(out[key]) >= _UNITLESS_MILLION_FLOOR
                    ):
                        out[key] = round(float(out[key]) / 1000.0, 4)
            else:
                out.pop("revenue", None)
        else:
            out.pop("revenue", None)
    return out


def _parse_raw_metric_number(
    context: str,
    *,
    document_scale: str | None = None,
    document_currency: str | None = None,
    period_hint: str | None = None,
) -> ExtractedAmount | None:
    """Raw extraction only; normalization is applied once via ``normalize_extracted_amount``.

    Forward context may contain a later table's caption; do not let that override the
    caller-provided local/document scale. Inline units on the number itself still win.
    """
    local_currency = detect_statement_currency(context) or document_currency
    local_period = detect_period_hint(context) or period_hint
    for num_match in re.finditer(r"[-$]?\s*([0-9][0-9,\.]+)", context):
        raw = num_match.group(1).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if context[num_match.start() : num_match.start() + 1] == "-" or (
            num_match.start() > 0 and context[num_match.start() - 1] == "-"
        ):
            value = -abs(value)
        suffix = context[num_match.end() : num_match.end() + 4].lstrip()
        if suffix.startswith("%"):
            continue
        if 2020 <= value <= 2035 and value == int(value):
            continue
        after = context[num_match.end() : num_match.end() + 24].strip().lower()
        has_billion = any(token in after for token in ("billion", "亿", "万亿", "bn"))
        has_million = any(token in after for token in ("million", "万", "mm")) and not has_billion
        if has_million:
            scale = "million"
            source = "inline_unit"
        elif has_billion:
            scale = "billion"
            source = "inline_unit"
        elif document_scale:
            scale = document_scale
            source = "table_caption"
        else:
            scale = None
            source = "unitless"
        return normalize_extracted_amount(
            value,
            raw_scale=scale,
            currency=local_currency,
            normalization_source=source,
            period_hint=local_period,
        )
    return None


def _extract_metric_amounts_raw(
    text: str,
    *,
    document_scale: str | None = None,
) -> dict[str, ExtractedAmount]:
    amounts: dict[str, ExtractedAmount] = {}
    lowered = text.lower()
    doc_scale = document_scale if document_scale is not None else detect_statement_scale(text)
    doc_currency = detect_statement_currency(text)
    period_hint = detect_period_hint(text)

    for metric, keywords in _METRIC_KEYWORDS:
        for kw in keywords:
            kw_match = re.search(kw, lowered, flags=re.IGNORECASE)
            if not kw_match:
                continue
            # Captions often sit before the label; keep numeric search forward-only.
            prefix = lowered[max(0, kw_match.start() - 160) : kw_match.start()]
            local_scale = detect_statement_scale(prefix) or doc_scale
            local_currency = detect_statement_currency(prefix) or doc_currency
            context = lowered[kw_match.end() : kw_match.end() + 200]
            # Preserve a leading minus that sits just before the match window.
            window_start = kw_match.end()
            if window_start > 0 and text[window_start - 1] == "-":
                context = "-" + context
            amount = _parse_raw_metric_number(
                context,
                document_scale=local_scale,
                document_currency=local_currency,
                period_hint=period_hint,
            )
            if amount is not None:
                amounts[metric] = amount
                break
    return amounts


def extract_metric_amounts(text: str) -> dict[str, ExtractedAmount]:
    return _extract_metric_amounts_raw(text)


def _compatibility_hints(amounts: dict[str, ExtractedAmount]) -> dict[str, float]:
    from .fundamentals import is_plausible_revenue_billion_usd

    out: dict[str, float] = {}
    for key, amount in amounts.items():
        if amount.normalized_unit != "billion_usd" or amount.normalized_value is None:
            continue
        out[key] = float(amount.normalized_value)
    revenue = out.get("revenue")
    if revenue is not None and not is_plausible_revenue_billion_usd(revenue):
        out.pop("revenue", None)
    return out


def extract_metric_hint_meta(text: str, metric: str = "revenue") -> dict[str, Any] | None:
    """Return auditable normalization metadata for one metric (Phase-2 contract)."""
    amounts = extract_metric_amounts(text)
    amount = amounts.get(metric)
    if amount is None:
        # Fall back to first number after an explicit metric-less window for unit tests.
        amount = _parse_raw_metric_number(
            text.lower(),
            document_scale=detect_statement_scale(text),
            document_currency=detect_statement_currency(text),
            period_hint=detect_period_hint(text),
        )
        if amount is None:
            return None
    return amount_to_meta(amount)


def _extract_metric_hints_raw(
    text: str,
    *,
    document_scale: str | None = None,
) -> dict[str, float]:
    """Extract raw absolute statement numbers (no billion projection)."""
    amounts = _extract_metric_amounts_raw(text, document_scale=document_scale)
    return {key: amount.raw_value for key, amount in amounts.items()}


def _extract_metric_hints(text: str) -> dict[str, float]:
    return _compatibility_hints(extract_metric_amounts(text))


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
    scaled = {
        company: normalize_metric_hints_to_billion_usd(hints, text=text)
        for company, hints in out.items()
        if hints
    }
    return {c: hints for c, hints in scaled.items() if hints}


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
    doc_scale = detect_statement_scale(text)
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
        amounts = _extract_metric_amounts_raw(window, document_scale=doc_scale)
        hints = _compatibility_hints(amounts)
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
        company: normalize_metric_hints_to_billion_usd(
            dict((base or {}).get(company) or {}),
            text=text,
        )
        for company in companies
    }
    for company in companies:
        if not merged.get(company):
            merged[company] = extract_metric_hints_for_company(text, company)
    columnar = extract_columnar_peer_metrics(text, companies)
    for company, hints in columnar.items():
        slot = merged.setdefault(company, {})
        slot.update(hints)
    return {c: h for c, h in merged.items() if h}


def extract_metric_amounts_for_company(text: str, company: str) -> dict[str, ExtractedAmount]:
    """Company-scoped structured amounts (mirrors extract_metric_hints_for_company)."""
    aliases = {company.lower()}
    for key, canonical in COMPANY_HINTS.items():
        if canonical == company:
            aliases.add(key.lower())
    lowered = text.lower()
    doc_scale = detect_statement_scale(text)
    windows: list[str] = []
    for alias in sorted(aliases, key=len, reverse=True):
        start = 0
        while True:
            idx = lowered.find(alias, start)
            if idx < 0:
                break
            windows.append(lowered[idx : idx + 360])
            start = idx + max(len(alias), 1)
    if not windows:
        return {}
    sentence_hits: dict[str, ExtractedAmount] = {}
    loose_hits: dict[str, ExtractedAmount] = {}
    for window in windows:
        amounts = _extract_metric_amounts_raw(window, document_scale=doc_scale)
        bucket = sentence_hits if _window_looks_sentence_metric(window) else loose_hits
        for metric, amount in amounts.items():
            bucket.setdefault(metric, amount)
    return {**loose_hits, **sentence_hits}


def merge_per_company_metric_hint_meta(
    text: str,
    companies: list[str],
    *,
    base: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-company ExtractedAmount metadata companion to float hint dicts."""
    merged: dict[str, dict[str, dict[str, Any]]] = {
        company: dict((base or {}).get(company) or {}) for company in companies
    }
    for company in companies:
        if not merged.get(company):
            amounts = extract_metric_amounts_for_company(text, company)
            merged[company] = {key: amount_to_meta(amount) for key, amount in amounts.items()}
    return {c: m for c, m in merged.items() if m}


def _first_metric_number(
    context: str,
    *,
    document_scale: str | None = None,
) -> float | None:
    """Compatibility wrapper: return once-normalized billion-USD magnitude when available."""
    amount = _parse_raw_metric_number(
        context,
        document_scale=document_scale,
        document_currency=detect_statement_currency(context),
        period_hint=detect_period_hint(context),
    )
    if amount is None or amount.normalized_value is None:
        return None
    if amount.normalized_unit != "billion_usd":
        return None
    return round(float(amount.normalized_value), 1)
