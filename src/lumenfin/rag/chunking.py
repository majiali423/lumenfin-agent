from __future__ import annotations

import re
from typing import Any


FINANCIAL_KEYWORDS = (
    "revenue",
    "ebitda",
    "margin",
    "profit",
    "r&d",
    "research",
    "supply chain",
    "risk",
    "cash flow",
    "operating income",
    "收入",
    "营收",
    "研发",
    "供应链",
    "风险",
    "利润",
)

# Common CN display names → canonical English keys used in detected_companies.
_COMPANY_ALIASES: dict[str, str] = {
    "apple": "Apple",
    "苹果": "Apple",
    "microsoft": "Microsoft",
    "微软": "Microsoft",
    "nvidia": "NVIDIA",
    "英伟达": "NVIDIA",
    "tsmc": "TSMC",
    "台积电": "TSMC",
    "oracle": "Oracle",
    "甲骨文": "Oracle",
    "amd": "AMD",
    "tesla": "Tesla",
    "特斯拉": "Tesla",
}

_NUMBER = r"[-+]?\d+(?:[.,]\d+)?"
_METRIC_ROW_RE = re.compile(
    rf"^(?P<metric>[A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff /&%'’.-]{{1,40}}?)"
    rf"(?:\s+|[:：]\s*)"
    rf"(?P<values>(?:{_NUMBER})(?:\s+(?:{_NUMBER}))+)\s*$"
)


def _classify_chunk(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("supply chain", "供应链")) and any(
        word in lowered for word in ("risk", "风险", "constraint", "concentration")
    ):
        return "risk_signal"
    hits = sum(1 for keyword in FINANCIAL_KEYWORDS if keyword in lowered)
    if hits >= 2:
        return "financial_metric"
    if any(word in lowered for word in ("risk", "supply chain", "风险", "供应链")):
        return "risk_signal"
    if any(word in lowered for word in ("revenue", "ebitda", "operating income", "r&d", "收入", "营收", "研发")):
        return "financial_metric"
    return "narrative"


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if paragraphs:
        return paragraphs
    return [line.strip() for line in text.splitlines() if line.strip()]


def _merge_small_chunks(parts: list[str], max_chars: int) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}\n\n{part}".strip() if buffer else part
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            merged.append(buffer)
        if len(part) <= max_chars:
            buffer = part
        else:
            for start in range(0, len(part), max_chars):
                merged.append(part[start : start + max_chars])
            buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


def _normalize_company_token(token: str) -> str | None:
    cleaned = token.strip().strip("|").strip()
    if not cleaned:
        return None
    return _COMPANY_ALIASES.get(cleaned.lower(), cleaned if cleaned[:1].isupper() else None)


def _companies_in_text(text: str, detected: list[str]) -> list[str]:
    if not detected:
        return []
    lowered = text.lower()
    found: list[str] = []
    for company in detected:
        aliases = [company.lower()]
        for alias, canonical in _COMPANY_ALIASES.items():
            if canonical == company:
                aliases.append(alias)
        if any(alias in lowered for alias in aliases):
            found.append(company)
    return found


def _detect_table_header(lines: list[str], detected: list[str]) -> tuple[int, list[str]] | None:
    if len(detected) < 2:
        return None
    for index, line in enumerate(lines[:12]):
        present = _companies_in_text(line, detected)
        # Header-like: at least two peer companies on one short line.
        if len(present) >= 2 and len(line) <= 120:
            # Preserve column order as they appear in the line.
            ordered: list[str] = []
            lower_line = line.lower()
            cursor = 0
            while cursor < len(lower_line):
                matched = None
                matched_at = None
                for company in present:
                    aliases = [company.lower()] + [
                        alias for alias, canonical in _COMPANY_ALIASES.items() if canonical == company
                    ]
                    for alias in aliases:
                        pos = lower_line.find(alias, cursor)
                        if pos < 0:
                            continue
                        if matched_at is None or pos < matched_at:
                            matched_at = pos
                            matched = company
                if matched is None or matched_at is None:
                    break
                if matched not in ordered:
                    ordered.append(matched)
                cursor = matched_at + 1
            if len(ordered) >= 2:
                return index, ordered
    return None


def _parse_metric_row(line: str, column_companies: list[str]) -> list[tuple[str, str, str]] | None:
    match = _METRIC_ROW_RE.match(line.strip())
    if not match:
        return None
    metric = match.group("metric").strip()
    # Skip header leftovers / section titles.
    if metric.lower() in {"metric", "指标", "item", "company", "companies"}:
        return None
    values = re.findall(_NUMBER, match.group("values"))
    if len(values) < len(column_companies):
        return None
    rows: list[tuple[str, str, str]] = []
    for company, value in zip(column_companies, values):
        rows.append((company, metric, value.replace(",", "")))
    return rows


def _split_table_page(
    page_text: str,
    *,
    detected_companies: list[str],
) -> list[tuple[str, list[str]]] | None:
    """Return (text, companies) parts when the page looks like a peer metrics table."""
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if len(lines) < 3:
        return None
    header = _detect_table_header(lines, detected_companies)
    if not header:
        return None
    header_index, column_companies = header

    parts: list[tuple[str, list[str]]] = []
    metric_rows = 0
    for line in lines[header_index + 1 :]:
        parsed = _parse_metric_row(line, column_companies)
        if parsed:
            metric_rows += 1
            for company, metric, value in parsed:
                parts.append(
                    (
                        f"{company} {metric}: {value} (peer table).",
                        [company],
                    )
                )
            continue
        # Narrative / appendix lines after the numeric block.
        mentioned = _companies_in_text(line, detected_companies)
        if mentioned:
            # Split multi-company narrative sentences when possible.
            sentences = [s.strip() for s in re.split(r"(?<=[。.;；])\s*", line) if s.strip()]
            if len(sentences) > 1:
                for sentence in sentences:
                    sent_companies = _companies_in_text(sentence, detected_companies) or mentioned
                    parts.append((sentence, sent_companies))
            else:
                parts.append((line, mentioned))
        else:
            parts.append((line, list(detected_companies)))

    # Require a real numeric table, not just a header coincidence.
    if metric_rows < 2:
        return None
    return parts


_METRIC_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "revenue",
        (
            "total net sales",
            "total revenue",
            "total revenues",
            "net sales",
            "revenues",
            "revenue",
            "净销售额",
            "营业收入",
            "营收",
            "收入",
        ),
    ),
    ("gross_margin", ("gross margin", "gross profit margin", "gross profit", "毛利率")),
    ("operating_margin", ("operating margin", "营业利润率", "经营利润率")),
    ("operating_income", ("operating income", "income from operations", "营业利润", "经营利润")),
    ("net_income", ("net income", "net earnings", "净利润", "净收益")),
    ("eps", ("diluted earnings per share", "earnings per share", "diluted eps", "basic eps", "eps")),
    ("ebitda", ("ebitda",)),
    ("r_and_d", ("research and development", "r&d", "r & d", "研发")),
    ("debt", ("total long-term debt", "long-term debt", "total debt", "长期债务")),
    ("operating_cash_flow", ("cash generated by operating activities", "net cash provided by operating activities", "operating cash flow", "经营活动现金流量")),
    ("capex", ("capital expenditures", "purchases of property, plant and equipment", "capex", "资本开支", "资本支出")),
    ("cash", ("cash and cash equivalents", "现金及现金等价物")),
)

_FACT_VALUE_RE = re.compile(
    r"(?P<value>\$\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$\s?\d+(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?)"
    r"(?:\s*(?P<unit>billion|million|bn|mm|亿|万))?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:fy\s*)?(?:20\d{2})\b", re.IGNORECASE)

_SEGMENT_HINTS = (
    "segment",
    "by product",
    "by category",
    "geographic",
    "americas",
    "greater china",
    "iphone",
    "mac ",
    "ipad",
    "wearables",
    "services",
    "data center",
    "gaming",
    "automotive",
    "reportable segment",
)
_CONSOLIDATED_HINTS = (
    "consolidated statements of operations",
    "consolidated statement of operations",
    "consolidated statements of income",
    "total net sales",
    "total revenue",
    "total revenues",
)


def _classify_fact_scope(label: str, page_text: str) -> tuple[str, str]:
    """Return (statement_type, scope) for ranking: consolidated > segment > narrative."""
    blob = f"{label} {page_text[:800]}".lower()
    statement = "narrative"
    if any(x in blob for x in ("net sales", "revenue", "operating income", "net income", "eps", "gross")):
        statement = "income_statement"
    elif any(x in blob for x in ("cash flow", "capital expenditure", "capex")):
        statement = "cash_flow"
    elif any(x in blob for x in ("debt", "cash and cash equivalents", "balance sheet")):
        statement = "balance_sheet"

    label_l = label.lower()
    if any(h in blob for h in _CONSOLIDATED_HINTS) or label_l.startswith("total "):
        return statement, "consolidated"
    if any(h in blob for h in _SEGMENT_HINTS) and "total" not in label_l:
        return statement, "segment"
    if "total" in label_l:
        return statement, "consolidated"
    return statement, "narrative"


def _fact_rank_key(fact: dict[str, Any]) -> tuple[int, int, int, float]:
    """Higher is better: consolidated > segment > narrative; prefer 'total'; prefer html; larger value."""
    meta = fact.get("financial_fact") or {}
    scope = str(meta.get("scope") or "narrative")
    scope_rank = {"consolidated": 3, "segment": 1, "narrative": 0}.get(scope, 0)
    label = str(meta.get("row_label") or meta.get("alias") or "").lower()
    total_bonus = 1 if "total" in label else 0
    source_bonus = 1 if meta.get("source") == "html_table" else 0
    try:
        value = abs(float(str(meta.get("value") or "0").replace(",", "")))
    except ValueError:
        value = 0.0
    unit = str(meta.get("unit") or "").lower()
    if unit in {"billion", "bn"}:
        value *= 1000.0
    elif unit in {"million", "mm"}:
        value *= 1.0
    return (scope_rank, total_bonus, source_bonus, value)


def _extract_financial_facts(
    page_text: str,
    *,
    page_number: int,
    issuers: list[str],
    document_id: str,
    filename: str,
    start_index: int,
) -> list[dict[str, Any]]:
    """Emit compact metric/period/value chunks for numeric RAG grounding.

    Collects multiple candidates per metric then keeps the best-ranked
    (consolidated / total preferred over segment / narrative).
    """
    candidates: list[dict[str, Any]] = []
    lowered = page_text.lower()
    years = [m.group(0).upper().replace(" ", "") for m in _YEAR_RE.finditer(page_text)]
    default_period = years[0] if years else ""
    chunk_index = start_index
    for metric_key, aliases in _METRIC_ALIASES:
        for alias in aliases:
            for match in re.finditer(re.escape(alias), lowered):
                window = page_text[match.start() : match.start() + 220]
                value_match = _FACT_VALUE_RE.search(window)
                if not value_match:
                    continue
                raw_value = value_match.group("value").replace("$", "").replace(",", "").strip()
                unit = (value_match.group("unit") or "").strip().lower()
                try:
                    numeric_mag = float(raw_value)
                except ValueError:
                    continue
                # Guard against tiny false positives near "net sales" prose.
                if metric_key == "revenue":
                    if unit in {"billion", "bn"} and numeric_mag < 50:
                        continue
                    if not unit and numeric_mag < 100:
                        continue
                year_in_window = _YEAR_RE.search(window)
                period = (year_in_window.group(0) if year_in_window else default_period) or "FY"
                period = period.upper().replace(" ", "")
                if not period.startswith("FY") and re.fullmatch(r"20\d{2}", period):
                    period = f"FY{period}"
                company = issuers[0] if len(issuers) == 1 else (
                    _companies_in_text(window, issuers)[0]
                    if _companies_in_text(window, issuers)
                    else (issuers[0] if issuers else "Unknown")
                )
                statement_type, scope = _classify_fact_scope(alias, page_text)
                display = (
                    f"{company} {metric_key} {period}: {raw_value}"
                    f"{' ' + unit if unit else ''} "
                    f"(filing fact; {scope}; {statement_type})."
                )
                candidates.append(
                    {
                        "chunk_id": f"{document_id}:p{page_number}:f{chunk_index}",
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_number,
                        "text": display,
                        "companies": [company] if company != "Unknown" else list(issuers),
                        "chunk_type": "financial_metric",
                        "char_count": len(display),
                        "financial_fact": {
                            "metric": metric_key,
                            "period": period,
                            "value": raw_value,
                            "unit": unit,
                            "company": company,
                            "statement_type": statement_type,
                            "scope": scope,
                            "alias": alias,
                            "row_label": alias,
                            "source": "text_window",
                        },
                    }
                )
                chunk_index += 1
                break  # one candidate per alias; ranking picks among aliases

    # Keep best-ranked fact per metric (consolidated Total > segment).
    best_by_metric: dict[str, dict[str, Any]] = {}
    for fact in candidates:
        metric = str((fact.get("financial_fact") or {}).get("metric") or "")
        prev = best_by_metric.get(metric)
        if prev is None or _fact_rank_key(fact) > _fact_rank_key(prev):
            best_by_metric[metric] = fact
    # Also keep runner-up segment facts when consolidated exists (debug / alternate).
    ranked = list(best_by_metric.values())
    for fact in candidates:
        metric = str((fact.get("financial_fact") or {}).get("metric") or "")
        best = best_by_metric.get(metric)
        if best is None or fact is best:
            continue
        scope = str((fact.get("financial_fact") or {}).get("scope") or "")
        if scope == "segment" and str((best.get("financial_fact") or {}).get("scope")) == "consolidated":
            ranked.append(fact)
    return ranked


def chunk_document(
    document: dict[str, Any],
    *,
    max_chunk_chars: int = 900,
    overlap_chars: int = 120,
) -> list[dict[str, Any]]:
    """Page-aware chunking with financial-signal tagging for hybrid retrieval.

    Peer metric tables are split into per-company/per-row chunks so retrieval
    for Apple does not reuse the same Microsoft row text (and vice versa).
    Issuer companies (primary entity) are preferred for chunk tags; body peer
    mentions alone do not expand live-lookup scope.
    """
    pages: list[str] = document.get("pages") or []
    if not pages and document.get("text"):
        pages = _split_paragraphs(document["text"])

    chunks: list[dict[str, Any]] = []
    issuers = list(
        document.get("issuer_companies")
        or document.get("detected_companies")
        or []
    )
    # Mention list only for in-line tagging; never fall back to expanding issuers.
    mention_pool = list(document.get("mentioned_companies") or issuers)
    tag_pool = issuers or mention_pool
    document_id = document.get("document_id", "unknown")
    filename = document.get("filename", "unknown")

    for page_number, page_text in enumerate(pages, start=1):
        table_parts = _split_table_page(page_text, detected_companies=tag_pool)
        if table_parts:
            paragraphs = table_parts
        else:
            merged = _merge_small_chunks(_split_paragraphs(page_text), max_chunk_chars)
            paragraphs = []
            for paragraph in merged:
                tagged = _companies_in_text(paragraph, tag_pool) or list(issuers) or tag_pool
                paragraphs.append((paragraph, tagged))

        page_start = len(chunks)
        for chunk_index, item in enumerate(paragraphs):
            if isinstance(item, tuple):
                paragraph, chunk_companies = item
            else:
                paragraph, chunk_companies = item, tag_pool
            if (
                not table_parts
                and overlap_chars
                and chunk_index > 0
                and len(paragraph) > overlap_chars
            ):
                paragraph = paragraph[max(0, overlap_chars // 2) :]
            chunk_id = f"{document_id}:p{page_number}:c{chunk_index}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number,
                    "text": paragraph,
                    "companies": list(chunk_companies),
                    "chunk_type": _classify_chunk(paragraph),
                    "char_count": len(paragraph),
                }
            )
        # Always attach compact financial facts for numeric grounding.
        facts = _extract_financial_facts(
            page_text,
            page_number=page_number,
            issuers=issuers or tag_pool,
            document_id=document_id,
            filename=filename,
            start_index=len(chunks) - page_start,
        )
        chunks.extend(facts)

    # Prefer native HTML DOM table facts when present (SEC HTML path).
    tables = document.get("tables") if isinstance(document.get("tables"), list) else []
    if tables:
        from ..sec_html import tables_to_financial_facts

        html_facts = tables_to_financial_facts(
            tables,
            issuers=issuers or tag_pool,
            document_id=document_id,
            filename=filename,
            page_offset=max(1, len(pages)),
        )
        # Dedupe: for same metric+period, keep higher-ranked (consolidated > segment).
        existing = {
            (
                str((c.get("financial_fact") or {}).get("metric")),
                str((c.get("financial_fact") or {}).get("period")),
            ): c
            for c in chunks
            if c.get("financial_fact")
        }
        for fact in html_facts:
            meta = fact.get("financial_fact") or {}
            key = (str(meta.get("metric")), str(meta.get("period")))
            prev = existing.get(key)
            if prev is None or _fact_rank_key(fact) > _fact_rank_key(prev):
                if prev is not None and prev in chunks:
                    chunks.remove(prev)
                chunks.append(fact)
                existing[key] = fact
    return chunks
