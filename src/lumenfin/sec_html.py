"""Native SEC HTML / DOM table extraction (prefer over HTML→PDF text loss).

Pipeline target:
  SEC HTML → DOM tables → financial facts (row/column/header/period)
  PDF remains a fallback for uploads that are already PDF-only.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .document_entity import resolve_document_entities
from .documents import (
    _extract_metric_hints,
    extract_metric_hints_for_company,
    merge_per_company_metric_hints,
)

_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6", "li", "section"}
)
_YEAR_RE = re.compile(r"\b(?:FY\s*)?(20\d{2})\b", re.IGNORECASE)
_NUM_RE = re.compile(
    r"^\(?\$?\s*-?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?$|^\(?\$?\s*-?\d+(?:\.\d+)?\)?$"
)

# Label → (metric_key, statement_type hint)
_ROW_METRIC_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^(?:total\s+)?net\s+sales$|^(?:total\s+)?revenue(?:s)?$", re.I), "revenue", "income_statement"),
    (re.compile(r"^gross\s+(?:margin|profit)$", re.I), "gross_margin", "income_statement"),
    (re.compile(r"^operating\s+(?:income|margin)|income\s+from\s+operations$", re.I), "operating_income", "income_statement"),
    (re.compile(r"^net\s+(?:income|earnings)$", re.I), "net_income", "income_statement"),
    (re.compile(r"^(?:diluted\s+)?(?:earnings|income)\s+per\s+(?:share|common\s+share)|^eps$", re.I), "eps", "income_statement"),
    (re.compile(r"^research\s+and\s+development|^r\s*&\s*d$", re.I), "r_and_d", "income_statement"),
    (re.compile(r"^capital\s+expenditures?|^purchases?\s+of\s+property|^capex$", re.I), "capex", "cash_flow"),
    (re.compile(r"^net\s+cash\s+(?:provided|used).*(?:operating)|cash\s+from\s+operations", re.I), "operating_cash_flow", "cash_flow"),
    (re.compile(r"^total\s+(?:long[- ]term\s+)?debt$|^long[- ]term\s+debt$", re.I), "debt", "balance_sheet"),
    (re.compile(r"^cash\s+and\s+cash\s+equivalents$", re.I), "cash", "balance_sheet"),
]


def _clean_cell(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return text.strip()


def _is_numeric_cell(text: str) -> bool:
    cleaned = _clean_cell(text).replace(",", "").replace("$", "").replace(" ", "")
    if not cleaned or cleaned in {"—", "-", "–", "n/a", "nm"}:
        return False
    return bool(_NUM_RE.match(_clean_cell(text))) or bool(
        re.fullmatch(r"\(?-?\d+(?:\.\d+)?\)?", cleaned)
    )


def _parse_number(text: str) -> str | None:
    cleaned = _clean_cell(text)
    neg = cleaned.startswith("(") and cleaned.endswith(")")
    digits = re.sub(r"[^\d.]", "", cleaned)
    if not digits:
        return None
    return f"-{digits}" if neg else digits


class _SecHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.plain_parts: list[str] = []
        self.tables: list[dict[str, Any]] = []
        self._in_script = False
        self._in_style = False
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._cell_tag = ""
        self._cell_buf: list[str] = []
        self._row: list[dict[str, Any]] = []
        self._rows: list[list[dict[str, Any]]] = []
        self._table_attrs: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if tag in {"script", "style", "noscript"}:
            self._in_script = True
            return
        if tag == "br":
            self.plain_parts.append("\n")
            if self._in_cell:
                self._cell_buf.append(" ")
            return
        if tag == "table":
            self._in_table = True
            self._rows = []
            self._table_attrs = attr_map
            self.plain_parts.append("\n\n")
            return
        if self._in_table and tag == "tr":
            self._in_row = True
            self._row = []
            return
        if self._in_table and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_tag = tag
            self._cell_buf = []
            return
        if tag in _BLOCK_TAGS and not self._in_table:
            self.plain_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._in_script = False
            return
        if tag in {"td", "th"} and self._in_cell:
            text = _clean_cell("".join(self._cell_buf))
            self._row.append({"tag": self._cell_tag, "text": text})
            self._in_cell = False
            self._cell_buf = []
            return
        if tag == "tr" and self._in_row:
            if any(c["text"] for c in self._row):
                self._rows.append(self._row)
            self._in_row = False
            self._row = []
            self.plain_parts.append("\n")
            return
        if tag == "table" and self._in_table:
            self.tables.append(
                {
                    "attrs": dict(self._table_attrs),
                    "rows": [
                        [cell["text"] for cell in row]
                        for row in self._rows
                    ],
                    "cells": self._rows,
                }
            )
            self._in_table = False
            self._rows = []
            self.plain_parts.append("\n\n")
            return
        if tag in _BLOCK_TAGS and not self._in_table:
            self.plain_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_script:
            return
        if self._in_cell:
            self._cell_buf.append(data)
            return
        if data:
            self.plain_parts.append(data)


def _classify_table_scope(header_blob: str, caption_blob: str, row_label: str) -> tuple[str, str]:
    """Return (statement_type, scope)."""
    blob = f"{caption_blob} {header_blob} {row_label}".lower()
    statement = "narrative"
    if any(x in blob for x in ("operations", "income", "earnings", "net sales", "revenue")):
        statement = "income_statement"
    elif any(x in blob for x in ("cash flow", "cash flows", "operating activities")):
        statement = "cash_flow"
    elif any(x in blob for x in ("balance sheet", "financial position", "assets", "liabilities")):
        statement = "balance_sheet"

    scope = "narrative"
    if any(
        x in blob
        for x in (
            "consolidated statements",
            "consolidated statement",
            "total net sales",
            "total revenue",
            "total revenues",
        )
    ) or re.search(r"\btotal\b", row_label, re.I):
        scope = "consolidated"
    if any(
        x in blob
        for x in (
            "segment",
            "reportable segment",
            "geographic",
            "by product",
            "net sales by",
            "revenue by",
            "iphone",
            "services",
            "americas",
            "europe",
            "greater china",
            "data center",
            "gaming",
        )
    ) and not re.search(r"\btotal\b", row_label, re.I):
        scope = "segment"
    if scope == "narrative" and statement != "narrative":
        scope = "consolidated" if "total" in row_label.lower() else "segment"
    return statement, scope


def _header_periods(header_row: list[str]) -> list[str | None]:
    periods: list[str | None] = []
    for cell in header_row:
        match = _YEAR_RE.search(cell or "")
        if match:
            year = match.group(1)
            periods.append(f"FY{year}")
        else:
            periods.append(None)
    return periods


def _row_numeric_values(row: list[str]) -> list[str]:
    """Extract numeric cells, skipping currency markers and blanks.

    SEC HTML often lays out: ``Total net sales | $ | 391,035 | $ | 383,285``.
    """
    values: list[str] = []
    for cell in row[1:]:
        cleaned = _clean_cell(cell)
        if not cleaned or cleaned in {"$", "%", "—", "-", "–"}:
            continue
        if not _is_numeric_cell(cleaned):
            continue
        # Skip lone percentages already partially filtered; skip tiny index nums
        parsed = _parse_number(cleaned)
        if parsed is None:
            continue
        values.append(parsed)
    return values


def _ordered_periods(header_row: list[str]) -> list[str]:
    periods = [p for p in _header_periods(header_row) if p]
    # Deduplicate while preserving order (some tables repeat year labels).
    seen: set[str] = set()
    ordered: list[str] = []
    for period in periods:
        if period not in seen:
            seen.add(period)
            ordered.append(period)
    return ordered


def tables_to_financial_facts(
    tables: list[dict[str, Any]],
    *,
    issuers: list[str],
    document_id: str,
    filename: str,
    page_offset: int = 1,
) -> list[dict[str, Any]]:
    """Convert DOM tables into ranked financial_fact chunks."""
    facts: list[dict[str, Any]] = []
    company = issuers[0] if issuers else "Unknown"
    idx = 0
    for table_i, table in enumerate(tables):
        rows: list[list[str]] = table.get("rows") or []
        if len(rows) < 2:
            continue
        # Find a header-ish row with years
        header_idx = 0
        for i, row in enumerate(rows[:8]):
            if sum(1 for c in row if _YEAR_RE.search(c or "")) >= 1:
                header_idx = i
                break
        header = rows[header_idx]
        periods = _ordered_periods(header)
        caption = " ".join(header)
        for row in rows[header_idx + 1 :]:
            if not row:
                continue
            label = _clean_cell(row[0])
            if not label or _is_numeric_cell(label):
                continue
            # Skip ratio / percentage rows that reuse net-sales wording.
            if "percentage" in label.lower() or label.strip().endswith("%"):
                continue
            metric_key = None
            statement_hint = "narrative"
            for pattern, key, stmt in _ROW_METRIC_PATTERNS:
                if pattern.search(label):
                    metric_key = key
                    statement_hint = stmt
                    break
            if not metric_key:
                continue
            values = _row_numeric_values(row)
            if not values:
                continue
            # Pair year columns left→right with numeric values left→right.
            pairs: list[tuple[str, str]] = []
            if periods:
                for period, value in zip(periods, values):
                    pairs.append((period, value))
            else:
                pairs.append(("FY", values[0]))
            for period, value in pairs:
                # Ignore tiny integers on revenue totals (often footnotes / counts).
                try:
                    numeric = abs(float(value))
                except ValueError:
                    continue
                if metric_key == "revenue" and numeric < 100 and "." not in value:
                    continue
                statement_type, scope = _classify_table_scope(caption, caption, label)
                if statement_type == "narrative":
                    statement_type = statement_hint
                # Totals on income statement tables are consolidated.
                if re.search(r"^total\s+(net\s+sales|revenue)", label, re.I):
                    scope = "consolidated"
                    statement_type = "income_statement"
                page = page_offset + (table_i // 3)
                display = (
                    f"{company} {metric_key} {period}: {value} "
                    f"(html table; {scope}; {statement_type}; row={label})."
                )
                facts.append(
                    {
                        "chunk_id": f"{document_id}:html:t{table_i}:f{idx}",
                        "document_id": document_id,
                        "filename": filename,
                        "page": page,
                        "text": display,
                        "companies": [company] if company != "Unknown" else list(issuers),
                        "chunk_type": "financial_metric",
                        "char_count": len(display),
                        "financial_fact": {
                            "metric": metric_key,
                            "period": period,
                            "value": value,
                            "unit": "million" if numeric >= 100 else "",
                            "company": company,
                            "statement_type": statement_type,
                            "scope": scope,
                            "row_label": label,
                            "source": "html_table",
                            "headers": header,
                        },
                    }
                )
                idx += 1
    return facts


def parse_sec_html_document(file_path: Path) -> dict[str, Any]:
    """Parse SEC filing HTML into document_context with preserved tables."""
    path = Path(file_path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    parser = _SecHtmlParser()
    try:
        parser.feed(raw)
        parser.close()
    except Exception:
        # Extremely broken HTML — fall back to tag strip.
        text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", raw)
        text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        pages = [text]
        entity = resolve_document_entities(text=text, pages=pages, filename=path.name)
        return {
            "document_id": path.stem,
            "filename": path.name,
            "path": str(path),
            "page_count": 1,
            "pages": pages,
            "text": text,
            "excerpt": text[:4000],
            "detected_companies": list(entity.get("detected_companies") or []),
            "issuer_companies": list(entity.get("issuer_companies") or []),
            "mentioned_companies": list(entity.get("mentioned_companies") or []),
            "primary_company": entity.get("primary_company"),
            "metric_hints": _extract_metric_hints(text),
            "tables": [],
            "source_type": "sec_html",
            "parse_fallback": "tag_strip",
        }

    text = re.sub(r"[ \t]+", " ", "".join(parser.plain_parts))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    # Paginate for chunker compatibility (~2800 chars ≈ prior PDF converter).
    page_size = 2800
    pages = [text[i : i + page_size] for i in range(0, max(1, len(text)), page_size)] or [text]
    entity = resolve_document_entities(text=text, pages=pages[:3], filename=path.name)
    issuers = list(entity.get("issuer_companies") or entity.get("detected_companies") or [])
    mentioned = list(entity.get("mentioned_companies") or issuers)
    tables = parser.tables
    # Attach linearized table blocks into pages so narrative chunking still sees numbers.
    table_pages: list[str] = []
    for t_i, table in enumerate(tables):
        rows = table.get("rows") or []
        lines = [" | ".join(r) for r in rows[:80]]
        table_pages.append(f"[HTML TABLE {t_i}]\n" + "\n".join(lines))
    if table_pages:
        # Append compact table pages after prose pages (facts also emitted separately).
        pages = pages + table_pages

    hint_scope = issuers or mentioned
    metric_hints = _extract_metric_hints(text)
    per_company = merge_per_company_metric_hints(
        text,
        hint_scope,
        base={c: extract_metric_hints_for_company(text, c) for c in hint_scope},
    )
    if len(issuers) == 1 and issuers[0] in per_company:
        metric_hints = {**metric_hints, **per_company[issuers[0]]}

    return {
        "document_id": path.stem,
        "filename": path.name,
        "path": str(path),
        "page_count": len(pages),
        "pages": pages,
        "text": text,
        "excerpt": text[:4000],
        "detected_companies": list(entity.get("detected_companies") or issuers),
        "issuer_companies": issuers,
        "mentioned_companies": mentioned,
        "primary_company": entity.get("primary_company"),
        "metric_hints": metric_hints,
        "per_company_metric_hints": per_company,
        "tables": tables,
        "source_type": "sec_html",
        "html_table_count": len(tables),
    }
