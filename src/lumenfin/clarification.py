from __future__ import annotations

from typing import Any


def merge_clarification_into_query(query: str, clarification: dict[str, Any]) -> str:
    """Append structured user clarification to the original query."""
    if not clarification:
        return query
    parts = [query.strip()]
    company = clarification.get("company") or clarification.get("companies")
    if company:
        if isinstance(company, list):
            parts.append(f"Target companies: {', '.join(company)}.")
        else:
            parts.append(f"Target company: {company}.")
    time_range = clarification.get("time_range") or clarification.get("fiscal_year")
    if time_range:
        parts.append(f"Time range: {time_range}.")
    output_format = clarification.get("output_format")
    if output_format:
        parts.append(f"Preferred output format: {output_format}.")
    free_text = clarification.get("notes") or clarification.get("answer")
    if free_text:
        parts.append(str(free_text))
    return " ".join(parts)
