from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_METRIC_KEY_MAP = {
    "revenue": "revenue",
    "revenue_2025": "revenue",
    "ebitda": "ebitda",
    "ebitda_2025": "ebitda",
    "r_and_d": "r_and_d",
    "r_and_d_2025": "r_and_d",
    "rd": "r_and_d",
    "operating_income": "operating_income",
    "operating_income_2025": "operating_income",
}


def normalize_metric_hints(metrics: dict[str, Any]) -> dict[str, float]:
    hints: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        mapped = _METRIC_KEY_MAP.get(key)
        if mapped is None:
            continue
        try:
            hints[mapped] = float(value)
        except (TypeError, ValueError):
            continue
    return hints


def structured_metrics_to_document_contexts(
    company_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn uploaded JSON/CSV-style metrics into document_contexts for retrieval/quant."""
    contexts: list[dict[str, Any]] = []
    for company, metrics in company_metrics.items():
        if not isinstance(metrics, dict):
            continue
        hints = normalize_metric_hints(metrics)
        serialized = json.dumps(metrics, ensure_ascii=False)
        contexts.append(
            {
                "document_id": f"structured_{company}",
                "filename": f"{company}_metrics.json",
                "detected_companies": [company],
                "metric_hints": hints,
                "text": serialized,
                "pages": [serialized],
                "excerpt": f"Structured financial metrics uploaded for {company}.",
                "source_type": "structured_json",
            }
        )
    return contexts


def load_metrics_json_file(path: Path) -> dict[str, dict[str, Any]]:
    """Load { \"Company\": { \"revenue_2025\": 1.0, ... } } from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Metrics JSON must be an object keyed by company name.")
    company_metrics: dict[str, dict[str, Any]] = {}
    for company, metrics in payload.items():
        if isinstance(metrics, dict):
            company_metrics[str(company)] = metrics
    if not company_metrics:
        raise ValueError("Metrics JSON must contain at least one company object.")
    return company_metrics
