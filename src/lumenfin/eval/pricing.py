"""Model price table. Missing usage or price is unknown, never zero-filled."""

from __future__ import annotations

from typing import Any

# Source: DeepSeek API docs pricing page. Confirm before a paid freeze run.
# https://api-docs.deepseek.com/quick_start/pricing
PRICING = {
    "schema_version": "lumenfin_model_pricing.v1",
    "currency": "USD",
    "models": {
        "deepseek-flash": {
            "input_per_million": 0.15,
            "output_per_million": 0.6,
            "source_url": "https://api-docs.deepseek.com/quick_start/pricing",
            "retrieved_at": "2026-09-10",
            "status": "flash_cache_miss_offpeak_estimate",
            "assumptions": {
                "tier": "cache_miss_offpeak",
                "peak_hours_utc": "01:00-04:00 and 06:00-10:00 Monday-Friday",
                "cache_hit_not_used": True,
            },
        },
        "deepseek-v4-flash": {
            "input_per_million": 0.15,
            "output_per_million": 0.6,
            "source_url": "https://api-docs.deepseek.com/quick_start/pricing",
            "retrieved_at": "2026-09-10",
            "status": "alias_of_deepseek-flash_cache_miss_offpeak_estimate",
            "assumptions": {
                "tier": "cache_miss_offpeak",
                "served_as": "DeepSeek-V4.1-Flash",
            },
        },
        "local-fallback": {
            "input_per_million": 0.0,
            "output_per_million": 0.0,
            "source_url": "local",
            "retrieved_at": "2026-09-10",
            "status": "offline_no_billable_tokens",
        },
    },
}


def estimate_cost_usd(
    *,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> dict[str, Any]:
    spec = PRICING["models"].get(model) or {
        "input_per_million": None,
        "output_per_million": None,
        "status": "unknown_model",
        "source_url": None,
    }
    if prompt_tokens is None or completion_tokens is None:
        return {
            "usd": None,
            "status": "unknown_usage",
            "model": model,
            "source_url": spec.get("source_url"),
        }
    inp = spec.get("input_per_million")
    out = spec.get("output_per_million")
    if inp is None or out is None:
        return {
            "usd": None,
            "status": spec.get("status") or "unknown_price",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "source_url": spec.get("source_url"),
        }
    usd = (prompt_tokens / 1_000_000) * float(inp) + (completion_tokens / 1_000_000) * float(out)
    return {
        "usd": round(usd, 8),
        "status": spec.get("status") or "priced",
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "source_url": spec.get("source_url"),
    }
