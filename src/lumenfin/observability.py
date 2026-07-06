from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# DeepSeek-chat rough public pricing (USD per 1M tokens) — for observability estimates only.
DEEPSEEK_INPUT_PER_M = 0.27
DEEPSEEK_OUTPUT_PER_M = 1.10


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    if model in {"deepseek", "deepseek-chat"}:
        return round(
            (prompt_tokens / 1_000_000) * DEEPSEEK_INPUT_PER_M
            + (completion_tokens / 1_000_000) * DEEPSEEK_OUTPUT_PER_M,
            6,
        )
    return 0.0


@dataclass
class StepTimer:
    step: str
    llm_client: Any
    started_at: str = field(default_factory=utc_now_iso)
    _t0: float = field(default_factory=time.perf_counter)

    def metrics(self) -> dict[str, Any]:
        usage = self.llm_client.usage_since_mark()
        latency_ms = round((time.perf_counter() - self._t0) * 1000, 2)
        model = getattr(self.llm_client, "model_name", self.llm_client.backend_name)
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return {
            "started_at": self.started_at,
            "ended_at": utc_now_iso(),
            "latency_ms": latency_ms,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "estimated_cost_usd": estimate_cost_usd(model, prompt_tokens, completion_tokens),
            "tool_calls": 0,
            "retry_count": 0,
        }


def merge_telemetry(existing: dict[str, Any] | None, event: dict[str, Any]) -> dict[str, Any]:
    telemetry = dict(existing or {})
    spans: list[dict[str, Any]] = list(telemetry.get("node_spans", []))
    spans.append(
        {
            "step": event.get("step"),
            "status": event.get("status"),
            "latency_ms": event.get("latency_ms"),
            "model": event.get("model"),
            "prompt_tokens": event.get("prompt_tokens", 0),
            "completion_tokens": event.get("completion_tokens", 0),
            "estimated_cost_usd": event.get("estimated_cost_usd", 0.0),
        }
    )
    telemetry["node_spans"] = spans
    telemetry["total_prompt_tokens"] = int(telemetry.get("total_prompt_tokens", 0)) + int(
        event.get("prompt_tokens", 0)
    )
    telemetry["total_completion_tokens"] = int(telemetry.get("total_completion_tokens", 0)) + int(
        event.get("completion_tokens", 0)
    )
    telemetry["total_latency_ms"] = round(
        float(telemetry.get("total_latency_ms", 0.0)) + float(event.get("latency_ms", 0.0)),
        2,
    )
    telemetry["total_estimated_cost_usd"] = round(
        float(telemetry.get("total_estimated_cost_usd", 0.0))
        + float(event.get("estimated_cost_usd", 0.0)),
        6,
    )
    return telemetry
