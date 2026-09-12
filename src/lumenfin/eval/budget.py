"""Single budget definition for document-task eval. Printer and runner share this."""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lumenfin.llm import BaseLLMClient

# Official Models & Pricing page captured 2026-09-10.
# https://api-docs.deepseek.com/quick_start/pricing
# Flash column: cache-hit / cache-miss / output; peak hours 01:00-04:00 and
# 06:00-10:00 UTC Mon-Fri. Intended model id is deepseek-flash (V4.1-Flash).
# Legacy deepseek-v4-flash is still accepted and billed at Flash.
PRICING_CAPTURE = {
    "source_url": "https://api-docs.deepseek.com/quick_start/pricing",
    "retrieved_at": "2026-09-10",
    "model_id": "deepseek-flash",
    "legacy_aliases": ["deepseek-v4-flash"],
    "served_as": "DeepSeek-V4.1-Flash",
    "currency": "USD",
    "unit": "per_1m_tokens",
    "flash": {
        "input_cache_hit_offpeak": 0.003,
        "input_cache_hit_peak": 0.006,
        "input_cache_miss_offpeak": 0.15,
        "input_cache_miss_peak": 0.3,
        "output_offpeak": 0.6,
        "output_peak": 1.2,
    },
    "peak_hours_utc": "01:00-04:00 and 06:00-10:00 Monday-Friday",
    "fee_hard_cap_supported": False,
    "fee_hard_cap_reason": (
        "Token usage is unknown before the run; cache hit/miss and peak/off-peak "
        "are not controllable. Request count is the hard cap. USD figures are "
        "estimates only and must not be treated as a dollar stop."
    ),
}

AUTH_TOKEN = "pilot24-lexical-v1-960http-2026-09-10"
SMOKE_TASK_IDS = ("dt-p01-nvda-oi", "dt-p21-narrative-refuse-oi")


def _b2_logical_upper_per_task() -> int:
    companies = 2
    planner_structure_attempts = 2
    profile = 1 * companies
    sentiment = 1 * companies
    critic = 1
    repair_rounds = 2
    repair_each = profile + sentiment + 1
    appendix_profiles = companies
    return (
        planner_structure_attempts
        + profile
        + sentiment
        + critic
        + repair_rounds * repair_each
        + appendix_profiles
    )


B2_LOGICAL_UPPER_PER_TASK = _b2_logical_upper_per_task()
PILOT_TASKS = 24
MAX_ATTEMPTS_PER_CALL = 2
LOGICAL_CALLS_UPPER = PILOT_TASKS * 1 + PILOT_TASKS * B2_LOGICAL_UPPER_PER_TASK
PROVIDER_REQUESTS_UPPER = LOGICAL_CALLS_UPPER * MAX_ATTEMPTS_PER_CALL

EVAL_BUDGET = {
    "id": AUTH_TOKEN,
    "pilot_tasks": PILOT_TASKS,
    "systems": ["b1_rag", "b2_agent"],
    "task_runs": PILOT_TASKS * 2,
    "smoke_task_ids": list(SMOKE_TASK_IDS),
    "model_id": PRICING_CAPTURE["model_id"],
    "model_aliases_accepted": [PRICING_CAPTURE["model_id"], *PRICING_CAPTURE["legacy_aliases"]],
    "allow_local_fallback": False,
    "retrieval_profile": "lexical_deterministic_eval",
    "b1_logical_calls_per_task": 1,
    "b2_logical_calls_upper_per_task": B2_LOGICAL_UPPER_PER_TASK,
    "logical_calls_upper": LOGICAL_CALLS_UPPER,
    "max_attempts_per_call": MAX_ATTEMPTS_PER_CALL,
    "max_provider_requests": PROVIDER_REQUESTS_UPPER,
    "max_output_tokens_per_call": 700,
    "max_input_chars_per_call": 12000,
    "timeout_seconds": 45,
    "max_inflight": 1,
    "bounded_repair_enabled": False,
    "profile_llm_max_attempts": 1,
    "critic_max_iterations": 2,
    "remote_embedding_calls": 0,
    "remote_rerank_calls": 0,
    "pricing": PRICING_CAPTURE,
    "do_not_estimate_as": "48 model calls or 432 HTTP requests",
    "authorization_env": "LUMENFIN_EVAL_BUDGET_AUTH",
    "confirm_budget_flag_is_not_authorization": True,
}


def estimate_usd_range(*, provider_requests: int) -> dict[str, Any]:
    """Range only. Not a hard dollar cap."""
    flash = PRICING_CAPTURE["flash"]
    out_cap = EVAL_BUDGET["max_output_tokens_per_call"]
    # Typical: 2k in / 200 out, cache-miss off-peak, all logical succeed once.
    typical_in = provider_requests * 2000
    typical_out = provider_requests * 200
    typical = (typical_in / 1_000_000) * flash["input_cache_miss_offpeak"] + (
        typical_out / 1_000_000
    ) * flash["output_offpeak"]
    # Conservative: full input clip ~4 chars/token, max output tokens, cache-miss peak.
    conservative_in = provider_requests * (EVAL_BUDGET["max_input_chars_per_call"] // 4)
    conservative_out = provider_requests * out_cap
    conservative = (conservative_in / 1_000_000) * flash["input_cache_miss_peak"] + (
        conservative_out / 1_000_000
    ) * flash["output_peak"]
    return {
        "typical_usd_cache_miss_offpeak": round(typical, 4),
        "conservative_usd_cache_miss_peak_max_tokens": round(conservative, 4),
        "assumptions": {
            "typical_input_tokens_per_request": 2000,
            "typical_output_tokens_per_request": 200,
            "conservative_input_tokens_per_request": EVAL_BUDGET["max_input_chars_per_call"] // 4,
            "conservative_output_tokens_per_request": out_cap,
            "provider_requests": provider_requests,
            "ignores_cache_hits": True,
        },
        "fee_hard_cap_supported": False,
        "unknown_is_not_zero": True,
    }


def budget_packet(*, n_tasks: int = 24, smoke: bool = False) -> dict[str, Any]:
    spec = dict(EVAL_BUDGET)
    spec["usd_estimate"] = estimate_usd_range(provider_requests=int(spec["max_provider_requests"]))
    spec["pricing"] = PRICING_CAPTURE
    spec["stage_note"] = (
        "Smoke and remainder share max_provider_requests; do not reset the cap per stage."
        if not smoke
        else "Smoke draws from the same 960 HTTP cap as the full pilot."
    )
    spec["smoke_only"] = bool(smoke)
    if smoke:
        spec["smoke_task_ids"] = list(SMOKE_TASK_IDS)
    return spec


class BudgetExhausted(RuntimeError):
    """No further provider requests are allowed."""


@dataclass
class EvalBudget:
    spec: dict[str, Any]
    provider_requests: int = 0
    logical_calls: int = 0
    exhausted: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def max_provider_requests(self) -> int:
        return int(self.spec["max_provider_requests"])

    def check_before_call(self) -> None:
        remaining = self.max_provider_requests - self.provider_requests
        if remaining < 1 or self.exhausted:
            self.exhausted = True
            raise BudgetExhausted(
                f"eval budget exhausted: used {self.provider_requests}/"
                f"{self.max_provider_requests} provider requests"
            )

    def consume_http(self) -> None:
        """Count one provider HTTP attempt, including failed retries. Call before the POST."""
        self.check_before_call()
        self.provider_requests += 1
        self.events.append({"http": True, "provider_requests": self.provider_requests})
        if self.provider_requests >= self.max_provider_requests:
            self.exhausted = True

    def record_logical(self, *, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        self.logical_calls += 1
        self.events.append(
            {
                "logical_calls": self.logical_calls,
                "provider_requests": self.provider_requests,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "auth_id": AUTH_TOKEN,
            "provider_requests": self.provider_requests,
            "logical_calls": self.logical_calls,
            "max_provider_requests": self.max_provider_requests,
            "exhausted": self.exhausted,
            "usd_unknown_until_usage": True,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        tmp.replace(path)


def clip_combined_input(system_prompt: str, user_prompt: str, *, max_chars: int) -> tuple[str, str]:
    """One limit for system+user. Does not give each side the full budget."""
    cap = max(1, int(max_chars))
    system_prompt = str(system_prompt or "")
    user_prompt = str(user_prompt or "")
    if len(system_prompt) + len(user_prompt) <= cap:
        return system_prompt, user_prompt
    sys_keep = min(len(system_prompt), max(0, cap // 4))
    system_prompt = system_prompt[:sys_keep]
    user_prompt = user_prompt[: max(0, cap - len(system_prompt))]
    return system_prompt, user_prompt


def fair_rerun_budget_report(state_path: Path) -> dict[str, Any]:
    """Accumulated 960 HTTP cap. Changing --out does not grant a new quota."""
    loaded = load_eval_budget(state_path)
    remaining = loaded.max_provider_requests - loaded.provider_requests
    typical_prior = loaded.provider_requests
    return {
        "auth_id": AUTH_TOKEN,
        "cap_http_total": loaded.max_provider_requests,
        "used_http": loaded.provider_requests,
        "remaining_http": remaining,
        "new_quota_granted": False,
        "out_file_does_not_reset_cap": True,
        "budget_state": str(state_path).replace("\\", "/"),
        "fair_rerun": {
            "tasks": PILOT_TASKS,
            "systems": ["b1", "b2"],
            "cases": PILOT_TASKS * 2,
            "logical_calls_upper": LOGICAL_CALLS_UPPER,
            "http_upper_if_cap_were_fresh": PROVIDER_REQUESTS_UPPER,
            "http_this_run_cannot_exceed_remaining": remaining,
            "typical_http_from_cumulative_used": typical_prior,
            "expected_http": {
                "typical_if_similar_to_completed_48_run": typical_prior,
                "hard_stop": remaining,
                "theoretical_upper_exceeds_remaining": PROVIDER_REQUESTS_UPPER > remaining,
            },
            "on_exhaustion": (
                "Stop new provider HTTP (BudgetExhausted). Mark remaining cases "
                "budget_exhausted, keep completed rows in the --out ledger, and "
                "persist used count in budget_state.json. Do not reset the 960 cap."
            ),
        },
        "authorization_request": (
            f"授权消耗同一累计额度 {AUTH_TOKEN} 中剩余的 {remaining} 次 DeepSeek HTTP，"
            f"用于一次完整 {PILOT_TASKS}×B1/B2 公平复跑。这不是新的 "
            f"{PROVIDER_REQUESTS_UPPER} 次额度。"
        ),
        "command": (
            "python scripts/run_document_task_eval.py --pilot --allow-live "
            "--out outputs/document_task_eval/pilot_live_fair_v2.json"
        ),
    }


def load_eval_budget(path: Path) -> EvalBudget:
    spec = dict(EVAL_BUDGET)
    if not path.is_file():
        return EvalBudget(spec=spec)
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("auth_id") or "") not in {AUTH_TOKEN, ""}:
        return EvalBudget(spec=spec)
    used = int(data.get("provider_requests") or 0)
    logical = int(data.get("logical_calls") or 0)
    budget = EvalBudget(spec=spec, provider_requests=used, logical_calls=logical)
    if used >= budget.max_provider_requests:
        budget.exhausted = True
    return budget


_BUDGET: ContextVar[EvalBudget | None] = ContextVar("lumenfin_eval_budget", default=None)


def set_eval_budget(budget: EvalBudget | None) -> None:
    _BUDGET.set(budget)


def get_eval_budget() -> EvalBudget | None:
    return _BUDGET.get()


class BudgetedLLMClient(BaseLLMClient):
    backend_name = "deepseek"

    def __init__(self, inner: BaseLLMClient, budget: EvalBudget, spec: dict[str, Any]) -> None:
        super().__init__()
        self._inner = inner
        self.budget = budget
        self.spec = spec
        self.backend_name = str(getattr(inner, "backend_name", "deepseek"))
        self.model_name = str(getattr(inner, "model_name", spec["model_id"]))
        self.last_attempts = 0

    def _bind_http_hook(self) -> None:
        primary = getattr(self._inner, "primary", None) or self._inner
        setattr(primary, "_before_provider_http", self.budget.consume_http)

    def fork_usage(self) -> "BudgetedLLMClient":
        inner = self._inner.fork_usage() if hasattr(self._inner, "fork_usage") else self._inner
        child = BudgetedLLMClient(inner, self.budget, self.spec)
        child._usage_totals = self._usage_totals
        child._usage_mark = self._usage_mark
        return child

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        self._bind_http_hook()
        primary = getattr(self._inner, "primary", None) or self._inner
        if not hasattr(primary, "settings"):
            self.budget.consume_http()
        system_prompt, user_prompt = clip_combined_input(
            system_prompt, user_prompt, max_chars=int(self.spec["max_input_chars_per_call"])
        )
        capped = min(int(max_tokens), int(self.spec["max_output_tokens_per_call"]))
        try:
            text = self._inner.chat(system_prompt, user_prompt, temperature=temperature, max_tokens=capped)
        except Exception:
            attempts = int(
                getattr(self._inner, "last_attempts", 0)
                or getattr(self._inner, "primary_attempts", 0)
                or 1
            )
            self.last_attempts = attempts
            self.budget.record_logical(prompt_tokens=None, completion_tokens=None)
            raise
        attempts = int(
            getattr(self._inner, "last_attempts", 0)
            or getattr(self._inner, "primary_attempts", 0)
            or 1
        )
        self.last_attempts = attempts
        usage = {}
        if hasattr(self._inner, "usage_since_mark"):
            try:
                usage = self._inner.usage_since_mark()
            except Exception:
                usage = {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        self._add_usage(prompt_tokens, completion_tokens)
        self.budget.record_logical(
            prompt_tokens=prompt_tokens or None,
            completion_tokens=completion_tokens or None,
        )
        return text
