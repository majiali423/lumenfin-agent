from __future__ import annotations

import json
import logging
import os
import time
from copy import copy
from dataclasses import dataclass
from typing import Any

import httpx

from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .provider_resilience import (
    InvalidProviderResponseError,
    ProviderCallContext,
    ProviderCallPolicy,
    acquire_provider_slot,
    call_with_policy,
    classify_provider_exception,
    close_shared_http_clients,
    get_shared_http_client,
    is_retryable_provider_exception,
)
from .tools import KNOWN_ALIASES, alias_mentioned

logger = logging.getLogger(__name__)


class EmptyVisibleCompletionError(RuntimeError):
    """Raised when a provider returns no user-visible assistant content."""


def _extract_companies_from_text(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for company in SAMPLE_FINANCIAL_DATA:
        if alias_mentioned(company.lower(), lowered, text) and company not in found:
            found.append(company)
    for alias, name in KNOWN_ALIASES.items():
        if alias_mentioned(alias, lowered, text) and name not in found:
            found.append(name)
    return found


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None
    base_url: str
    model: str
    timeout_seconds: float
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5

    @classmethod
    def from_env(cls) -> "LLMSettings":
        raw_key = os.getenv("DEEPSEEK_API_KEY")
        api_key = raw_key.strip() if raw_key and raw_key.strip() else None
        base_url = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip()
        model = (os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-flash").strip()
        # Prefer MAS_LLM_MAX_ATTEMPTS when set; DEEPSEEK_MAX_RETRIES remains total attempts.
        attempts_raw = os.getenv("MAS_LLM_MAX_ATTEMPTS") or os.getenv("DEEPSEEK_MAX_RETRIES") or "3"
        timeout_str = os.getenv("DEEPSEEK_TIMEOUT_SECONDS") or "45"
        backoff_str = os.getenv("DEEPSEEK_RETRY_BACKOFF_SECONDS") or "0.5"
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=float(timeout_str),
            max_retries=max(1, int(attempts_raw)),
            retry_backoff_seconds=max(0.0, float(backoff_str)),
        )


class BaseLLMClient:
    backend_name = "unknown"
    model_name = "unknown"

    def __init__(self) -> None:
        self._usage_totals: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self._usage_mark: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        raise NotImplementedError

    def mark_usage_start(self) -> None:
        self._usage_mark = dict(self._usage_totals)

    def usage_since_mark(self) -> dict[str, int]:
        return {
            "prompt_tokens": self._usage_totals["prompt_tokens"] - self._usage_mark["prompt_tokens"],
            "completion_tokens": self._usage_totals["completion_tokens"] - self._usage_mark["completion_tokens"],
        }

    def _add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._usage_totals["prompt_tokens"] += prompt_tokens
        self._usage_totals["completion_tokens"] += completion_tokens

    def fork_usage(self) -> "BaseLLMClient":
        """Share immutable provider configuration, but not mutable usage state."""
        forked = copy(self)
        forked._usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
        forked._usage_mark = {"prompt_tokens": 0, "completion_tokens": 0}
        return forked


class DeepSeekChatClient(BaseLLMClient):
    """DeepSeek HTTP chat client.

    Retry owner: ``call_with_policy`` inside ``chat`` (single layer only).
    Transport: process-local shared ``httpx.Client``.
    """

    backend_name = "deepseek"
    _CLIENT_KEY = "deepseek-chat"

    def __init__(
        self,
        settings: LLMSettings,
        *,
        http_client: httpx.Client | None = None,
        call_context: ProviderCallContext | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.model_name = settings.model
        self._http_client = http_client
        self._owns_client = http_client is None
        self._call_context = call_context
        self.extra_headers: dict[str, str] = {}
        self.last_attempts = 0
        self.last_trace: list[dict[str, Any]] = []

    def bind_call_context(self, context: ProviderCallContext | None) -> None:
        self._call_context = context

    def _client(self, timeout: httpx.Timeout | float) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        return get_shared_http_client(self._CLIENT_KEY, timeout=timeout)

    def close(self) -> None:
        # Shared process client is closed via close_shared_http_clients().
        if self._http_client is not None and self._owns_client and not self._http_client.is_closed:
            self._http_client.close()

    def fork_usage(self) -> "DeepSeekChatClient":
        forked = DeepSeekChatClient(
            self.settings,
            http_client=self._http_client,
            call_context=None,
        )
        forked._usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
        forked._usage_mark = {"prompt_tokens": 0, "completion_tokens": 0}
        forked._owns_client = False
        forked.extra_headers = dict(self.extra_headers)
        return forked

    def _policy(self) -> ProviderCallPolicy:
        return ProviderCallPolicy(
            provider="deepseek",
            operation="chat",
            max_attempts=self.settings.max_retries,
            connect_timeout_seconds=min(5.0, self.settings.timeout_seconds),
            read_timeout_seconds=self.settings.timeout_seconds,
            write_timeout_seconds=self.settings.timeout_seconds,
            pool_timeout_seconds=min(5.0, self.settings.timeout_seconds),
            base_backoff_seconds=self.settings.retry_backoff_seconds,
            max_backoff_seconds=max(self.settings.retry_backoff_seconds * 8, 8.0),
            jitter_ratio=0.2,
        )

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking": {"type": "disabled"},
        }
        policy = self._policy()
        context = self._call_context or ProviderCallContext.create()
        if context.trace_sink is None:
            context.trace_sink = []
        self.last_trace = context.trace_sink

        def _once() -> tuple[str, dict]:
            remaining = context.remaining_seconds()
            timeout = policy.httpx_timeout(remaining_seconds=remaining)
            client = self._client(timeout)
            response = client.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise InvalidProviderResponseError(f"malformed JSON: {exc}") from exc
            choice = data["choices"][0]
            visible_content = (choice["message"].get("content") or "").strip()
            if not visible_content:
                finish_reason = str(choice.get("finish_reason") or "unknown")
                raise EmptyVisibleCompletionError(
                    "DeepSeek returned empty visible content "
                    f"(finish_reason={finish_reason})."
                )
            return visible_content, data

        def _retryable(exc: BaseException) -> bool:
            if isinstance(exc, EmptyVisibleCompletionError):
                return True
            return is_retryable_provider_exception(exc)

        before_events = len(context.trace_sink)
        release = None
        try:
            release = acquire_provider_slot(
                "llm",
                max_inflight=max(1, int(os.getenv("MAS_LLM_MAX_INFLIGHT_PER_PROCESS", "8"))),
                context=context,
                acquire_timeout_seconds=float(
                    os.getenv("MAS_PROVIDER_ACQUIRE_TIMEOUT_SECONDS", "5")
                ),
            )
            visible_content, data = call_with_policy(
                _once,
                policy=policy,
                context=context,
                is_retryable=_retryable,
            )
        except Exception:
            self.last_attempts = max(1, len(context.trace_sink) - before_events)
            self.last_trace = list(context.trace_sink[before_events:])
            raise
        finally:
            if release is not None:
                release()
        self.last_attempts = max(1, len(context.trace_sink) - before_events)
        self.last_trace = list(context.trace_sink[before_events:])
        usage = data.get("usage", {})
        self._add_usage(
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )
        return visible_content


def _is_company_extract_prompt(prompt_lower: str) -> bool:
    """True only for dedicated company-extraction prompts (not planner structure JSON)."""
    if "公司名称提取" in prompt_lower or "company name extractor" in prompt_lower:
        return True
    return "返回 json" in prompt_lower and '"companies"' in prompt_lower and "time_range" not in prompt_lower


class LocalFallbackLLMClient(BaseLLMClient):
    backend_name = "local-fallback"
    model_name = "local-fallback"

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        prompt = f"{system_prompt}\n{user_prompt}"
        prompt_lower = prompt.lower()
        companies = _extract_companies_from_text(prompt)

        if _is_company_extract_prompt(prompt_lower):
            content = json.dumps({"companies": companies}, ensure_ascii=False)
        elif "executive summary" in prompt_lower or "执行摘要" in prompt_lower:
            if len(companies) >= 2:
                content = (
                    f"本次尽调对比 {companies[0]} 与 {companies[1]}："
                    f"量化指标由 AST 引擎基于样本/文档证据计算，供应链与管理层语气已完成结构化采集，"
                    f"具体优劣需结合报告正文指标与风险评分综合判断。"
                )
            elif len(companies) == 1:
                company = companies[0]
                content = (
                    f"本次对 {company} 的尽调已完成多 Agent 流水线："
                    f"检索、量化、情绪与合规审计结果已汇入报告。"
                    f"建议结合毛利率、研发强度与供应链风险分项阅读下文。"
                )
            else:
                content = "本次分析已完成编排与合规检查；请补充明确公司与财年后获取完整量化结论。"
        elif "合规" in prompt_lower or "compliance" in prompt_lower:
            content = "报告包含数据来源与风险免责声明，当前未发现明显合规缺口。"
        elif "peer comparison" in prompt_lower or "定量分析师" in prompt_lower or "quantitative analyst" in prompt_lower:
            content = f"基于当前样本指标，{('、'.join(companies) if companies else '目标公司')} 的盈利能力与研发强度存在可比对差异。"
        elif "sentiment" in prompt_lower or "语气" in prompt_lower or "psychologist" in prompt_lower:
            content = "管理层整体语气偏积极，对需求与执行力表述较为自信，少量措辞提及供应链与监管不确定性。"
        elif "profile" in prompt_lower or "公司简介" in prompt_lower or "equity research" in prompt_lower:
            target = companies[0] if companies else "该公司"
            content = f"{target} 主营核心业务增长稳健，近期战略重点围绕产品组合、供应链韧性与资本回报展开。"
        else:
            target = companies[0] if companies else "目标公司"
            content = f"已完成 {target} 相关金融分析文本生成。"
        self._add_usage(max(len(prompt) // 4, 1), max(len(content) // 4, 1))
        return content


class ResilientLLMClient(BaseLLMClient):
    def __init__(
        self,
        primary: BaseLLMClient | None,
        fallback: BaseLLMClient | None = None,
        *,
        allow_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.primary = primary
        self.fallback = fallback or LocalFallbackLLMClient()
        self.allow_fallback = allow_fallback
        self.backend_name = primary.backend_name if primary else self.fallback.backend_name
        self.model_name = getattr(primary, "model_name", self.fallback.model_name)
        self.last_error: str | None = None
        self.used_fallback: bool = primary is None and allow_fallback
        self.degraded: bool = self.used_fallback
        self.primary_provider: str | None = primary.backend_name if primary else None
        self.primary_error_class: str | None = None
        self.primary_attempts: int = 0
        self.provider_trace: list[dict[str, Any]] = []

    def mark_usage_start(self) -> None:
        self._usage_mark = dict(self._usage_totals)
        active = self._active_client()
        active.mark_usage_start()
        self._usage_mark = {
            "prompt_tokens": self._usage_totals["prompt_tokens"],
            "completion_tokens": self._usage_totals["completion_tokens"],
        }

    def usage_since_mark(self) -> dict[str, int]:
        return {
            "prompt_tokens": self._usage_totals["prompt_tokens"] - self._usage_mark["prompt_tokens"],
            "completion_tokens": self._usage_totals["completion_tokens"] - self._usage_mark["completion_tokens"],
        }

    def _active_client(self) -> BaseLLMClient:
        if self.primary is not None:
            return self.primary
        if not self.allow_fallback:
            raise RuntimeError("No primary LLM configured and local fallback is disabled.")
        return self.fallback

    def bind_call_context(self, context: ProviderCallContext | None) -> None:
        bind = getattr(self.primary, "bind_call_context", None)
        if callable(bind):
            bind(context)

    def fork_usage(self) -> "ResilientLLMClient":
        forked = ResilientLLMClient(
            primary=fork_llm_client(self.primary) if self.primary is not None else None,
            fallback=fork_llm_client(self.fallback),
            allow_fallback=self.allow_fallback,
        )
        forked.last_error = None
        forked.used_fallback = self.primary is None and self.allow_fallback
        forked.degraded = forked.used_fallback
        forked.primary_error_class = None
        forked.primary_attempts = 0
        forked.provider_trace = []
        return forked

    def fallback_audit(self) -> dict[str, Any]:
        return {
            "provider": self.primary_provider or "none",
            "operation": "chat",
            "used_fallback": self.used_fallback,
            "degraded": self.degraded,
            "error_class": self.primary_error_class,
            "attempts": self.primary_attempts,
        }

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        if self.primary is None:
            if not self.allow_fallback:
                raise RuntimeError("No primary LLM configured and local fallback is disabled.")
            self.backend_name = self.fallback.backend_name
            self.model_name = self.fallback.model_name
            self.used_fallback = True
            self.degraded = True
            self.primary_error_class = "not_configured"
            self.primary_attempts = 0
            before = dict(self.fallback._usage_totals)
            content = self.fallback.chat(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
            self._sync_usage_from(self.fallback, before)
            return content
        try:
            self.backend_name = self.primary.backend_name
            self.model_name = self.primary.model_name
            before = dict(self.primary._usage_totals)
            content = self.primary.chat(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
            self._sync_usage_from(self.primary, before)
            self.used_fallback = False
            self.degraded = False
            self.last_error = None
            self.primary_error_class = None
            self.primary_attempts = int(getattr(self.primary, "last_attempts", 1) or 1)
            self.provider_trace = list(getattr(self.primary, "last_trace", []) or [])
            return content
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.primary_error_class = classify_provider_exception(exc)
            self.primary_attempts = int(getattr(self.primary, "last_attempts", 0) or 0) or int(
                getattr(getattr(self.primary, "settings", None), "max_retries", 1) or 1
            )
            self.provider_trace = list(getattr(self.primary, "last_trace", []) or [])
            if not self.allow_fallback:
                raise
            logger.warning(
                "Primary LLM (%s) failed (%s); falling back to %s",
                self.primary.backend_name,
                self.last_error,
                self.fallback.backend_name,
            )
            self.backend_name = self.fallback.backend_name
            self.model_name = self.fallback.model_name
            self.used_fallback = True
            self.degraded = True
            before = dict(self.fallback._usage_totals)
            content = self.fallback.chat(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
            self._sync_usage_from(self.fallback, before)
            if self.provider_trace is not None:
                self.provider_trace.append(
                    {
                        "provider": self.primary_provider or self.primary.backend_name,
                        "operation": "chat",
                        "status": "fallback",
                        "used_fallback": True,
                        "degraded": True,
                        "error_class": self.primary_error_class,
                        "attempts": self.primary_attempts,
                    }
                )
            return content

    def _sync_usage_from(self, client: BaseLLMClient, before: dict[str, int]) -> None:
        delta_prompt = client._usage_totals["prompt_tokens"] - before["prompt_tokens"]
        delta_completion = client._usage_totals["completion_tokens"] - before["completion_tokens"]
        self._add_usage(delta_prompt, delta_completion)


def build_llm_client(
    settings: LLMSettings | None = None,
    *,
    allow_local_fallback: bool = True,
) -> ResilientLLMClient:
    settings = settings or LLMSettings.from_env()
    primary = DeepSeekChatClient(settings) if settings.api_key else None
    return ResilientLLMClient(
        primary=primary,
        fallback=LocalFallbackLLMClient(),
        allow_fallback=allow_local_fallback,
    )


def fork_llm_client(client: BaseLLMClient | None) -> BaseLLMClient | None:
    """Create a run-local usage tracker while reusing provider configuration."""
    if client is None:
        return None
    fork = getattr(client, "fork_usage", None)
    if callable(fork):
        return fork()
    forked = copy(client)
    if hasattr(forked, "_usage_totals"):
        forked._usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
    if hasattr(forked, "_usage_mark"):
        forked._usage_mark = {"prompt_tokens": 0, "completion_tokens": 0}
    return forked


def shutdown_llm_http_clients() -> None:
    close_shared_http_clients()
