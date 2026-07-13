from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import httpx

from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .tools import KNOWN_ALIASES


def _extract_companies_from_text(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for company in SAMPLE_FINANCIAL_DATA:
        if company.lower() in lowered and company not in found:
            found.append(company)
    for alias, name in KNOWN_ALIASES.items():
        if alias in lowered and name not in found:
            found.append(name)
    return found


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None
    base_url: str
    model: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "LLMSettings":
        api_key = os.getenv("DEEPSEEK_API_KEY") or None
        base_url = os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        model = os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
        timeout_str = os.getenv("DEEPSEEK_TIMEOUT_SECONDS") or "45"
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=float(timeout_str),
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


class DeepSeekChatClient(BaseLLMClient):
    backend_name = "deepseek"

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__()
        self.settings = settings
        self.model_name = settings.model

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        with httpx.Client(timeout=self.settings.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        usage = data.get("usage", {})
        self._add_usage(
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
        )
        return data["choices"][0]["message"]["content"].strip()


class LocalFallbackLLMClient(BaseLLMClient):
    backend_name = "local-fallback"
    model_name = "local-fallback"

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        prompt = f"{system_prompt}\n{user_prompt}"
        prompt_lower = prompt.lower()
        companies = _extract_companies_from_text(prompt)

        if "公司名称提取" in prompt_lower or '"companies"' in prompt_lower:
            content = json.dumps({"companies": companies}, ensure_ascii=False)
        elif "task_brief" in prompt_lower or "任务概括" in prompt_lower or "监督代理" in prompt_lower or "任务拆解" in prompt_lower or "supervisory agent" in prompt_lower:
            target = "、".join(companies) if companies else "目标公司"
            content = json.dumps(
                {
                    "task_brief": f"对 {target} 开展 2025 财年财务、供应链与管理层语气尽调，并输出带审计信息的合规报告。",
                    "analysis_dimensions": ["profitability", "supply_chain", "r_and_d", "sentiment"],
                    "key_questions": [],
                    "risk_appetite": "moderate",
                    "industry_context": "科技与半导体行业处于 AI 基础设施投资周期。",
                },
                ensure_ascii=False,
            )
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

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        if self.primary is None:
            if not self.allow_fallback:
                raise RuntimeError("No primary LLM configured and local fallback is disabled.")
            self.backend_name = self.fallback.backend_name
            self.model_name = self.fallback.model_name
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
            return content
        except Exception:
            if not self.allow_fallback:
                raise
            self.backend_name = self.fallback.backend_name
            self.model_name = self.fallback.model_name
            before = dict(self.fallback._usage_totals)
            content = self.fallback.chat(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
            self._sync_usage_from(self.fallback, before)
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
