"""Rerank providers for hybrid RAG.

The lexical provider remains deterministic and offline-safe.  The optional
DashScope provider calls ``qwen3-rerank`` through its OpenAI-compatible HTTP
endpoint and is always wrapped with lexical fallback by ``build_reranker``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Protocol

import httpx

from ..provider_resilience import (
    InvalidProviderResponseError,
    ProviderCallContext,
    ProviderCallPolicy,
    acquire_provider_slot,
    call_with_policy,
    classify_provider_exception,
    get_shared_http_client,
    redact_provider_message,
    summarize_provider_trace,
)
from .lexical import lexical_overlap, query_has_any, tokenize_text

logger = logging.getLogger(__name__)

DEFAULT_RERANK_INSTRUCT = (
    "Given a financial due diligence query, retrieve passages that directly "
    "answer it. Prefer the correct company, reporting period, metric, scope, "
    "and filing context over merely topical passages."
)
QWEN3_MAX_DOCUMENTS = 500
QWEN3_MAX_QUERY_CHARS = 4000
# A conservative character proxy for the official 120k-token request limit.
# The application's default 20 candidates x 4k chars remains below this cap.
QWEN3_MAX_REQUEST_CHARS = 100_000


class Reranker(Protocol):
    """Provider contract consumed by ``HybridEvidenceRetriever``."""

    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


def lexical_rerank_score(query: str, hit: dict[str, Any]) -> float:
    text = str(hit.get("text") or "")
    overlap = lexical_overlap(query, text)
    if overlap <= 0.0 and not tokenize_text(query):
        return 0.0
    query_tokens = tokenize_text(query)
    chunk_type = str(hit.get("chunk_type") or "narrative")
    if chunk_type == "financial_metric" and query_has_any(
        query_tokens,
        "revenue",
        "ebitda",
        "margin",
        "operating_margin",
        "r_and_d",
        "研发",
        "收入",
        "gpu",
        "利润率",
    ):
        overlap += 0.12
    if chunk_type == "risk_signal" and query_has_any(
        query_tokens,
        "risk",
        "supply",
        "supply_chain",
        "风险",
        "供应链",
        "capex",
    ):
        overlap += 0.12
    if "rrf" in str(hit.get("retrieval_method") or ""):
        overlap += 0.03
    prior = float(hit.get("fusion_score") or hit.get("score") or 0.0)
    prior_norm = min(1.0, prior if prior <= 1.0 else prior / 10.0)
    return min(1.0, overlap * 0.85 + prior_norm * 0.15)


def _enrich_hit(
    hit: dict[str, Any],
    *,
    score: float,
    provider: str,
    model: str,
    method_suffix: str,
    fallback: bool = False,
) -> dict[str, Any]:
    enriched = dict(hit)
    enriched["rerank_score"] = round(float(score), 8)
    enriched["rerank_provider"] = provider
    enriched["rerank_model"] = model
    enriched["rerank_fallback"] = bool(fallback)
    base_method = str(hit.get("retrieval_method") or "retrieve")
    enriched["retrieval_method"] = f"{base_method}+{method_suffix}"
    return enriched


class LexicalReranker:
    """Deterministic CJK/English lexical reranker and provider fallback."""

    provider_name = "lexical"
    model_name = "lexical-financial-v1"

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = time.perf_counter()
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for original_rank, hit in enumerate(hits):
            score = lexical_rerank_score(query, hit)
            enriched = _enrich_hit(
                hit,
                score=score,
                provider=self.provider_name,
                model=self.model_name,
                method_suffix="lexical_rerank",
            )
            scored.append((score, original_rank, enriched))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [item[2] for item in scored[: max(1, int(top_k))]]
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return selected, {
            "rerank_provider": self.provider_name,
            "rerank_model": self.model_name,
            "rerank_latency_ms": latency_ms,
            "rerank_attempts": 1 if hits else 0,
            "rerank_tokens": 0,
            "rerank_fallback": False,
            "rerank_error_type": "",
            "rerank_mode_suffix": "lexical_rerank",
        }


class DashScopeQwen3Reranker:
    """DashScope ``qwen3-rerank`` provider with bounded retry and bulkhead."""

    provider_name = "dashscope"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen3-rerank",
        instruct: str = DEFAULT_RERANK_INSTRUCT,
        timeout_seconds: float = 12.0,
        max_attempts: int = 2,
        backoff_seconds: float = 0.25,
        max_inflight: int = 2,
        acquire_timeout_seconds: float = 5.0,
        max_document_chars: int = 4000,
        client: httpx.Client | None = None,
        sleep: Any | None = None,
        jitter_ratio: float = 0.2,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY") or ""
        ).strip()
        self._base_url = (
            base_url
            if base_url is not None
            else os.getenv("DASHSCOPE_RERANK_BASE_URL") or ""
        ).strip().rstrip("/")
        self._model = (model or "qwen3-rerank").strip()
        self._instruct = (instruct or DEFAULT_RERANK_INSTRUCT).strip()
        self._timeout_seconds = max(0.1, float(timeout_seconds))
        self._max_attempts = max(1, int(max_attempts))
        self._backoff_seconds = max(0.0, float(backoff_seconds))
        self._max_inflight = max(1, int(max_inflight))
        self._acquire_timeout_seconds = max(0.05, float(acquire_timeout_seconds))
        self._max_document_chars = max(1, int(max_document_chars))
        self._client = client
        self._sleep = sleep
        self._jitter_ratio = max(0.0, float(jitter_ratio))

    @property
    def model_name(self) -> str:
        return self._model

    def _validate_configuration(self) -> None:
        if not self._api_key:
            raise ValueError("DASHSCOPE_API_KEY is required for qwen3 rerank")
        if not self._base_url:
            raise ValueError(
                "DASHSCOPE_RERANK_BASE_URL is required for qwen3 rerank; "
                "use the workspace-compatible API base URL"
            )
        if not self._base_url.startswith("https://"):
            raise ValueError("DASHSCOPE_RERANK_BASE_URL must use https://")

    def _parse_response(
        self,
        response: httpx.Response,
        *,
        hits: list[dict[str, Any]],
        expected_count: int,
    ) -> tuple[list[dict[str, Any]], int, str]:
        response.raise_for_status()
        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise InvalidProviderResponseError(f"malformed rerank JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise InvalidProviderResponseError("rerank response must be an object")
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise InvalidProviderResponseError("rerank response is missing results")
        if len(raw_results) != expected_count:
            raise InvalidProviderResponseError(
                f"rerank result count mismatch: expected {expected_count}, got {len(raw_results)}"
            )

        seen: set[int] = set()
        ranked: list[dict[str, Any]] = []
        previous_score: float | None = None
        for item in raw_results:
            if not isinstance(item, dict):
                raise InvalidProviderResponseError("rerank result item must be an object")
            try:
                index = int(item["index"])
                score = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidProviderResponseError(
                    "rerank result requires numeric index and relevance_score"
                ) from exc
            if index < 0 or index >= len(hits) or index in seen:
                raise InvalidProviderResponseError(f"invalid or duplicate rerank index: {index}")
            if not 0.0 <= score <= 1.0:
                raise InvalidProviderResponseError(
                    f"rerank relevance_score must be within [0, 1], got {score}"
                )
            if previous_score is not None and score > previous_score + 1e-12:
                raise InvalidProviderResponseError(
                    "rerank results must be sorted by relevance_score descending"
                )
            seen.add(index)
            previous_score = score
            ranked.append(
                _enrich_hit(
                    hits[index],
                    score=score,
                    provider=self.provider_name,
                    model=self.model_name,
                    method_suffix="qwen3_rerank",
                )
            )
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        try:
            total_tokens = max(0, int(usage.get("total_tokens") or 0))
        except (TypeError, ValueError) as exc:
            raise InvalidProviderResponseError("rerank usage.total_tokens must be numeric") from exc
        request_id = str(payload.get("id") or "")
        return ranked, total_tokens, request_id

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not hits:
            return [], {
                "rerank_provider": self.provider_name,
                "rerank_model": self.model_name,
                "rerank_latency_ms": 0.0,
                "rerank_attempts": 0,
                "rerank_tokens": 0,
                "rerank_fallback": False,
                "rerank_error_type": "",
                "rerank_mode_suffix": "qwen3_rerank",
            }
        self._validate_configuration()
        if len(hits) > QWEN3_MAX_DOCUMENTS:
            raise ValueError(
                f"qwen3 rerank accepts at most {QWEN3_MAX_DOCUMENTS} documents"
            )
        query_text = str(query or "")
        if not query_text.strip():
            raise ValueError("qwen3 rerank query must not be empty")
        if len(query_text) > QWEN3_MAX_QUERY_CHARS:
            raise ValueError(
                f"qwen3 rerank query exceeds {QWEN3_MAX_QUERY_CHARS} characters"
            )
        expected_count = min(max(1, int(top_k)), len(hits))
        documents = [str(hit.get("text") or "")[: self._max_document_chars] for hit in hits]
        if any(not document.strip() for document in documents):
            raise InvalidProviderResponseError("rerank candidates must contain non-empty text")
        request_chars = len(query_text) * len(documents) + sum(map(len, documents))
        if request_chars > QWEN3_MAX_REQUEST_CHARS:
            raise ValueError(
                "qwen3 rerank request exceeds the conservative local size guard"
            )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "query": query_text,
            "documents": documents,
            "top_n": expected_count,
            "instruct": self._instruct,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/reranks"
        context = ProviderCallContext.create()
        if self._sleep is not None:
            context.sleep = self._sleep
        policy = ProviderCallPolicy(
            provider="rerank",
            operation="qwen3_rerank",
            max_attempts=self._max_attempts,
            read_timeout_seconds=self._timeout_seconds,
            write_timeout_seconds=self._timeout_seconds,
            base_backoff_seconds=self._backoff_seconds,
            max_backoff_seconds=max(self._backoff_seconds * 4.0, 1.0),
            jitter_ratio=self._jitter_ratio,
        )
        started = time.perf_counter()
        release = acquire_provider_slot(
            "rerank",
            max_inflight=self._max_inflight,
            context=context,
            acquire_timeout_seconds=self._acquire_timeout_seconds,
        )
        try:
            def _call() -> tuple[list[dict[str, Any]], int, str]:
                client = self._client or get_shared_http_client(
                    "dashscope-rerank", timeout=self._timeout_seconds
                )
                response = client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=policy.httpx_timeout(
                        remaining_seconds=context.remaining_seconds()
                    ),
                )
                return self._parse_response(
                    response,
                    hits=hits,
                    expected_count=expected_count,
                )

            try:
                ranked, total_tokens, request_id = call_with_policy(
                    _call,
                    policy=policy,
                    context=context,
                )
            except Exception as exc:
                trace = summarize_provider_trace(list(context.trace_sink or []))
                setattr(
                    exc,
                    "rerank_attempts",
                    int(trace.get("physical_provider_attempts") or 0),
                )
                setattr(
                    exc,
                    "rerank_latency_ms",
                    round((time.perf_counter() - started) * 1000.0, 2),
                )
                raise
        finally:
            release()
        trace = summarize_provider_trace(list(context.trace_sink or []))
        return ranked, {
            "rerank_provider": self.provider_name,
            "rerank_model": self.model_name,
            "rerank_latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "rerank_attempts": int(trace.get("physical_provider_attempts") or 0),
            "rerank_tokens": total_tokens,
            "rerank_fallback": False,
            "rerank_error_type": "",
            "rerank_request_id": request_id,
            "rerank_mode_suffix": "qwen3_rerank",
        }


class FallbackReranker:
    """Run a primary reranker and fall back to lexical ranking on any failure."""

    def __init__(self, primary: Reranker, fallback: Reranker | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or LexicalReranker()

    @property
    def provider_name(self) -> str:
        return self.primary.provider_name

    @property
    def model_name(self) -> str:
        return self.primary.model_name

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            return self.primary.rerank(query, hits, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Rerank provider %s failed; using lexical fallback: %s",
                self.primary.provider_name,
                redact_provider_message(str(exc)),
            )
            ranked, meta = self.fallback.rerank(query, hits, top_k=top_k)
            for hit in ranked:
                method = str(hit.get("retrieval_method") or "")
                if method.endswith("+lexical_rerank"):
                    hit["retrieval_method"] = f"{method[:-len('+lexical_rerank')]}+lexical_rerank_fallback"
                hit["rerank_fallback"] = True
                hit["rerank_requested_provider"] = self.primary.provider_name
                hit["rerank_requested_model"] = self.primary.model_name
            meta.update(
                {
                    "rerank_requested_provider": self.primary.provider_name,
                    "rerank_requested_model": self.primary.model_name,
                    "rerank_fallback": True,
                    "rerank_latency_ms": round(
                        float(getattr(exc, "rerank_latency_ms", 0.0))
                        + float(meta.get("rerank_latency_ms") or 0.0),
                        2,
                    ),
                    "rerank_attempts": int(getattr(exc, "rerank_attempts", 0)),
                    "rerank_error_type": classify_provider_exception(exc),
                    "rerank_error": redact_provider_message(str(exc)),
                    "rerank_mode_suffix": "lexical_rerank_fallback",
                }
            )
            return ranked, meta


def build_reranker(
    provider_name: str,
    *,
    model: str = "qwen3-rerank",
    base_url: str = "",
    instruct: str = DEFAULT_RERANK_INSTRUCT,
    timeout_seconds: float = 12.0,
    max_attempts: int = 2,
    backoff_seconds: float = 0.25,
    max_inflight: int = 2,
    acquire_timeout_seconds: float = 5.0,
    max_document_chars: int = 4000,
) -> Reranker:
    normalized = (provider_name or "lexical").strip().lower()
    if normalized in {"lexical", "local", "offline"}:
        return LexicalReranker()
    if normalized in {"qwen3", "dashscope", "dashscope-qwen3"}:
        primary = DashScopeQwen3Reranker(
            base_url=base_url,
            model=model,
            instruct=instruct,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            max_inflight=max_inflight,
            acquire_timeout_seconds=acquire_timeout_seconds,
            max_document_chars=max_document_chars,
        )
        return FallbackReranker(primary, LexicalReranker())
    raise ValueError(
        f"Unsupported MAS_RAG_RERANK_PROVIDER={provider_name!r}; choose lexical or qwen3"
    )


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Backward-compatible lexical rerank helper."""
    ranked, _meta = LexicalReranker().rerank(query, hits, top_k=top_k)
    return ranked


__all__ = [
    "DEFAULT_RERANK_INSTRUCT",
    "DashScopeQwen3Reranker",
    "FallbackReranker",
    "LexicalReranker",
    "Reranker",
    "build_reranker",
    "lexical_rerank_score",
    "rerank_hits",
]
