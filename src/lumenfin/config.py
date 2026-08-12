from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .env_bootstrap import assert_no_env_conflicts, bootstrap_dotenv
from .llm import LLMSettings

# Prefer project-root .env (stable regardless of cwd), then cwd as secondary.
# Process env wins; conflicting non-empty process vs .env values fail fast.
_PROJECT_ROOT = bootstrap_dotenv(strict_conflicts=True)

_RAG_INDEX_MODES = frozenset({"sync_on_run", "async_on_upload"})
_RAG_RERANK_PROVIDERS = frozenset({"lexical", "qwen3"})


def _normalize_rag_index_mode(raw: str | None) -> str:
    mode = (raw or "sync_on_run").strip().lower() or "sync_on_run"
    if mode not in _RAG_INDEX_MODES:
        return "sync_on_run"
    return mode


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalize_rag_rerank_provider(raw: str | None) -> str:
    provider = (raw or "lexical").strip().lower() or "lexical"
    aliases = {
        "local": "lexical",
        "offline": "lexical",
        "dashscope": "qwen3",
        "dashscope-qwen3": "qwen3",
    }
    provider = aliases.get(provider, provider)
    if provider not in _RAG_RERANK_PROVIDERS:
        raise ValueError(
            "MAS_RAG_RERANK_PROVIDER must be one of: lexical, qwen3"
        )
    return provider


def _is_sqlite_url(database_url: str) -> bool:
    return database_url.strip().lower().startswith("sqlite:")


def resolve_database_url(*, app_env: str, raw_db_path: str) -> str:
    """PostgreSQL-first resolution with explicit SQLite opt-in outside tests."""
    configured = os.getenv("MAS_DATABASE_URL")
    if configured and configured.strip():
        database_url = configured.strip()
    else:
        database_url = f"sqlite:///{raw_db_path.replace(os.sep, '/')}"

    allow_sqlite_dev = os.getenv("MAS_ALLOW_SQLITE_DEV", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    using_sqlite = _is_sqlite_url(database_url)

    if app_env in {"production", "integration"} and using_sqlite:
        raise RuntimeError(
            "PostgreSQL is required for production/integration. "
            "Set MAS_DATABASE_URL=postgresql+psycopg://..."
        )
    if app_env == "dev" and using_sqlite and not allow_sqlite_dev:
        raise RuntimeError(
            "SQLite is disabled by default for APP_ENV=dev. "
            "Set MAS_DATABASE_URL=postgresql+psycopg://... "
            "or explicitly opt in with MAS_ALLOW_SQLITE_DEV=true."
        )
    return database_url


@dataclass(frozen=True)
class AppConfig:
    output_dir: Path
    upload_dir: Path
    db_path: Path
    database_url: str
    redis_url: str | None
    redis_queue_name: str
    redis_index_queue_name: str
    redis_job_max_attempts: int
    redis_reclaim_idle_seconds: int
    redis_retry_backoff_seconds: float
    market_data_provider: str
    market_data_fallback: str
    market_cache_ttl_seconds: int
    alphavantage_api_key: str | None
    host: str
    port: int
    api_key: str | None
    api_key_client_id: str
    api_key_tenant_id: str
    api_key_principals_json: str | None
    app_env: str
    data_mode: str
    allow_local_fallback: bool | None
    allow_sqlite_dev: bool
    max_upload_bytes: int
    max_upload_files: int
    llm: LLMSettings
    rag_enabled: bool
    milvus_uri: str
    milvus_collection: str
    milvus_isolate: bool
    embedding_provider: str
    embedding_dimension: int
    rag_top_k: int
    rag_index_mode: str
    rag_tenant_id: str
    rag_require_ready: bool
    rag_index_lease_seconds: int
    embedding_max_retries: int
    embedding_backoff_seconds: float
    embedding_timeout_seconds: float
    rag_min_score: float
    rag_degrade_on_vector_error: bool
    rag_bm25_enabled: bool
    rag_bm25_rrf_weight: float
    rag_sanitize_hits: bool
    rag_rerank_enabled: bool
    rag_rerank_candidates: int
    rag_rerank_provider: str
    rag_rerank_model: str
    rag_rerank_base_url: str
    rag_rerank_instruct: str
    rag_rerank_timeout_seconds: float
    rag_rerank_max_attempts: int
    rag_rerank_backoff_seconds: float
    rag_rerank_max_inflight_per_process: int
    rag_rerank_max_document_chars: int
    critic_max_iterations: int
    company_parallelism: int
    profile_llm_max_attempts: int
    input_guardrail_enabled: bool
    input_guardrail_mode: str
    tool_backend: str
    fetch_live_fundamentals: bool
    fetch_sec_fundamentals: bool
    analysis_deadline_seconds: float
    index_job_deadline_seconds: float
    llm_max_inflight_per_process: int
    embedding_max_inflight_per_process: int
    market_data_max_inflight_per_process: int
    provider_acquire_timeout_seconds: float

    def allows_sample_data(self) -> bool:
        return self.data_mode == "demo"

    def allows_local_fallback(self) -> bool:
        if self.allow_local_fallback is not None:
            return self.allow_local_fallback
        if self.requires_api_key():
            return False
        return self.data_mode == "demo" or self.app_env in {"dev", "test", "integration"}

    def requires_api_key(self) -> bool:
        return self.app_env not in {"dev", "test", "integration"}

    def uses_sqlite(self) -> bool:
        return _is_sqlite_url(self.database_url)

    def principal_directory(self):
        from .api.auth import build_principal_directory

        return build_principal_directory(
            legacy_api_key=self.api_key,
            legacy_client_id=self.api_key_client_id,
            legacy_tenant_id=self.api_key_tenant_id or self.rag_tenant_id,
            principals_json=self.api_key_principals_json,
        )

    def anonymous_principal(self):
        from .api.auth import AuthenticatedPrincipal

        return AuthenticatedPrincipal(
            client_id="anonymous",
            tenant_id=(self.rag_tenant_id or "default").strip() or "default",
        )

    @classmethod
    def from_env(cls) -> "AppConfig":
        # Re-check on every load so preflight and runtime share the same path.
        assert_no_env_conflicts(root=_PROJECT_ROOT)
        raw_output_dir = os.getenv("MAS_OUTPUT_DIR", "outputs")
        raw_db_path = os.getenv("MAS_DB_PATH", "data/lumenfin.db")
        app_env = os.getenv("APP_ENV", "dev").strip().lower() or "dev"
        default_data_mode = "demo" if app_env in {"dev", "test", "integration"} else "live"
        data_mode = os.getenv("DATA_MODE", default_data_mode).strip().lower()
        if data_mode not in {"demo", "live"}:
            data_mode = default_data_mode
        allow_raw = os.getenv("ALLOW_LOCAL_FALLBACK")
        allow_local_fallback = None
        if allow_raw is not None and allow_raw.strip() != "":
            allow_local_fallback = allow_raw.strip().lower() in {"1", "true", "yes"}
        allow_sqlite_dev = os.getenv("MAS_ALLOW_SQLITE_DEV", "false").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        database_url = resolve_database_url(app_env=app_env, raw_db_path=raw_db_path)
        return cls(
            output_dir=Path(raw_output_dir),
            upload_dir=Path(os.getenv("MAS_UPLOAD_DIR", "uploads")),
            db_path=Path(raw_db_path),
            database_url=database_url,
            redis_url=os.getenv("MAS_REDIS_URL"),
            redis_queue_name=os.getenv("MAS_REDIS_QUEUE_NAME", "finance-analysis"),
            redis_index_queue_name=os.getenv("MAS_REDIS_INDEX_QUEUE_NAME", "rag-document-index"),
            redis_job_max_attempts=_positive_int_env("MAS_REDIS_JOB_MAX_ATTEMPTS", 3),
            redis_reclaim_idle_seconds=_positive_int_env("MAS_REDIS_RECLAIM_IDLE_SECONDS", 10),
            redis_retry_backoff_seconds=float(os.getenv("MAS_REDIS_RETRY_BACKOFF_SECONDS", "1")),
            market_data_provider=os.getenv("MAS_MARKET_DATA_PROVIDER", "yahoo"),
            market_data_fallback=os.getenv("MAS_MARKET_DATA_FALLBACK", "yahoo"),
            market_cache_ttl_seconds=int(os.getenv("MAS_MARKET_CACHE_TTL_SECONDS", "60")),
            alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY"),
            host=os.getenv("MAS_HOST", "127.0.0.1"),
            port=int(os.getenv("MAS_PORT", "8000")),
            api_key=os.getenv("MAS_API_KEY") or None,
            api_key_client_id=(os.getenv("MAS_API_KEY_CLIENT_ID", "default-client").strip() or "default-client"),
            api_key_tenant_id=(
                os.getenv("MAS_API_KEY_TENANT_ID") or os.getenv("MAS_RAG_TENANT_ID", "default")
            ).strip()
            or "default",
            api_key_principals_json=os.getenv("MAS_API_KEY_PRINCIPALS") or None,
            app_env=app_env,
            data_mode=data_mode,
            allow_local_fallback=allow_local_fallback,
            allow_sqlite_dev=allow_sqlite_dev,
            max_upload_bytes=int(os.getenv("MAS_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
            max_upload_files=int(os.getenv("MAS_MAX_UPLOAD_FILES", "5")),
            llm=LLMSettings.from_env(),
            rag_enabled=os.getenv("MAS_RAG_ENABLED", "true").lower() in {"1", "true", "yes"},
            milvus_uri=os.getenv("MAS_MILVUS_URI", "data/milvus_lite.db"),
            milvus_collection=os.getenv("MAS_MILVUS_COLLECTION", "lumenfin_chunks"),
            milvus_isolate=os.getenv("MAS_MILVUS_ISOLATE", "true").lower() in {"1", "true", "yes"},
            embedding_provider=os.getenv("MAS_EMBEDDING_PROVIDER", "deterministic"),
            embedding_dimension=int(os.getenv("MAS_EMBEDDING_DIMENSION", "384")),
            rag_top_k=int(os.getenv("MAS_RAG_TOP_K", "5")),
            rag_index_mode=_normalize_rag_index_mode(os.getenv("MAS_RAG_INDEX_MODE", "sync_on_run")),
            rag_tenant_id=(os.getenv("MAS_RAG_TENANT_ID", "default").strip() or "default"),
            rag_require_ready=os.getenv("MAS_RAG_REQUIRE_READY", "false").lower() in {"1", "true", "yes"},
            rag_index_lease_seconds=_positive_int_env("MAS_RAG_INDEX_LEASE_SECONDS", 300),
            embedding_max_retries=max(1, int(os.getenv("MAS_EMBEDDING_MAX_RETRIES", "3"))),
            embedding_backoff_seconds=float(os.getenv("MAS_EMBEDDING_BACKOFF_SECONDS", "0.5")),
            embedding_timeout_seconds=float(
                os.getenv("DASHSCOPE_EMBEDDING_TIMEOUT")
                or os.getenv("MAS_EMBEDDING_TIMEOUT_SECONDS", "60")
            ),
            rag_min_score=float(os.getenv("MAS_RAG_MIN_SCORE", "0")),
            rag_degrade_on_vector_error=os.getenv("MAS_RAG_DEGRADE_ON_VECTOR_ERROR", "true").lower()
            in {"1", "true", "yes"},
            # Staged rollout: keep dense-only collections usable until the operator
            # explicitly switches to a BM25-capable versioned collection.
            rag_bm25_enabled=os.getenv("MAS_RAG_BM25_ENABLED", "false").lower()
            in {"1", "true", "yes"},
            rag_bm25_rrf_weight=max(
                0.0,
                float(os.getenv("MAS_RAG_BM25_RRF_WEIGHT", "1.1")),
            ),
            rag_sanitize_hits=os.getenv("MAS_RAG_SANITIZE_HITS", "true").lower() in {"1", "true", "yes"},
            rag_rerank_enabled=os.getenv("MAS_RAG_RERANK_ENABLED", "true").lower() in {"1", "true", "yes"},
            rag_rerank_candidates=max(1, int(os.getenv("MAS_RAG_RERANK_CANDIDATES", "20"))),
            rag_rerank_provider=_normalize_rag_rerank_provider(
                os.getenv("MAS_RAG_RERANK_PROVIDER", "lexical")
            ),
            rag_rerank_model=(
                os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank").strip()
                or "qwen3-rerank"
            ),
            rag_rerank_base_url=os.getenv("DASHSCOPE_RERANK_BASE_URL", "").strip(),
            rag_rerank_instruct=(
                os.getenv(
                    "MAS_RAG_RERANK_INSTRUCT",
                    "Given a financial due diligence query, retrieve passages that "
                    "directly answer it. Prefer the correct company, reporting period, "
                    "metric, scope, and filing context over merely topical passages.",
                ).strip()
            ),
            rag_rerank_timeout_seconds=max(
                0.1, float(os.getenv("MAS_RAG_RERANK_TIMEOUT_SECONDS", "12"))
            ),
            rag_rerank_max_attempts=_positive_int_env(
                "MAS_RAG_RERANK_MAX_ATTEMPTS", 2
            ),
            rag_rerank_backoff_seconds=max(
                0.0, float(os.getenv("MAS_RAG_RERANK_BACKOFF_SECONDS", "0.25"))
            ),
            rag_rerank_max_inflight_per_process=_positive_int_env(
                "MAS_RAG_RERANK_MAX_INFLIGHT_PER_PROCESS", 2
            ),
            rag_rerank_max_document_chars=_positive_int_env(
                "MAS_RAG_RERANK_MAX_DOCUMENT_CHARS", 4000
            ),
            critic_max_iterations=int(os.getenv("MAS_CRITIC_MAX_ITERATIONS", "2")),
            company_parallelism=int(os.getenv("MAS_COMPANY_PARALLELISM", "4")),
            profile_llm_max_attempts=max(
                0,
                min(3, int(os.getenv("MAS_PROFILE_LLM_MAX_ATTEMPTS", "1"))),
            ),
            input_guardrail_enabled=os.getenv("MAS_INPUT_GUARDRAIL_ENABLED", "true").lower() in {"1", "true", "yes"},
            input_guardrail_mode=os.getenv("MAS_INPUT_GUARDRAIL_MODE", "sanitize").lower(),
            tool_backend=os.getenv("MAS_TOOL_BACKEND", "local").lower(),
            fetch_live_fundamentals=(
                os.getenv(
                    "MAS_FETCH_LIVE_FUNDAMENTALS",
                    "true" if data_mode == "live" else "false",
                )
                .strip()
                .lower()
                in {"1", "true", "yes"}
            ),
            fetch_sec_fundamentals=(
                os.getenv(
                    "MAS_FETCH_SEC_FUNDAMENTALS",
                    "true" if data_mode == "live" else "false",
                )
                .strip()
                .lower()
                in {"1", "true", "yes"}
            ),
            analysis_deadline_seconds=float(os.getenv("MAS_ANALYSIS_DEADLINE_SECONDS", "120")),
            index_job_deadline_seconds=float(os.getenv("MAS_INDEX_JOB_DEADLINE_SECONDS", "180")),
            llm_max_inflight_per_process=_positive_int_env("MAS_LLM_MAX_INFLIGHT_PER_PROCESS", 8),
            embedding_max_inflight_per_process=_positive_int_env(
                "MAS_EMBEDDING_MAX_INFLIGHT_PER_PROCESS", 4
            ),
            market_data_max_inflight_per_process=_positive_int_env(
                "MAS_MARKET_DATA_MAX_INFLIGHT_PER_PROCESS", 8
            ),
            provider_acquire_timeout_seconds=float(
                os.getenv("MAS_PROVIDER_ACQUIRE_TIMEOUT_SECONDS", "5")
            ),
        )
