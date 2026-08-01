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


@dataclass(frozen=True)
class AppConfig:
    output_dir: Path
    upload_dir: Path
    db_path: Path
    database_url: str
    redis_url: str | None
    redis_queue_name: str
    redis_index_queue_name: str
    neo4j_uri: str | None
    neo4j_username: str | None
    neo4j_password: str | None
    market_data_provider: str
    market_data_fallback: str
    market_cache_ttl_seconds: int
    alphavantage_api_key: str | None
    host: str
    port: int
    api_key: str | None
    app_env: str
    data_mode: str
    allow_local_fallback: bool | None
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
    rag_sanitize_hits: bool
    rag_rerank_enabled: bool
    rag_rerank_candidates: int
    critic_max_iterations: int
    company_parallelism: int
    profile_llm_max_attempts: int
    input_guardrail_enabled: bool
    input_guardrail_mode: str
    tool_backend: str
    fetch_live_fundamentals: bool
    fetch_sec_fundamentals: bool

    def allows_sample_data(self) -> bool:
        return self.data_mode == "demo"

    def allows_local_fallback(self) -> bool:
        if self.allow_local_fallback is not None:
            return self.allow_local_fallback
        if self.requires_api_key():
            return False
        return self.data_mode == "demo" or self.app_env in {"dev", "test"}

    def requires_api_key(self) -> bool:
        return self.app_env not in {"dev", "test"}

    @classmethod
    def from_env(cls) -> "AppConfig":
        # Re-check on every load so preflight and runtime share the same path.
        assert_no_env_conflicts(root=_PROJECT_ROOT)
        raw_output_dir = os.getenv("MAS_OUTPUT_DIR", "outputs")
        raw_db_path = os.getenv("MAS_DB_PATH", "data/lumenfin.db")
        app_env = os.getenv("APP_ENV", "dev").strip().lower() or "dev"
        default_data_mode = "demo" if app_env in {"dev", "test"} else "live"
        data_mode = os.getenv("DATA_MODE", default_data_mode).strip().lower()
        if data_mode not in {"demo", "live"}:
            data_mode = default_data_mode
        allow_raw = os.getenv("ALLOW_LOCAL_FALLBACK")
        allow_local_fallback = None
        if allow_raw is not None and allow_raw.strip() != "":
            allow_local_fallback = allow_raw.strip().lower() in {"1", "true", "yes"}
        return cls(
            output_dir=Path(raw_output_dir),
            upload_dir=Path(os.getenv("MAS_UPLOAD_DIR", "uploads")),
            db_path=Path(raw_db_path),
            database_url=os.getenv("MAS_DATABASE_URL", f"sqlite:///{raw_db_path.replace(os.sep, '/')}"),
            redis_url=os.getenv("MAS_REDIS_URL"),
            redis_queue_name=os.getenv("MAS_REDIS_QUEUE_NAME", "finance-analysis"),
            redis_index_queue_name=os.getenv("MAS_REDIS_INDEX_QUEUE_NAME", "rag-document-index"),
            neo4j_uri=os.getenv("MAS_NEO4J_URI"),
            neo4j_username=os.getenv("MAS_NEO4J_USERNAME"),
            neo4j_password=os.getenv("MAS_NEO4J_PASSWORD"),
            market_data_provider=os.getenv("MAS_MARKET_DATA_PROVIDER", "yahoo"),
            market_data_fallback=os.getenv("MAS_MARKET_DATA_FALLBACK", "yahoo"),
            market_cache_ttl_seconds=int(os.getenv("MAS_MARKET_CACHE_TTL_SECONDS", "60")),
            alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY"),
            host=os.getenv("MAS_HOST", "127.0.0.1"),
            port=int(os.getenv("MAS_PORT", "8000")),
            api_key=os.getenv("MAS_API_KEY") or None,
            app_env=app_env,
            data_mode=data_mode,
            allow_local_fallback=allow_local_fallback,
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
            rag_sanitize_hits=os.getenv("MAS_RAG_SANITIZE_HITS", "true").lower() in {"1", "true", "yes"},
            rag_rerank_enabled=os.getenv("MAS_RAG_RERANK_ENABLED", "true").lower() in {"1", "true", "yes"},
            rag_rerank_candidates=max(1, int(os.getenv("MAS_RAG_RERANK_CANDIDATES", "20"))),
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
        )
