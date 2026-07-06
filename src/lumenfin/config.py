from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .llm import LLMSettings

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    output_dir: Path
    upload_dir: Path
    db_path: Path
    database_url: str
    redis_url: str | None
    redis_queue_name: str
    neo4j_uri: str | None
    neo4j_username: str | None
    neo4j_password: str | None
    market_data_provider: str
    alphavantage_api_key: str | None
    host: str
    port: int
    api_key: str | None
    llm: LLMSettings
    rag_enabled: bool
    milvus_uri: str
    milvus_collection: str
    embedding_provider: str
    embedding_dimension: int
    rag_top_k: int
    critic_max_iterations: int
    company_parallelism: int
    input_guardrail_enabled: bool
    input_guardrail_mode: str
    tool_backend: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        raw_output_dir = os.getenv("MAS_OUTPUT_DIR", "outputs")
        raw_db_path = os.getenv("MAS_DB_PATH", "data/lumenfin.db")
        return cls(
            output_dir=Path(raw_output_dir),
            upload_dir=Path(os.getenv("MAS_UPLOAD_DIR", "uploads")),
            db_path=Path(raw_db_path),
            database_url=os.getenv("MAS_DATABASE_URL", f"sqlite:///{raw_db_path.replace(os.sep, '/')}"),
            redis_url=os.getenv("MAS_REDIS_URL"),
            redis_queue_name=os.getenv("MAS_REDIS_QUEUE_NAME", "finance-analysis"),
            neo4j_uri=os.getenv("MAS_NEO4J_URI"),
            neo4j_username=os.getenv("MAS_NEO4J_USERNAME"),
            neo4j_password=os.getenv("MAS_NEO4J_PASSWORD"),
            market_data_provider=os.getenv("MAS_MARKET_DATA_PROVIDER", "yahoo"),
            alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY"),
            host=os.getenv("MAS_HOST", "127.0.0.1"),
            port=int(os.getenv("MAS_PORT", "8000")),
            api_key=os.getenv("MAS_API_KEY"),
            llm=LLMSettings.from_env(),
            rag_enabled=os.getenv("MAS_RAG_ENABLED", "true").lower() in {"1", "true", "yes"},
            milvus_uri=os.getenv("MAS_MILVUS_URI", "data/milvus_lite.db"),
            milvus_collection=os.getenv("MAS_MILVUS_COLLECTION", "lumenfin_chunks"),
            embedding_provider=os.getenv("MAS_EMBEDDING_PROVIDER", "deterministic"),
            embedding_dimension=int(os.getenv("MAS_EMBEDDING_DIMENSION", "384")),
            rag_top_k=int(os.getenv("MAS_RAG_TOP_K", "5")),
            critic_max_iterations=int(os.getenv("MAS_CRITIC_MAX_ITERATIONS", "2")),
            company_parallelism=int(os.getenv("MAS_COMPANY_PARALLELISM", "4")),
            input_guardrail_enabled=os.getenv("MAS_INPUT_GUARDRAIL_ENABLED", "true").lower() in {"1", "true", "yes"},
            input_guardrail_mode=os.getenv("MAS_INPUT_GUARDRAIL_MODE", "sanitize").lower(),
            tool_backend=os.getenv("MAS_TOOL_BACKEND", "local").lower(),
        )
