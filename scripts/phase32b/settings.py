from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "docker-compose.integration.yml"
ENV_FILE = ROOT / ".env.integration.example"
OUTPUT_DIR = ROOT / "outputs" / "phase32b_integration"


@dataclass(frozen=True)
class IntegrationSettings:
    project: str = "lumenfin-it"
    postgres_user: str = "lumenfin_it"
    postgres_password: str = "lumenfin_it_pass"
    postgres_db: str = "lumenfin_integration"
    postgres_host_port: int = 5433
    redis_host_port: int = 6380
    milvus_host_port: int = 19531
    api_a_port: int = 18080
    api_b_port: int = 18081
    redis_index_queue: str = "rag-document-index-it"
    milvus_collection: str = "lumenfin_chunks_it"
    lease_seconds: int = 5

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@127.0.0.1:{self.postgres_host_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://127.0.0.1:{self.redis_host_port}/0"

    @property
    def milvus_uri(self) -> str:
        return f"http://127.0.0.1:{self.milvus_host_port}"

    @property
    def api_a_url(self) -> str:
        return f"http://127.0.0.1:{self.api_a_port}"

    @property
    def api_b_url(self) -> str:
        return f"http://127.0.0.1:{self.api_b_port}"

    @classmethod
    def from_env(cls) -> "IntegrationSettings":
        return cls(
            project=os.getenv("COMPOSE_PROJECT_NAME", "lumenfin-it"),
            postgres_user=os.getenv("POSTGRES_USER", "lumenfin_it"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", "lumenfin_it_pass"),
            postgres_db=os.getenv("POSTGRES_DB", "lumenfin_integration"),
            postgres_host_port=int(os.getenv("POSTGRES_PORT", "5433")),
            redis_host_port=int(os.getenv("REDIS_PORT", "6380")),
            milvus_host_port=int(os.getenv("MILVUS_PORT", "19531")),
            api_a_port=int(os.getenv("API_A_PORT", "18080")),
            api_b_port=int(os.getenv("API_B_PORT", "18081")),
            redis_index_queue=os.getenv("MAS_REDIS_INDEX_QUEUE_NAME", "rag-document-index-it"),
            milvus_collection=os.getenv("MAS_MILVUS_COLLECTION", "lumenfin_chunks_it"),
            lease_seconds=int(os.getenv("MAS_RAG_INDEX_LEASE_SECONDS", "5")),
        )
