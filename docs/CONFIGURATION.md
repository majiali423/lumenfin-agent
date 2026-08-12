# Configuration

LumenFin reads environment variables through `AppConfig` and `LLMSettings`.
Copy `.env.example` to `.env` for local use. Never commit `.env`.

## Required by mode

| Variable | Offline/dev | Controlled live deployment | Default |
|----------|-------------|----------------------------|---------|
| `APP_ENV` | `dev` / `test` | non-dev value such as `production` | `dev` |
| `DATA_MODE` | `demo` | `live` | demo in dev/test; live otherwise |
| `DEEPSEEK_API_KEY` | Optional (local fallback) | Required | empty |
| `DEEPSEEK_MODEL` | Optional | Required model must exist | `deepseek-v4-flash` |
| `DASHSCOPE_API_KEY` | Not needed with deterministic embeddings | Required when embedding provider is DashScope | empty |
| `SEC_USER_AGENT` | Optional local fallback | **Required**, include operator contact | no production default |
| `MAS_API_KEY` | Optional in dev/test | Required API authentication | empty |

Missing production credentials do not enable sample fundamentals. LLM/provider
failures are surfaced as degraded/incomplete paths; live mode must not silently
fall back to demo data.

## LLM and provider settings

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible API root |
| `DEEPSEEK_TIMEOUT_SECONDS` | `45` | LLM request timeout |
| `DEEPSEEK_MAX_RETRIES` | `3` | Transient retries |
| `DEEPSEEK_RETRY_BACKOFF_SECONDS` | `0.5` | Exponential backoff base |
| `DASHSCOPE_EMBEDDING_MODEL` | `text-embedding-v4` | Embedding model |
| `DASHSCOPE_BASE_URL` | DashScope compatible endpoint | Embedding API root |
| `MAS_EMBEDDING_TIMEOUT_SECONDS` | `60` | Embedding timeout |
| `MAS_MARKET_DATA_PROVIDER` | `yahoo` in code; example recommends Alpha Vantage | Primary market snapshot |
| `MAS_MARKET_DATA_FALLBACK` | `yahoo` | Market fallback |
| `ALPHAVANTAGE_API_KEY` | empty | Optional Alpha Vantage key |

## Data, persistence and checkpoints

| Variable | Default | Notes |
|----------|---------|-------|
| `MAS_DATABASE_URL` | SQLite derived from `MAS_DB_PATH` only when allowed | Job / checkpoint / RAG metadata |
| `MAS_DB_PATH` | `data/lumenfin.db` | Local SQLite path when SQLite is allowed |
| `MAS_ALLOW_SQLITE_DEV` | `false` | Explicit opt-in for SQLite when `APP_ENV=dev` |
| `MAS_REDIS_URL` | empty | Optional reliable job/index queues |
| `MAS_REDIS_JOB_MAX_ATTEMPTS` | `3` | Max deliveries before dead-letter |
| `MAS_REDIS_RECLAIM_IDLE_SECONDS` | `10` | Stale processing reclaim threshold |
| `MAS_REDIS_RETRY_BACKOFF_SECONDS` | `1` | Delay before requeue poll continues |
| `MAS_OUTPUT_DIR` | `outputs` | Generated artifacts; ignored |
| `MAS_UPLOAD_DIR` | `uploads` | Local uploads; ignored |

### PostgreSQL-first database policy

| `APP_ENV` | Database rule |
|-----------|---------------|
| `production` | PostgreSQL required (`MAS_DATABASE_URL=postgresql+psycopg://...`) |
| `integration` | PostgreSQL required |
| `dev` | PostgreSQL recommended; SQLite only with `MAS_ALLOW_SQLITE_DEV=true` |
| `test` | SQLite allowed for fast unit tests (`sqlite:///:memory:` or temp files) |

Local recommended runtime: PostgreSQL via Docker Compose.
Unit-test backend: SQLite.
Production/integration: PostgreSQL required (fail-fast on SQLite).

Request execution state is request-scoped. SQLite is for unit tests / explicit
dev opt-in only; it is not a distributed strongly consistent checkpoint service.

## RAG / Milvus

| Variable | Default | Notes |
|----------|---------|-------|
| `MAS_RAG_ENABLED` | `true` | Enables document retrieval |
| `MAS_RAG_INDEX_MODE` | `sync_on_run` | `async_on_upload` for controlled service use |
| `MAS_MILVUS_URI` | `data/milvus_lite.db` | Lite file or Milvus server URI |
| `MAS_MILVUS_ISOLATE` | `true` | PID-isolate Lite files |
| `MAS_EMBEDDING_PROVIDER` | `deterministic` | Example live profile uses DashScope |
| `MAS_EMBEDDING_DIMENSION` | `384` | Must match provider/database |
| `MAS_RAG_REQUIRE_READY` | `false` | Fail when referenced documents are not ready |
| `MAS_RAG_INDEX_LEASE_SECONDS` | `300` | Lease before abandoned index work can be reclaimed |
| `MAS_RAG_RERANK_ENABLED` | `true` | Rerank retrieved candidates before final top-K |
| `MAS_RAG_RERANK_CANDIDATES` | `20` | Candidate count supplied to the reranker |
| `MAS_RAG_RERANK_PROVIDER` | `lexical` | Safe code/example default; the approved local production profile uses `qwen3` |
| `DASHSCOPE_RERANK_MODEL` | `qwen3-rerank` | Remote rerank model when provider is `qwen3` |
| `DASHSCOPE_RERANK_BASE_URL` | empty | Workspace compatible API base URL required by Qwen3 |
| `MAS_RAG_RERANK_TIMEOUT_SECONDS` | `12` | Qwen3 per-attempt timeout |
| `MAS_RAG_RERANK_MAX_ATTEMPTS` | `2` | Total Qwen3 attempts, including the first |
| `MAS_RAG_RERANK_MAX_INFLIGHT_PER_PROCESS` | `2` | Qwen3 concurrency bulkhead per process |

Milvus Lite is single-machine, single-writer infrastructure for local/dev runs.
Do not share one Lite file among independent API/CLI/worker processes. Validated
multi-process runs (Phase 3.2B / 3.3A) use **Milvus Server** over the Compose
network with `MAS_MILVUS_URI=http://milvus:19530`; tenant filtering is pushed
down to Milvus metadata (see
[MULTI_TENANCY_BOUNDARY.md](MULTI_TENANCY_BOUNDARY.md)).

The production Compose profile fixes the embedding provider to DashScope
`text-embedding-v4` with 1024-dimensional vectors. API, analysis worker and
index worker all connect to the same Milvus Server and use
`lumenfin_chunks_v4_bm25`, whose schema combines dense vectors with a native
Milvus BM25 sparse function. Changing the embedding model, dimension, analyzer,
or BM25 schema requires a versioned collection or full re-index; an existing
collection must not be reused with an incompatible schema.

The optional Qwen3 reranker sends the query and candidate document text to
DashScope. Its synthetic live preflight and local deployment cutover were
approved and passed on 2026-08-12; the code and example defaults remain
`lexical` so a fresh checkout does not send document text externally without
an operator decision. See [QWEN3_RERANK.md](QWEN3_RERANK.md).

## Cross-repository release gate

| Variable | Default | Purpose |
|----------|---------|---------|
| `FINAGENTBENCH_DIR` | sibling `finagentbench-demo` | Local repository discovery |
| `LUMENFIN_ROOT` | sibling `lumenfin-agent` | Benchmark-side discovery |
| `FINAGENTBENCH_REF` | `v0.1.0-rc.3` | Released benchmark tag/SHA |
| `FINAGENTBENCH_PROFILE` | `ci` | Deterministic benchmark profile |

The GitHub workflow permits manual `FINAGENTBENCH_REF` override and records the
resolved commit SHA. Normal push/PR validation uses the pinned release tag.

## Production Compose

`docker-compose.yml` forces `APP_ENV=production` and `DATA_MODE=live`. A
`migrator` service applies PostgreSQL migrations and must complete successfully
before the API, analysis worker or RAG index worker start. The stack includes
Milvus Server with internal etcd and MinIO services; uploaded documents placed
on the Redis index queue are consumed by `lumenfin-index-worker`.

Compose configuration fails before startup unless these operator-owned values
are set:

- `MAS_API_KEY`
- `DEEPSEEK_API_KEY`
- `DASHSCOPE_API_KEY`
- `SEC_USER_AGENT`
- `POSTGRES_PASSWORD`
- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`

Postgres, Redis, etcd, MinIO and Milvus are available only on the internal
Compose network by default; only the LumenFin API port is published. Redis uses
AOF with a named volume. Milvus metadata, object data and vector state use
separate named Docker volumes, so rebuilding an application container does not
discard queue or indexed state.

The API Compose healthcheck uses `/ready`, which requires PostgreSQL, Redis, and
the configured Milvus collection to be reachable. MinIO has its own live
healthcheck and Milvus waits for it before startup. See
[PRODUCTION_BACKUP_RESTORE.md](PRODUCTION_BACKUP_RESTORE.md) for the verified
backup and restore-rehearsal boundary.

## Offline versus live

**Offline:** deterministic embeddings, local fallback LLM/fake market providers,
fixture FinRuns, no API keys.

**Live:** real LLM, embeddings, SEC/Yahoo/market network access. Provider quota
or network failure must be classified separately from Agent-quality failure.
