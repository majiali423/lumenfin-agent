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
| `MAS_DATABASE_URL` | SQLite derived from `MAS_DB_PATH` | Job and workflow checkpoint repository |
| `MAS_DB_PATH` | `data/lumenfin.db` | Local SQLite only |
| `MAS_REDIS_URL` | empty | Optional RQ job/index queues |
| `MAS_NEO4J_URI` | empty | Optional knowledge store |
| `MAS_OUTPUT_DIR` | `outputs` | Generated artifacts; ignored |
| `MAS_UPLOAD_DIR` | `uploads` | Local uploads; ignored |

Request execution state is request-scoped. SQLite persists pause/resume
snapshots but is not a distributed strongly consistent checkpoint service.

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

Milvus Lite is single-machine infrastructure. Do not share one Lite file among
independent API/CLI/worker processes.

## Cross-repository release gate

| Variable | Default | Purpose |
|----------|---------|---------|
| `FINAGENTBENCH_DIR` | sibling `finagentbench-demo` | Local repository discovery |
| `LUMENFIN_ROOT` | sibling `lumenfin-agent` | Benchmark-side discovery |
| `FINAGENTBENCH_REF` | `v0.1.0-rc.1` | Released benchmark tag/SHA |
| `FINAGENTBENCH_PROFILE` | `ci` | Deterministic benchmark profile |

The GitHub workflow permits manual `FINAGENTBENCH_REF` override and records the
resolved commit SHA. Normal push/PR validation uses the pinned release tag.

## Production Compose

`docker-compose.yml` forces `APP_ENV=production` and `DATA_MODE=live`. Compose
configuration fails before startup unless these operator-owned values are set:

- `MAS_API_KEY`
- `DEEPSEEK_API_KEY`
- `SEC_USER_AGENT`
- `POSTGRES_PASSWORD`
- `NEO4J_PASSWORD`

Postgres, Redis and Neo4j are available only on the internal Compose network by
default; only the LumenFin API port is published.

## Offline versus live

**Offline:** deterministic embeddings, local fallback LLM/fake market providers,
fixture FinRuns, no API keys.

**Live:** real LLM, embeddings, SEC/Yahoo/market network access. Provider quota
or network failure must be classified separately from Agent-quality failure.
