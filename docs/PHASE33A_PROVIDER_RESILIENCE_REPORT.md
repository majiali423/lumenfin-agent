# Phase 3.3A Provider Resilience Report

Desensitized validation record for unified provider call policy, request
deadlines, Retry-After/jitter, fallback degradation, dual-API multi-process
fault injection, and embedding failure compensation. Not a production-ready
certification. Not exactly-once.

## Test metadata (closure patch)

| Field | Value |
|-------|-------|
| Test date (UTC) | 2026-08-04 |
| Closure base HEAD (pushed) | `2c90f85` |
| Closure patch HEAD | `b12e9ab` |
| Suite A–G deterministic | **pass** (`scripts/run_phase33a_provider_resilience.py`) |
| Dual-API Scenario G | **pass** (`scripts/run_phase33a_dual_api_scenario_g.py`, dual OS processes) |
| Offline unit tests | **453 passed, 1 skipped** (`scripts/run_tests.py`) |
| Phase 3.2B regression (current) | **pass** — run id `20260804T095357Z` |
| Live smoke | **skipped** (no live key exercise this session) |

Historical Phase 3.3A deterministic PASS (pre-closure) is retained under
`outputs/phase33a_provider_resilience/`; current evidence is the closure
re-run plus dual-API and Phase 3.2B regression above.

---

## 1. Four provider retry owners (single owner per logical call)

| Provider | Logical entry | Retry owner | Physical transport | Max attempts config |
|----------|---------------|-------------|--------------------|---------------------|
| DeepSeek | `DeepSeekChatClient.chat` | `call_with_policy` only | process-local shared `httpx.Client` (`deepseek-chat`) | `MAS_LLM_MAX_ATTEMPTS` / `DEEPSEEK_MAX_RETRIES` (total attempts) |
| DashScope | `ResilientEmbeddingProvider.embed` | `call_with_policy` only (inner DashScope does not retry) | shared/injected `httpx.Client` | `MAS_EMBEDDING_MAX_ATTEMPTS` / `MAS_EMBEDDING_MAX_RETRIES` |
| SEC | `_get_json_with_retries` / `fetch_sec_companyfacts_fundamentals` | `call_with_policy` only (custom loop removed) | shared `httpx.Client` (`sec-edgar`) + required User-Agent | `MAS_MARKET_DATA_MAX_ATTEMPTS` (default 3) |
| Yahoo | `fetch_yahoo_fundamentals` → `_load_yahoo_income` | `call_with_policy` only | yfinance loader (not raw HTTP) | `MAS_MARKET_DATA_MAX_ATTEMPTS` (default 3) |

Proof contract (unit tests): `max_attempts=3` + always-transient → physical
attempts **== 3** (not 6/9). Covered by SEC / Yahoo / DeepSeek / embedding tests.

---

## 2. SEC unified policy

- Internal retry loop deleted; SEC uses `ProviderCallPolicy` + `ProviderCallContext` + `call_with_policy`
- Process-local shared client; SEC User-Agent preserved (`SEC_USER_AGENT` / safe local fallback in dev/test/integration)
- Deadline, Retry-After, bounded jitter; `404` → `not_found` (1 attempt); `400/401/403` no retry; `429/5xx/timeout/connection` limited retry
- Trace records attempts; no response bodies / headers / env secrets in traces
- Market-data bulkhead on SEC path; `provider_busy` is non-retryable for HTTP

Tests: `tests/test_sec_provider_policy.py`, `tests/test_sec_retry_ownership.py`

---

## 3. Yahoo retry ownership

- Sole retry owner: `call_with_policy` around `_load_yahoo_income`
- Empty DataFrame / missing symbols → `truly_missing`, no transient retry
- Deadline stops further loader calls
- Sample fallback after live provider errors is marked `degraded` + `data_fallback=sample_db` (does not hide `provider_errors`)

Tests: `tests/test_yahoo_retry_ownership.py`

---

## 4. Dual-API Scenario G (multi-process)

Mode: **dual OS processes** (`api-a`, `api-b`) + local `provider-stub`.
Docker overlay available: `docker-compose.phase33a.yml` (compose with integration stack).
Note: **per-process bulkhead ≠ cross-process global rate limit**.

| Field | Value |
|-------|-------|
| API A | worker_id=`api-a`, PID=`32948`, container_id=`null` (OS process) |
| API B | worker_id=`api-b`, PID=`14956`, container_id=`null` (OS process) |
| request_count | 20 |
| api_a_count / api_b_count | 10 / 10 |
| logical_provider_calls | 20 |
| physical_provider_attempts | 25 |
| retry_amplification_ratio | 1.25 |
| success_count | 17 |
| degraded_count / fallback_count | 2 / 2 |
| expected_failure_count | 1 |
| unexpected_failure_count | 0 |
| deadline_exceeded_count | 0 |
| provider_busy_count | 0 |
| p50 / p95 / max latency (ms) | 481.21 / 929.3 / 940.05 |
| provider stub request count | **25** (matches physical attempts) |
| cross-process shared client ids | **none** |
| unique request / logical call ids | 20 / 20 |

Artifact: `outputs/phase33a_provider_resilience/dual_api_summary.json`

---

## 5. Bulkhead coverage

| Lane | Wired paths |
|------|-------------|
| LLM | `DeepSeekChatClient.chat` → `acquire_provider_slot("llm")` |
| embedding | `ResilientEmbeddingProvider.embed` → `acquire_provider_slot("embedding")` |
| market-data | SEC `_get_json_with_retries`, Yahoo `fetch_yahoo_fundamentals` → `acquire_provider_slot("market-data")` |

- Semaphores are **per-process** only
- Acquire respects request deadline; timeout → `provider_busy` (no HTTP retry)
- Release on exception paths; unit test `tests/test_provider_bulkhead.py`

---

## 6. Embedding failure compensation

| Case | Result |
|------|--------|
| embedding_count_mismatch / dimension_mismatch / malformed_json | `invalid_response`, provider attempts=1 (no transient retry) |
| timeout until deadline | provider retry stops before max_attempts |
| indexer path | document **not** `ready`; chunks cleared; partial vectors deleted; orphan chunks/vectors = 0 in compensation test |

**Layer separation**

- **Provider retry:** re-issues the external embed HTTP call inside one index job
- **Redis job retry:** redelivers the whole document index job after job-level failure

Tests: `tests/test_embedding_provider_compensation.py`

---

## 7. Phase 3.2B latest regression (current evidence)

| Field | Value |
|-------|-------|
| Run id | `20260804T095357Z` |
| Status | **pass** |
| Artifact | `outputs/phase32b_integration/20260804T095357Z/summary.json` |
| postgres_migrations | pass |
| checkpoint CAS / duplicate Redis / kill reclaim | covered (manual_redelivery=false) |
| stale fencing / tenant isolation / DLQ / Redis restart / limited load | covered |
| orphan_chunk_count / orphan_vector_count | **0 / 0** |
| unexpected_error_count | 0 |

Prior Phase 3.2B PASS remains historical only; this run is the closure basis.

---

## 8. Deterministic Phase 3.3A suite (re-run)

| Scenario | Result |
|----------|--------|
| A 503 then success | pass |
| B 429 + Retry-After | pass |
| C permanent 400 | pass |
| D deadline | pass |
| E fallback | pass |
| F embedding invalid | pass |
| G concurrency | pass — logical=10, physical=14, ratio=1.4 |

---

## 9. Offline / live smoke

- Offline: **453 passed, 1 skipped**
- `scripts/validate_concurrency.py`: pass (this session)
- Live smoke: **skipped**

---

## 10. Remaining risks (not verified / out of scope)

- Dual-API evidence here used **OS processes**, not Docker containers (compose overlay exists but was not the measured run)
- Intra-process `get_shared_http_client` has no lock; rare races under extreme concurrency possible
- No cross-process Redis distributed semaphore / shared circuit breaker (intentionally out of scope)
- No large-scale real provider soak; live DeepSeek/DashScope smoke not run
- Not exactly-once; not production-ready
