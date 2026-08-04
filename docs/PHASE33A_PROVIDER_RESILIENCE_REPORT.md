# Phase 3.3A Provider Resilience Report

**Phase 3.3A: PASS**

Desensitized validation record for unified provider call policy, request
deadlines, Retry-After/jitter, fallback degradation, dual-API multi-process
fault injection (OS process + **Docker containers**), and embedding failure
compensation. Not a production-ready certification. Not exactly-once.

## Test metadata (final closure)

| Field | Value |
|-------|-------|
| Test date (UTC) | 2026-08-04 |
| Pushed baseline before Docker evidence | `193c097` |
| Suite A–G deterministic | **pass** |
| Dual-API Scenario G (OS process) | **pass** (earlier closure) |
| Dual-API Scenario G (**Docker**) | **pass** — run id `docker_20260804T100817Z` |
| Offline unit tests | **453 passed, 1 skipped** |
| Phase 3.2B latest regression | **pass** — `20260804T095357Z` (same code baseline as `193c097`; not re-run this Docker-only round) |
| Live smoke | **skipped** |

---

## Docker Dual-API Validation

| Field | Value |
|-------|-------|
| Docker run id | `docker_20260804T100817Z` |
| Test commit (image/build baseline at run start) | `193c097` (+ local harness fixes validated in-run) |
| Compose files | `docker-compose.integration.yml` + `docker-compose.phase33a.yml` |
| Runner | `scripts/run_phase33a_dual_api_docker.py` (`--mode docker` only; no OS-process fallback) |
| Mode | `docker` |
| API A short container ID | `5f8518e619cd` |
| API B short container ID | `cadce5e69547` |
| API A hostname / worker_id / PID | `api-a` / `api-a` / `1` |
| API B hostname / worker_id / PID | `api-b` / `api-b` / `1` |
| API A / B request count | 10 / 10 |
| logical_provider_calls | 20 |
| physical_provider_attempts | 25 |
| provider stub request count | **25** (exact match) |
| retry_amplification_ratio | 1.25 |
| p50 / p95 / max latency (ms) | 617.88 / 755.39 / 756.83 |
| success / degraded / expected failure / unexpected | 17 / 2 / 1 / **0** |
| fallback_count | 2 |
| deadline_exceeded_count | 0 |
| provider_context_leakage_count | 0 |
| duplicate_logical_call_id_count | 0 |
| configured per-process LLM inflight limit | 4 |
| API A / B max observed inflight | 4 / 3 |
| HTTP client reuse | each API: **1** stable `http_client_instance_id`; cross-container IDs **disjoint** |
| Lifespan shutdown | both APIs logged cleanup; `unclosed_client_warnings=0`; `shutdown_tracebacks=0` |
| Unexpected container restarts | 0 / 0 |
| postgres / redis / milvus error signals in API logs | 0 / 0 / 0 |
| Artifact dir | `outputs/phase33a_provider_resilience/docker_20260804T100817Z/docker/` |

**Explicit semantics:** per-process bulkhead and process-local shared HTTP client
**≠** cross-process global rate limiting. Combined observed inflight across
containers may reach the sum of per-process limits.

---

## 1. Four provider retry owners (single owner per logical call)

| Provider | Logical entry | Retry owner | Physical transport | Max attempts config |
|----------|---------------|-------------|--------------------|---------------------|
| DeepSeek | `DeepSeekChatClient.chat` | `call_with_policy` only | process-local shared `httpx.Client` (`deepseek-chat`) | `MAS_LLM_MAX_ATTEMPTS` / `DEEPSEEK_MAX_RETRIES` |
| DashScope | `ResilientEmbeddingProvider.embed` | `call_with_policy` only | shared/injected `httpx.Client` | `MAS_EMBEDDING_MAX_*` |
| SEC | `_get_json_with_retries` | `call_with_policy` only | shared `httpx.Client` (`sec-edgar`) | `MAS_MARKET_DATA_MAX_ATTEMPTS` |
| Yahoo | `fetch_yahoo_fundamentals` | `call_with_policy` only | yfinance loader | `MAS_MARKET_DATA_MAX_ATTEMPTS` |

---

## 2–3. SEC / Yahoo ownership

Unchanged from prior closure: SEC and Yahoo use a single `call_with_policy`
owner; sample fallback is marked degraded; physical-attempt unit tests pass.

---

## 4. Bulkhead coverage

LLM / embedding / market-data (SEC+Yahoo) use per-process semaphores.
Acquire respects request deadline; `provider_busy` is non-retryable for HTTP.

---

## 5. Embedding failure compensation

`invalid_response` → attempts=1; timeout bounded by deadline; failed index jobs
leave orphan chunks/vectors = 0. Provider retry ≠ Redis job retry.

---

## 6. Phase 3.2B latest regression (referenced)

| Field | Value |
|-------|-------|
| Run id | `20260804T095357Z` |
| Status | **pass** |
| Note | Validated on HEAD `193c097` before this Docker evidence round. Docker harness did not change queueing/checkpoint/Milvus cleanup shared paths beyond compose overlay env for provider stub; Phase 3.2B was **not** re-executed this round. |

---

## 7. Deterministic Phase 3.3A suite

Scenarios A–G: **pass** (prior closure re-run retained).

---

## 8. Offline / live smoke

- Offline: **453 passed, 1 skipped**
- Live smoke: **skipped**

---

## 9. Known limitations / remaining risks

- Live DeepSeek/DashScope smoke not run
- No cross-process Redis semaphore / shared circuit breaker (out of scope)
- No large-scale real provider soak
- Docker Scenario G uses deterministic stub + local fallback; not production traffic
- Not exactly-once; not production-ready
