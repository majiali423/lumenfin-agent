# Phase 3.3A Provider Resilience Report

Desensitized validation record for unified provider call policy, request
deadlines, Retry-After/jitter, fallback degradation, and deterministic fault
stub evidence. Not a production-ready certification.

## Test metadata

| Field | Value |
|-------|-------|
| Test date (UTC) | 2026-08-04 |
| Suite | Phase 3.3A deterministic (`scripts/run_phase33a_provider_resilience.py`) |
| Suite status | **pass** |
| Live smoke | **skipped** (`MAS_PHASE33A_LIVE_SMOKE` unset) |
| Offline unit tests | **437 passed, 1 skipped** (`scripts/run_tests.py`) |
| Phase 3.2B regression | **not re-run in this session** (prior PASS at Phase 3.2B closure; deferred) |
| Base commit before work | `a2a1493e1a89550bc6c9c1556c590ca82024cc77` |

## 1. Files added / modified

**Added:** `src/lumenfin/provider_resilience.py`, `scripts/provider_stub/`, `scripts/run_phase33a_provider_resilience.py`, `tests/test_provider_resilience.py`, `docs/PHASE33A_PROVIDER_RESILIENCE_REPORT.md`

**Modified:** `llm.py`, `provider_retry.py`, `rag/embeddings.py`, `config.py`, `service.py`, `api/app.py`, `api/schemas.py`, related tests

## 2–3. Provider call list and retry owners

| Provider | Operation | Retry owner |
|----------|-----------|-------------|
| DeepSeek | chat | `DeepSeekChatClient` → `call_with_policy` (single layer) |
| DashScope | embed | `ResilientEmbeddingProvider` → `call_with_policy`; DashScope itself does not retry |
| SEC | HTTP JSON | still custom loop (compat); classification via `classify_exception` |
| Yahoo | fundamentals | `call_with_transient_retry` → delegates to `call_with_policy` |

## 4–7. Semantics / deadlines / Retry-After / jitter

- `max_attempts` / historical `*_MAX_RETRIES` = **total physical attempts** (e.g. 3 → ≤3 calls, ≤2 sleeps)
- New optional aliases: `MAS_LLM_MAX_ATTEMPTS`, `MAS_EMBEDDING_MAX_ATTEMPTS`
- Request deadlines: `MAS_ANALYSIS_DEADLINE_SECONDS` (default 120), `MAS_INDEX_JOB_DEADLINE_SECONDS` (default 180)
- Distinguishes `timeout` vs `deadline_exceeded`
- `Retry-After` integer seconds honored as `max(backoff, retry_after)` capped by `max_backoff` and remaining deadline
- Bounded multiplicative jitter (`jitter_ratio`, default 0.2); tests inject `jitter_ratio=0` / fixed RNG

## 8–9. HTTP client reuse / shutdown

- Process-local shared `httpx.Client` via `get_shared_http_client` (`trust_env=False` to avoid proxy hijack of localhost)
- `fork_usage()` reuses transport; does not clone connection pools unsafely
- FastAPI lifespan closes shared clients; `shutdown_llm_http_clients()` / `close_shared_http_clients()`

## 10–11. Fallback / trace

- Fallback sets `used_fallback`, `degraded`, `primary_error_class`, `primary_attempts`
- API `AnalyzeResponse` exposes `degraded`, `provider_degraded`, `provider_call_summary`
- Trace events redact secrets; no prompts/PDF bodies

Example (desensitized):

```json
{
  "provider": "deepseek",
  "operation": "chat",
  "attempt": 2,
  "status": "retry",
  "status_code": 429,
  "error_class": "rate_limited",
  "retry_after_ms": 1000,
  "used_fallback": false
}
```

## 12–17. Scenario results (deterministic stub)

| Scenario | Result |
|----------|--------|
| A 503 then success | pass — attempts=3, fallback=false |
| B 429 + Retry-After | pass — Retry-After recognized, attempts=2 |
| C permanent 400 | pass — attempts=1, `client_error` |
| D deadline | pass — `deadline_exceeded` |
| E fallback | pass — degraded=true, primary error retained |
| F embedding invalid | pass — `invalid_response`, attempts=1 |
| G concurrency (10) | pass |

## 18–24. Concurrency aggregates (scenario G)

| Metric | Value |
|--------|-------|
| logical_provider_calls | 10 |
| physical_provider_attempts | 14 |
| retry_amplification_ratio | 1.4 |
| p50 / p95 / max latency (ms) | 4.06 / 5.61 / 5.62 |
| fallback_count | 1 |
| deadline_exceeded_count | 0 |
| unexpected_error_count | 0 |

## 25–28. Suites

- Offline: 437 passed, 1 skipped
- Phase 3.2B: not re-executed this session
- Phase 3.3A summary: `outputs/phase33a_provider_resilience/summary.json` (gitignored under `outputs/`)
- Live smoke: skipped

## 29–31. Git / risks

- **HEAD:** `f97119aa25fcbbee65d22e451175b16d5c344acd`
- Worktree clean after commit
- Remaining risks: SEC still has a parallel custom retry loop; dual-API Docker compose stub profile not required for this deterministic PASS; per-process bulkheads are implemented as helpers but not yet wired into every market-data path; Phase 3.2B docker regression deferred; no live DeepSeek/DashScope smoke; not exactly-once; not production-ready
