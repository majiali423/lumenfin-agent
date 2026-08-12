# Qwen3 rerank controlled rollout

## Current status

The approved local deployment configuration was switched to Qwen3 on
2026-08-12 after the live hard-negative gate passed:

```text
MAS_RAG_RERANK_ENABLED=true
MAS_RAG_RERANK_PROVIDER=qwen3
```

The code and example-configuration default remains `lexical`, and lexical also
remains the automatic runtime fallback. The production Compose stack was later
validated with synthetic data through native BM25 + dense RRF + Qwen3 with zero
fallback/degradation. No model is downloaded: the selected provider calls
DashScope `qwen3-rerank` over HTTPS.

## Retrieval path

The dense and native-BM25 branches first produce up to
`MAS_RAG_RERANK_CANDIDATES` candidates. The selected reranker orders those
candidates and returns `MAS_RAG_TOP_K` evidence items.

The Qwen3 response is accepted only when it contains the expected number of
unique, in-range candidate indexes, numeric scores in `[0, 1]`, and descending
score order. Returned document text is never trusted; indexes are mapped back
to the original local evidence objects.

On missing configuration, timeout, rate limit, network error, server error, or
invalid response, the request falls back to the deterministic lexical
reranker. Telemetry records the requested provider, actual provider, model,
latency, attempts, token usage, fallback status, and error class. A fallback
also marks retrieval degraded.

## Configuration

| Variable | Code default | Purpose |
|----------|--------------|---------|
| `MAS_RAG_RERANK_PROVIDER` | `lexical` | Approved local deployment uses `qwen3`; fresh deployments require their own egress decision |
| `DASHSCOPE_RERANK_MODEL` | `qwen3-rerank` | DashScope rerank model |
| `DASHSCOPE_RERANK_BASE_URL` | empty | Workspace compatible API base URL ending in `/compatible-api/v1` |
| `DASHSCOPE_API_KEY` | empty | Shared DashScope credential; required for Qwen3 |
| `MAS_RAG_RERANK_TIMEOUT_SECONDS` | `12` | Per-attempt timeout |
| `MAS_RAG_RERANK_MAX_ATTEMPTS` | `2` | Total attempts, including the first |
| `MAS_RAG_RERANK_BACKOFF_SECONDS` | `0.25` | Retry backoff base |
| `MAS_RAG_RERANK_MAX_INFLIGHT_PER_PROCESS` | `2` | Per-process concurrency bulkhead |
| `MAS_RAG_RERANK_MAX_DOCUMENT_CHARS` | `4000` | Conservative local proxy for the per-item token limit |
| `MAS_RAG_RERANK_INSTRUCT` | financial diligence instruction | Ranking instruction |

## Data-egress boundary

Qwen3 is a remote reranker. Enabling it sends the query and candidate document
text to DashScope. The live preflight therefore requires explicit approval for
document-data egress and must use an operator-owned workspace, endpoint, and
API key. Secrets must stay in `.env` or the deployment secret store and must
not be committed.

## Offline evaluation

The hard-negative corpus deliberately includes wrong company, reporting
period, metric scope, filing section, and negation distractors:

```powershell
python scripts/run_rerank_eval.py --provider lexical --gate
```

The Qwen3 comparison is intentionally blocked unless remote egress is made
explicit:

```powershell
python scripts/run_rerank_eval.py --provider qwen3 --allow-remote --gate
```

The formal live gate requires zero fallbacks and Qwen3 metrics no worse than
the lexical baseline for top-1 accuracy, MRR, and nDCG@K. API contract,
timeout, 429 retry, malformed response, missing-key fallback, concurrency, and
telemetry behavior are covered by offline tests.

## Live gate evidence

The approved 2026-08-12 preflight used only the 10 synthetic cases in
`data/eval_rag/rerank_cases.json`. No uploaded or recruitment document was
sent. The final result was:

| Ranker | Top-1 | MRR | nDCG@5 |
|--------|------:|----:|-------:|
| Candidate order | 0.0000 | 0.4167 | 0.6118 |
| Lexical | 0.5000 | 0.6617 | 0.7460 |
| Qwen3 | 1.0000 | 1.0000 | 0.9711 |

All 10 Qwen3 calls used one attempt, returned provider request IDs and token
usage, and completed without fallback or provider error. Observed latency was
188.33–897.17 ms and total token usage was 3,873. The temporary JSON artifact
records only non-secret telemetry and does not persist full request IDs,
endpoint details, credentials, queries, or candidate text.

## Rollback

Rollback is configuration-only:

```text
MAS_RAG_RERANK_PROVIDER=lexical
```

No Milvus collection rebuild is needed because reranking occurs after dense +
BM25 retrieval and does not change stored vectors or sparse fields.
