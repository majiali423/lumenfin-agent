# Native BM25 rollout

LumenFin uses Milvus-native BM25 as the sparse retrieval branch. The production
cutover uses a versioned v4 collection because the retained
`lumenfin_chunks_v3` collection is dense-only and cannot be altered in place to
add a BM25 function.

## Retrieval design

Each row in the BM25-capable collection contains:

- a DashScope dense vector in `vector`;
- raw chunk text in the analyzer-enabled `text` field;
- a Milvus-generated sparse vector in `sparse`;
- explicit tenant, source document, company, page, chunk type, and financial
  fact fields required by both Milvus Server and Milvus Lite.

The primary path is dense search plus native BM25, fused with weighted
reciprocal rank fusion. Dense weight is `1.0`; BM25 defaults to `1.1` to break
cross-rank ties in favor of exact financial terms. The approved local
deployment then applies Qwen3 model rerank; lexical remains the safe code
default and automatic fallback. See `docs/QWEN3_RERANK.md`.

The analyzer is explicitly configured as `jieba` plus `cnalphanumonly` instead
of the `chinese` shortcut. That form works in both Milvus Server 3.0 and Milvus
Lite 3.1. Lite additionally requires the locked `jieba==0.42.1` dependency.

## Production feature flag

After the approved production cutover, production Compose uses:

```text
MAS_RAG_BM25_ENABLED=true
MAS_RAG_BM25_RRF_WEIGHT=1.1
MAS_MILVUS_COLLECTION=lumenfin_chunks_v4_bm25
```

The preserved rollback configuration is:

```text
MAS_RAG_BM25_ENABLED=false
MAS_MILVUS_COLLECTION=lumenfin_chunks_v3
```

Enabling BM25 against `lumenfin_chunks_v3` fails fast with a schema error. This
is deliberate: never reuse or reset the dense-only rollback collection.

## Preflight and rebuild

Run preflight with the target environment before any reset:

```powershell
python scripts/rebuild_rag_vector_index.py
```

The output must report the expected durable document and chunk counts and
`bm25_enabled=true`. The execution form requires an exact collection-name
confirmation:

```powershell
python scripts/rebuild_rag_vector_index.py --execute --confirm-reset lumenfin_chunks_v4_bm25
```

Every rebuilt document must pass both its first dense search and its first BM25
search. After the application starts, run:

```powershell
python scripts/verify_rag_first_search.py --expect-collection lumenfin_chunks_v4_bm25
```

## Degradation modes

- Dense and BM25 succeed: `hybrid_dense_bm25_rrf`.
- Embedding/dense search fails: `bm25_only_degraded`.
- BM25 fails but dense and durable local chunks remain:
  `hybrid_dense_lexical_fallback_rrf_degraded`.
- Milvus retrieval paths return no usable result: local lexical fallback.

Telemetry exposes `vector_hits`, `bm25_hits`, `lexical_fallback_hits`, the
backward-compatible `keyword_hits`, and the effective retrieval mode.

## Verification

The BM25 release gate compares dense, BM25, and hybrid retrieval on Chinese,
exact filing identifiers, rare accelerator names, and financial metrics:

```powershell
python scripts/run_bm25_eval.py --gate
```

Production cutover requires this gate, the isolated Milvus Server first-read
test, the full test suite, a PostgreSQL backup, and preservation of the v3
collection and volumes for rollback.
