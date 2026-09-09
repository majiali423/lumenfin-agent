# LEDGER public_holdout rc5 index

Document-only index for the one-shot `public_holdout` E2E. This is a
resource authorization before holdout consumption, not retrieval tuning.

## Isolation

Index construction reads only company/document identity and filing page
text through Arrow column projection. It does not read `query_text`,
`value`/`gold`, or `qrels`. Document text is corpus, not holdout
consumption.

## Identity

- Product: `v0.1.0-rc.5` / `31e8680aa89636f1fd897d7aa5ed7ca86317bd73`
- Chunker: rc5 `chunk_document` (`max_chunk_chars=900`, `overlap_chars=120`)
- Embedding: DashScope `text-embedding-v4`, dimension 1024
- Retrieval schema: dense + BM25 (`dense_bm25_v1`)
- Collection: `lumenfin_ledger_public_holdout_rc5_v1`
- Session/tenant: `ledger-public-holdout-rc5-index-v1`

## Commands

```text
python scripts/run_ledger_public_holdout_index.py --dry-run
python scripts/run_ledger_public_holdout_index.py --build --allow-remote
python scripts/run_ledger_public_holdout_index.py --build --allow-remote --resume
```

The large Milvus Lite files stay gitignored under
`outputs/ledger_public_holdout_rc5_index_v1/`.

Status: **SEALED**. Corpus: 26 companies, 10,895 pages, 81,376 chunks.
Synthetic canary (no holdout queries) returned dense/BM25 hits with
`row_count=81376`. Holdout questions, gold values, and qrels were not
opened during indexing.

Tracked files:

- [`../data/eval_rag/ledger_public_holdout_index_v1_dryrun.json`](../data/eval_rag/ledger_public_holdout_index_v1_dryrun.json)
- [`../data/eval_rag/ledger_public_holdout_index_v1.json`](../data/eval_rag/ledger_public_holdout_index_v1.json)

## v1 E2E remains historical

The original zero-embedding blocked contract is unchanged:

- `data/eval_rag/ledger_public_holdout_e2e_contract.json`
- `data/eval_rag/ledger_public_holdout_e2e_authorization.json`
- `data/eval_rag/ledger_public_holdout_e2e_result.json`

Unlocked E2E continues as v2. See
[`LEDGER_PUBLIC_HOLDOUT_E2E.md`](LEDGER_PUBLIC_HOLDOUT_E2E.md).
