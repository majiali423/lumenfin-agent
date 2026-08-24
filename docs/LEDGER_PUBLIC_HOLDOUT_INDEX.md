# LEDGER public_holdout rc5 index

Document-only index for a later one-shot `public_holdout` E2E. This is a
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

## Commands

```text
python scripts/run_ledger_public_holdout_index.py --dry-run
python scripts/run_ledger_public_holdout_index.py --build --allow-remote
```

The large Milvus Lite files stay gitignored under
`outputs/ledger_public_holdout_rc5_index_v1/`. The tracked seal is
`data/eval_rag/ledger_public_holdout_index_v1.json` after a successful
build.

## v1 E2E remains blocked

The original zero-embedding contract is unchanged:

- `data/eval_rag/ledger_public_holdout_e2e_contract.json`
- `data/eval_rag/ledger_public_holdout_e2e_authorization.json`
- `data/eval_rag/ledger_public_holdout_e2e_result.json`
