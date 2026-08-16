# FinanceBench external retrieval evaluation

This harness scores **LumenFin retrieval** on the open 150-question
[FinanceBench](https://github.com/patronus-ai/financebench) set. It is **not**
a production retrieval path, and it is **not** a substitute for the existing
4/5/10 synthetic RAG/BM25/Qwen3 gates.

Those synthetic gates remain release checks. They are **not** external financial
RAG accuracy.

## What is measured

Four modes share one corpus, one chunker (900/120, 1-indexed `page`), one
query set, one qrel set, one top-k, one split, and one embedding/schema:

| Mode | Dense | BM25 | RRF | Qwen3 |
|---|---|---|---|---|
| `bm25` | no | yes | no | no |
| `dense` | yes | no | no | no |
| `hybrid` | yes | yes | yes | no |
| `hybrid-qwen3` | yes | yes | yes | yes |

Metrics are reported at **page** and **chunk** level: Hit@K, Recall@K, MRR,
nDCG, bootstrap 95% CI (seed `20260816`, 1000 samples), taxonomy breakdowns,
and rank movements across modes.

Gold pages use FinanceBench `evidence_page_num` (**0-indexed**) mapped to
LumenFin `page` (**1-indexed**): `lumenfin_page = evidence_page_num + 1`.
A chunk is relevant only when it is the **same document** and either covers
that page or has an **auditable span overlap**. Missing page metadata is
fail-closed (the chunk is not relevant). No fuzzy substitute.

The held-out split is **50 dev / 100 test**, assigned by
`sha256("lumenfin-financebench-split-v1|{financebench_id}")` (order-independent).
`--tune` is refused on `--split test`. Do not fit thresholds on the test split.

Phase 4 end-to-end answer evaluation is **not implemented** (`NOT_RUN`).

## Dataset (not committed)

Do **not** commit PDFs, the JSONL, API keys, or fabricated scores.

1. HuggingFace merged questions (150 rows, includes `doc_link`):
   `https://huggingface.co/datasets/PatronusAI/financebench`
2. Official GitHub dump when reachable:
   `https://github.com/patronus-ai/financebench`
   Some environments cannot clone GitHub and should use the HuggingFace file.
3. Place files under gitignored `data/external/financebench-src/`:
   - `data/financebench_merged.jsonl` **or**
     `financebench_open_source.jsonl` + `financebench_document_information.jsonl`
   - `pdfs/{doc_name}.pdf` (84 unique filings)

```bash
python scripts/fetch_financebench_pdfs.py --dataset-root data/external/financebench-src
python scripts/prepare_financebench_eval.py --dataset-root data/external/financebench-src
```

FinanceBench is CC-BY-NC-4.0 (Patronus AI et al.). See `THIRD_PARTY_NOTICES.md`.

## Offline default

Remote DashScope embeddings and Qwen3 rerank require **`--allow-remote`**.
Default runs are offline (deterministic embeddings + BM25).

```bash
# Offline smoke (selected PDFs only; not the official 84-doc comparison)
APP_ENV=test python scripts/run_financebench_retrieval_eval.py \
  --mode bm25 --split test --limit 2 --index-scope selected \
  --embedding-provider deterministic
```

## Frozen four-mode run (approved remote)

Index **once**, then score all four modes on the held-out **test 100**.
Do not pass `--tune`. If `DASHSCOPE_RERANK_BASE_URL` is empty, `hybrid-qwen3`
records lexical fallback instead of pretending Qwen3 ran.

```bash
python scripts/run_financebench_retrieval_eval.py \
  --mode all --split test --allow-remote \
  --embedding-provider dashscope \
  --dataset-root data/external/financebench-src \
  --out-dir outputs/financebench_eval \
  --fetch-pdfs --index-scope corpus
```

Outputs (gitignored): `outputs/financebench_eval/{bm25,dense,hybrid,hybrid-qwen3}/`
plus `compare_modes.json`. Unrun numbers stay `NOT_RUN` / `UNVERIFIED`.

Eval indexes into a **shared session** with company tag `FinanceBenchEval` so
all four modes search the same 84-document collection through production
retriever APIs. Production hybrid retrieval, RRF weights, FinRun schema, and
API routes are unchanged.
