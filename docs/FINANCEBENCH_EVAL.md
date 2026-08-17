# FinanceBench external RAG evaluation

This is an **external retrieval evaluation harness**. It is not FinAgentBench,
not a production API path, and not a license to change retrieval thresholds.

The 2026-08-16 test-100 four-mode corpus run is an **exploratory baseline /
exposed test-100**. It has been scored, read, and used for failure analysis.
It is **not** an unseen held-out. Later company-scope runs on the same 100
questions are **post-hoc paired diagnostics** only. Confirmation-50 was a
one-shot unseen set at execution time and is now **consumed / exposed**.
Recorded page-level numbers (Hit@5 0.50, Hit@10 0.62, MRR 0.2955, nDCG@10
0.3461) are **not product accuracy** and **not end-to-end QA**. Do not rerun
or retune from them. Phase 4 answer metrics remain `NOT_RUN`.

## Why this exists

Existing LumenFin RAG numbers are **synthetic gates**:

| Gate | Size | What it is |
|------|------|------------|
| `scripts/run_rag_eval.py` | 4 term-overlap cases | Internal hybrid recall/citation regression |
| `scripts/run_bm25_eval.py` | 5 cases | Dense vs BM25 vs hybrid on synthetic docs |
| `scripts/run_rerank_eval.py` | 10 hard negatives | Candidate vs lexical vs Qwen3 ranking |

The published Qwen3 row (Top-1 1.0000 / MRR 1.0000 / nDCG@5 0.9711, 10/10 no
fallback) is a **synthetic hard-negative gate**. It is **not** FinanceBench
accuracy. Do not mix the two.

FinanceBench is CC-BY-NC-4.0, 150 open questions, with human gold answers,
evidence spans, document names, and **zero-indexed** evidence pages. The
GitHub repo `patronus-ai/financebench` returned **404** at clone time; the
local checkout was materialized from HuggingFace `PatronusAI/financebench`
(`data/external/financebench-hf/`) into `data/external/financebench-src/`
(gitignored).

## Audit (2026-08-16)

### What we reuse

- `lumenfin.documents.parse_pdf_document` — PyMuPDF page list, 1:1 with PDF order
- `lumenfin.rag.chunking.chunk_document` — `page` is 1-indexed; `filename#p{page}`
- `MilvusRAGStore.vector_search` / `bm25_search` — isolated dense / BM25 modes
- `HybridEvidenceRetriever` — hybrid RRF and hybrid+Qwen3 **only when
  `--index-scope company`**
- `DeterministicEmbeddingProvider` — default offline embedding
- Citation / tenant / source-document metadata already stored on chunks

### Page provenance

**No production chunk-schema change is required.** Chunks already persist
`page`. FinanceBench `evidence_page_num` is zero-indexed; LumenFin pages are
one-indexed:

```text
lumenfin_page = evidence_page_num + 1
```

If a future chunk path dropped `page`, the harness **fails closed** and records
`page_provenance_gap`. It will not silently substitute fuzzy string matching.

### What we do not change

Production retrieval logic, RRF weights, Qwen3 thresholds, FinRun schema, and
API behavior are unchanged. Mode isolation lives only in
`src/lumenfin/eval/financebench/retrieval.py`.

Windows Milvus Lite cannot `os.rename` over `manifest.json`. Production
`milvus_store.py` must not patch process-global `os.rename`. On Windows Lite
only, `Manifest.save` is wrapped to use `os.replace`; Linux/server paths are
unchanged. Index/flush errors are verified by schema and searchability, not
by matching `"183"` / `"already exists"`. FinanceBench remote runs should
prefer Linux Docker or standalone Milvus. DashScope error redaction in
`embeddings.py` is a separate production hardening, not part of the eval
protocol.

`--index-scope corpus` hybrid is **eval-only RRF** (`_retrieve_corpus_hybrid`).
It searches the full 84-document index with **no company filter**. That is
stricter than production hybrid, which filters by company. Do not treat
corpus-scope hybrid as a production hybrid score.

### Gold relevance

Page-level qrels:

```text
query_id → evidence_doc_name → evidence_page_num_one → evidence_text
```

Chunk-level relevance is split:

1. **page-derived** (`page_chunk_*`): same document AND same gold page
2. **evidence-span** (`span_chunk_*`): same document AND auditable overlap
   with human `evidence_text`
3. Legacy `chunk_*` is the union of the two and is **deprecated**

Unmapped evidence spans are counted as `span_qrel_unmapped`. Do not generate
gold labels from embedding similarity.

Zero-chunk documents (including `JOHNSON_JOHNSON_2022Q4_EARNINGS`) are
`ingestion_failure`, not retrieval miss. Main tables keep all 100 questions;
real-PDF vs fallback cohorts are sensitivity analysis only.

### Split

`sha256("lumenfin-financebench-split-v1|{financebench_id}")` lexicographic sort,
independent of JSONL order:

- development / **confirmation-50**: first 50 ids (recorded 2026-08-16; now consumed/exposed)
- exposed test-100: remaining 100

Test-100 is frozen and **exposed**. Confirmation-50 is **recorded and
consumed**. `--tune` is rejected on every split, including confirmation. Do
not fit thresholds on confirmation-50.

## Dataset layout

Do not commit FinanceBench PDFs or the full JSONL. Place a local checkout at
for example `data/external/financebench-src/` (gitignored via `data/**`):

```text
financebench/
  data/financebench_open_source.jsonl
  data/financebench_document_information.jsonl
  pdfs/*.pdf
```

The 2026-08-16 checkout has **84 PDFs**. Seventy came from `doc_link`; eight
after retry; **six gold-page fallbacks** (evidence text rendered as PDF) in
`pdf_fallback.json`:

- `AMD_2015_10K`
- `KRAFTHEINZ_2019_10K`
- `JOHNSON_JOHNSON_2022_10K`
- `JOHNSON_JOHNSON_2022Q4_EARNINGS` (indexed **0 chunks**)
- `JOHNSON_JOHNSON_2023_8K_dated-2023-08-30`
- `JOHNSON_JOHNSON_2023Q2_EARNINGS`

Prepare into another gitignored directory:

```powershell
python scripts/prepare_financebench_eval.py `
  --source-dir <financebench-checkout> `
  --output-dir data/external/financebench `
  --require-pdfs
```

## Retrieval commands

Offline default (no DashScope / Qwen3). Do **not** use `--split dev` or
`--split confirmation` for smoke: both are the consumed confirmation-50.
Use the unit-test fixtures, or `--split test` (already exposed):

```powershell
python -m unittest tests.test_financebench_retrieval_eval tests.test_financebench_loader -v
python scripts/run_financebench_retrieval_eval.py --dataset-dir <checkout> --split test --mode bm25 --limit 2
python scripts/run_financebench_retrieval_eval.py --dataset-dir <checkout> --split test --mode dense --limit 2
python scripts/run_financebench_retrieval_eval.py --dataset-dir <checkout> --split test --mode hybrid --limit 2
```

Frozen corpus four-mode run (the recorded exploratory baseline):

```powershell
python scripts/run_financebench_retrieval_eval.py `
  --dataset-dir data\external\financebench-src `
  --mode all `
  --split test `
  --allow-remote `
  --embedding-provider dashscope `
  --index-scope corpus `
  --keep-index
```

`--mode all` shares one index across `bm25`, `dense`, `hybrid`, and
`hybrid-qwen3`. Remote DashScope embedding or Qwen3 **must** pass
`--allow-remote`. Exposed test-100 is for a frozen configuration only. Do not
fit thresholds on it. Company-scope on the same 100 questions is post-hoc.

## Ablation matrix

| Mode | Dense | BM25 | RRF | Qwen3 | Default remote |
|---|---:|---:|---:|---:|---|
| bm25 | no | yes | no | no | blocked |
| dense | yes | no | no | no | blocked unless DashScope |
| hybrid | yes | yes | yes | no | blocked unless DashScope |
| hybrid-qwen3 | yes | yes | yes | yes | requires `--allow-remote` |

Compare rank movement after four result directories exist:

```powershell
python scripts/run_financebench_retrieval_eval.py `
  --dataset-dir <checkout> `
  --mode hybrid `
  --compare-dirs outputs/financebench_eval/bm25 outputs/financebench_eval/dense outputs/financebench_eval/hybrid outputs/financebench_eval/hybrid-qwen3
```

## Outputs

```text
outputs/financebench_eval/
  results.json
  ablation.json
  rank_movements.jsonl
  <mode>/
    manifest.json
    environment.json
    results.json
    results.md
    per_case.jsonl
    failures.jsonl
```

`environment.json` records commit, dirty/clean worktree, dataset hash, split
hash, Python version, embedding/rerank models, chunk size/overlap, collection,
BM25/RRF parameters, top-k, timestamp, and whether remote calls were enabled.
It does not record API keys, full provider request IDs, or private endpoints.

Per-case rows keep citations and ranks, not document body.

## Metrics

Page level: Hit@1/3/5/10, Recall@1/3/5/10, MRR, nDCG@5, nDCG@10.

Page-derived chunk metrics: `page_chunk_hit_at_k` / `page_chunk_recall_at_k`.

Evidence-span chunk metrics: `span_chunk_hit_at_k` / `span_chunk_recall_at_k`.

Legacy `chunk_*` is the union of those labels and is deprecated.

Independent bootstrap 95% CIs describe one system's uncertainty (1000
resamples, seed `20260816`). They are **not** a significance test against
another system on the same queries. Same-query comparisons use:

- paired bootstrap ΔHit@5 / ΔHit@10 / ΔMRR / ΔnDCG@10
- McNemar exact on Hit@5 and Hit@10

Report effect size, paired CI, and p-value. A significant p-value on an
exposed or dirty run does not authorize a held-out claim.

## Recorded exploratory baseline / exposed test-100 (2026-08-16)

Configuration:

- split `test` (100 / 100 succeeded, 0 provider errors, Qwen3 fallback 0)
- `--mode all --index-scope corpus --allow-remote --embedding-provider dashscope`
- 84 documents, 52,518 chunks
- DashScope `text-embedding-v4`, dimension 1024
- commit `5877be8555bd72f411225b809ed75454607618bd`, **dirty worktree**
- dataset hash `5e961c0aa84a5ed578bdc2cea4f2ef8e33aa6ffe9394fc6c2508b303bf10fdeb`
- BM25 RRF weight 1.1, top-k 10

Page-level (primary). Hit@5 CI is bootstrap 95%:

| Mode | Hit@1 | Hit@5 | Hit@10 | MRR | nDCG@10 | Hit@5 CI |
|---|---:|---:|---:|---:|---:|---|
| bm25 | 0.11 | 0.21 | 0.30 | 0.1603 | 0.1763 | 0.14–0.29 |
| dense | 0.12 | 0.37 | 0.60 | 0.2465 | 0.3099 | 0.28–0.46 |
| hybrid | 0.13 | 0.23 | 0.38 | 0.1836 | 0.2137 | 0.15–0.32 |
| hybrid-qwen3 | 0.19 | 0.47 | 0.58 | 0.3044 | 0.3479 | 0.37–0.57 |

Chunk Recall@10 is 0.079 / 0.131 / 0.095 / 0.131 for the four modes. That is
expected: page Hit@K can fire if any chunk on the gold page ranks; chunk
Recall@K requires the evidence span itself.

Ablation vs BM25: improved 41, degraded 10, **never-retrieved 33**.

**How to read this:**

- Harness, split freeze, and four-mode isolation: **合格**.
- Absolute quality: honest **page-level baseline**, not product accuracy.
- `hybrid-qwen3` has the best point estimate. Paired Hit@5 counts vs dense
  were 33 / 4 / 14 / 49 (both / dense-only / qwen3-only / neither); McNemar
  exact two-sided p ≈ 0.031. Independent CIs overlap and are the wrong test.
  Because the worktree is dirty and test-100 is now exposed, **do not publish
  a statistically significant Qwen3 win**.
- Corpus hybrid is **weaker than dense**. That is a real finding of this
  scope, not a production hybrid regression.
- Do not copy these rows into the root README as published accuracy.

Raw artifacts: `outputs/financebench_eval/`.

### Why corpus hybrid lost to dense

Production hybrid filters by company. This run used `--index-scope corpus`,
so eval-only RRF fused unfiltered BM25 and dense lists with BM25 weight
**1.1**. Across 84 10-K/10-Q filings, lexical hits on shared boilerplate
(gross margin, working capital, inventories) outrank the gold page.

Pairwise on the same 100 questions:

- dense better first-relevant rank: **40**; hybrid better: **16**; tie: **44**
- dense Hit@5 and hybrid miss: **20**; reverse: **6**
- of those 20, all used `hybrid_dense_bm25_rrf`; 12 top-5 were a
  **different document**, 8 were the right document on the wrong page

Example (`fb-financebench_id_00005`, Corning): dense top-5 are all
`CORNING_2022_10K`; hybrid top-5 are 3M / CVS / Netflix. Dense gold rank 3;
hybrid miss-all.

Company-scope hybrid (`--index-scope company`) was the next frozen comparison.
This corpus run answers “does unfiltered RRF help on a multi-issuer 10-K
pile?” — and the answer is no. See the post-hoc company-scope section below.

### The 33 never-retrieved cases

Never-retrieved means **no mode** placed a gold page in top-10.

| Slice | Count |
|---|---:|
| miss_all (gold page never in top-10) | 28 |
| wrong_document | 5 |
| single gold page / multi gold page | 29 / 4 |
| fallback PDFs in the 33 | 4 of 6 fallback docs (vs 29/94 real PDFs) |
| period_disambiguation | 28/33 (vs 86/100 overall — not distinctive) |
| question_type | domain-relevant 13, novel-generated 12, metrics-generated 8 |
| companies | J&J 4, Pfizer 3, Amcor 3 |

Period labels do **not** explain the ceiling. The 33 are mostly
qualitative / multi-hop MD&A questions (working capital usefulness, capital
intensity, “what drove gross margin, or explain why the metric is not
useful”) whose gold pages are not lexically unique in an 84-doc corpus.
Fallback PDF quality is a second, smaller bucket: four of six fallback
documents appear in the 33, including `JOHNSON_JOHNSON_2022Q4_EARNINGS`
(0 indexed chunks).

Sample questions (gold page missed by every mode):

- Paypal FY2022: does it have positive working capital, or is the metric
  not useful?
- Verizon FY2022: is it capital intensive?
- Pfizer: expected USD millions to spin off Upjohn?
- Ulta Beauty FY2023: what drove the merchandise inventories increase?
- J&J (fallback 10-K): are FY2022 financials those of a high-growth company?

This 33/100 is a hard ceiling for the current chunker + top-10 + full-corpus
setup. Raising Hit@5 without changing scope, PDF quality, or chunking cannot
pass 0.67.

## Post-hoc paired diagnostic: company scope (2026-08-16)

**Role:** `post_hoc_paired_diagnostic` on the **exposed test-100**. Not
held-out. Dirty worktree. Not product accuracy. Phase 4 remains `NOT_RUN`.

The only intended experimental change versus the corpus baseline:

```text
index_scope: corpus → company
```

Recorded command:

```powershell
python scripts/run_financebench_retrieval_eval.py `
  --dataset-dir data\external\financebench-src `
  --output-dir outputs\financebench_eval_company `
  --mode all --split test --allow-remote `
  --embedding-provider dashscope --embedding-dimension 1024 `
  --index-scope company --keep-index --top-k 10
```

Identity:

- documents 84, chunks 52,518 (same as corpus)
- dataset hash `5e961c0aa84a5ed578bdc2cea4f2ef8e33aa6ffe9394fc6c2508b303bf10fdeb` (same)
- case-id overlap with corpus test-100: **100 / 100**
- split-manifest hash changed (`e410e13e…` → `b58dbba8…`) because the manifest
  now includes governance fields; assignment IDs did not move
- embedding DashScope `text-embedding-v4` / 1024, chunk 900/120, RRF 1.1, top-k 10
- 0 provider errors, Qwen3 fallback 0 / 100
- `JOHNSON_JOHNSON_2022Q4_EARNINGS` still 0 chunks (`ingestion_failure`; 2 questions)

Page-level company-scope (all 100 questions):

| Mode | Hit@1 | Hit@5 | Hit@10 | MRR | nDCG@10 | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.15 | 0.39 | 0.47 | 0.2494 | 0.2794 | 80 | 147 |
| dense | 0.12 | 0.37 | 0.58 | 0.2455 | 0.3048 | 319 | 375 |
| hybrid | 0.17 | 0.39 | 0.48 | 0.2575 | 0.2933 | 409 | 789 |
| hybrid-qwen3 | 0.22 | 0.48 | 0.65 | 0.3391 | 0.3940 | 721 | 1190 |

Four-mode never-retrieved: **28** (corpus 33). Span qrels: 50 mapped / 71
unmapped. Main table keeps all 100 questions.

### Paired corpus → company (same 100 queries)

Independent CIs are not the test. Paired bootstrap 1000 / seed `20260816`;
McNemar exact two-sided on Hit@5 and Hit@10.

| Pair | ΔHit@5 (CI) | ΔHit@10 (CI) | ΔMRR (CI) | McNemar Hit@5 | McNemar Hit@10 |
|---|---|---|---|---|---|
| Dense | 0.00 (−0.03, 0.03) | −0.02 (−0.05, 0.00) | −0.001 (−0.030, 0.021) | 36/1/1/62, p=1.0 | 58/2/0/40, p=0.50 |
| Hybrid | **+0.16 (0.09, 0.23)** | **+0.10 (0.04, 0.16)** | **+0.074 (0.040, 0.111)** | 23/0/16/61, p=0.000031 | 38/0/10/52, p=0.001953 |
| Hybrid+Qwen3 | +0.01 (−0.04, 0.07) | **+0.07 (0.02, 0.12)** | **+0.035 (0.008, 0.065)** | 44/3/4/49, p=1.0 | 58/0/7/35, p=0.015625 |

McNemar cells are `both / baseline_only / candidate_only / neither`. Rank
movement (first gold-page rank): dense 2/2/96; hybrid 22/1/77; Qwen3 12/6/82
(improved / degraded / tied).

**Read-out:** company scope repaired **BM25 and unfiltered-RRF hybrid**. It
did **not** move Dense Hit@5. Hybrid+Qwen3 Hit@5 was already high on corpus
and did not move; Hit@10 and MRR did.

### Paired within company scope

| Pair | ΔHit@5 (CI) | ΔHit@10 (CI) | McNemar Hit@5 | McNemar Hit@10 |
|---|---|---|---|---|
| Dense vs Hybrid | +0.02 (−0.08, 0.12) | **−0.10 (−0.20, −0.01)** | 25/12/14/49, p=0.845 | 40/18/8/34, p=0.076 |
| Hybrid vs Hybrid+Qwen3 | **+0.09 (0.01, 0.17)** | **+0.17 (0.10, 0.25)** | 33/6/15/46, p=0.078 | 47/1/18/34, p=0.000076 |

After company filtering, Hybrid Hit@5 matches BM25 (0.39) and is no longer
worse than Dense at Hit@5. Dense still has a higher Hit@10 than Hybrid
without rerank. Qwen3 is what lifts company Hybrid to Hit@10 0.65.

### Cohorts (sensitivity; not the main table)

| Mode | all n=100 Hit@5 / @10 | real PDF n=94 | fallback n=6 |
|---|---|---|---|
| bm25 | 0.39 / 0.47 | 0.3723 / 0.4574 | 0.6667 / 0.6667 |
| dense | 0.37 / 0.58 | 0.3830 / 0.6064 | 0.1667 / 0.1667 |
| hybrid | 0.39 / 0.48 | 0.3723 / 0.4681 | 0.6667 / 0.6667 |
| hybrid-qwen3 | 0.48 / 0.65 | 0.4787 / 0.6596 | 0.5000 / 0.5000 |

Do not drop the two `ingestion_failure` J&J questions from the main 100.

Raw artifacts: `outputs/financebench_eval_company/` including
`paired_vs_corpus.json`.

Because this is post-hoc on an exposed, dirty run: **do not publish
“company scope significantly wins” as a held-out or product claim.**

## End-to-end answer evaluation (phase 4)

**Status: `NOT_RUN` / not implemented in this change.** Retrieval must stabilize
first. Planned deterministic metrics (not an LLM-judge substitute):

- numeric accuracy with relative tolerance
- exact / normalized match
- citation document/page precision and recall
- evidence support rate
- unit/currency and reporting-period consistency
- unsupported numeric claims
- correct abstention / `incomplete_data` precision and recall

If an LLM judge is added later it is supplementary only, with judge model and
prompt version recorded.

## Result ledger

| Result | Status |
|--------|--------|
| Synthetic 10-case Qwen3 gate | already published; **synthetic**, not FinanceBench |
| FinanceBench loader / split / qrels unit tests | implemented; run with `python scripts/run_tests.py` |
| Offline smoke on 4 synthetic PDF questions | unit tests; **synthetic** |
| FinanceBench 150 load + 50/100 split on real JSONL | **recorded** (HF materialization; GitHub 404; 6 fallback PDFs) |
| Exposed test-100 corpus four-mode | **recorded** 2026-08-16; exploratory baseline; dirty tree |
| Company-scope post-hoc diagnostic on test-100 | **recorded** 2026-08-16; post-hoc; dirty tree |
| Frozen config used for confirmation | `data/eval_rag/financebench/frozen_config.json`; hash `18a483f604f3a5420264e746d9219e77e3c9bddbd91c5c50252025b40ccb1ee7`; tag `financebench-confirmation-v1` |
| Confirmation-50 (formerly dev-50) | **RECORDED** 2026-08-16; one-shot unseen at execution; now consumed/exposed; Hit@5 0.50, Hit@10 0.62, MRR 0.2955, nDCG@10 0.3461; **not** product accuracy; retune forbidden |
| Phase 4 end-to-end answer metrics | `NOT_RUN` |

Do not copy synthetic gate scores **or** these FinanceBench page-level rows
into README as product accuracy.

## Confirmation-50

**Status: `RECORDED` / consumed.** Machine-readable lock:
`data/eval_rag/financebench/frozen_config.json`. Aggregate (git-tracked, no
raw questions/qrels/per-case):
`data/eval_rag/financebench/confirmation_result.json`. Raw artifacts remain
local and gitignored under `outputs/financebench_eval_confirmation/`.

At execution this split was a one-shot unseen confirmation set. It is now
consumed/exposed. Do **not** run it again. Do **not** retune from it.

Canonical hash: UTF-8 `json.dumps(..., sort_keys=True, ensure_ascii=False,
separators=(',', ':'))` SHA-256. `config_hash` and timestamp keys
(`proposed_at`, `executed_at`, `frozen_at`, `timestamp`) are excluded.
The published digest includes the original `notes` and identity fields;
dropping `notes` does **not** reproduce the digest, and this freeze will not
mint a replacement hash.

```text
config_hash: 18a483f604f3a5420264e746d9219e77e3c9bddbd91c5c50252025b40ccb1ee7
tag: financebench-confirmation-v1
commit: 379a8b053256fd43260ecf031cdf675af7c3be4b
index_scope: company
mode: hybrid-qwen3
chunk: 900 / 120
embedding: dashscope text-embedding-v4, 1024
BM25 RRF weight: 1.1
top_k: 10
rerank_candidates: 20
rerank: qwen3-rerank
query_rewriting: off
executed_at: 2026-08-16T18:49:44Z
cases: 50
page Hit@1: 0.14
page Hit@5: 0.50 (95% CI 0.36-0.64)
page Hit@10: 0.62
page MRR: 0.2955 (95% CI 0.1978-0.3897)
page nDCG@10: 0.3461 (95% CI 0.244-0.4417)
Hit@5 count: 25/50
Hit@10 count: 31/50
top10_missed: 19/50
miss_all: 16
wrong_document: 2
ingestion_failure: 1
never_retrieved_across_modes: NOT_APPLICABLE
```

These are frozen-config **page-level retrieval** numbers on a one-shot unseen
confirmation-50. They are **not product accuracy** and **not end-to-end QA
accuracy**. Hit@10 bootstrap CI was not stored in the original
`results.json` summary; persisting it is a later maintenance item.

`--split confirmation` and `--split dev` remain the same protected split.
They still require `--frozen-config` and `--confirm-held-out`. Do not rerun
the recorded command. `--limit` and `--mode all` stay rejected.

Historical command (already executed; do not run again):

```powershell
python scripts/run_financebench_retrieval_eval.py `
  --dataset-dir data\external\financebench-src `
  --output-dir outputs\financebench_eval_confirmation `
  --split confirmation `
  --mode hybrid-qwen3 `
  --index-scope company `
  --embedding-provider dashscope `
  --embedding-dimension 1024 `
  --top-k 10 `
  --allow-remote `
  --frozen-config data\eval_rag\financebench\frozen_config.json `
  --confirm-held-out
```

Do not retune from that result. Further system changes need a new dataset.

Confirmation-50 was recorded under the previous narrative chunker, which did
**not** implement sliding-window overlap (`overlap_chars` stripped a prefix
instead of sharing a suffix/prefix). A later chunker fix changes indexed
text. Post-fix retrieval numbers are a **new system**, not a re-run of the
frozen confirmation. Do not rewrite `frozen_config.json` or treat those
scores as the same freeze. The harness can still run (`--split confirmation`
remains protected; test-100 remains diagnostic only). Confirmation-50 is no
longer unseen.

Candidate-depth attempt 1 (`outputs/financebench_candidate_depth_test100/`) is
**invalid**: the historical company index was written with
`session_id=financebench-eval`, but the diagnostic queried
`session_id=financebench-candidate-depth`. Those all-zero rows are a session
filter mismatch, not a retrieval-quality result. Do not copy them into
benchmark tables. A later valid run must use a new directory
(`outputs/financebench_candidate_depth_test100_v2/`).

Candidate-depth preflight attempt 1
(`outputs/financebench_candidate_depth_test100_v2_preflight/`) is **invalid**:
copied collection was released before query. Do not treat that directory as
`PREFLIGHT_OK` and do not write retrieval scores from it. A later preflight
must use `outputs/financebench_candidate_depth_test100_v2_preflight2/`.

## License

FinanceBench is CC-BY-NC-4.0. LumenFin does not relicense or vend the PDFs.
See `THIRD_PARTY_NOTICES.md`.
