---
title: FinanceBench next phase
---

# Next phase: reranker, page diversity, section/parent retrieval

**LEDGER `public_dev` status (2026-08-19): sealed and stopped.**
Chain seal: `data/eval_rag/holdout/ledger_public_dev_chain_seal.json`.
Do not embed a page-parent index. Do not rescore the frozen 5×50 suffix.
FinanceBench Phase 4 remains `NOT_RUN`. Production RAG defaults unchanged.

A structured-citation **public/dev shadow harness** is frozen under
`data/eval_rag/structured_citation_shadow_config.json`. It is not a new
LEDGER benchmark, not held-out, and is not authorized to run in this
stage. Do not open `public_holdout`. Do not retune from exposed public/dev.

This protocol starts **after** the sealed exposed test-100 A/B/C ablation.
It is not a license to change production retrieval.

Sealed source of truth:
`data/eval_rag/financebench/candidate_pool_ablation_result.json`

## Why this phase exists

The recorded ablation said:

- Keep production **arm A**.
- Do not adopt **B** (50+50 retrieval, rerank still 20).
- Treat **C** as a next-generation *candidate*, not an authorized default.
- C moved more gold into the rerank pool, then left **11** questions
  `gold_in_pool_not_in_final_top10`. Ranking, not pool size, is now the
  primary bottleneck.
- C still missed **21** questions with gold outside the pool. Those need
  section/parent retrieval or better chunk identity, not another pool-size
  sweep on the same 100 questions.

Current chunk records have `page` and `chunk_type`. They do **not** have
section titles or parent-chunk ids. Eval indexes report
`section_metadata=NOT_AVAILABLE`.

## Hard bans

| Ban | Reason |
| --- | --- |
| Do not score or retune on FinanceBench test-100 | Already exposed; `retuning_on_test100_forbidden=true` |
| Do not rerun confirmation-50 | Consumed / exposed |
| Do not edit `frozen_config.json` | That lock is confirmation-50 arm A |
| Do not change production RAG defaults | `production_change_authorized=false` |
| Do not enlarge candidate pool as the next experiment on old data | B had no gain; C is already recorded |
| Do not treat a win on a new set as automatic production cutover | Needs a separate authorization |

A later production cutover, if any, is a **new decision** after a sealed
unseen result. It is not part of building the prototype.

## Dataset: must be new and frozen first

FinanceBench's 150 questions are fully consumed (test-100 + confirmation-50).
There is no remaining unseen FinanceBench split.

**Recommended path:** a new internal holdout, not another pass over Patronus
questions.

| Option | Use | Do not use if |
| --- | --- | --- |
| Internal holdout (LumenFin diligence questions + filings) | Default. Matches product: issuer isolation, page-level evidence | Questions are written after seeing retrieval |
| Time-split new 10-K/10-Q pages | Extra distribution-shift check | Gold pages are guessed from model output |
| Other public QA sets (TAT-QA, ConvFinQA, …) | Optional later, different task | Treated as a FinanceBench replacement |

Freeze **before** any retrieval score:

1. Write questions and gold `filename` + 1-indexed page (and optional span)
   without looking at hybrid/Qwen3 ranks.
2. Hash the question file and qrels. Record `dataset_hash`.
3. Choose a one-shot holdout (if n is small, do not also carve a tuning split
   from the same items).
4. Lock `experiment_role=unseen_holdout` and refuse `--split test` /
   `--split confirmation` from FinanceBench.

If the set is smaller than ~40 questions, treat it as **one-shot
confirmation**, not as a tuning loop.

### Public development benchmark

Before a private holdout exists, use
[LEDGER](https://github.com/artefactory/LEDGER) for page-retrieval
development. LEDGER code is MIT and its data is CC BY 4.0. It is public, so
foundation-model training exposure is unknown and no result may be called a
truly unseen or product-accuracy score.

The adapter under `src/lumenfin/eval/holdout/ledger.py` accepts the official
`artefactory/ledger-long-context-KPI-QA` `eval/test` parquet schema. It:

- requires an immutable 40-character source revision and local snapshot hash;
- preserves TREC `doc_id` qrels exactly rather than guessing PDF page numbers;
- splits by exchange+ticker, never by individual question;
- exposes repeatable `public_dev` and locally held-back `public_holdout`;
- records only counts and identity hashes in the manifest, not questions,
  report text, or qrel ids;
- keeps scoring and remote calls disabled.

The public holdout only protects against local tuning leakage. Once its results
influence a change, mark it consumed and treat later runs as regression tests.
FinRank may be added later as a separate passage-reranking stress test; its
CC BY-NC 4.0 license must not be treated as commercial-use permission.

A local snapshot of the official LEDGER `eval/test` config is pinned at source
revision `b7085dc6cb16b3ec8149a9baf6dd2d3416cf7619`. The tracked identity-only
manifest is `data/eval_rag/holdout/ledger_public_manifest.json`: 10,000
questions across 111 companies, split into `public_dev` (7,616 questions,
85 companies) and unconsumed `public_holdout` (2,384 questions, 26 companies).
Raw parquet and report text remain gitignored. Primary Qwen3 scoring and all
`public_holdout` scoring remain disabled; an offline BM25 preflight is enabled
for `public_dev` only.

The frozen public-dev corpus audit follows the official indexer's rule that
blank OCR page segments are not indexed. It records 116 zero-relevance qrels
to blank pages as ignored and 72 positive qrels to blank pages across 69
queries as unavailable. One query has no reachable positive page and is
excluded by frozen ID hash; 7,615 queries remain scorable. For the other 68
affected queries, only the unavailable blank-page labels are removed. This is
a corpus/qrel availability policy fixed before retrieval ranks are inspected,
not a retrieval-result exclusion.

LEDGER intentionally grades pages from other report years for the same
exchange+ticker (comparative KPI evidence is common). The adapter therefore
locks qrels to the query company, not to only the row's nominal year; pages
from another company still fail closed.

## Workstreams (priority)

Work them in this order. Each is a separate locked arm on the **new** set.
Do not combine instruct + parent-child + pool-50 into one first run.

### 1. Reranker ranking (first)

**Hypothesis:** gold is often already in the C-sized pool, but Qwen3 does not
put the gold page in final Top-10.

Candidate eval-only arms (names are local to this phase):

| Arm | What changes | What stays |
| --- | --- | --- |
| `A_prod` | nothing | production A: channels 20, RRF 20, Qwen3 20 → 10 |
| `R_page` | collapse production Top-20 to unique pages before Qwen3; rerank page representatives | same source Top-20 as A; no deeper backfill, so the rerank pool may be smaller |
| `R_fuse` | keep RRF score visible to the final cut, or weighted fuse with Qwen3 | same candidates |
| `R_window` | send heading/query-window text to Qwen3 instead of raw first 4000 chars | same candidates |

Success is page Hit@5 / Hit@10 / MRR / nDCG@10 on the **new** holdout, with
McNemar + paired bootstrap, `primary_comparison_valid`, and zero Qwen3
fallback if claiming a Qwen3 ranking win.

Do not retune `MAS_RAG_RERANK_INSTRUCT` against FinanceBench test-100.

### 2. Page diversity

**Hypothesis:** final Top-10 wastes slots on the same page, so a gold page at
rank 11 never appears.

Eval-only ideas:

- Unique-page cap in the final 10 (keep best chunk per page).
- MMR / same-page penalty after Qwen3.
- Report `unique_pages_top10` on every run. Page uniqueness can be computed
  today from `page`; it does not require section metadata.

This workstream does not require a new chunker.

### 3. Section / parent retrieval (second system)

**Hypothesis:** remaining pool misses need a larger *semantic* window, not a
larger k.

This **is** a new indexed system (like the overlap-fix chunker). Do not mix
its scores with confirmation-50 or the sealed A/B/C numbers.

Minimum schema additions (eval collection first, not production):

- `section_id` / `section_title` (10-K Item 1, 1A, 7, 8, … when detectable)
- `parent_chunk_id` or parent text window (retrieve child, return parent page
  or section)
- Keep existing `page` and issuer tags

Do not rebuild the production `lumenfin_chunks_v4_bm25` collection until a
holdout result is sealed **and** production change is separately authorized.

## Protocol for a scoring run

Same discipline as confirmation-50 / A/B/C:

```text
freeze dataset + config hash
  → synthetic / unit tests (no remote)
  → preflight on a copied index (no Qwen3)
  → one --allow-remote run
  → seal aggregate JSON (no raw questions)
  → production still unchanged
```

Call accounting, resume, and fail-closed empty retrieval stay as in the
candidate-pool harness. New output directories only; never
`outputs/financebench_candidate_pool_ablation_test100/`.

Arm C may appear **once** on the new holdout as `C_candidate` versus `A_prod`.
That run answers “does C still help on unseen data?”. It does not authorize
shipping C. Do not also retune instruct or parent-child on that same holdout.

## First build slice (no remote, no production)

Do this before collecting the holdout gold if needed, but **do not score
FinanceBench again**:

1. Unit tests: unique-page collapse and `unique_pages_top10`.
2. Schema tests: optional `section_title` / `parent_chunk_id` round-trip;
   missing metadata stays `NOT_AVAILABLE` rather than guessed.
3. Dataset template under `data/eval_rag/` (gitignored raw PDFs; tracked
   schema example only).
4. Harness flag: refuse FinanceBench `test` / `confirmation` for this phase.

Implementation status: the validate-only scaffold now lives under
`src/lumenfin/eval/holdout/`, with
`scripts/run_holdout_ranking_eval.py` and a tracked schema example under
`data/eval_rag/holdout/`. It refuses FinanceBench `test`, `dev`,
`confirmation`, and `all`, path escape, and all remote calls. `A_prod` and
`R_page` pool construction plus page-level metrics are available for
synthetic/offline tests only.

The next slice adds `scripts/run_ledger_public_dev_ranking.py`. It verifies the
pinned parquet hash, split salt, and complete `public_dev` query/company
identity before indexing. Each non-empty zero-based LEDGER page becomes one
eval document whose `document_id` is the official qrel ID; the existing
production chunker and Milvus-native BM25 implementation are reused without
changing production retrieval behavior. Deterministic embeddings only satisfy
the isolated BM25 index schema and are not used for BM25 ranking.

```powershell
python scripts/run_ledger_public_dev_ranking.py `
  --parquet-path data/external/ledger-long-context-KPI-QA/b7085dc6cb16b3ec8149a9baf6dd2d3416cf7619/eval-test `
  --manifest data/eval_rag/holdout/ledger_public_manifest.json `
  --split-salt lumenfin-ledger-public-v1 `
  --output-dir outputs/ledger_public_dev_bm25_preflight_5 `
  --max-cases 5
```

This preflight retrieves one shared BM25 Top-20 per query. `A_prod` keeps the
chunk order; `R_page` collapses duplicates only inside that same Top-20, with
no deeper backfill. It writes redacted per-case metrics and an aggregate with
`remote_calls=0` and `qwen3_calls=0`. It is not yet a production-hybrid or
Qwen3 result, so `primary_comparison_valid=false`. Limited preflights select
queries by deterministic company round-robin rather than taking a
single-company lexical prefix.

The monolithic CLI is capped at 10 cases. A full 7,615-query attempt exceeded
the practical single-process Milvus memory boundary and produced no aggregate;
full public-dev scoring therefore uses
`scripts/run_ledger_public_dev_bm25_sharded.py`. It partitions the 85 companies
deterministically, runs shards in sequential child processes, removes each
released index, supports completed-shard resume, verifies full 7,615-query
coverage, and only then publishes the combined aggregate.

```powershell
python scripts/run_ledger_public_dev_bm25_sharded.py `
  --parquet-path data/external/ledger-long-context-KPI-QA/b7085dc6cb16b3ec8149a9baf6dd2d3416cf7619/eval-test `
  --manifest data/eval_rag/holdout/ledger_public_manifest.json `
  --split-salt lumenfin-ledger-public-v1 `
  --output-dir outputs/ledger_public_dev_bm25_full_sharded_v1 `
  --shard-count 17
```

Do not bypass the monolithic cap or interpret its failed resource probe as an
evaluation result.

The completed 17-shard aggregate is sealed at
`data/eval_rag/holdout/ledger_public_dev_bm25_baseline.json`:

- 7,615/7,615 scorable queries, 85 companies, 36,215 pages, and 269,454
  production-chunker chunks; remote calls and Qwen3 calls are both zero.
- BM25 `A_prod`-shape Top-10 page Hit is 31.24%; same-Top-20 `R_page` is
  33.08% (+1.84 percentage points, 140 gains and zero losses).
- Unique pages in Top-10 rise from 8.9913 to 9.9825, while duplicate occupancy
  falls from 10.09% to zero.
- Pool Hit@20 is only 44.12%; 4,255 queries miss every positive page in the
  pool. Page collapse cannot repair those misses.

This is an offline BM25/prerank baseline, not production hybrid+Qwen3:
`primary_comparison_valid=false`. The evidence supports retaining page
deduplication as a low-risk candidate, but the next experiment must measure
hybrid candidate recall on `public_dev` before spending remote Qwen3 calls.

The same CLI now has an eval-only Hybrid canary mode. It requires all three of
`--mode hybrid`, `--embedding-provider dashscope`, and `--allow-remote`; it is
limited to at most 100 questions from one explicit company. It embeds that
company's page corpus with `text-embedding-v4`, retrieves Dense Top-20 and BM25
Top-20, fuses them with the production RRF weights, and passes the same fused
Top-20 to both ranking arms. Dense failure and missing physical HTTP-call
accounting fail closed. Qwen3 remains disabled.

The first 50-question single-company canary (`amex:brn`) indexed 650 pages /
3,840 chunks (2,298,062 characters). It recorded 389 document-embedding HTTP
calls plus 50 query-embedding calls and zero Qwen3 calls. Against the exact
same 50 queries from the sealed BM25 baseline:

- Hybrid Pool Hit@20 increased from 42% to 48%.
- `A_prod`-shape Hit@10 increased from 32% to 38%; `R_page` Hit@10 increased
  from 34% to 38%.
- MRR and nDCG@10 were approximately flat/slightly lower, so this is candidate
  recall evidence only—not evidence that final ranking improved.

Because one company is not representative of all 85 public-dev companies,
`scripts/run_ledger_public_dev_hybrid_stratified.py` freezes five companies at
evenly spaced positions in the sorted public-dev company identity, takes the
first 50 frozen queries per company, validates the sealed full BM25 per-case
artifact before any remote work, and runs each company sequentially. Completed
children are resumable only after exact dataset, source, query, company, index,
physical-call, and qrel-audit validation.

The sealed result is
`data/eval_rag/holdout/ledger_public_dev_hybrid_stratified_5x50.json`:

- 250 paired queries, 5 companies, 2,427 pages, and 15,829 chunks.
- Hybrid Pool Hit@20 increased from 54.8% to 72.8% (+18.0 points; 48 gains,
  3 losses).
- `A_prod`-shape Hit@10 increased from 39.6% to 56.4% (+16.8 points; 47 gains,
  5 losses).
- `R_page` Hit@10 increased from 42.8% to 57.6% (+14.8 points; 41 gains,
  4 losses).
- `A_prod` MRR increased from 0.1596 to 0.2544 and nDCG@10 from 0.1149 to
  0.1727. `R_page` MRR increased from 0.1671 to 0.2579 and nDCG@10 from
  0.1250 to 0.1808.
- Physical accounting exactly matched the no-retry plan: 1,601 document
  embedding calls plus 250 query embedding calls. Reranker and Qwen3 calls
  remained zero; indexes were removed.

An exact-configuration repeat kept Pool Hit@20 identical and changed Hit@10 on
3/250 cases (six cases changed reciprocal rank/nDCG). The sealed values above
come from the fingerprint-complete repeat. This small run-to-run variation is
recorded as a Milvus ranking-stability limitation and reinforces persisting
candidate identity hashes in the rerank experiment.

This broader sample supported Hybrid as the candidate-retrieval arm for the
next public-dev experiment. It did not by itself authorize a production default
change because Qwen3 had not yet run.

`scripts/run_ledger_public_dev_qwen3_paired.py` then repeated the locked Hybrid
retrieval on the same 250 query identities and split execution into three
explicit phases: freeze full candidates and their ordered identity hashes,
freeze a local-only Qwen3 request plan, then rerank. The rerank phase refuses
candidate, plan, settings, source, or completed-row divergence before new
provider calls. Complete cases are atomically resumable with at-least-once
billing semantics.

The sealed paired result is
`data/eval_rag/holdout/ledger_public_dev_qwen3_paired_5x50.json`:

- Candidate Pool Hit@20 was 72.8% for both arms and remained unchanged by
  reranking, as required by the frozen-pool design.
- `A_prod` Hit@5 increased from 36.8% to 61.6% (+24.8 points; 72 gains,
  10 losses; exact paired sign p=1.02e-12). Hit@10 increased from 55.6% to
  70.0% (+14.4 points; 40 gains, 4 losses; p=1.71e-8).
- `R_page` Hit@5 increased from 37.6% to 62.0% (+24.4 points; 71 gains,
  10 losses; p=1.80e-12). Hit@10 increased from 56.8% to 70.4% (+13.6
  points; 39 gains, 5 losses; p=1.41e-7).
- `A_prod` MRR increased from 0.2494 to 0.4663 and nDCG@10 from 0.1708 to
  0.3141. `R_page` MRR increased from 0.2528 to 0.4488 and nDCG@10 from
  0.1789 to 0.3067.
- All 500 logical Qwen3 reranks completed in 500 physical attempts with
  2,392,888 provider-reported tokens, zero retries, zero fallbacks, and zero
  terminal provider errors. Therefore `primary_comparison_valid=true`.
- Candidate generation used 1,601 document-embedding calls plus 250 query
  embedding calls. Qwen3 cost was approximately CNY 1.20 at the documented
  CNY 0.0005 per thousand input tokens, before any account-specific free quota.

This is valid final-ranking evidence on a public development benchmark, not a
private holdout and not end-to-end answer accuracy. `A_prod` is the preferred
next development arm: it is within 0.4 points of `R_page` Hit@10 and has higher
MRR, while retaining the production-shaped pool.

`scripts/run_ledger_public_dev_e2e_canary.py` then reused the frozen Hybrid
Top-20 cache, kept `A_prod` only, took the first 10 queries per frozen company
(50 cases), and generated numeric answers with DeepSeek after lexical ranking
and after Qwen3. Gold KPI values were loaded from parquet in a new scoring
module; ranking fingerprint sources were not edited. Plan freeze is local-only;
the run requires `--allow-remote`. Complete cases are atomically resumable with
at-least-once billing. This is not FinanceBench Phase 4.

The sealed canary is
`data/eval_rag/holdout/ledger_public_dev_e2e_canary_5x10.json`:

- 50 paired generations, 5 companies, A_prod Top-10 after the frozen Hybrid
  Top-20. Candidate embedding calls remained zero because the Qwen3-paired
  cache was reused.
- Numeric match (1% relative tolerance, including common scale factors) rose
  from 24% lexical to 32% Qwen3 (+8 points; 6 gains, 2 losses, 42 unchanged).
  That paired sign test is not significant at n=50; treat it as a directional
  canary, not a ranking-style sealed win.
- Abstain was high: 74% lexical, 62% Qwen3. Citation support is not a usable
  signal here (0% / 2%): the generator often emitted passage indices such as
  `"1"` instead of `chunk_id`.
- Eval latency is rerank plus generate, not production multi-agent latency.
  Lexical p50/p95 total 711 ms / 942 ms; Qwen3 1,029 ms / 1,369 ms. Mean
  generate is ~735 ms on both arms; Qwen3 adds ~351 ms mean rerank.
- All 50 Qwen3 reranks and 100 generations completed in 50 + 100 physical
  attempts, zero retries, zero live Qwen3 fallbacks, and zero generate errors.
  Therefore `primary_comparison_valid=true`. Injected Qwen3 failure still
  ranks with lexical and generates; that path was not observed live.

Production retrieval and reranker defaults remain unchanged.

`scripts/run_ledger_public_dev_e2e_failure_taxonomy.py` then classified those
50 Qwen3 generations locally against the frozen Top-20 texts actually shown
to the generator (4,000-character truncation). It did not call remote
providers. Gold digits are detected with the same 1% / scale-factor rule as
scoring. The sealed counts are
`data/eval_rag/holdout/ledger_public_dev_e2e_failure_taxonomy_5x10.json`:

- Qwen3 Hit@10 on this prefix was 33/50; Pool Hit@20 was 34/50. Ranking
  leftover is 1 case (`gold_in_pool_not_in_final_top10`).
- 15/50 numeric matches had the gold number in the final Top-10 context.
  There were **zero** `generation_abstain` and **zero** `generation_miss`
  cases: when the digits were in the prompt, DeepSeek extracted them.
- The remaining leaks are retrieval/packing: 15 `retrieval_pool_miss`, 16
  `evidence_gap_number_absent` (gold page in Top-10, KPI digits not in any
  gold-page chunk that reached the pool), 2 `evidence_gap_unselected_chunk`,
  and 1 unsupported match that must not be credited to generation.
- Locked recommendation: `section_parent_retrieval` as a **new eval index**,
  not a production cutover and not a prompt retune on these 50 questions.

The high canary abstain rate was therefore mostly missing evidence, not a
timid generator. Do not switch the default reranker on this basis.

`scripts/run_ledger_public_dev_parent_pack_probe.py` then packed existing
Qwen3 Top-10 pages locally (no re-embedding, no generation). It asks whether
the gold KPI digits would have been in the prompt if the retriever returned
parent pages instead of chunks. The sealed counts are
`data/eval_rag/holdout/ledger_public_dev_parent_pack_probe_5x10.json`:

- Chunk context contains the number in 16/50 cases.
- Replacing each retrieved chunk with its full source page lifts that to
  38/50. A ±1 page window reaches 44/50. The gold-page oracle is 46/50.
- All 16 `evidence_gap_number_absent` cases recover under full-page return.
  The 15 pool misses still have the digits on the gold page in the corpus;
  returning retrieved pages recovers only 3, and ±1 neighbors recover 9.
- Locked next implementation: **retrieve child, return parent page** on an
  eval collection. Do not re-generate on this same 50-question prefix. Do
  not change production chunking.

`scripts/run_ledger_public_dev_parent_pack_suffix.py` then applied the same
local packing check to the frozen remaining 40 queries per company (200
unseen cases; query identity
`7bd0906ea034b7c2a679957ac0ad82f1583934872f689c82385d5cc7a9aa9c33`). It reused
the Qwen3 A_prod ranks and candidate cache, with zero embeddings, Qwen3
calls, or generation. The sealed counts are
`data/eval_rag/holdout/ledger_public_dev_parent_pack_suffix_5x40.json`:

- Chunk context contains the number in 107/200 cases.
- Full retrieved-page return lifts that to 144/200. A ±1 page window reaches
  160/200. The gold-page oracle is 166/200.
- That held-out packing lift authorized a generate canary on this suffix,
  not a production chunker change and not a re-score of the original 50.

`scripts/run_ledger_public_dev_parent_page_e2e.py` then generated numeric
answers from the same frozen Qwen3 Top-10: candidate chunks (4,000-character
cap) versus eval-only parent pages (`retrieved_page_full`, no 4,000-character
truncation). Plan freeze is local-only; the run requires `--allow-remote`.
Complete cases are atomically resumable with at-least-once billing. This is
not FinanceBench Phase 4 and not a product-accuracy claim.

The sealed generate canary is
`data/eval_rag/holdout/ledger_public_dev_parent_page_e2e_5x40.json`:

- 200 paired generations, 5 companies, A_prod Top-10 after the frozen Hybrid
  Top-20. Candidate embedding calls and Qwen3 calls remained zero because
  ranks were reused.
- Numeric match rose from 51.0% chunk to 71.5% parent page (+20.5 points;
  41 gains, 0 losses, 159 unchanged). Abstain fell from 40% to 21%.
- Prompt size rose from 1.38M to 5.70M characters. Mean generate latency
  rose from 790 ms to 1,193 ms. Parent-page generation used 1,745,293
  tokens versus 435,100 on chunks.
- All 400 logical generations completed in 400 physical attempts with zero
  retries and zero generate errors. Therefore
  `primary_comparison_valid=true`.
- Packing recoverability on this suffix was 107→144; generate match was
  102→143. Those are related but not interchangeable.

`scripts/run_ledger_public_dev_parent_page_e2e_taxonomy.py` then classified
those 200 chunk and parent-page generations locally against the frozen
Hybrid Top-20 and Qwen3 identity. It did not call remote providers. The
sealed counts are
`data/eval_rag/holdout/ledger_public_dev_parent_page_e2e_taxonomy_5x40.json`:

- Parent numeric match 143 splits into 132 gold-page-supported matches and
  11 unsupported hits. Chunk is 98 supported plus 4 unsupported. Do not
  treat 71.5% as supported accuracy.
- Digits in the actual prompt stay 107 chunk / 144 parent, matching the
  packing probe. Returning the page removes `evidence_gap_unselected_chunk`
  (3→0) and most `evidence_gap_number_absent` (30→3).
- Parent leftover is recall: 45 `retrieval_pool_miss`, 2 ranking misses,
  3 gold pages that still lack the KPI digits, and 7 generation leftovers
  when the digits were already in the page.
- Locked recommendation: `section_parent_retrieval` as a **new eval
  index**, not a production cutover, not a prompt retune, and not a
  re-score of this 200.

`scripts/run_ledger_public_dev_section_parent.py` then indexed the five
frozen companies' pages as BM25 parent units (2,427 pages; one page =
one retrieval unit; `section_title=NOT_AVAILABLE`). It scored Pool
Hit@20 on the same 200 suffix queries against the frozen Hybrid chunk
pool. Local only: zero embeddings, Qwen3, or generation. The sealed
counts are
`data/eval_rag/holdout/ledger_public_dev_section_parent_bm25_5x40.json`:

- Hybrid chunk Pool Hit@20 is 148/200. Page-parent BM25 is 103/200
  (delta −45; 12 gains, 57 losses, 131 unchanged). Hit@10 on the page
  index is 66/200.
- Of the 45 taxonomy `retrieval_pool_miss` cases, page BM25 recovers
  11. That does not offset the 57 cases Hybrid already had and the
  page index lost.
- Locked recommendation: `do_not_embed_page_parent_index`. Do not start
  DashScope hybrid embeddings on pages. Keep **child-retrieve +
  parent-page return** on the existing Hybrid chunk first-stage.

This BM25 page recall is not answer accuracy. Production retrieval,
chunking, and the default lexical reranker remain unchanged. Keep
parent-page return eval-only.

The schema example is not a real evaluation set and must never produce a
score. Stop after this slice unless independently authored questions and
human-confirmed qrels have been frozen before inspecting retrieval ranks.

## Out of scope

- FinanceBench Phase 4 end-to-end metrics remain `NOT_RUN`. The LEDGER
  public-dev generation canary is a separate numeric KPI check, not Phase 4
  and not a product-accuracy claim
- Switching the default reranker model
- Multi-agent retrieval
- Re-opening pool-size A/B/C on test-100
- DashScope / hybrid embeddings on a page-parent eval index
- README product-accuracy claims
