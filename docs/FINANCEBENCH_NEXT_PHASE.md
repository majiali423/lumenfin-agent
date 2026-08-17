---
title: FinanceBench next phase
---

# Next phase: reranker, page diversity, section/parent retrieval

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
| `R_page` | collapse to unique pages before Qwen3; rerank page representatives | same pool size as A unless later authorized |
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
- Report `unique_pages_top10` on every run (this metric is currently
  unavailable because section metadata is missing; page uniqueness *can* be
  computed today from `page`).

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

Stop after the slice unless a holdout file is frozen.

## Out of scope

- End-to-end answer metrics (FinanceBench Phase 4 remains `NOT_RUN`)
- Switching the default reranker model
- Multi-agent retrieval
- Re-opening pool-size A/B/C on test-100
- README product-accuracy claims
