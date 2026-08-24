# LEDGER public_holdout E2E

Attempted one-shot **LEDGER public_holdout held-out end-to-end verified
task success** against LumenFin `v0.1.0-rc.5`
(`31e8680aa89636f1fd897d7aa5ed7ca86317bd73`).

## Status

**BLOCKED_BEFORE_PREFLIGHT.** `holdout_consumed=false`. No official
preflight. No remote run. No evaluation tag.

Stop reason: `no_compatible_prebuilt_index`.

## Why it stopped

rc5 production retrieval is DashScope hybrid + Qwen3 arm A
(`final_k=10`). A true E2E run must retrieve live against a compatible
index, not replay the consumed public/dev candidate cache.

`public_holdout` is company-disjoint (2,384 questions / 26 companies).
Every existing LEDGER index and candidate cache is `public_dev` only.
No parquet snapshot or holdout-company DashScope index is present.

Building that index now would require document embedding. The frozen
budget is `document_reembedding_calls=0`. The public/dev analog is
1,601 document embedding calls for five companies. This task will not
spend that cost or silently downgrade to candidate replay.

Holdout questions, gold values, and qrel bodies were not opened.

## Frozen contract

- Claim name only: LEDGER public_holdout held-out end-to-end verified
  task success
- `dataset_specific=true`, `held_out=true`, `single_use=true`,
  `general_product_accuracy_claim=false`
- Primary: `verified_e2e_success = answer_correct AND
  structured_contract_valid AND citation_supported AND provider_success`
- Strict rate uses all selected cases; empty qrels are `NOT_EVALUABLE`,
  not zero
- No LLM judge as the primary metric
- Config hash
  `be89d18d77da01e0f2e3938ecf6c5235887e81720f5defc3f1ed55a45644674e`

Tracked files:

- [`../data/eval_rag/ledger_public_holdout_e2e_contract.json`](../data/eval_rag/ledger_public_holdout_e2e_contract.json)
- [`../data/eval_rag/ledger_public_holdout_e2e_authorization.json`](../data/eval_rag/ledger_public_holdout_e2e_authorization.json)
- [`../data/eval_rag/ledger_public_holdout_e2e_result.json`](../data/eval_rag/ledger_public_holdout_e2e_result.json)
