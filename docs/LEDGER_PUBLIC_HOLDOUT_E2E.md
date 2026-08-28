# LEDGER public_holdout E2E

One-shot **LEDGER public_holdout held-out end-to-end verified
task success** against LumenFin `v0.1.0-rc.5`
(`31e8680aa89636f1fd897d7aa5ed7ca86317bd73`).

This is **dataset-specific**, **single-use**, and **not** a general product
accuracy claim. Do not mix it with FinAgentBench fixture scores or with the
webpage Run Manifest Evaluator.

## Status

### v1 (historical)

**BLOCKED_BEFORE_PREFLIGHT.** `holdout_consumed=false`. Stop reason:
`no_compatible_prebuilt_index`. Do not overwrite:

- [`../data/eval_rag/ledger_public_holdout_e2e_contract.json`](../data/eval_rag/ledger_public_holdout_e2e_contract.json)
- [`../data/eval_rag/ledger_public_holdout_e2e_authorization.json`](../data/eval_rag/ledger_public_holdout_e2e_authorization.json)
- [`../data/eval_rag/ledger_public_holdout_e2e_result.json`](../data/eval_rag/ledger_public_holdout_e2e_result.json)

### v2 (official, consumed)

**SEALED.** Single-use held-out E2E completed after the compatible rc5
DashScope index sealed. Live hybrid retrieve + Qwen3 arm A + DeepSeek
citation-alias generation. No public_dev candidate cache.
`document_reembedding_calls=0`.

| Field | Value |
| --- | --- |
| Cases (denominator) | **100** (all selected cases) |
| Strict verified E2E success | 35 / 100 = **0.35** |
| Wilson 95% CI | [0.264, 0.447] |
| `answer_correct` | 37 / 100 |
| `structured_contract_valid` | 47 / 100 |
| `citation_supported` | 40 / 100 |
| `provider_success` | 100 / 100 |
| `abstain` | **54 / 100** (current main gap) |
| Config hash | `233605b6…de01d` |
| Holdout consumed | **true** (do not re-open, retune, or rerun) |

Tracked v2 files:

- [`../data/eval_rag/ledger_public_holdout_e2e_selection_v2.json`](../data/eval_rag/ledger_public_holdout_e2e_selection_v2.json)
- [`../data/eval_rag/ledger_public_holdout_e2e_contract_v2.json`](../data/eval_rag/ledger_public_holdout_e2e_contract_v2.json)
- [`../data/eval_rag/ledger_public_holdout_e2e_authorization_v2.json`](../data/eval_rag/ledger_public_holdout_e2e_authorization_v2.json)
- [`../data/eval_rag/ledger_public_holdout_e2e_result_v2.json`](../data/eval_rag/ledger_public_holdout_e2e_result_v2.json)

Raw outputs stay gitignored under `outputs/ledger_public_holdout_e2e_v2/`.
The result seal records size and SHA-256 for those files. CI without those
bytes still checks the tracked ledger; it does **not** claim to have
recomputed the original bytes.

Index seal: [`LEDGER_PUBLIC_HOLDOUT_INDEX.md`](LEDGER_PUBLIC_HOLDOUT_INDEX.md).

## Strict success

Primary metric:

`verified_e2e_success = answer_correct AND structured_contract_valid AND citation_supported AND provider_success`

The denominator is **all 100 selected cases**, not the non-abstain subset.
Empty qrels are `NOT_EVALUABLE`, not zero. No LLM judge as the primary
metric. Fifty-four abstains are the current main failure mode; they are
unsuccessful under the strict definition.

## Claims boundary

- `dataset_specific=true`, `held_out=true`, `single_use=true`
- `general_product_accuracy_claim=false`
- FinAgentBench is an **offline contract / regression gate** on exported
  FinRun fixtures. It is not this holdout and not product accuracy.
- The webpage Run Manifest **Evaluator** is a **local lightweight check** of
  the current analysis run. It is not this holdout and not product accuracy.

## Governance

Holdout is consumed (`holdout_consumed=true`, `retuning_forbidden=true`,
`second_fresh_run_forbidden=true`). Official CLI is default-deny: a second
fresh run, `--resume`, a different output directory, or an authorizing
environment variable must fail before credentials, holdout text, providers,
directory creation, or remote requests.

```text
python -m unittest tests.test_ledger_public_holdout_e2e tests.test_ledger_public_holdout_e2e_v2 tests.test_ledger_public_holdout_e2e_v2_result -v
```
