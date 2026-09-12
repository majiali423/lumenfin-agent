# LumenFin Validation Commands

Supported local and release validation entrypoints. Live RC orchestration
lives in the sibling FinAgentBench repository.

## Environment

- Python **3.12** (CI pin)
- Install via `requirements-lock.txt` then `pip install -e . --no-deps`
- Set `FINAGENTBENCH_DIR` for product tests (sibling discovery is opt-in via `LUMENFIN_ALLOW_SIBLING_FAB=1`)

## 1. Minimal offline validation (solo, no evaluator)

```bash
python scripts/run_tests.py --fast
python scripts/run_tests.py --skip-joint
python scripts/run_portfolio_demo.py
```

Documentation checks use only Python's standard library:

```bash
python scripts/check_doc_links.py
python -m unittest tests.test_report_path_portability tests.test_version_consistency -v
```

Document-task evaluation (independent gold; does not consume holdout):

```bash
python -m unittest tests.test_document_tasks tests.test_tracing tests.test_document_task_baselines tests.test_eval_contract_policy
python scripts/run_document_task_eval.py --freeze-check
python scripts/run_document_task_eval.py --prepare-live
python scripts/run_document_task_eval.py --pilot --offline
```

`--pilot --offline` reports execution coverage and scorer behavior only.
LocalFallback gold-pass counts are diagnostics, not model accuracy, and must
not be copied into README. `--freeze-check` must fail until test gold is
reviewed. Authorized live B1/B2 used the **lexical + deterministic** retrieval
profile (not production DashScope/Qwen3) under
`LUMENFIN_EVAL_BUDGET_AUTH=pilot24-lexical-v1-960http-2026-09-10`.
`--confirm-budget` is not authorization. The hard cap is provider HTTP
requests including retries (960; currently 357 used). USD is an estimate
only. Do not reset the ledger or start another paid run unless newly
authorized. `fair_v4` resumed after a harness interrupt. Later product fixes
were not live-retested.


CI runs documentation contracts independently of the runtime jobs. Fast tests
gate the full offline regression, product v3 and both frozen FinRun lanes. All
jobs remain enabled for every push and pull request; documentation changes do
not bypass runtime tests. Dependency downloads are cached against the lockfile,
and a newer run supersedes an older run on the same branch or pull request.

On an offline regression failure, download `offline-ci-logs` for the complete
test output. Logging uses Bash `pipefail`, so `tee` cannot hide a failing test.

## 2. Joint product validation (explicit FinAgentBench checkout)

For the published baseline, use evaluator commit
`40f7599e408f317515583405cb90249b811179c0`. The independent full suite above
does not require the evaluator.

```bash
export FINAGENTBENCH_DIR=/absolute/path/finagentbench-demo
python scripts/run_tests.py --joint-only
```

From the evaluator checkout:

```bash
cd "$FINAGENTBENCH_DIR"
python -m unittest discover -s tests -v
python scripts/run_mutation_suite.py
python scripts/run_offline_demo.py
```

## 3. Cross-repository gate

CI uses absolute `LUMENFIN_ROOT` / `FINAGENTBENCH_DIR` (do not rely on a
neighbor folder being present):

```bash
export FINAGENTBENCH_DIR=/absolute/path/finagentbench-demo
python scripts/run_cross_repo_ci.py --profile ci --require-clean-lumenfin
```

Local equivalent from FinAgentBench:

```bash
cd "$FINAGENTBENCH_DIR"
export LUMENFIN_ROOT=/absolute/path/lumenfin-agent
python scripts/validate_cross_repo.py --profile ci
```

## 4. Live RC

```bash
cd ../finagentbench-demo
python scripts/run_rc_validation.py --help
python scripts/run_rc_validation.py --dry-run
python scripts/run_rc_validation.py
```

Do not run live RC until offline gates and dry-run pass. Distinguish provider
infrastructure failure from Agent failure in any report.

## 5. Mutation suite

```bash
cd ../finagentbench-demo
python scripts/run_mutation_suite.py
```

## Portable path helpers

```bash
python scripts/repo_paths.py
```

Discovers sibling repositories without hard-coded absolute paths.

## 6. Docker integration harnesses (manual)

Queue/worker multi-process:

```powershell
python scripts/run_queue_worker_integration.py
```

Provider resilience (deterministic stub + optional Docker dual-API):

```powershell
python scripts/validate_provider_resilience.py
python scripts/validate_provider_resilience_docker.py
```

Requires Docker Compose resources; see
[QUEUE_WORKER_INTEGRATION.md](QUEUE_WORKER_INTEGRATION.md) and
[PROVIDER_RESILIENCE.md](PROVIDER_RESILIENCE.md).

## 7. FinanceBench retrieval eval

Requires a local FinanceBench checkout (JSONL + PDFs). Raw files stay
gitignored. Remote embedding/rerank is blocked unless `--allow-remote` is set.

Exposed test-100 four-mode ablation was **recorded** 2026-08-16 as an
exploratory baseline (corpus scope, dirty worktree). Company-scope on the
same 100 questions was then recorded as a **post-hoc paired diagnostic**
(`outputs/financebench_eval_company/`). Neither is product accuracy.
Confirmation-50 (`--split confirmation` / `--split dev`) is **RECORDED**
(2026-08-16, tag `financebench-confirmation-v1`). At execution it was a
one-shot unseen set; it is now consumed/exposed. Page-level Hit@5 0.50,
Hit@10 0.62, MRR 0.2955, nDCG@10 0.3461. These are **not** product accuracy
and **not** end-to-end QA. Do not rerun or retune. Aggregate:
`data/eval_rag/financebench/confirmation_result.json`. Details:
[FINANCEBENCH_EVAL.md](FINANCEBENCH_EVAL.md).

Candidate-pool / Qwen3 A/B/C on exposed test-100 is **RECORDED**
(2026-08-17, tag `financebench-candidate-pool-ablation-v1`). It is post-hoc,
not held-out, and **not** a production change. Keep arm A in production.
Do not retune on test-100 and do not rerun the scoring directory. Aggregate:
`data/eval_rag/financebench/candidate_pool_ablation_result.json`.

Offline harness smoke (no confirmation-50 rerun, no remote providers).
`--split confirmation` and `--split dev` are the consumed confirmation set;
do **not** run them, including with `--limit 2`. Use unit tests / synthetic
fixtures, or the already exposed test split:

```powershell
python -m unittest `
  tests.test_financebench_loader `
  tests.test_financebench_split `
  tests.test_financebench_qrels `
  tests.test_financebench_metrics `
  tests.test_financebench_retrieval_eval `
  tests.test_financebench_frozen `
  tests.test_financebench_confirmation_result `
  tests.test_financebench_candidate_pool_ablation_result -v
python scripts/run_financebench_retrieval_eval.py --dataset-dir <checkout> --split test --mode bm25 --limit 2
```

Recorded confirmation-50 command (already executed; do not run again):

```powershell
python scripts/run_financebench_retrieval_eval.py `
  --dataset-dir data\external\financebench-src `
  --output-dir outputs\financebench_eval_confirmation `
  --split confirmation --mode hybrid-qwen3 --index-scope company `
  --embedding-provider dashscope --embedding-dimension 1024 --top-k 10 `
  --allow-remote `
  --frozen-config data\eval_rag\financebench\frozen_config.json `
  --confirm-held-out
```

Recorded corpus exploratory-baseline command (do not retune from it):

```powershell
python scripts/run_financebench_retrieval_eval.py `
  --dataset-dir data\external\financebench-src `
  --mode all --split test --allow-remote `
  --embedding-provider dashscope --index-scope corpus --keep-index
```

Recorded company-scope post-hoc command (do not retune from it):

```powershell
python scripts/run_financebench_retrieval_eval.py `
  --dataset-dir data\external\financebench-src `
  --output-dir outputs\financebench_eval_company `
  --mode all --split test --allow-remote `
  --embedding-provider dashscope --embedding-dimension 1024 `
  --index-scope company --keep-index --top-k 10
```

## 8. LEDGER public-dev (sealed / stopped)

LEDGER scores are a **public development canary**, not product accuracy and
not FinanceBench Phase 4. Do not open `public_holdout`. Do not embed a
page-parent index. Do not rescore the frozen 5×50 suffix.

Tracked aggregates: `data/eval_rag/holdout/ledger_public_dev_*.json`.
Chain provenance: `data/eval_rag/holdout/ledger_public_dev_chain_seal.json`
(hashes only). Annotated tag `ledger-public-dev-chain-v1` peels to
`ec4d9e40d45a536ec00cbdd8fbdadf6e051e4e8c`. Protocol:
[FINANCEBENCH_NEXT_PHASE.md](FINANCEBENCH_NEXT_PHASE.md).

Offline identity / unit checks only (do not re-run remote scoring or rewrite
the tracked manifest):

```powershell
python -m unittest tests.test_ledger_public_benchmark tests.test_ledger_section_parent tests.test_ledger_parent_page_e2e tests.test_ledger_public_dev_chain_seal -v
```

## 8b. Structured citation synthetic canary (offline)

Contract-only. Not product accuracy, RAG recall, FinanceBench, or LEDGER
benchmark. Refuses `public_holdout` and remote providers.

```powershell
python -m unittest tests.test_citation_alias tests.test_structured_citation_canary tests.test_structured_answer tests.test_claim_binding tests.test_finrun_export tests.test_ledger_e2e_canary -v
python scripts/run_structured_citation_canary.py --output-dir outputs/structured_citation_canary_v1
```

Slim tracked record:
[`../data/eval_rag/structured_citation_canary_result.json`](../data/eval_rag/structured_citation_canary_result.json)
(`config_hash` `6f85a617a16446afc17b940919bc57c10b397b588279466aa824e93e8536f2fa`;
not product accuracy).

## 8c. LEDGER structured-citation public/dev shadow (recorded)

Exposed public/dev `sealed_candidate_replay_shadow` is recorded. Not live
production retrieval, not held-out, not product accuracy, not a LEDGER
benchmark, not rc5. Execution gate passed; structured-citation quality
gate failed. Do not rerun or resume. Do not retune from this result.

Tracked ledger:
[`../data/eval_rag/ledger_structured_citation_shadow_result.json`](../data/eval_rag/ledger_structured_citation_shadow_result.json).
Frozen config:
[`../data/eval_rag/structured_citation_shadow_config.json`](../data/eval_rag/structured_citation_shadow_config.json).
Cache identity:
[`../data/eval_rag/structured_citation_shadow_cache_manifest.json`](../data/eval_rag/structured_citation_shadow_cache_manifest.json).
Raw outputs stay gitignored under
`outputs/ledger_structured_citation_shadow_v1/`. The raw JSON has no
`status` or `executed_at` field; the ledger uses
`seal_status=RECORDED_COMPLETE`. `execution_time` is inferred from
`summary.json` mtime and is not authoritative.
22/50 structured answers, 18/29 unknown citations, and 11
citation-validation failures. Recorded `supported_claims=0` is the raw
sealed fact, not a valid support rate (official scorer received empty
qrels). The “7 might be supported” count is a read-only mechanism
diagnosis, not an official repaired score. Do not rerun public/dev to
refresh numbers. Audit:
[`../data/eval_rag/ledger_structured_citation_shadow_audit.json`](../data/eval_rag/ledger_structured_citation_shadow_audit.json).
v3 authorized that one sealed shadow and cannot authorize a later
commit. Goal C hash `5b259515…` never executed (`preflight=0`,
`shadow=0`) and is superseded before preflight. V4 was never executed.
Hash `7db41564…` is the Goal A contract implementation identity only.
V5 was never executed and is retired before preflight. The tracked
execution ledger is the only grant source; unknown or missing hashes
default to deny. Do not run V5 preflight or shadow. Goal A was
validated by offline contract tests only. No repaired public/dev score
exists. rc5 must not claim reliable structured citations from this run.

```powershell
python -m unittest tests.test_citation_alias tests.test_ledger_structured_citation_shadow tests.test_ledger_structured_citation_shadow_execution tests.test_ledger_structured_citation_shadow_result tests.test_ledger_structured_citation_qrel_binding tests.test_structured_citation_canary tests.test_ledger_e2e_canary -v
```

## 8d. Synthetic remote alias-compliance canary (sealed)

Independent fictional suite. Not product accuracy, not LEDGER, not
FinanceBench, not holdout. Official preflight and remote ran once and
are consumed. Do not rerun the official CLI.

```powershell
python -m unittest tests.test_synthetic_alias_compliance tests.test_citation_alias tests.test_ledger_structured_citation_shadow_execution -v
```

Frozen config:
[`../data/eval_rag/synthetic_alias_compliance_config.json`](../data/eval_rag/synthetic_alias_compliance_config.json).
Dataset:
[`../data/eval_rag/synthetic_alias_compliance_cases.json`](../data/eval_rag/synthetic_alias_compliance_cases.json).
Authorization (closed):
[`../data/eval_rag/synthetic_alias_compliance_authorization.json`](../data/eval_rag/synthetic_alias_compliance_authorization.json).
Sealed ledger:
[`../data/eval_rag/synthetic_alias_compliance_result.json`](../data/eval_rag/synthetic_alias_compliance_result.json).
Official commands stay refused (`ONE_SHOT_CONSUMED`).

```powershell
python scripts/run_synthetic_alias_compliance_canary.py --preflight-only
python scripts/run_synthetic_alias_compliance_canary.py --confirm-synthetic-alias-compliance --allow-remote
```

## 8e. LEDGER public_holdout E2E

v1 remains the historical blocked ledger (exit `2` /
`PREFLIGHT_BLOCKED`). v2 is sealed and consumed — do not re-run remote
consume. Governance tests only:

```powershell
python -m unittest tests.test_ledger_public_holdout_e2e tests.test_ledger_public_holdout_e2e_v2 tests.test_ledger_public_holdout_e2e_v2_result -v
python scripts/run_ledger_public_holdout_e2e.py --preflight-only
```

Tracked ledgers:
[`../data/eval_rag/ledger_public_holdout_e2e_result.json`](../data/eval_rag/ledger_public_holdout_e2e_result.json)
(v1 blocked),
[`../data/eval_rag/ledger_public_holdout_e2e_result_v2.json`](../data/eval_rag/ledger_public_holdout_e2e_result_v2.json)
(v2 sealed). Protocol:
[`LEDGER_PUBLIC_HOLDOUT_E2E.md`](LEDGER_PUBLIC_HOLDOUT_E2E.md).
Not a general product accuracy claim.
