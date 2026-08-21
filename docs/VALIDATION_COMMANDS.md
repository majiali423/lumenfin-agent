# LumenFin Validation Commands

Supported local and release validation entrypoints. Live RC orchestration
lives in the sibling FinAgentBench repository.

## Environment

- Python **3.12** (CI pin)
- Install via `requirements-lock.txt` then `pip install -e . --no-deps`
- Sibling layout or `LUMENFIN_ROOT` / `FINAGENTBENCH_DIR`

## 1. Minimal offline validation

```bash
python scripts/run_tests.py
```

## 2. Full offline validation

```bash
python scripts/run_tests.py
# concurrency / HITL are included in the unit suite
```

From sibling FinAgentBench:

```bash
cd ../finagentbench-demo
python -m unittest discover -s tests -v
python scripts/run_mutation_suite.py
python scripts/run_offline_demo.py
python -m unittest tests.test_rc_runner_import -v
```

## 3. Cross-repository gate

CI uses the LumenFin orchestrator with absolute paths (sibling layout):

```bash
python scripts/run_cross_repo_ci.py --profile ci --require-clean-lumenfin
```

Local equivalent still available from FinAgentBench:

```bash
cd ../finagentbench-demo
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
python -m unittest tests.test_structured_citation_canary tests.test_structured_answer tests.test_claim_binding tests.test_finrun_export tests.test_ledger_e2e_canary -v
python scripts/run_structured_citation_canary.py --output-dir outputs/structured_citation_canary_v1
```

Slim tracked record:
[`../data/eval_rag/structured_citation_canary_result.json`](../data/eval_rag/structured_citation_canary_result.json)
(`config_hash` `6f85a617a16446afc17b940919bc57c10b397b588279466aa824e93e8536f2fa`;
not product accuracy).

## 8c. LEDGER structured-citation public/dev shadow (tools only)

Exposed public/dev shadow harness. Not held-out, not product accuracy, not a
LEDGER benchmark, not rc5. Refuses `public_holdout`. Do **not** run official
preflight or `--allow-remote` in this stage.

Frozen config:
[`../data/eval_rag/structured_citation_shadow_config.json`](../data/eval_rag/structured_citation_shadow_config.json).

```powershell
python -m unittest tests.test_ledger_structured_citation_shadow tests.test_structured_citation_canary tests.test_ledger_e2e_canary -v
```

Future official CLI (do not execute yet):

```powershell
python scripts/run_ledger_structured_citation_shadow.py `
  --split public-dev `
  --frozen-config data/eval_rag/structured_citation_shadow_config.json `
  --confirm-exposed-shadow `
  --preflight-only
```
