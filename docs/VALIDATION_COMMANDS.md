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

## 7. FinanceBench retrieval eval (optional, not a release gate)

Synthetic 4/5/10 RAG cases remain the offline RAG gates. FinanceBench is an
external accuracy harness; see [FINANCEBENCH_EVAL.md](FINANCEBENCH_EVAL.md).

```bash
python scripts/prepare_financebench_eval.py
python scripts/run_financebench_retrieval_eval.py --mode bm25 --split test --limit 2 --index-scope selected --embedding-provider deterministic
# Remote four-mode held-out run (explicit opt-in only):
# python scripts/run_financebench_retrieval_eval.py --mode all --split test --allow-remote --embedding-provider dashscope --fetch-pdfs
```
