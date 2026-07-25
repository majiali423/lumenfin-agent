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
