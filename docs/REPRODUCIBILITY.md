# Reproducibility

Supported release environment: Python 3.12, LumenFin `0.1.0rc1`,
FinAgentBench `0.1.0rc1`, FinRun schema `1.0`.

Clone repositories as siblings:

```text
workspace/
  lumenfin-agent/
  finagentbench-demo/
```

Environment variables may replace sibling discovery:
`LUMENFIN_ROOT` and `FINAGENTBENCH_DIR`.

## Install

```bash
cd lumenfin-agent
python -m venv .venv
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps

cd ../finagentbench-demo
python -m pip install -e .
```

## Offline Quick Validation

```bash
cd lumenfin-agent
python scripts/run_tests.py

cd ../finagentbench-demo
python -m unittest discover -s tests -v
python scripts/run_mutation_suite.py
python scripts/validate_cross_repo.py --profile ci
python scripts/run_offline_demo.py
```

These commands require no real API key. They validate LumenFin, the FinRun
contract, the deterministic benchmark gate and all four mutations.

## Live RC Validation

Configure `.env` using [CONFIGURATION.md](CONFIGURATION.md), then:

```bash
cd finagentbench-demo
python scripts/run_rc_validation.py
```

Live validation requires:

- DeepSeek API/model availability
- DashScope embedding API when configured
- SEC and Yahoo/market-provider network access
- operator-owned `SEC_USER_AGENT` outside dev/test
- RC PDF/HTML fixtures in the sibling LumenFin repository

Quota, authentication, model removal and network errors are **infrastructure
failures**. They must not be recorded as Agent-quality passes.

## Docker

```bash
cd lumenfin-agent
docker build -t lumenfin:0.1.0rc1 .
```

The Dockerfile installs `requirements-lock.txt`. Local build verification still
depends on Docker being available; CI and local Python validation do not prove
container runtime health by themselves.

## Release order

1. Commit/tag FinAgentBench `v0.1.0-rc.1`.
2. Verify its mutation and unit workflows.
3. Commit LumenFin with CI pinned to that tag.
4. Run LumenFin cross-repository and live RC gates.
5. Tag LumenFin `v0.1.0-rc.1`.

Do not tag a dirty working tree.
