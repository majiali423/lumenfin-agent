# Reproducibility

Supported **source-candidate** environment: Python 3.12, LumenFin `0.1.0rc5`
(intended tag `v0.1.0-rc.5` — **not created**). Latest published LumenFin
release remains `0.1.0rc4` / `v0.1.0-rc.4`.

FinAgentBench versioning (do not conflate):

| Role | Version |
|------|---------|
| LumenFin CI **authoritative frozen pin** | FinAgentBench tag `v0.1.0-rc.3` / package `0.1.0rc3` |
| LumenFin CI **latest published compatibility** | FinAgentBench tag `v0.1.0-rc.4` / package `0.1.0rc4` |
| Current FinAgentBench **package tag** | `v0.1.0-rc.4` / `0.1.0rc4` (pins producer LumenFin `v0.1.0-rc.3`) |
| FinRun schema | `1.0` |

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
- DashScope workspace-compatible Qwen3 rerank endpoint when configured
- SEC and Yahoo/market-provider network access
- operator-owned `SEC_USER_AGENT` outside dev/test
- RC PDF/HTML fixtures in the sibling LumenFin repository

Quota, authentication, model removal and network errors are **infrastructure
failures**. They must not be recorded as Agent-quality passes.

## Docker

```bash
cd lumenfin-agent
docker build -t lumenfin-agent:0.1.0rc5 .
```

When the default PyPI route is slow, select a trusted mirror for that build
without changing the project default:

```bash
docker build \
  --build-arg PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -t lumenfin-agent:0.1.0rc5 .
```

Do not include credentials in `PIP_INDEX_URL`, because Docker build arguments
can appear in image build metadata.

The Dockerfile installs `requirements-lock.txt` and the current project package.
Application processes run as the unprivileged `lumenfin` user with fixed
UID/GID `10001`. On native Linux, prepare the bind-mounted write directories
before starting Compose:

```bash
mkdir -p outputs uploads
sudo chown -R 10001:10001 outputs uploads
```

Docker Desktop handles these bind mounts without the Linux ownership step.
Local build verification still depends on Docker being available; CI and local
Python validation do not prove container runtime health by themselves.

## Release order

1. Run both repositories' full offline, mutation, and cross-repository gates.
2. Run the approved live RC profile and classify infrastructure failures
   separately from Agent-quality failures.
3. Freeze release reports with the observed commits and test counts.
4. Create fresh commits in each repository only after human diff review.
5. Inspect existing remote tags before selecting new immutable RC tags; never
   move or reuse an existing tag.
6. Push, tag, and create releases only with explicit approval.

Do not tag a dirty working tree.
