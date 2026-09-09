# LumenFin

**English** | [中文](README.zh-CN.md)

**Financial research answers with numbers you can trace back to the filing.**
Upload a report, ask a question, and inspect the source page, fiscal period and
calculation behind the answer. Missing evidence produces an explicit data gap.

[![CI](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml)

[Try the demo](#try-it-locally) · [Architecture](docs/ARCHITECTURE.md) ·
[Evidence](docs/EVIDENCE_INDEX.md) · [Documentation](docs/README.md)

## What to try

| Scenario | Expected behavior |
|---|---|
| Upload the NVIDIA FY2025 excerpt | Operating income **81.453 billion USD**, linked to page 1 |
| Upload the narrative-only fixture | Explain the missing financial data; no sample-data backfill |
| Read a multi-page filing | Keep each number's original page and fiscal period; an unspecified period stays unknown |
| Alter a number, fiscal year or citation in an exported report | FinAgentBench v3 reports the unsupported assertion |

These are reproducible fixture scenarios. The offline demo uses local model
fallback and deterministic providers, so it does not measure live-model answer
quality.

## Try it locally

Use **Python 3.12** and a source checkout. Dependencies must be downloaded once;
running the offline demo requires no API keys.

```bash
git clone https://github.com/majiali423/lumenfin-agent.git
cd lumenfin-agent
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` in a POSIX shell. Reuse an existing environment only
if it matches the lockfile.

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip check
python scripts/start_offline_demo_api.py
```

Open **http://127.0.0.1:8000/**, upload
[`nvda_fy2025_10k_excerpt.pdf`](tests/fixtures/sec/derived/nvda_fy2025_10k_excerpt.pdf),
and ask:

> Using uploaded files only, what is NVIDIA FY2025 operating income from the filing facts?

Inspect the answer and its page citation. Refreshing the `?job=` URL restores
the same job. Then try
[`nvda_narrative_only.txt`](tests/fixtures/sec/minimal/nvda_narrative_only.txt)
with the same question to see the missing-data path.

A terminal-only demo is available with
`python scripts/run_portfolio_demo.py`; its A/B/C cases use built-in fixtures.
For a walkthrough, see the [demo guide](docs/DEMO_GUIDE.md) and
[90-second presentation script](docs/AUTUMN_RECRUITING_DEMO_SCRIPT.md).

## How it works

```text
Question + documents → task plan → retrieval → calculation when required
                     → critic / bounded retry → claim–evidence binding
                     → answer + FinRun → FinAgentBench evaluation
```

- **LangGraph** coordinates specialist nodes sharing one state.
- **FastAPI and Redis** provide queued jobs, polling, retry and clarification.
- **PostgreSQL** stores jobs, checkpoints and document metadata; **Milvus**
  supports hybrid retrieval. SQLite and Milvus Lite support local tests.
- **Deterministic financial calculations** keep arithmetic outside the LLM.
- **Field-level provenance** binds values to their company, unit, period and
  source page. TaskSpec lets evidence-backed risk questions proceed without
  unrelated financial ratios.

[Architecture and diagrams](docs/ARCHITECTURE.md) explain the control flow.
[Design decisions](docs/architecture_decisions.md) cover the trade-offs.

## LumenFin and FinAgentBench

LumenFin produces answers; [FinAgentBench](https://github.com/majiali423/finagentbench-demo)
replays the exported **FinRun 1.0** artifact. Its opt-in **scoring v3** checks
visible financial assertions in prose and tables, including the claims ledger.

Separate packages make the interface and evaluator changes reviewable. Both
repositories are maintained by the same author; passing these contract tests
is not an independent benchmark of real-world answer accuracy.

## Validation and versions

The **2026-09-09 published baseline** passed all five LumenFin CI jobs:
Fast, Offline regression, Product quality v3 and both frozen FinRun contracts.
[Inspect the run](https://github.com/majiali423/lumenfin-agent/actions/runs/34329999879).

| Role | Version / evidence |
|---|---|
| Validated product source | [`60e4ed6`](https://github.com/majiali423/lumenfin-agent/commit/60e4ed6a06d7afb6fce907413d2359cbf89eae44) |
| Product v3 evaluator | [`40f7599`](https://github.com/majiali423/finagentbench-demo/commit/40f7599e408f317515583405cb90249b811179c0), pinned by `FINAGENTBENCH_PRODUCT_REF` |
| Historical package/tag | LumenFin `0.1.0rc5` / `v0.1.0-rc.5`; the tag does not include subsequent source fixes |
| Frozen evaluator compatibility | FinAgentBench `v0.1.0-rc.3` and `v0.1.0-rc.4` |
| Current branch status | CI badge above; the linked baseline is an immutable historical run |

Main-project checks need no evaluator:

```bash
python scripts/run_tests.py --fast
python scripts/run_tests.py --skip-joint
python scripts/check_doc_links.py
```

Joint tests use an explicitly installed FinAgentBench checkout and
`python scripts/run_tests.py --joint-only`.
[Reproduction instructions](docs/REPRODUCIBILITY.md) include the pinned evaluator
and both platform setup paths. [Validation commands](docs/VALIDATION_COMMANDS.md)
separate offline, joint and optional infrastructure checks.

## Boundaries

- Queue delivery is **at-least-once**, with leases and execution fencing.
- Clarification uses application checkpoints plus an in-process LangGraph
  checkpointer; general cross-process node replay is not certified.
- Bounded repair is implemented and **off by default** after the dev-set
  ablation. No production SLA or broad live-model accuracy is claimed.
- The UI ships through source checkout or Docker; a wheel alone excludes
  `static/`. Third-party dependencies retain their own licenses.
- Historical FinanceBench / LEDGER results remain sealed and are indexed
  separately from current product and contract tests.

[Operational limitations](docs/PRODUCTION_LIMITATIONS.md) ·
[Tenant boundaries](docs/MULTI_TENANCY_BOUNDARY.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md)

## Explore the implementation

| Path | Responsibility |
|---|---|
| `src/lumenfin/agents/`, `graph.py` | Planning, retrieval, calculation, validation and synthesis |
| `src/lumenfin/claims/`, `finrun.py` | Claim binding and evaluation export |
| `src/lumenfin/rag/` | Document indexing, page identity and retrieval |
| `src/lumenfin/api/`, `worker.py` | API and job execution |
| `tests/` | Offline, fault-injection and product regressions |
| `docs/` | Architecture, operation and evidence |

For engineering examples, see [reliability decisions](docs/architecture_decisions.md#11-current-reliability-examples).
The full navigation lives in the [documentation index](docs/README.md).

LumenFin-owned source is [MIT licensed](LICENSE). Financial output is for
research and demonstration and requires human review.
