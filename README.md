# LumenFin

**Evidence-grounded multi-agent financial diligence** — LangGraph orchestration,
structured grounding, claim→evidence binding, and an independent FinAgentBench
reliability gate.

[![CI](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml)

Release candidate: **`0.1.0rc2`** | FinRun schema: `1.0` | FinAgentBench pin:
`v0.1.0-rc.1`

LumenFin is a **portfolio release candidate** validated under controlled
multi-process and deterministic fault-injection conditions. These results are
**not** a certification of unrestricted production readiness.

[Docs](docs/README.md) · [Architecture](docs/FINAL_ARCHITECTURE.md) ·
[Limitations](docs/PRODUCTION_LIMITATIONS.md) · [Demo](docs/DEMO_GUIDE.md) ·
[Release report](docs/PORTFOLIO_RELEASE_REPORT.md)

---

## 1. One-line positioning

Turn a research query (and optional filings) into a **checkable** diligence
report that fails closed when data is missing — then prove reliability with an
external FinRun gate.

## 2. Core problem

Typical financial RAG demos often:

- promote peers out of a 10-K body into issuer scope;
- invent ratios without structured inputs;
- emit fluent claims without citations;
- look “correct” when only the final paragraph is judged.

LumenFin makes those failure modes **visible** and **fail-closed**.

## 3. System architecture (two paths)

### Agent / evidence path

```text
Query / PDF
→ LangGraph orchestration
→ SEC / Yahoo / hybrid RAG
→ structured financial grounding
→ claim builder
→ evidence binder
→ verified-only synthesis
→ FinRun
→ FinAgentBench
```

### Runtime / infrastructure path

```text
Client
→ API instance(s)
→ PostgreSQL checkpoint / job / RAG metadata
→ Redis reliable queues (pending / processing / DLQ)
→ index worker(s)
→ Milvus Server
→ provider resilience layer
```

Details: [docs/FINAL_ARCHITECTURE.md](docs/FINAL_ARCHITECTURE.md)

## 4. Trusted financial analysis chain

- Issuer-only entity resolution and multi-company isolation
- SEC companyfacts / Yahoo fundamentals with provenance
- Hybrid RAG (keyword + vector) with page citations when uploads exist
- Typed claims bound to evidence before synthesis
- `incomplete_data` when fundamentals are absent (no forged numerics)

## 5. Engineering reliability

| Concern | Design |
|---------|--------|
| Persistence | PostgreSQL-first (SQLite only for `test` / explicit dev opt-in) |
| Queues | Redis pending → processing → dead-letter; reclaim without manual redelivery |
| Workers | Index lease + attempt fencing; kill → automatic reclaim |
| Providers | Single retry owner, deadline, Retry-After, jitter, degraded fallback, per-process bulkhead |
| Tenancy | RAG data-plane tenant-aware logical isolation ([boundary](docs/MULTI_TENANCY_BOUNDARY.md)) |

## 6. Validated results (separate gates)

Do **not** merge these into one “accuracy” number.

| Gate | What it measures | Result |
|------|------------------|--------|
| **Unit regression** | Offline Python tests | **453 passed, 1 skipped** |
| **Infrastructure integration** | Phase 3.2B multi-process Docker | **PASS** (`20260804T095357Z`) |
| Worker-kill manual redelivery | Must stay false | **false** |
| Tenant leakage | Cross-tenant RAG read | **0** |
| Orphan chunks / vectors | Index compensation | **0 / 0** |
| **Provider fault validation** | Phase 3.3A + Docker dual-API | **PASS** (`docker_20260804T100817Z`) |
| Dual-API logical / physical calls | Stub reconciliation | **20 / 25** |
| Provider unexpected failures | Scenario G | **0** |
| **Benchmark reliability** | FinAgentBench completed-case mean | **92.97** (informational) |
| Mutation detection | Wrong entity/number/citation/risk | **4/4** |

Evidence: [PHASE32B](docs/PHASE32B_INTEGRATION_REPORT.md) ·
[PHASE33A](docs/PHASE33A_PROVIDER_RESILIENCE_REPORT.md) ·
[RC reliability](reports/current/LumenFin_RC_Final_Reliability_Report.md) ·
[Compatibility](reports/current/Joint_Compatibility_Report.md)

## 7. Three demo narratives

| Demo | Story |
|------|--------|
| **A** Trusted normal analysis | Issuer scope, grounded claims, citations, FinRun export |
| **B** Isolation & error detection | Multi-company isolation; mutation 4/4; tenant leakage 0 |
| **C** Fail-closed | Missing fundamentals → `incomplete_data`; no forged numerics |

```powershell
python scripts/run_portfolio_demo.py
```

Interview walkthrough: [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)

Optional Docker recovery story (not in default demo): worker A killed →
automatic reclaim → worker B attempt=2 → ready
([Phase 3.2B](docs/PHASE32B_INTEGRATION_REPORT.md)).

## 8. Quick start

Supported CI Python: **3.12**. Prefer the lockfile path.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
copy .env.example .env
.\.venv\Scripts\python scripts\run_tests.py
.\.venv\Scripts\python scripts\run_portfolio_demo.py
```

Live providers need keys in `.env` (never commit them). See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

FinAgentBench (optional sibling / published tag):
[majiali423/finagentbench-demo](https://github.com/majiali423/finagentbench-demo)

## 9. Boundaries

- Portfolio RC / controlled deployment candidate — **not** unrestricted production-ready
- At-least-once queues — **not** exactly-once
- Per-process bulkhead — **not** cross-process global rate limit
- Live provider smoke: **skipped** in current release evidence
- Not investment advice; human review required
- PyMuPDF license limits public image redistribution

Full text: [docs/PRODUCTION_LIMITATIONS.md](docs/PRODUCTION_LIMITATIONS.md)

## 10. Documentation map

| Doc | Purpose |
|-----|---------|
| [docs/README.md](docs/README.md) | Doc index |
| [docs/FINAL_ARCHITECTURE.md](docs/FINAL_ARCHITECTURE.md) | Agent + runtime architecture |
| [docs/MULTI_TENANCY_BOUNDARY.md](docs/MULTI_TENANCY_BOUNDARY.md) | Tenant isolation scope |
| [docs/PORTFOLIO_RELEASE_REPORT.md](docs/PORTFOLIO_RELEASE_REPORT.md) | Freeze evidence |
| [CHANGELOG.md](CHANGELOG.md) | Version history |
| [docs/VALIDATION_COMMANDS.md](docs/VALIDATION_COMMANDS.md) | Supported commands |

## Repository layout

```text
src/lumenfin/     Agent runtime, grounding, claims, FinRun, RAG, providers
tests/            Offline regression
scripts/          Tests, demos, Phase 3.2B/3.3A harnesses
docs/             Architecture and release docs
reports/current/  Authoritative RC evidence packs
```

## License / disclaimer

No public open-source license has been selected. Research output is for
engineering evaluation only and is **not investment advice**.
Third-party notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
