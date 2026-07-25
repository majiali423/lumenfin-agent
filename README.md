# LumenFin

**A Trustworthy Financial Research Agent**

LumenFin turns a research query and optional SEC filings into an evidence-backed
diligence report. It combines issuer-only entity resolution, SEC / market
financial grounding, claim → evidence binding and an independent FinAgentBench
reliability gate.

Release candidate: `0.1.0rc1` | FinRun schema: `1.0` | FinAgentBench pin:
`v0.1.0-rc.1`
Project status: **Release Candidate / Internal Portfolio Release**

[Docs index](docs/README.md) | [Architecture](docs/FINAL_ARCHITECTURE.md) |
[Limitations](docs/PRODUCTION_LIMITATIONS.md) |
[Demo guide](docs/DEMO_GUIDE.md)

---

## Why this project

Typical financial RAG demos often:

- pull peer companies out of a 10-K body;
- compute ratios from unsupported numbers;
- emit fluent claims without citations;
- invent precision when data is missing;
- look correct when only the final answer is judged.

LumenFin is built to make those failure modes visible and fail closed.

## Key capabilities

- SEC 10-K / Company Facts and live market snapshots
- Issuer-only entity resolution and multi-company isolation
- Hybrid RAG with structured financial facts
- Verified claim objects bound to evidence before synthesis
- Fail-closed `incomplete_data` when fundamentals are absent
- Request-scoped runtime (concurrent issuer isolation)
- FinRun export evaluated by FinAgentBench

## Architecture

```text
Query
  → LangGraph orchestration
  → Retrieval / tools / hybrid RAG
  → Issuer financial grounding
  → Claim binder
  → Report synthesizer
  → FinRun export → FinAgentBench
```

```mermaid
flowchart LR
  Q[Query] --> O[Orchestration]
  O --> R[Retrieval]
  R --> G[Financial Grounding]
  G --> C[Claim + Evidence]
  C --> S[Synthesizer]
  S --> F[FinRun]
  F --> B[FinAgentBench]
```

Details: [docs/FINAL_ARCHITECTURE.md](docs/FINAL_ARCHITECTURE.md)

## Reliability results

Observed on the current controlled RC pack (not a universal accuracy claim):

| Gate | Result |
|------|--------|
| Live RC pack | 8/8 PASS |
| Completed-case FAB mean | 92.97 |
| Entity / numeric / evidence floors | 100 / 100 / 100 |
| Mutation detection | 4/4 |
| Offline unit tests | 267 PASS, 1 skipped |

Evidence: [reports/current/LumenFin_RC_Final_Reliability_Report.md](reports/current/LumenFin_RC_Final_Reliability_Report.md) |
[reports/current/Joint_Compatibility_Report.md](reports/current/Joint_Compatibility_Report.md)

## Quick start

Supported environment: **Python 3.12** (CI). Local 3.11 may work but is not
the release pin. Prefer the lockfile install path below.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
copy .env.example .env
.\.venv\Scripts\python scripts\run_tests.py
```

Offline reliability demo (sibling FinAgentBench repo):

```powershell
cd ..\finagentbench-demo
python scripts\run_offline_demo.py
python scripts\validate_cross_repo.py --profile ci
```

Live configuration needs `DEEPSEEK_API_KEY`, optional DashScope embedding key,
and `SEC_USER_AGENT` outside `dev`/`test`. See
[docs/CONFIGURATION.md](docs/CONFIGURATION.md).

Supported validation commands:
[docs/VALIDATION_COMMANDS.md](docs/VALIDATION_COMMANDS.md).

Full live RC pack (requires live providers; do not confuse infra failure with
Agent failure):

```powershell
cd ..\finagentbench-demo
python scripts\run_rc_validation.py --help
python scripts\run_rc_validation.py --dry-run
python scripts\run_rc_validation.py
```

## Example workflow: NVIDIA 10-K

```text
Query: Analyze NVIDIA using the uploaded FY2025 10-K
  → companies == ["NVIDIA"]
  → SEC grounding fills AST-computable fundamentals
  → verified claims with evidence
  → report citations / provenance
  → FinAgentBench issuer gate
```

Interview demos A/B/C: [docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md)

## Repository structure

```text
src/lumenfin/          Agent runtime, grounding, claims, FinRun, RAG
tests/                 Unit and reliability regression tests
tests/fixtures/sec/    Minimized manifested SEC extracts
scripts/               Supported tests, diagnostics and fixture builders
docs/                  Current architecture and guides
reports/current/       Authoritative RC evidence
reports/history/       Superseded engineering evidence
tools/archived_audits/ Unsupported historical runners
mcp_layer/             Optional MCP boundary (not production PDF RAG)
fixtures/stress/       Small synthetic stress PDFs
```

## Limitations

- Ready for **controlled** production deployment only
- Milvus Lite and SQLite are not HA multi-tenant infrastructure
- External LLM / embedding / SEC / Yahoo availability and quotas apply
- Not an automated investment or legal decision system
- No public license grant yet; repository is internal/portfolio-oriented
- PyMuPDF (AGPL/commercial terms) blocks public Docker image redistribution
  until an explicit compliance decision is made

Full boundary: [docs/PRODUCTION_LIMITATIONS.md](docs/PRODUCTION_LIMITATIONS.md)

## License / disclaimer

No public open-source license has been selected. Research output is for
engineering evaluation only and is **not investment advice**.
Third-party and SEC fixture notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
