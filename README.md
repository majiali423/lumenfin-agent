# LumenFin
**Financial document analysis with an auditable path from question to evidence to answer.**

**English** | [中文](README.zh-CN.md)

LumenFin turns uploaded filings and research questions into cited answers and
recomputable financial analysis. It connects **task planning, hybrid retrieval,
deterministic calculation, claim verification and asynchronous execution** in one
application. [FinAgentBench](https://github.com/majiali423/finagentbench-demo)
provides the companion evaluator for exported results.

[![CI](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml)

[Run the demo](#run-the-demo) · [Follow the code](#the-decisions-behind-an-answer) ·
[Evaluation design](docs/evaluation_strategy.md) · [Documentation](docs/README.md)

## A concrete workflow

Upload the [NVIDIA FY2025 excerpt](tests/fixtures/sec/derived/nvda_fy2025_10k_excerpt.pdf)
and ask for operating income. The source shows **81,453 USD millions**;
the expected answer is **81.453 billion USD**, with a citation to page 1.
The [hand-read gold record](tests/fixtures/sec/nvda_fy2025_operating_income_gold.json)
keeps the source hash, original unit, fiscal year and page together.

That example exercises a specific sequence: identify the issuer and period,
extract a financial field, normalize its scale, preserve its source, and expose
the answer with a checkable citation. A
[narrative-only upload](tests/fixtures/sec/minimal/nvda_narrative_only.txt)
must instead explain why that financial number cannot be established.

The same workflow supports ratio questions, company comparisons, evidence-backed
risk summaries and clarification when the request is incomplete.

## The decisions behind an answer

```text
Question + uploaded files
  → input checks → planner → TaskSpec
      ├─ missing information → clarification checkpoint → resume planner
      └─ retrieval within document / company / tenant scope
           ├─ required financial inputs missing → report the data gap
           └─ calculation when needed → risk analysis → critic / bounded correction
  → claim–evidence binding → synthesis → answer, citations and audit artifacts
  → FinRun export → optional FinAgentBench replay / CI gate
```

LangGraph runs specialist nodes over shared state. The branches, limits and
resume behavior are explicit in [graph.py](src/lumenfin/graph.py).

### 1. Plan the task before deciding which tools are necessary

The planner extracts company scope, time range, analysis dimensions and missing
information. [TaskSpec](src/lumenfin/task_spec.py) translates that plan into
execution requirements such as `requires_ast_ratios` and `skip_quant`.

A profitability comparison needs structured financial inputs. A supported
supply-chain risk question can proceed without an unrelated EBITDA ratio.
Missing information can pause the graph and resume through an application
checkpoint. See [clarification behavior](docs/HITL_CLARIFICATION.md).

### 2. Keep financial identity through retrieval

[Document parsing](src/lumenfin/documents.py) and
[chunking](src/lumenfin/rag/chunking.py) preserve original page identity.
[Hybrid retrieval](src/lumenfin/rag/hybrid_retriever.py) combines vector and
keyword retrieval, with configurable reranking.

A number's company, fiscal period, unit and source page travel with the evidence.
A page-2 fact must keep its own locator; an unstated period stays unknown rather
than inheriting the cover's year. Query-named issuers stay the subjects even
when the upload is a different filing. Nearby fiscal years in a compare question
bind to the nearest company name, not the first year in the window.
[Page-identity regressions](tests/test_rag_page_identity.py)
cover parsing, indexing and retrieval together.

Asking Apple operating income against an NVIDIA-only upload must refuse the
NVIDIA figure rather than treat it as Apple's answer. Comparing Microsoft
FY2024 R&D with NVIDIA FY2025 R&D must keep both issuers' field years. Those
failures were found on the 24-task diagnostic and fixed in product code;
later live scores are not a substitute for that root-cause work.

### 3. Compute ratios from explicit inputs

The [quantitative node](src/lumenfin/agents/quantitative.py) consumes structured
fields. The [restricted formula evaluator](src/lumenfin/safe_formula.py) handles
arithmetic, while [quant contracts](src/lumenfin/quant_contract.py) identify
computable inputs and incomplete comparisons.

For example, an operating margin needs operating income and revenue with
compatible financial context. A fluent explanation cannot supply a missing
denominator.

### 4. Bind claims before synthesis, then inspect the visible result

[Claim models](src/lumenfin/claims/models.py) carry the entity, metric, value,
unit, period, evidence references and verification status.
[Binding](src/lumenfin/claims/binding.py) checks numerical support before the
[synthesizer](src/lumenfin/agents/synthesis.py) builds the report.

[FinRun export](src/lumenfin/finrun.py) packages the execution and final output
for replay. FinAgentBench's opt-in v3 scorer also inspects supported financial
assertions in prose and tables: correct internal metrics do not excuse a wrong
number or year in the final answer. The
[product-quality test](tests/test_product_quality_loop.py) checks a real product
export against independent fixture gold, then injects a visible error.

### 5. Treat a long-running analysis as a recoverable job

FastAPI exposes submission, polling and clarification APIs. The UI restores
a job through its `?job=` URL.
[Redis reservations](src/lumenfin/queueing.py) and the
[worker](src/lumenfin/worker.py) use ownership tokens and leases;
database job state and an outbox handle enqueue failures.

Delivery is **at-least-once**. Stale workers must not overwrite or acknowledge
another execution. [Worker recovery tests](tests/test_analysis_worker_resilience.py)
exercise duplicate delivery, lease expiry and failure recovery.

## Architecture choices

| Component | Responsibility | Why it is here |
|---|---|---|
| LangGraph | Conditional workflow, shared state, bounded routing | Make decisions and recovery paths inspectable |
| FastAPI + Redis | User requests and background execution | Keep long analyses outside the request lifecycle |
| PostgreSQL / SQLite | Jobs, application checkpoints and document metadata | Shared service state / lightweight local operation |
| Milvus / Milvus Lite | Document retrieval | Hybrid retrieval with separate service and local profiles |
| Typed claims + restricted arithmetic | Financial assertions and calculations | Keep identity and inputs available for verification |
| FinRun + FinAgentBench | Export and replay evaluation | Review producer behavior and scoring rules separately |

[Architecture](docs/ARCHITECTURE.md) and
[design decisions](docs/architecture_decisions.md) explain the trade-offs.
Both repositories belong to the same project and author; the separate evaluator
is an interface boundary, not third-party certification.

## Run the demo

Use **Python 3.12** and a source checkout:

```bash
git clone https://github.com/majiali423/lumenfin-agent.git
cd lumenfin-agent
python -m venv .venv
```

Activate with `source .venv/bin/activate` on POSIX, or
`.\.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip check
python scripts/start_offline_demo_api.py
```

Open **http://127.0.0.1:8000/**, upload the excerpt linked above, and enter:

> Using uploaded files only, what is NVIDIA FY2025 operating income from the filing facts?

Inspect the answer, source page and financial field, then refresh the job URL.
Use the narrative-only fixture to inspect the missing-data path.

The offline profile uses a **rule-based fallback client** and deterministic
providers, with no API keys. It demonstrates execution behavior; live-model
quality requires a separate evaluation.
For a terminal walkthrough, run `python scripts/run_portfolio_demo.py`.
See the [demo guide](docs/DEMO_GUIDE.md) and
[90-second walkthrough](docs/AUTUMN_RECRUITING_DEMO_SCRIPT.md).

## How the project is evaluated

Two questions are kept separate:

- **Does the implementation preserve its contracts?** CI runs document checks,
  fast tests, full offline regression, the product v3 loop and frozen FinRun
  compatibility lanes. [Recorded passing run](https://github.com/majiali423/lumenfin-agent/actions/runs/34340156893).
- **Does the Agent answer a user's task correctly with supporting evidence?**
  LumenFin runs the product task. Optional LangSmith records traces when
  configured; it does not score answers. FinAgentBench checks exported FinRun
  consistency (layer A). An independent catalog checks source facts (layer B).
  The 24-task development pilot in
  [`tests/fixtures/document_tasks/lumenfin_document_tasks_v1.json`](tests/fixtures/document_tasks/lumenfin_document_tasks_v1.json)
  is **candidate gold**: derived excerpts, repeated-page stress PDFs, and
  lexical+deterministic retrieval. Scoring policy `lumenfin_eval_contract.v1`
  keeps FinAgentBench on `evaluate_run` (layer A) and gold on source facts
  (layer B). `diagnostic_pass` remains gold-only; `eval_acceptance_v1` is a
  new development field and is not formal accuracy. It is not independent
  human review, not a freeze, and not formal accuracy. Same-model B1/B2 live contrast was run as
  an authorized HTTP-capped diagnostic (`fair_v2`/`v3`/`v4`); `fair_v4`
  resumed after a harness interrupt and includes two B1 transport failures.
  Those counts are not production DashScope/Qwen3 results and must not be
  quoted as accuracy. Product fixes after `fair_v4` (company-period binding,
  issuer-mismatch copy, unlabeled-period summary consistency) are offline-only
  and are not claimed from the v4 ledger. p24 remains a recorded scorer
  suspicion, not a gold change.
  Offline LocalFallback runs are coverage diagnostics, not accuracy.
  Details: [evaluation strategy](docs/evaluation_strategy.md).

```bash
python scripts/run_tests.py --fast
python scripts/run_tests.py --skip-joint
python scripts/check_doc_links.py
```

[Joint setup](docs/REPRODUCIBILITY.md) explains the explicit evaluator checkout;
[validation commands](docs/VALIDATION_COMMANDS.md) cover optional infrastructure
checks. Historical experiments and their full limitations remain in the
[evidence index](docs/EVIDENCE_INDEX.md).

## Operating scope

The UI is delivered through source checkout or Docker. General cross-process
LangGraph node replay is outside the certified recovery boundary; application
checkpoints handle the supported clarification flow. Bounded data repair is
implemented and **off by default**. See
[operational limits](docs/PRODUCTION_LIMITATIONS.md) and
[tenant boundaries](docs/MULTI_TENANCY_BOUNDARY.md).

Package metadata remains `0.1.0rc5`; historical tag `v0.1.0-rc.5` predates later
source fixes. Frozen evaluator tags and the product v3 source pin are listed in
[reproducibility](docs/REPRODUCIBILITY.md).

[MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).
Financial outputs are research artifacts requiring human review.
