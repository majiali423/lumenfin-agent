# LumenFin
**Upload filings, ask a research question, and get a cited financial answer that can be checked against the source.**

**English** | [中文](README.zh-CN.md)

LumenFin is a financial-document Agent: it plans the task, retrieves inside
the uploaded files, computes ratios from explicit inputs, binds claims to
evidence, and returns an answer with citations and an audit trail.
[FinAgentBench](https://github.com/majiali423/finagentbench-demo) is the
companion evaluator for exported FinRun records.

[![CI](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/majiali423/lumenfin-agent/actions/workflows/ci.yml)

The badge is the latest default-branch workflow. It is not a pass for
unpushed local changes. The last published `main` run failed Offline
regression on a Windows/Linux HTML hash mismatch
([34691494024](https://github.com/majiali423/lumenfin-agent/actions/runs/34691494024)).
An earlier green run
([34340156893](https://github.com/majiali423/lumenfin-agent/actions/runs/34340156893))
is historical and does not describe the current commit.

[Run offline](#run-it-offline) · [Three cases](#three-inspectable-cases) ·
[Code](#how-an-answer-is-produced) ·
[Evaluation](docs/evaluation_strategy.md) · [Docs](docs/README.md)

## What a run looks like

The committed demo path is the real API and UI, started with a **rule-based
fallback LLM** and deterministic retrieval. It is not a live-model quality
demo.

![Control flow from question and uploads through retrieval, calculation, claim binding, FinRun export and FinAgentBench replay](docs/assets/lumenfin-control-flow.png)

A concrete source fact, independent of any Agent output: upload
[`nvda_fy2025_10k_excerpt.pdf`](tests/fixtures/sec/derived/nvda_fy2025_10k_excerpt.pdf)
and ask for NVIDIA FY2025 operating income. The excerpt states **81,453 USD
millions** on page 1; the expected answer is **81.453 billion USD**. The
[hand-read gold record](tests/fixtures/sec/nvda_fy2025_operating_income_gold.json)
stores the file hash, original unit, fiscal year and page.

A UI screenshot of that completed offline job:

![Offline demo showing NVIDIA FY2025 operating income 81.453 billion USD with a page citation and LOCAL-FALLBACK badge](docs/assets/offline-demo-nvda-operating-income.png)

The capture is from `scripts/start_offline_demo_api.py` (rule-based fallback).
Gold for this excerpt is page 1; the eight-page derived PDF repeats the same
facts, so this run cited `#p8`. That is the actual offline retrieval result,
not a live-model score and not a complete 10-K.

These PDFs and HTML extracts are **minimized derived fixtures**, not complete
official 10-Ks.

## Three inspectable cases

| Case | Input | Expected behavior | Where to check |
|---|---|---|---|
| Factual answer | NVIDIA FY2025 excerpt + operating-income question | Report **81.453 billion USD** with a page-1 citation, not sample-library 72.4 | Gold JSON above; catalog `dt-p01-nvda-oi` |
| Issuer mismatch | Same NVIDIA file, question asks for **Apple** FY2025 operating income | Explain that the upload cannot answer Apple; do not substitute NVIDIA 81.453 | Catalog `dt-p14-aapl-not-in-nvda-file` |
| Different fiscal years | Microsoft FY2024 excerpt + NVIDIA FY2025 excerpt, compare R&D | Keep **Microsoft FY2024 29.51 billion** and **NVIDIA FY2025 12.914 billion** as separate facts | Catalog `dt-p15-msft-vs-nvda-rd` |

The 24-task development catalog is
[`lumenfin_document_tasks_v1.json`](tests/fixtures/document_tasks/lumenfin_document_tasks_v1.json).
It is candidate gold for a diagnostic, not formal accuracy.

## How an answer is produced

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

| Decision | Why it is here | Code |
|---|---|---|
| Plan before choosing tools | A profitability compare needs structured fields; an evidenced risk question should not fail for a missing unrelated ratio | [`graph.py`](src/lumenfin/graph.py), [`task_spec.py`](src/lumenfin/task_spec.py) |
| Keep financial identity in retrieval | Company, period, unit and page travel with the evidence; query-named issuers stay the subjects | [`chunking.py`](src/lumenfin/rag/chunking.py), [`hybrid_retriever.py`](src/lumenfin/rag/hybrid_retriever.py), [`query_focus.py`](src/lumenfin/query_focus.py) |
| Compute from explicit inputs | Restricted arithmetic; a fluent sentence cannot invent a denominator | [`quantitative.py`](src/lumenfin/agents/quantitative.py), [`safe_formula.py`](src/lumenfin/safe_formula.py) |
| Bind claims before the visible answer | Entity, metric, value, period and citation are checked before synthesis | [`claims/binding.py`](src/lumenfin/claims/binding.py), [`synthesis.py`](src/lumenfin/agents/synthesis.py) |
| Treat analysis as a recoverable job | At-least-once delivery; stale workers must not overwrite another run | [`worker.py`](src/lumenfin/worker.py), [`queueing.py`](src/lumenfin/queueing.py) |

Asking Apple operating income against an NVIDIA-only upload must refuse the
NVIDIA figure. Comparing Microsoft FY2024 R&D with NVIDIA FY2025 R&D must
keep both issuers' field years. Those failures were found on the 24-task
diagnostic and fixed in product code; later live scores are not a substitute
for that work.

Architecture trade-offs: [ARCHITECTURE.md](docs/ARCHITECTURE.md),
[design decisions](docs/architecture_decisions.md).

## Run it offline

Use **Python 3.12** and a source checkout. This path needs no API keys.

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

Open **http://127.0.0.1:8000/**, upload the NVIDIA excerpt, and enter:

> Using uploaded files only, what is NVIDIA FY2025 operating income from the filing facts?

Then upload [`nvda_narrative_only.txt`](tests/fixtures/sec/minimal/nvda_narrative_only.txt)
with the same question to see the missing-data path.

The offline profile uses a **rule-based fallback client** and deterministic
providers. It demonstrates execution, citations and refusal behavior. It is
not live-model answer quality.

Terminal walkthrough: `python scripts/run_portfolio_demo.py`
(also rule-based). See the [demo guide](docs/DEMO_GUIDE.md) and
[90-second script](docs/AUTUMN_RECRUITING_DEMO_SCRIPT.md).

## Evaluation, briefly

Two questions stay separate:

- **Does the implementation keep its contracts?** CI runs document checks,
  fast tests, full offline regression, the product v3 loop and frozen FinRun
  lanes. Product-quality and the two frozen contract lanes already passed on
  GitHub against evaluator pin `40f7599`. That does not make the current
  unpublished working tree remotely green.
- **Did the Agent complete the user's task with supporting evidence?**
  LumenFin runs the product. FinAgentBench checks export consistency
  (layer A). An independent catalog checks source facts (layer B). Scoring
  policy `lumenfin_eval_contract.v1` keeps those layers apart.
  `diagnostic_pass` is gold-only. `eval_acceptance_v1` is a development
  field. Neither is formal accuracy.

The 24-task set is a development diagnostic: derived excerpts, repeated-page
stress PDFs, lexical + deterministic retrieval. Do not quote 17/24, B1 13/24,
or similar live counts as accuracy. Authorized `fair_v2`/`v3`/`v4` ledgers,
budget notes and the p24 scorer suspicion stay in
[evaluation strategy](docs/evaluation_strategy.md).
Historical experiments: [evidence index](docs/EVIDENCE_INDEX.md).

```bash
python scripts/run_tests.py --fast
python scripts/run_tests.py --skip-joint
python scripts/check_doc_links.py
```

[Joint setup](docs/REPRODUCIBILITY.md) and
[validation commands](docs/VALIDATION_COMMANDS.md) cover optional
infrastructure. Do not present LocalFallback coverage as model accuracy.

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
