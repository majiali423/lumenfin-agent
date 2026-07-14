# Architecture Decisions

This document records the main engineering choices in the project: why each component exists, what problem it solves, and what complexity was intentionally avoided.

## 1. Why LangGraph

The workflow is naturally graph-shaped:

```text
retrieval gap -> replanner
metric gap -> replanner
replanner exhausted -> degraded report
critic -> synthesizer
```

LangGraph is used because it provides explicit state, nodes, conditional edges, and checkpointing. This is a better fit than a free-form agent loop for a financial diligence workflow where each phase must be explainable and auditable.

Why not a pure ReAct loop:

- ReAct is useful for open-ended exploration, but this workflow has stable business stages.
- Financial reports need predictable checks, especially metric computation and compliance review.
- A graph makes routing, retry, and degraded-mode behavior easier to inspect.

Why not a heavy multi-agent framework:

- The project needs state-machine control more than role-play abstraction.
- LangGraph keeps the orchestration explicit without hiding the control flow.

## 2. Why Add A Query Planner

Raw user requests are often ambiguous: "compare Apple and Microsoft risk", "analyze this PDF", or "help me review this company". If every downstream agent parses the raw query independently, the system becomes inconsistent.

The `query_planner` node converts natural language into a structured task plan:

```text
intent
companies
analysis_dimensions
output_format
required_skills
missing_fields
clarification_questions
```

This solves three concrete problems:

- Downstream agents receive a stable interpretation of the task.
- The audit trail records how the system understood the user request.
- Missing information, such as an absent company name, is captured explicitly instead of silently defaulting to demo companies.

The current version does not pause the workflow for human clarification. It records missing fields in state first. A later UI/API layer can turn those fields into an interactive clarification step without changing the core workflow.

## 3. Why Lightweight Skills Registry

The project has several tool capabilities: document parsing, market data retrieval, deterministic ratio computation, sentiment analysis, compliance review, and report synthesis. If these capabilities only exist as scattered functions, it is hard to answer what the agent can do and which node owns each capability.

`SKILL_REGISTRY` describes each capability with:

```text
name
description
inputs
outputs
owner_node
```

This is intentionally a lightweight registry, not a plugin marketplace. For this project size, dynamic plugins, permissions, and remote skill loading would add complexity without improving the core workflow. The registry is enough to support planning, auditability, and maintainability.

## 4. Why FastAPI

FastAPI provides typed request/response models, automatic OpenAPI documentation, file upload support, and straightforward deployment as a service.

Using only a CLI would make the project feel like a script. The API layer makes it integrable with a UI, scheduler, queue worker, or external system.

## 5. Why DeepSeek With Rule-Based Fallback

DeepSeek is used as the primary LLM backend because it is cost-effective and works well for Chinese/English financial analysis text.

The local fallback is not a local LLM. It is a rule-based degraded client used only to keep the workflow testable when no API key is configured or the remote model fails. This avoids coupling the orchestration demo to one external service.

## 6. Why PyMuPDF For PDF Parsing

Most digital annual reports contain extractable text. PyMuPDF is fast, local, and simple to deploy for this use case.

## 6b. Fail-loud When Fundamentals Are Missing

Demo sample rows and uploaded PDFs are the only structured inputs for AST ratios. If a company resolves (e.g. `腾讯控股` → `Tencent`) but has neither sample fundamentals nor extractable PDF metrics, retrieval sets `fatal_data_gap` and the graph routes `retrieval → synthesizer` with `workflow_status=incomplete_data`. The synthesizer writes an honest incomplete report and does **not** ask the LLM to invent numbers. FinAgentBench is expected to fail-closed on that export (`structured_source=none`). This is preferred to silent degraded loops through replanner/quant/critic.

OCR is intentionally not part of the MVP because it adds cost, deployment complexity, and another source of extraction error. OCR can be added later for scanned documents.

## 7. Why Numerical Computation Is Not Delegated To The LLM

LLMs can summarize and reason over text, but they are not reliable calculators. The project uses a restricted AST-based expression evaluator for financial ratios.

This keeps metric computation:

- deterministic
- testable
- safer than arbitrary Python execution
- traceable in exported state

## 8. Why Trace Evaluation

Agent evaluation should not only inspect the final prose. A polished report can still hide a broken workflow. The evaluator checks:

- whether required agent steps ran
- whether the report contract was satisfied
- whether each company has evidence coverage
- whether degraded mode or compliance findings remain

This makes the project easier to regression-test after prompt, model, or tool changes.

## 9. Why SQLite First, PostgreSQL Optional

SQLite keeps local development simple. PostgreSQL is available for a production-like deployment. This is a progressive architecture choice:

```text
local demo -> SQLite
service deployment -> PostgreSQL
async jobs -> Redis/RQ
graph memory extension -> Neo4j
```

The project does not require every infrastructure component to run for the core workflow to be useful.
