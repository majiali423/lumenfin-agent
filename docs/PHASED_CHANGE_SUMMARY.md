# Phased change summary (0–9)

Work logged in [PHASED_IMPROVEMENT_LOG.md](PHASED_IMPROVEMENT_LOG.md).
No git reset/clean/commit of the user's dirty tree. Holdout scores not rewritten.

## Phase 0 — Baseline

Reproduced review findings without product fixes: `999999%` still 100 on
metric-only scoring; concurrent `begin` while `running`; ACK without owner
token; unsanitized HTML; upload/outbox gaps. Documented `.venv` drift
(rc3 vs rc5, milvus-lite 3.0 vs 3.1.0). Offline env helper; do not set
`PYTHON_DOTENV_DISABLED=1`.

## Phase 1 — Runtime / I/O boundary

HTML sanitizer; queue reservation token; job lease; chunked uploads; DB↔Redis
outbox so a successful DB write cannot stay `pending` after Redis enqueue
failure.

## Phase 2 — Visible output quality gate

FinAgentBench scoring v3 + `visible_supported_claims` (opt-in, not default v1).
Offline product graph → FinRun. `execution_path` labels. VOI still does not
catch `999999%` unless that path is enabled.

## Phase 3 — Boundaries by responsibility

Lazy `fitz` so FinRun export does not hard-import PyMuPDF. `quant_contract.py`,
`safe_formula.py`, `RuntimeDependencies`, `api/responses.py`, `report_gap.py`.

## Phase 4 — TaskSpec + product-dev set

TaskSpec gating default **on**: missing ratios must not fatal-block evidenced
risk. Bounded repair default **off**. Catalog 36 items; ablation on **dev**
only with LocalFallbackLLM. No-sample risk slice: legacy fatal-gap vs TaskSpec.

## Phase 5 — Honest demo UI

Removed 800ms fake progress. `POST /api/v1/jobs` + poll; `?job=` restore.
Default concise tab. No CDN for core UI. Bench drawer: clean FinRun → inject
`999999%` → gate locates; 14/14 is not product accuracy.

## Phase 6 — Story, CI split, packaging honesty

Unified README homepage order (problem → visible result → 5-minute offline →
trade-offs → cost/limits → deep docs). Historical tables moved to
[EVIDENCE_INDEX.md](EVIDENCE_INDEX.md). Resume draft. CI **fast** job required
before full offline suite and FinRun contract. Document: UI is source/Docker,
not wheel. Do not claim LangGraph multi-process node-level durable recovery.

## Compatibility kept

FastAPI, LangGraph, Redis, Milvus. At-least-once. FinAgentBench sibling pin.
No extra agent framework.

## R4 wrap-up (2026-09-08)

Fast CI no longer imports FinAgentBench. Visible scoring checks evidence
unit/currency/value and Markdown table cells. Offline graph reports keep
raw `lumenfin:` citations so the strict product-quality loop has a real
positive. Frozen FinRun pins remain published rc.3 / rc.4. Product scorer
is still unpublished; set `FINAGENTBENCH_PRODUCT_REF` before trusting remote
offline CI. Sample/gate 100 is a contract score, not product accuracy.

## R5 wrap-up (2026-09-08)

Isolated venv (not the drifted `.venv`) installs lock + editable rc5,
milvus-lite 3.1.0, `pip check` OK. UTF-8 subprocess tests keep 中文 and
nonzero status. Dual-repo tests take `FINAGENTBENCH_DIR` (sibling opt-in).
NVIDIA FY2025 excerpt upload → 81.453 operating income, strict v3 gate,
sparse refuse, UI `?job=` restore. Remote product-scorer CI remains
**unverified** until a real published `FINAGENTBENCH_PRODUCT_REF`.

## R6 wrap-up (2026-09-08)

Field periods bind to the number's page, not the cover FY, **when pages have
distinct document IDs** (R6 helper default). Claim Ledger Statement/Source
rows are scored; retrieval catalogs are not. Offline CI always runs
`--skip-joint`; Product quality v3 fail-closes without a real v3 ref
(rc.3/rc.4 are not substitutes). Remote Actions still **unverified**.
`validate_cross_repo --profile ci` 100 remains a contract score.

## R7 wrap-up (2026-09-08)

Same `document_id` multi-page PDF: page-2 FY2024 operating income stays
FY2024 + `#p2`; missing page-2 period does not inherit cover FY2025; locator
does not take the first prefix match. FinRun metric period follows field
provenance. Strict v3 rejects an OI-line FY/citation swap. **Non-RAG upload
positives do not cover RAG chunk identity.** Remote Actions and a published
product-scorer ref remain **unverified**.

## R8 wrap-up (2026-09-09)

Expanded one-page RAG contexts keep original `page` in `chunk_id` and
citations (`#p2`). Indexer SQLite path reaches `ready`. Unknown field
periods export as `unknown`, not `latest`; v3 `wrong_period` rejects a
year claimed against that evidence while an honest “period not stated”
line still passes. R8 `--skip-joint` was **not** all-green because tests
compared working-tree `chunking.py` to rc5 pins (evaluator SHA includes
LumenFin `rag/chunking.py`, not FinAgentBench `visible_supported_claims.py`).
Remote Actions still **unverified**.

## R9 wrap-up (2026-09-09)

`--skip-joint` no longer imports FinAgentBench via RAG poll helpers.
Historical holdout/hybrid seals are checked against Git objects at
`PRODUCT_COMMIT`; official `assert_rc5_sources` still rejects a drifted
worktree. Local gates (blocked skip-joint, joint-only, FinAgentBench,
doc links, cross-repo ci) passed. Remote Actions and a published
`FINAGENTBENCH_PRODUCT_REF` remain **待发布验证**.
