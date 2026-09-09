# Resume draft (honest, interview-ready)

Use as bullets. Do **not** invent users, production SLA, third-party independent
benchmarks, or autonomous multi-agent swarms.

## One-line product

Built **LumenFin**, an evidence-grounded financial research agent: LangGraph
specialist nodes (plan → retrieve → AST-safe quant when required → critic →
claim bind → synthesize) behind FastAPI, Redis at-least-once workers, and
Milvus. Sibling repo **FinAgentBench** replays exported **FinRun** traces as a
versioned contract gate — author-owned, not a market leaderboard.

## What I actually built

- Fail-closed numeric claims when structured inputs are missing; risk answers
  can still complete when TaskSpec says ratios are not required.
- Claim–evidence binding before the visible answer; HTML sanitization of
  rendered markdown.
- Job lease + queue reservation token (at-least-once, not exactly-once);
  chunked uploads; DB↔Redis outbox so enqueue failure cannot leave a silent
  pending job.
- Offline 5-minute path: portfolio A (evidenced) / B (injected errors caught) /
  C (missing-data refuse) without live keys.
- UI: real `audit_log` via job polling, not a fake 800ms node timer; default
  concise answer; optional FinRun mutation story that does not treat 14/14 as
  product accuracy.

## Bugs I found and fixed (examples)

| Bug | Fix |
|-----|-----|
| Visible `999999%` still scored 100 because metrics ignored prose | Opt-in FinAgentBench `visible_supported_claims` + product graph export; UI demo injects then locates |
| `begin_job_execution` while `running` returned another `run` | Job lease / single owner |
| Queue ACK ignored worker reservation token | Lua / token match |
| Markdown → `innerHTML` unsanitized | Server + client sanitize |
| Upload size check after full `read()`; leftover files; enqueue fail still `pending` | Chunked limits, cleanup, outbox |
| Missing EBITDA blocked evidenced **risk** answers | TaskSpec gating (default on) |
| Exporter imported `fitz` so a thin evaluator venv could not load FinRun | Lazy `fitz` |

## Trade-offs I can defend

- Specialist **nodes**, not independent agents with their own memory loops.
- At-least-once over exactly-once (Redis reclaim + fencing).
- Bounded repair **implemented, default off** after an honest offline ablation.
- HITL uses in-process LangGraph checkpointer **plus** an app checkpoint table;
  I do **not** claim tested multi-process LangGraph node-level durable replay.
- `requires-python >= 3.11`; **required CI is Python 3.12**.
- Wheel does not include `static/`; UI is checkout or Docker.

## Numbers I will say — and the caveat in the same breath

- Offline portfolio A/B/C must pass on a source checkout.
- Cross-repo FinRun pin `v0.1.0-rc.3` (compatibility also `v0.1.0-rc.4`).
- FinanceBench confirmation-50 Hit@10 **0.62** on a **consumed** split — retrieval
  canary, not product QA; do not rerun to tune.
- LEDGER public_holdout v2 **35/100** — sealed, dataset-specific.
- Product-dev **dev** ablation is a small hand set; test split frozen; not 80%.

## Unfinished (say this first if asked)

- Existing workspace `.venv` can drift from the lockfile (rc3 metadata vs rc5
  source); isolated clean LumenFin install on Windows was not the Phase 0 proof.
- No claimed production users or SLA.
- Full Chrome click-through of every UI path was not the Phase 5 proof
  (TestClient + static assertions were).
- SSE job streaming not implemented; `audit_log` appears after `analyze()`
  finishes.
- Bounded repair not on by default.
- FinanceBench / LEDGER holdouts will not be re-run as a scoreboard.
