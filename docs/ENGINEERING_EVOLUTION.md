# Engineering Evolution

Honest chronology of how LumenFin + FinAgentBench became a reliability-gated agent system.

Not a marketing narrative. Each phase records a **failure mode that was observed**, then what changed.

---

## Phase 1 — Initial Agent

**What existed:** LangGraph diligence pipeline (planner → retrieval → quant → sentiment → critic → synthesizer), Milvus hybrid RAG, AST quant, audit export.

**Problems that showed up in real runs:**

| Failure | Symptom |
|---------|---------|
| Entity leakage | Upload / issuer analysis pulled peer names into the entity set (e.g. NVDA run mentioning AMD/Intel as analyzed entities) |
| Unsupported claims | Fluent report sentences with numbers or conclusions that had no checkable evidence |
| Weak grounding | PDF narrative retrieval looked “cited,” but structured ratios had empty or non-recomputable inputs |

**Lesson:** A working demo pipeline is not a trustworthy research agent.

---

## Phase 2 — FinAgentBench Introduction

**Why:** Final-answer reading and internal golden evals were insufficient. A polished Markdown report could hide broken entity sets, empty checkable metrics, or missing steps.

**What was introduced (separate repo / calibration layer):**

- Neutral **FinRun** schema (entities, steps, metrics, evidence, market_data, final_output)
- **Replay evaluation** (score exported traces; do not call the agent inside the bench)
- Reliability metrics with CI / regression gates
- Adapter boundary so LumenFin remains the generator

**Lesson:** Reliability needs an **external** gate with deterministic checks, not more prompts.

---

## Phase 3 — Entity Grounding

**Problem:** Issuer vs peer confusion — document entity resolution and supervisor expansion treated peers as in-scope companies.

**Fix direction:** Issuer isolation — uploads expand `issuer_companies` only; compare cases use explicit expected/forbidden entities; FinAgentBench `entity_leakage` metric.

**Validation style:** Issuer NVDA case must not leak AMD; compare NVDA–AMD must allow AMD only when requested.

**Lesson:** Entity reliability is a first-class product requirement, not a prompt footnote.

---

## Phase 4 — Financial Grounding

**Problem:** Many SEC 10-K PDF excerpts do not yield AST-computable structured facts. Early exit on “any market_data present” left **checkable metrics at 0** while reports still sounded complete.

**Fix direction:**

- Treat “has computable fundamentals” as the gate, not “has any market blob”
- Issuer-only SEC companyfacts / Yahoo gap-fill when uploads are incomplete
- Document wins on overlapping fields; `prefer_uploaded_only` still refuse backfill

**Observed effect (NVIDIA PDF validation):** checkable fundamentals recovered (e.g. 0 → 3); numeric correctness floors held without lowering bench thresholds.

**Lesson:** Grounding is a **financial fact layer**, not a retrieval-quality tweak.

---

## Phase 5 — Claim → Evidence Binding

**Problem:** Even with better facts, synthesis could still emit claims without bound evidence, or invent ratios under sparse / fail-closed paths.

**Fix direction:**

- Internal claim objects (numeric / growth / risk / investment)
- `claim_binder` node before synthesizer
- Synthesizer consumes **verified** claims only
- Under `fatal_data_gap` / `structured_source=none`, block verified numeric claims

**Observed effect:** Page-anchored citations and verified-in-report coverage improved on issuer PDF runs; sparse/OpenAI paths no longer mint verified numeric claims.

**Lesson:** Binding claims to evidence is a graph stage, not a “please cite sources” instruction.

---

## Phase 6 — Production Hardening + RC Validation

**What was validated (no new claim rules, no threshold gaming):**

| Scenario | Intent |
|----------|--------|
| Long document | MSFT long 10-K completes with claim coverage |
| Multi-company | AAPL–MSFT and NVDA–AMD entity parity |
| Fail-closed | OpenAI live + sparse upload-only → `incomplete_data`, checkable=0 |
| Expanded RC pack | Apple, NVIDIA, Tesla, Microsoft, compares, negatives — **8/8** |

**RC outcome:** Offline gates PASS; FinAgentBench floors held on completed cases; readiness **GO** for Release Candidate packaging.

**Lesson:** Prove reliability under stress cases; do not add features to inflate scores.

---

## What deliberately did not happen

Across these phases the team repeatedly refused:

- Adding agents for the sake of architecture diagrams
- Prompt-forcing citations
- Lowering FinAgentBench thresholds to “pass”
- Replacing AST/SEC facts with pure RAG answers

Those refusals are part of the engineering story.
