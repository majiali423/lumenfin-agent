# FinAgentBench Technical Design

FinAgentBench is a **Financial Agent Reliability Evaluation Framework** — not a shell script wrapped around `assert "Apple" in report`.

Primary implementation lives in the sibling repo `finagentbench-demo`. LumenFin integrates by exporting FinRun (or adapter-compatible state). This document is the design brief for interview and packaging; schema details remain authoritative in FinAgentBench’s own docs.

---

## Motivation

Financial agents fail in **different places**: wrong companies, empty retrieval, invented ratios, citations that do not match inputs, missing risk/compliance sections, silent market-data failure.

A single final-answer score (or human skim of Markdown) hides those failures. Teams then “fix” by polishing prose.

FinAgentBench exists to:

1. Score a **normalized execution trace** (FinRun)
2. Localize **which reliability dimension** failed
3. Act as a **CI / regression gate** independent of the agent framework

---

## Design Principles

| Principle | Meaning |
|-----------|---------|
| **Replay-first** | Evaluate exported artifacts; do not invoke the live agent inside the bench |
| **Framework independent** | Core metrics know FinRun, not LangGraph / AutoGen internals |
| **Deterministic metrics first** | Recompute formulas, check entity sets, step presence, citation linkage |
| **Semantic judge optional** | LLM-as-judge is additive, not the CI floor |
| **Fail closed** | Empty check sets must not score as perfect (`require_checkable_metrics`) |

---

## FinRun Schema

Agents emit heterogeneous state. FinRun is the **contract**:

| Field | Role |
|-------|------|
| `run_id` | Stable id for the evaluated run |
| `query` | User request (optional but useful) |
| `entities` | Companies actually in scope |
| `steps` | Intermediate stages with status |
| `metrics` | Deterministic calculations with formula + inputs |
| `evidence` | Retrieved snippets / citations |
| `market_data` | Snapshots or provider failures |
| `final_output` | User-facing report text |
| `metadata` | Agent/model/version stamps |

**Why unify?** Without a shared format, every agent invents its own “eval JSON,” metrics cannot be reused, and mutation tests cannot be applied consistently. FinRun keeps LumenFin as generator and FinAgentBench as calibrator.

---

## Metrics (by reliability class)

### Entity reliability

- Expected entities present (`entity_coverage`)
- Forbidden peers absent (`entity_leakage`)
- Compare cases allow only requested peers

### Retrieval provenance

- Evidence items present for claims / sections (`evidence_coverage`, retrieval provenance)
- Citations consistent with metric inputs where checkable (`evidence_consistency`)

### Numeric correctness

- Recompute `formula` from `inputs`; compare to declared `value`
- Unit / currency / period consistency
- Prefer cases with `require_checkable_metrics` so “nothing to check” ≠ 100

### Citation coverage

- Material numeric claims should be evidence-backed in export/report
- Page markers (`#pN`) when document-grounded; live fundamentals use structured source citations (not fake pages)

### Compliance

- Required report sections (risk, disclaimer, methodology, …)
- Unsafe investment language / missing limitation disclosure checks
- Step presence for required pipeline nodes (including `claim_binder` when expected)

---

## Mutation Testing

Mutation testing answers: **can the benchmark catch a known bad export?**

Typical mutations (validated in FinAgentBench correctness work):

| Mutation | Expected catch |
|----------|----------------|
| Wrong revenue (or swapped inputs) | `numeric_correctness` fails |
| Wrong / leaked company | `entity_leakage` / entity coverage fails |
| Missing citation / empty evidence for checkable claims | evidence metrics fail |
| Missing risk / required section | section / compliance metrics fail |

If a mutation still scores “green,” the metric or case definition is insufficient — fix the **bench**, do not lower thresholds to hide agent bugs.

---

## Integration with LumenFin

```text
LumenFin analyze
  → state.json + report
  → export_finrun_state() / scripts/export_finrun.py
  → finagentbench evaluate | gate | benchmark
```

RC pack runner: `finagentbench-demo/scripts/run_rc_validation.py` (live cases + offline gates; **no** threshold relaxation).

---

## Related

- FinAgentBench `docs/architecture.md`, `docs/finrun_schema.md`
- LumenFin [FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md), [FINAL_RESULTS.md](FINAL_RESULTS.md)
