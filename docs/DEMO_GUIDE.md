# Release Demo Guide

Use the offline demo first. Live demos require configured providers and should
never print API keys.

## Report length mode (UI / API)

Optional explicit `output_format`: `research_report` (default) |
`executive_summary` (brief) | `table_summary`. Only the UI button / API field
shortens the report; query keywords such as「简版」do **not** auto-trim.

## Offline reliability demo

```bash
cd finagentbench-demo
python scripts/run_offline_demo.py
```

Expected: baseline PASS; wrong number/entity and missing citation/risk detected.

## Demo A — NVIDIA 10-K

**Input**

```text
Analyze NVIDIA investment risk using the uploaded FY2025 10-K and current
market valuation. Cite filing pages where possible.
```

Fixture: `fixtures/e2e_real/nvda_fy2025_10k_sec.pdf`.

**Expected behavior**

- `companies == ["NVIDIA"]`
- no AMD/Intel peer promoted to issuer scope
- SEC financial grounding fills missing AST-computable fundamentals
- verified numeric/risk claims have evidence
- document evidence includes `#pN` citations
- FinAgentBench issuer gate passes

**Trace to show:** query planner, retrieval provenance, quant formulas/inputs,
`claim_binder`, verified claim ledger, FinRun evidence.

**External risk:** SEC/LLM/embedding quota or network failure. Classify such a
failure as infrastructure, not a quality pass.

## Demo B — Apple versus Microsoft

**Input**

```text
Compare Apple and Microsoft FY2024 profitability, operating margin and R&D
intensity using live fundamentals.
```

**Expected behavior**

- exact entity set: Apple + Microsoft
- independent numeric claims for both entities
- no unrequested peer entity leakage
- formulas and evidence are entity-aligned
- compare FinAgentBench gate passes

**Trace to show:** planner company scope, parallel retrieval/quant results,
per-entity claim counts, entity coverage/leakage metrics.

**External risk:** one provider may return partial data for one issuer. The run
must expose partial comparability instead of silently substituting samples.

## Demo C — OpenAI or sparse upload

**Input**

```text
Analyze OpenAI FY2025 annual profitability using live fundamentals only. Do not
invent estimates if data is unavailable.
```

Alternative fixture: `fixtures/stress/oracle_sparse_fluff.pdf` with
upload-only instructions.

**Expected behavior**

- `workflow_status == "incomplete_data"`
- zero AST-checkable fundamentals
- no verified numeric claims
- explicit data-limitation risk claim
- no crash

**Trace to show:** retrieval provider errors/gap classification,
`fatal_data_gap → claim_binder → synthesizer`, empty numeric checks and
fail-closed report.

**External risk:** distinguish a transient provider outage from a genuinely
unavailable private-company financial dataset.

## Recommended interview order

1. 30-second system spine from `FINAL_ARCHITECTURE.md`
2. Offline mutation demo
3. NVIDIA grounded success
4. Multi-company isolation
5. Fail-closed negative control
6. End with `PRODUCTION_LIMITATIONS.md`
