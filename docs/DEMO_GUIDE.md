# Release Demo Guide

Use the offline demo first. Live demos require configured providers and should
never print API keys.

## Portfolio demo (default entrypoint)

Deterministic, offline, no API key, non-zero exit on failure:

```powershell
python scripts/run_portfolio_demo.py
```

Covers three narratives in one run:

| Demo | Story | Assertion |
|------|-------|-----------|
| A | Trusted normal analysis | issuer-only scope, grounded claims, internal contract score, FinRun-exportable state |
| B | Isolation + error detection | Apple/Microsoft only; wrong number / wrong entity / missing citation / missing risk all rejected (**local claim-binder checks 4/4**; stamped Phase 3.2B tenant leakage `0`) |
| C | Fail-closed | forced missing SEC+Yahoo → `incomplete_data`, zero numeric claims |

The portfolio demo's "evaluator score" is an **internal LumenFin contract
score**, not a FinAgentBench completed-case mean. For the external evaluator,
use FinAgentBench `scripts/run_offline_demo.py` or the pinned cross-repo gate.

The demo also prints validated references (Phase 3.2B run, Phase 3.3A Docker
run, tenant leakage 0) and the optional Docker recovery story:
`worker A killed → automatic reclaim → worker B attempt=2 → ready`
([Phase 3.2B evidence](PHASE32B_INTEGRATION_REPORT.md)). The full Docker stack
is **not** started by the default demo.

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

## Optional live demos (operator workstation only)

These are **not** part of the public clean-clone path. They need provider keys
in a local `.env` and often use **gitignored** filing fixtures under
`fixtures/e2e_real/` (not shipped on GitHub). Prefer
`python scripts/run_portfolio_demo.py` for any shareable walkthrough.

### Live 1 — NVIDIA 10-K (requires local fixture)

**Input**

```text
Analyze NVIDIA investment risk using the uploaded FY2025 10-K and current
market valuation. Cite filing pages where possible.
```

Fixture (local only): `fixtures/e2e_real/nvda_fy2025_10k_sec.pdf` — **not in
the public clone**.

**Expected behavior**

- `companies == ["NVIDIA"]`
- no AMD/Intel peer promoted to issuer scope
- SEC financial grounding fills missing AST-computable fundamentals
- verified numeric/risk claims have evidence
- document evidence includes `#pN` citations
- FinAgentBench issuer gate passes when FinRun is exported

**External risk:** SEC/LLM/embedding quota or network failure. Classify such a
failure as infrastructure, not a quality pass.

### Live 2 — Apple versus Microsoft

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

**External risk:** one provider may return partial data for one issuer. The run
must expose partial comparability instead of silently substituting samples.

### Live 3 — OpenAI or sparse upload

**Input**

```text
Analyze OpenAI FY2025 annual profitability using live fundamentals only. Do not
invent estimates if data is unavailable.
```

Alternative tracked stress fixture: `fixtures/stress/oracle_sparse_fluff.pdf`
(if present in your checkout) with upload-only instructions.

**Expected behavior**

- `workflow_status == "incomplete_data"`
- zero AST-checkable fundamentals
- no verified numeric claims
- explicit data-limitation risk claim
- no crash

**External risk:** distinguish a transient provider outage from a genuinely
unavailable private-company financial dataset.

## Recommended demo order

1. System spine from `FINAL_ARCHITECTURE.md`
2. `python scripts/run_portfolio_demo.py` (offline A/B/C in one run)
3. Optional live 1–3 on an operator machine with fixtures/keys
4. Runtime reliability references: Phase 3.2B worker-kill recovery + Phase 3.3A dual-API
5. End with `PRODUCTION_LIMITATIONS.md`

Published tags for this guide: LumenFin `v0.1.0-rc.3`, FinAgentBench
evaluator pin `v0.1.0-rc.3` (current FinAgentBench package tag `v0.1.0-rc.4`).
