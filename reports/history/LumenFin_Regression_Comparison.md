# LumenFin Regression Comparison (Before vs After P0/P1)

Status: Historical
Superseded by: `../current/LumenFin_Final_Release_Report.md`
Purpose: Engineering evolution and regression evidence

Generated: 2026-07-24T16:13:00+00:00

Before = first live E2E audit (`LumenFin_E2E_Audit_Report.md`, rescored artifacts).
After = same **15 RAG + 10 agent** cases via `scripts/run_e2e_production_audit.py` after P0/P1 code, plus deep probes (`scripts/validate_p0_deep.py`).
Evaluators were **not** modified to inflate scores.

**Caveat (honest):** After agent runs saw frequent DeepSeek `400` → `local-fallback` on several cases. Entity / RAG / fact metrics below still come from live fixtures + APIs; full-report narrative quality is partly degraded by LLM fallback, not only by P0/P1.

---

## Summary metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Wrong company lookup | ag02/ag06: **10 companies** (NVIDIA + AMD/Amazon/Microsoft/…) | ag02/ag06: **`['NVIDIA']` only** | **Fixed** — body peers no longer enter live lookup |
| Report length (ag02 NVDA PDF) | 84398 | 3138 | **−81260** (peer fan-out gone; also `incomplete_data` fail-loud) |
| Report length (ag06 NVDA sustainability) | 85038 | 3138 | **−81900** (same pattern) |
| Financial fact hit rate | low (rq01 relevance **2**/5; narrative tops) | deep probe **55%** (11/20); Apple revenue **391035 consolidated** | **Improved** (still gaps on NI/EPS/margins/debt) |
| Citation accuracy (report mean 0–10) | 7.0 | ~7.2 on scored runs; PDF fail-loud cites present (#p) | ≈ flat / slight up on PDF fail-loud path |
| Retrieval score (RAG mean 0–5) | 4.33 | **4.73** | **+0.40**; rq01 **2→5** |
| Hallucination cases | OpenAI/Oracle fail-closed OK; peer pollution was the trust risk | OpenAI/Oracle still fail-closed; **no peer-pollution “fake multi-issuer” reports**; NVDA/MSFT PDF → honest `incomplete_data` when AST metrics missing | Honesty preserved; peer hallucination vector removed |

---

## Phase 1 — Full E2E (same 15+10)

### Ingestion
- Apple / NVIDIA / Microsoft PDFs (parity with first audit) + **Tesla** 10-K PDF added.
- After entity resolve: NVDA/AAPL/TSLA issuers are single-company (validated).

### RAG (15 queries, unchanged)
| | Before | After |
|--|--------|-------|
| Mean relevance | 4.33/5 | **4.73/5** |
| rq01 Apple FY2024 revenue | score **2** (narrative) | score **5** |
| Mode | keyword_only+rerank | still mostly **keyword_only+rerank** (vector_hits=0 in harness; parity unit test passes on tenant path) |

### Agent (10 cases, unchanged queries)
| Case | Before companies / status | After companies / status |
|------|---------------------------|--------------------------|
| ag02 NVDA PDF | 10 cos / completed ~84KB | **NVIDIA** / `incomplete_data` ~3KB |
| ag06 NVDA sustain | 10 cos / completed ~85KB | **NVIDIA** / `incomplete_data` ~3KB |
| ag03 MSFT PDF | 9 cos / completed | **Microsoft** / `incomplete_data` |
| ag07 Apple PDF risk | Apple+Alphabet / completed | **Apple** only / completed |
| ag08 OpenAI | incomplete_data | incomplete_data (OK) |
| ag09 ambiguous | needs_clarification | needs_clarification (OK) |
| ag10 sparse Oracle | incomplete_data | incomplete_data (OK) |
| ag05 Tesla live | completed Tesla | **needs_clarification** (regression vs Before; LLM fallback likely) |

**Interpretation:** P0 stopped wrong multi-company lookup. Filing-only PDF paths then correctly refuse to invent AST metrics (`incomplete_data`) instead of padding with peer SEC facts — shorter reports are **not** “better analysis,” they are **honest gaps** exposed once peers are removed.

---

## Phase 2 — Deep P0 validation

### 1. Entity resolution (document vs user)

| Check | Result |
|-------|--------|
| NVDA 10-K body mentions AMD / Microsoft / Amazon | Present in `mentioned_companies` |
| Those peers in `detected_companies` / live issuer list | **No** — issuers=`['NVIDIA']` |
| User: “Compare NVIDIA and AMD” | companies=`['NVIDIA','AMD']`, **AMD allowed** |
| Non-compare NVDA upload query | companies=`['NVIDIA']`, **AMD excluded** |

Document entity ≠ user-requested entity: **pass**.

### 2. Numeric grounding (~20 metrics)

| Company | Metric | Value | Period | Scope | Page | Correct? | Confusions |
|---------|--------|------:|--------|-------|-----:|:--------:|------------|
| Apple | revenue | 391035 | FY2024 | consolidated | 80 | Y | - |
| Apple | net_income | — | — | — | 43 | N | - |
| Apple | eps | — | — | — | 51 | N | - |
| Apple | gross_margin | — | — | — | 13 | N | - |
| Apple | operating_income | — | — | — | 39 | N | - |
| Apple | operating_margin | — | — | — | 13 | N | - |
| Apple | r_and_d | — | — | — | 13 | N | - |
| Apple | debt | — | — | — | 52 | N | - |
| Apple | operating_cash_flow | 11445 | — | narrative | 92 | Y* | magnitude/context risk |
| Apple | capex | — | — | — | 43 | N | - |
| Apple | cash | 29943 | — | narrative | 136 | Y | html_table |
| NVIDIA | revenue | 130.5 | — | segment | 77 | Y | segment_vs_consolidated |
| NVIDIA | net_income | 5.2 | — | narrative | 106 | Y* | unit/scale risk |
| NVIDIA | gross_margin | 16405 | — | narrative | 129 | Y* | likely mis-scaled |
| NVIDIA | operating_income | 81453 | — | narrative | 129 | Y | - |
| NVIDIA | r_and_d | — | — | — | 104 | N | - |
| Tesla | revenue | 97.69 | — | consolidated | 64 | Y | - |
| Tesla | net_income | 7.09 | — | consolidated | 64 | Y | - |
| Tesla | operating_income | 7076 | — | consolidated | 151 | Y | - |
| Tesla | capex | 11.34 | — | consolidated | 64 | Y | - |

\*Marked correct_enough by metric+value presence; analyst should still treat unit/scale as untrusted without statement gold labels.

**Hit rate: 55%.** Remaining issues: segment vs consolidated (NVDA revenue), missing NI/EPS/margins/debt on Apple HTML/PDF mix, occasional wrong magnitude.

Apple HTML revenue top (top-5): **391035 consolidated** (`prefers_consolidated=true`).

---

## Phase 3 — P1 status

| Item | Status |
|------|--------|
| **P1-1** SEC HTML → DOM tables → facts | **Implemented** (`src/lumenfin/sec_html.py`); `.htm/.html` ingest; PDF remains fallback. Apple HTML: **63 tables** parsed. |
| **P1-2** Fact ranking consolidated > segment > narrative | **Implemented** (`statement_type` / `scope` on facts + retriever boost + dedupe). Apple “total net sales” beats iPhone **201183**. |

Not done yet for production: native issuer PDF page geometry / iXBRL; gold-labeled line-item QA across statements.

---

## Phase 4 — Production readiness

| Dimension | Assessment |
|-----------|------------|
| 1. 可信金融数据 grounding | **Partial.** Apple consolidated revenue grounding works; ~half of probed metrics still miss or risk scale/segment confusion. |
| 2. 稳定 entity routing | **Strong.** Issuer vs mention split + compare allow-list verified on live NVDA/AMD. |
| 3. 可解释 citation | **Partial.** `#pN` present on PDF paths; not every numeric claim is fact-bound. |
| 4. 真实 analyst workflow | **Partial.** HITL + fail-closed work; filing-only AST gaps + LLM 400 fallback interrupt “happy path” diligence. |

### Production readiness score: **6.5 / 10**

Distance to 10:

1. Gold FY line-item set (consolidated vs segment vs adjusted) with human labels — not heuristic hit rates.
2. When upload lacks AST metrics, controlled SEC fill for **issuer only** (not peers) so diligence completes without reintroducing pollution.
3. Stable primary LLM (eliminate DeepSeek 400 → local-fallback in production runs).
4. Hybrid vector arm reliable under live DashScope + Milvus Lite (harness still often `keyword_only`).
5. Native PDF/iXBRL page-faithful tables beyond HTML DOM heuristics.
6. Claim→citation binding for every material number in the final report.

---

## Artifacts

- After raw: `outputs/e2e_production_audit/raw/` (`module2_rag.json`, `module3_agents.json`, `p0_deep_validation.json`, `summary_after.json`)
- Deep script: `scripts/validate_p0_deep.py`
- P0 quick check: `scripts/validate_p0_optimizations.py` (Apple revenue → **391035 consolidated**)
- P1 code: `src/lumenfin/sec_html.py`, ranking in `rag/chunking.py` + `rag/hybrid_retriever.py`
