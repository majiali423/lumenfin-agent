# LumenFin Optimization Report (Post-Audit)

Status: Historical
Superseded by: `../current/LumenFin_Final_Release_Report.md`
Purpose: Engineering evolution and regression evidence

Generated after implementing P0 engineering fixes from `LumenFin_E2E_Audit_Report.md`.

Validation basis:

- Unit regressions: `tests/test_document_primary_entity.py`, `tests/test_rag_hybrid_harness_parity.py`
- Fixture checks: `scripts/validate_p0_optimizations.py` on SEC EDGAR Apple/NVIDIA 10-K PDFs
- Full E2E re-audit script remains: `scripts/run_e2e_production_audit.py` (+ `rescore_e2e_production_audit.py`)

---

## Before vs After

| Issue | Before | After | Improvement |
|-------|--------|-------|-------------|
| P0-1 10-K peer over-detect → multi-company live lookup | NVDA PDF `detected_companies` included AMD/Amazon/Microsoft/…; agent expanded to ~10 issuers; ~80KB+ reports | `resolve_document_entities` sets **issuer = NVIDIA only** (filename/cover/peer-table aware); supervisor expands **issuer_companies only**; query-only scope skips upload peer merge | **Fixed for issuer filings**. Peer-table PDFs still keep both columns. |
| P0-2 Numeric grounding | Revenue queries retrieved thematic narrative without metric/value facts | Chunker emits `financial_fact` chunks `{metric, period, value}`; keyword path boosts them for numeric queries | **Improved**. Top hit for Apple revenue query is now a fact chunk (fixture extract may surface segment totals before consolidated line — see Remaining). |
| P0-3 Standalone RAG `keyword_only` | Audit harness often `keyword_only+rerank` while agent path `hybrid_rrf+rerank` | Shared retriever: relax company filter on empty vector hits + warning; parity test asserts **hybrid_rrf** on tenant+`source_document_ids` path | **Fixed in shared retriever + regression**. Harness and agent use same `HybridEvidenceRetriever`. |
| Report bloat from peers | Live SEC/Yahoo fan-out for every body mention | Live fan-out limited to issuers / explicit compare query companies | **Fixed** (depends on P0-1). |
| Citation section | Already present when `rag_evidence` non-empty | Unchanged; fewer spurious companies → cleaner citation set | Indirect win |

---

## What was changed (code)

| File | Change |
|------|--------|
| `src/lumenfin/document_entity.py` | **New** Document Primary Entity Resolver |
| `src/lumenfin/documents.py` | PDF parse attaches `primary_company` / `issuer_companies` / `mentioned_companies`; `detected_companies` = issuers |
| `src/lumenfin/document_ingest.py` | All uploads go through entity resolver (unless explicit companies) |
| `src/lumenfin/planning.py` | `_upload_companies` prefers `issuer_companies` |
| `src/lumenfin/agents.py` | Supervisor no longer unions body peer mentions; respects query-only scope |
| `src/lumenfin/rag/chunking.py` | Financial fact chunks + issuer-preferred tagging |
| `src/lumenfin/rag/hybrid_retriever.py` | Fact boost; vector empty → relax company filter; keyword_only warning |
| `tests/test_document_primary_entity.py` | Regressions for NVDA issuer-only + revenue facts |
| `tests/test_rag_hybrid_harness_parity.py` | Tenant index/retrieve must be `hybrid_rrf` |
| `scripts/validate_p0_optimizations.py` | Fast fixture validation |

API surface (`LumenFinAnalysisService.analyze`, schemas) unchanged.

---

## Answers

### 1. Which problems are solved?

- **SEC 10-K competitor mentions no longer become live-lookup companies** for single-issuer filings (NVIDIA/Apple fixtures → one issuer).
- **Supervisor/upload company expansion** no longer blindly unions all `COMPANY_HINTS` hits from the body.
- **Numeric queries** can retrieve structured `financial_fact` chunks instead of only prose.
- **Standalone vs agent RAG** share one hybrid retriever; empty company-filtered vector search no longer silently traps the harness in keyword-only without a second chance / warning.
- Regressions cover the NVDA primary-entity case and hybrid parity.

### 2. Which problems remain?

- **HTML→text PDF preparation** still loses native table geometry (P1). Production should ingest native PDF/HTML/iXBRL.
- **Fact extractor can latch onto the first large number near “net sales”** (e.g. segment `201183` before consolidated `391035` if the latter is later/missing in the excerpt). Needs statement-priority / “Total net sales” ranking (P1).
- **CIK/ticker enrichment** on `primary_company` is stubbed (`ticker=None`, `cik=None`) — filename/cover confidence only for now.
- **P1 report controller** (max length, dedupe) not implemented yet.
- **Full live E2E re-score** (15 queries + 10 agent cases) not re-run in this pass; unit + fixture validation only. Recommend: `python scripts/run_e2e_production_audit.py` then rescore.

### 3. Distance to production-ready

| Area | Status |
|------|--------|
| Honesty (HITL, fail-closed, live source labels) | Strong |
| Issuer scoping on filings | **Much closer** after P0-1 |
| Exact FY line-item retrieval | Better, not analyst-grade yet |
| Table/layout fidelity | Gap (parser/ingest) |
| Multi-tenant vector ops | Lite OK for single process; Server still deferred |
| Eval harness (Recall@K / gold cites) | Partial; expand gold set |

**Verdict:** LumenFin is no longer “demo that accidentally analyzes every competitor named in a 10-K.” It is closer to a **reliable single-issuer diligence agent**, but not yet production-ready for unsupervised numeric claims without a fact-ranking / native filing parser pass.

---

## Recommended next steps (P1)

1. Rank financial facts: prefer `Total net sales` / consolidated statements over segment lines.
2. Native table extraction (pdfplumber / Docling) for SEC PDFs.
3. Re-run full `run_e2e_production_audit.py` and attach a Before/After scoreboard to this file.
4. Report controller: cap companies, cap length, require issuer-only unless compare intent.
