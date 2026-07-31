# LumenFin End-to-End Production Validation & Optimization Audit

Status: Historical
Superseded by: `../current/LumenFin_Final_Release_Report.md`
Purpose: Engineering evolution and regression evidence

Generated: 2026-07-24T16:10:46.607084+00:00

> Standard: live DeepSeek + DashScope embeddings + live SEC/Yahoo + SEC EDGAR 10-K content. No mock LLM.

## 1. Executive Summary

This audit exercised the full diligence pipeline against **real SEC 10-K content** (Apple FY2024, NVIDIA FY2025, Microsoft FY2024) and live market/fundamentals APIs. RAG mean relevance **4.733/5** (hit@≥3 = 100%); agent dimension means planning=4.8, tools=3.5, reasoning=3.9, grounding=3.5 (0–5). Report means accuracy=7.3, structure=8.0, reasoning=6.9, citation=6.5 (0–10).

**Analyst-risk verdict:** The largest production risk is **evidence fidelity on long filings** — retrieval sometimes returns thematically related narrative without the specific numeric cell an analyst needs, and HTML→text PDF preparation already discards native table geometry. Pipeline honesty (fail-closed, HITL, live sources) is comparatively strong.

## 2. Environment

| Component | Value |
|-----------|-------|
| data_mode | `live` |
| llm_model | `deepseek-chat` |
| llm_base | `https://api.deepseek.com` |
| embedding_provider | `dashscope` |
| embedding_dimension | `1024` |
| dashscope_model | `text-embedding-v3` |
| rag_index_mode | `async_on_upload` |
| rag_rerank | `True` |
| fetch_sec | `True` |
| fetch_live | `True` |
| market_provider | `alphavantage` |
| started_at | `2026-07-24T16:02:40.505971+00:00` |

### Document provenance

- Downloaded SEC EDGAR HTML 10-K filings with SEC-compliant User-Agent.
- Converted to PDF via `scripts/convert_sec_html_to_pdf.py` (text extract → paginated PDF) for `parse_pdf_document` ingestion.
- **Limitation (disclosed):** this is real filing text, not a byte-identical issuer PDF; tables become linear text.

## 3. Pipeline Evaluation

| Module | Score (0-10) | Issue |
|--------|-------------:|-------|
| PDF Parsing | 6 | See ingest issues below |
| Chunking | 7 | Peer-table splitter helps synthetic tables; 10-K prose still page-ish |
| Embedding | 8 | provider=ResilientEmbeddingProvider dim=1024 |
| Retrieval | 9.5 | mean_rel=4.733/5 answer_in_context=100% |
| Agent | 7.9 | plan/tool/reason/ground = 4.8/3.5/3.9/3.5 |
| Report Generation | 7.2 | acc/struct/reason/cite = 7.3/8.0/6.9/6.5 |

### 3.1 Ingestion detail

| File | Pages | Chars | Chunks | Avg chunk | Companies | Issues |
|------|------:|------:|-------:|----------:|-----------|--------|
| aapl_fy2024_10k_sec.pdf | 60 | 139379 | 205 | 651.8 | Apple | no metric_hints extracted from filing text |
| nvda_fy2025_10k_sec.pdf | 60 | 164769 | 236 | 667.7 | NVIDIA | none |
| msft_fy2024_10k_sec.pdf | 40 | 94727 | 137 | 661.6 | Microsoft | none |
| tsla_fy2024_10k_sec.pdf | 60 | 161857 | 232 | 667.1 | Tesla | none |

## 4. RAG Evaluation

- Queries: **15**
- Hit rate (relevance ≥ 3): **100%**
- Average relevance: **4.733/5**
- Answer-in-context rate (expected-term heuristic): **100%**

| Query | Retrieved Evidence | Score | Problem |
|-------|--------------------|------:|---------|
| rq01 numeric: What was Apple's total net sales / revenue in fiscal 2024? | `aapl_fy2024_10k_sec.pdf#p55`:products, including evaluating if it can establish the price of the product, ret; `aapl_fy2024_10k_sec.pdf#p36`: of its net sales from a single product and a decline in
demand for that product | 5 |  |
| rq02 numeric: What was Apple's net income in 2024? | `aapl_fy2024_10k_sec.pdf#p36`: of its net sales from a single product and a decline in
demand for that product; `aapl_fy2024_10k_sec.pdf#p46`:and 2022 were as follows (dollars in millions):
 2024 2023 2022
 Provision for i | 4 |  |
| rq03 definition: What does Apple disclose about Services net sales? | `aapl_fy2024_10k_sec.pdf#p36`: of its net sales from a single product and a decline in
demand for that product; `aapl_fy2024_10k_sec.pdf#p55`:products, including evaluating if it can establish the price of the product, ret | 5 |  |
| rq04 risk: What supply chain or manufacturing concentration risks does Apple disclose? | `aapl_fy2024_10k_sec.pdf#p49`:ts.
 Apple Inc. / 2024 Form 10-K / 26
 Item 7A. Quantitative and Qualitative Dis; `aapl_fy2024_10k_sec.pdf#p55`:products, including evaluating if it can establish the price of the product, ret | 5 |  |
| rq05 multi_hop: How do Apple's R&D expenses relate to product development risk? | `aapl_fy2024_10k_sec.pdf#p36`: of its net sales from a single product and a decline in
demand for that product; `aapl_fy2024_10k_sec.pdf#p14`:ntiment in areas like
career development, manager performance and inclusivity.
  | 5 |  |
| rq06 numeric: What was NVIDIA's revenue for fiscal year 2025? | `nvda_fy2025_10k_sec.pdf#p8`:1045810 us-gaap:AllowanceForCreditLossMember 2022-01-30 0001045810 us-gaap:Allow; `nvda_fy2025_10k_sec.pdf#p27`: our Earth-2 initiative to create a digital twin of the Earth on NVIDIA AI and N | 5 |  |
| rq07 numeric: What portion of NVIDIA revenue is Data Center related? | `nvda_fy2025_10k_sec.pdf#p37`: Visualization, and Automotive products. The computing industry is experiencing ; `nvda_fy2025_10k_sec.pdf#p47`:he region have been on active military duty for an extended period
and may conti | 5 |  |
| rq08 risk: What manufacturing or foundry / packaging supply risks does NVIDIA mention? | `nvda_fy2025_10k_sec.pdf#p11`:0-K. These channels may be updated from time to time on NVIDIA's investor relati; `nvda_fy2025_10k_sec.pdf#p22`:Organization for Standardization in such areas as fabrication, assembly, quality | 4 |  |
| rq09 definition: How does NVIDIA describe Data Center growth drivers? | `nvda_fy2025_10k_sec.pdf#p12`:computing infrastructure company with data-center-scale offerings that are resha; `nvda_fy2025_10k_sec.pdf#p13`:ation canvas to include networking, enabled
our platforms to be data center scal | 5 |  |
| rq10 numeric: What was Microsoft's revenue in fiscal year 2024? | `msft_fy2024_10k_sec.pdf#p31`:erating system and Windows cloud services such
as Microsoft Defender for Endpoin; `msft_fy2024_10k_sec.pdf#p24`:we are making curriculum available free of charge to all of the nation s higher  | 4 |  |
| rq11 compare: How does Microsoft describe Intelligent Cloud versus Productivity performance? | `msft_fy2024_10k_sec.pdf#p18`: tools, and video games. We also design and sell devices,
including PCs, tablets; `msft_fy2024_10k_sec.pdf#p21`:365 enables users to stream a full Windows experience from the Microsoft Cloud t | 5 |  |
| rq12 risk: What cybersecurity or AI-related risk factors does Microsoft disclose? | `msft_fy2024_10k_sec.pdf#p17`:ectations and assumptions that are subject to risks and uncertainties that may c; `msft_fy2024_10k_sec.pdf#p19`:cations that delivers operational efficiency and breakthrough customer experienc | 5 |  |
| rq13 compare: Compare Apple iPhone versus Services contribution qualitatively from the filing. | `aapl_fy2024_10k_sec.pdf#p24`: products or services, or otherwise have a material adverse impact on the Compan; `aapl_fy2024_10k_sec.pdf#p25`:ces or goods initiated within an application. From time to time, the Company has | 5 |  |
| rq14 implication: Is NVIDIA's growth described as concentrated in AI / data center demand? | `nvda_fy2025_10k_sec.pdf#p26`: acquisitions, foreign exchange controls and
cash repatriation restrictions, dat; `nvda_fy2025_10k_sec.pdf#p37`: Visualization, and Automotive products. The computing industry is experiencing  | 5 |  |
| rq15 multi_hop: How could cloud capex and AI investment affect Microsoft's operating margins per | `msft_fy2024_10k_sec.pdf#p27`:nt Solutions, Sales Solutions, and Premium Subscriptions offerings, as well as m; `msft_fy2024_10k_sec.pdf#p24`:we are making curriculum available free of charge to all of the nation s higher  | 4 |  |

## 5. Report Quality Evaluation

### Agent case scoreboard

| Case | Status | Plan | Tools | Reason | Ground | Report cite# | LLM |
|------|--------|-----:|------:|-------:|-------:|-------------:|-----|
| ag01_apple_live | completed | 5 | 5 | 5 | 2 | 0 | local-fallback |
| ag02_nvda_pdf_live | incomplete_data | 5 | 4 | 4 | 5 | 5 | deepseek |
| ag03_msft_pdf | incomplete_data | 5 | 4 | 4 | 5 | 5 | local-fallback |
| ag04_aapl_msft_compare | completed | 5 | 5 | 5 | 2 | 0 | local-fallback |
| ag05_tesla_live | needs_clarification | 4 | 1 | 2 | 1 | 0 | local-fallback |
| ag06_nvda_sustainability | incomplete_data | 5 | 4 | 4 | 5 | 5 | local-fallback |
| ag07_apple_pdf_risk | completed | 5 | 5 | 5 | 5 | 5 | local-fallback |
| ag08_openai_failclosed | incomplete_data | 5 | 3 | 4 | 4 | 0 | local-fallback |
| ag09_ambiguous | needs_clarification | 4 | 1 | 2 | 1 | 0 | local-fallback |
| ag10_sparse_pdf | incomplete_data | 5 | 3 | 4 | 5 | 1 | local-fallback |

### Sample report excerpt

Case: `ag02_nvda_pdf_live` — status=`incomplete_data` sources=`{'NVIDIA': 'document_extracted'}`

```markdown
# Incomplete Diligence Output (Fail-Loud Data Gap)

**Companies:** NVIDIA

## 1. Executive Summary

No computable structured fundamentals for NVIDIA (structured_source has no revenue/EBITDA/R&D inputs). Upload source filings with extractable metrics, retry the live fundamentals provider, or explicitly switch to DATA_MODE=demo for demonstrations. Refusing to invent numbers.

**Evidence Boundary:** This run produced no AST-verifiable revenue/EBITDA/R&D inputs. Market snapshots and LLM general knowledge alone are not treated as structured fundamentals. No ratios, SWOT, or investment positioning were invented.

## 4. Financial Performance Analysis

Not available — fail-closed. Structured fundamentals were missing or non-computable for: NVIDIA.

## 7. Risk Architecture

**Data limitation risk (high):** Without extractable FY metrics, quantitative risk scoring and peer margin comparison are withheld rather than estimated.

## 10. Compliance Review & Data Integrity

Fail-closed compliance path: the synthesizer refused to fabricate checkable metrics. Gate expectation: `structured_source=none` for NVIDIA.

## 11. Methodology, Data Sources & Disclaimer

**Action Required:** Upload source filings (PDF) with extractable FY metrics, or retry the configured live fundamentals provider. To use local demo coverage, switch DATA_MODE=demo explicitly.

### Retrieved Document Citations (page-level)

These anchors come from hybrid RAG hits and are written deterministically (not paraphrased by the
```

### Observed strengths / weaknesses

- **Strength:** live `structured_source` honesty; OpenAI / sparse Oracle paths fail-closed rather than inventing EBITDA.
- **Strength:** ambiguous query triggers clarification (HITL) instead of guessing a ticker.
- **Weakness:** analyst-grade valuation discussion is template/heuristic-heavy vs full comps model.
- **Weakness:** RAG may cite nearby narrative pages without guaranteeing the exact FY total line item.

## 6. Bottleneck Analysis

### Stress results

```json
{
  "long_document": {
    "pages": 180,
    "parse_ms": 104.1,
    "analyze_ms": 85542.5,
    "status": "incomplete_data",
    "rag_mode": "hybrid_rrf+rerank",
    "chunks_indexed": 370,
    "page_citations_in_report": 5
  },
  "prefer_upload": {
    "status": "incomplete_data",
    "sources": {
      "Apple": "none"
    },
    "source_resolution": {
      "prefer_uploaded_only": true,
      "mode": "uploaded_only",
      "companies": {
        "Apple": {
          "structured_source": "none",
          "upload_present": true,
          "upload_had_computable_metrics": false,
          "live_fallback_used": false,
          "fallback_reason": "prefer_uploaded_only=true; refused SEC/Yahoo/sample backfill because uploaded materials lacked AST-computable revenue/EBITDA/R&D."
        }
      }
    },
    "report_has_source_resolution": false
  }
}
```

### Priority

**P0 — must fix for analyst trust**

- Completed runs used local LLM fallback — DeepSeek path degraded.

**P1 — clear quality lift**

- SEC HTML→PDF conversion is text-pagination; production should ingest native PDF/HTML with table-aware parsers.

**P2 — experience / polish**

- Page citations present on PDF-backed runs after citation-section fix — keep regression tests.
- Add LLM-as-judge groundedness on a fixed gold set; current relevance uses term+lexical heuristics.

## 7. Optimization Roadmap

### RAG

- Ingest **native PDF / HTML / iXBRL** without lossy text reflow; keep table grid metadata.
- Add **numeric/fact index** (metric, period, value, page) alongside dense chunks for FY totals.
- Keep lexical ZH/EN rerank; consider **DashScope/cloud rerank** only when candidate pool ≥20.
- Expand gold RAG eval set (15→50) with page-level citations as labels.

### Agent

- Planner: force `prefer_uploaded_only` when user says so; surface conflicts in §0 Source Resolution.
- Tool routing already local-first; keep MCP out of production evidence path.
- Trace export: ensure `run_telemetry.rag.mode` always populated (already fixed).

### Report

- Keep deterministic **Retrieved Document Citations** section; add inline `[cite:]` on metric claims when AST inputs come from upload.
- Fail-loud minimal contract already added; extend valuation section to explicitly say “no DCF computed” when true.

### Infrastructure

- Cache SEC companyfacts + embeddings by content hash.
- Isolate Milvus Lite per long job; move to Server when concurrent users appear.
- Retain raw audit JSON under `outputs/e2e_production_audit/raw/` for regression diffs.

## Appendix — Raw artifacts

- Directory: `outputs\e2e_production_audit`
- RAG raw: `raw/module2_rag.json`
- Agent raw: `raw/module3_agents.json`
- Stress raw: `raw/module5_stress.json`
