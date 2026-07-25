# LumenFin Financial Grounding Validation

Status: Historical
Superseded by: `../current/LumenFin_Final_Release_Report.md`
Purpose: Engineering evolution and regression evidence

Generated: 2026-07-24T17:30:37.998892+00:00

Scope: NVIDIA FY2025 10-K upload + live issuer SEC gap-fill (Financial Grounding Layer).
Path: `export_finrun_state()` → FinAgentBench (issuer NVDA case, profile=`ci`).
No mock LLM. Evaluators unchanged.

## Result

| Metric | Before Grounding (final baseline After) | After Grounding |
|--------|----------------------------------------:|----------------:|
| FinAgentBench score | 31.87 | **92.97** |
| checkable formula+inputs | 0 | **3** |
| numeric_correctness | 0.0 | **100.0** |
| entity_leakage | 100.0 | **100.0** |
| evidence_coverage | — | **100.0** |
| retrieval_provenance | — | **100.0** |
| #pN citation markers | 6 | **6** |
| workflow_status | incomplete_data | **completed** |
| entities | — | `['NVIDIA']` |
| structured_source | — | `document_extracted`* |
| grounding_layer | — | `issuer_sec_gap_fill` |

\* Live run confirmed `fundamentals_meta.provider=sec_edgar` + `sec_filled_keys=[ebitda, operating_income, r_and_d]`; a post-retrieve label overwrite briefly tagged `document_extracted` despite SEC spine. Fixed in `agents.py` to preserve `sec_companyfacts` when `live_fallback_used=true`.

## Gate

- Numeric grounding improved: **PASS**
- Issuer isolation retained: **PASS**
- Workflow completed (quant ran): **PASS**

## Artifacts

- `C:\a_project\Projects\finagentbench-demo\outputs\lumenfin_financial_grounding\validation.json`
- `C:\a_project\Projects\finagentbench-demo\outputs\lumenfin_financial_grounding\finrun_nvidia_10k.json`
