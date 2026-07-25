# Archived Audit Scripts

These scripts preserve historical engineering evidence. They are unsupported,
not imported by current runners, not executed by CI, and must not be run
against production fixtures.

| Script | Historical phase | Problem investigated | Current replacement | Last schema/status |
|--------|------------------|----------------------|---------------------|-------------------|
| `build_regression_comparison.py` | P0/P1 before/after | Report-level regression comparison | Current release reports + `run_rc_validation.py` | Pre-FinRun 1.0; archived |
| `finalize_regression_readiness.py` | P0/P1 readiness | Ad-hoc readiness score from audit outputs | `Release_Checklist.md` and deterministic gates | Pre-FinRun 1.0; archived |
| `rescore_e2e_production_audit.py` | Initial E2E audit | Heuristic rescoring of stored states/reports | FinAgentBench cases and RC runner | Legacy heuristic; unsupported |
| `validate_p0_deep.py` | P0 grounding | Issuer leakage, compare intent and ~20 fact probes | Unit tests, issuer/compare FAB cases, RC pack | Historical fixture layout; archived |
| `validate_p0_optimizations.py` | P0 quick check | Apple/NVIDIA entity and consolidated revenue ranking | `test_document_primary_entity.py`, `test_sec_html_facts.py`, RC pack | Historical fixture layout; archived |
| `run_e2e_acceptance.py` | Early acceptance | Query/upload acceptance matrix | FinAgentBench RC runner | Historical outputs; archived |
| `run_e2e_production_audit.py` | Initial E2E | Live production-oriented audit | FinAgentBench RC runner | Historical fixtures; archived |
| `run_live_quality_audit.py` | Live quality | PDF/RAG optimization audit | RC runner + live showcase | Historical outputs; archived |
| `run_live_table_phrasing_qa.py` | Table QA | English phrasing matrix | Current table tests | Historical outputs; archived |
| `run_live_zh_table_phrasing_qa.py` | Table QA | Chinese phrasing/mismatch matrix | Current table/mismatch tests | Historical outputs; archived |
| `convert_sec_html_to_pdf.py` | Fixture generation | Full-HTML text pagination | Manifested minimal SEC fixtures | Unsafe for current fixtures; archived |

Supported release validation interfaces are limited to the FinAgentBench
repository:

1. `scripts/run_offline_demo.py`
2. `scripts/run_mutation_suite.py`
3. `scripts/validate_cross_repo.py`
4. `scripts/run_rc_validation.py`

Do not add archived scripts to CI or reference them as current release gates.
