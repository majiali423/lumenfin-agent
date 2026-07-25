# LumenFin RC Commit Plan

Status: Historical
Superseded by: `../current/LumenFin_Final_Release_Report.md`
Purpose: Engineering evolution and release-commit planning evidence

Base HEAD: `f13ec3d867fa53de0594a6a8c992e9c2ba1e6f6f`
Target package: `0.1.0rc1`
Suggested tag (not created): `v0.1.0-rc.1`

This plan is **not executable yet**. LumenFin commits are held until the actual
FinAgentBench `v0.1.0-rc.1` tag exists. No `git add .` is permitted.

## 1. `feat(runtime): isolate request-scoped analysis execution`

- `src/lumenfin/service.py`
- `src/lumenfin/graph.py`
- `src/lumenfin/state.py`
- `src/lumenfin/database.py`
- `src/lumenfin/artifacts.py`
- `src/lumenfin/api/{app,schemas}.py`
- `tests/test_service_concurrency.py`
- related existing API/system/failure-injection tests changed in this diff

Purpose: request-scoped memory/audit/checkpoint execution and concurrency
isolation.

## 2. `feat(grounding): bind financial claims to issuer evidence`

- `src/lumenfin/{agents,claims,critic_checks,critic_repair,document_entity,documents,document_ingest,data_ingest,evaluation,finrun,fundamentals,metrics_schema,planning,reporting,repair_policies,sec_fundamentals,sec_html,ticker_resolve,tools}.py`
- `src/lumenfin/data/sample_financial_data.py`
- grounding, claim-binding, document, metric, ticker, partial-quant and honesty
  tests under `tests/`

Purpose: previously validated financial grounding, issuer scoping, claim
binding and fail-closed reporting. This commit must not change thresholds.

## 3. `feat(rag): harden hybrid retrieval and indexing`

- `src/lumenfin/rag/**`
- `src/lumenfin/provider_retry.py`
- `src/lumenfin/input_guardrail.py`
- RAG/profile/retry/guardrail tests under `tests/`
- `scripts/run_rag_eval.py`
- `scripts/run_rag_index_worker.py`
- `scripts/demo_rag_ranking.py`

Purpose: Milvus abstraction, lexical/dedupe/rerank, async indexing and
observability.

## 4. `feat(mcp): enforce scoped deterministic tool adapters`

- `mcp_layer/**`
- MCP adapter and document-search tests
- `docs/MCP.md`
- `mcp_layer/README.md`

Purpose: optional MCP boundary. It remains separate from production PDF RAG.

## 5. `ci: pin FinAgentBench RC and add portable release gates`

Only after the FAB tag resolves:

- `.github/workflows/test.yml`
- `scripts/repo_paths.py`
- `scripts/run_tests.py`
- `scripts/e2e_verify_gates.py`
- `scripts/run_linked_coverage.py`
- `scripts/run_live_multi_stability.py`
- `scripts/run_stress_coverage.py`

Purpose: fixed `FINAGENTBENCH_REF`, locked install and diagnostic SHA output.

## 6. `fix(security): fail closed in production configuration`

- `.env.example`
- `.gitignore`
- `.dockerignore`
- `Dockerfile`
- `docker-compose.yml`
- `src/lumenfin/config.py`
- `src/lumenfin/api/app.py`
- `src/lumenfin/sec_fundamentals.py`
- `tests/test_production_mindset_guards.py`
- `tests/test_sec_fundamentals.py`
- `requirements.txt`
- `requirements-lock.txt`

Purpose: production/live Compose, required credentials, private data-service
ports, package version parity and locked dependencies.

## 7. `test(release): add controlled RC validation tooling`

Maintained release/demo scripts:

- `scripts/run_e2e_acceptance.py`
- `scripts/run_e2e_production_audit.py`
- `scripts/run_live_quality_audit.py`
- `scripts/run_live_showcase.py`
- `scripts/run_live_table_phrasing_qa.py`
- `scripts/run_live_zh_table_phrasing_qa.py`
- `scripts/check_dashscope_embedding.py`
- `scripts/check_deepseek.py`
- `run_demo.py`
- `start_api.py`

Uncertain and excluded pending owner review:

- `scripts/build_regression_comparison.py`
- `scripts/finalize_regression_readiness.py`
- `scripts/rescore_e2e_production_audit.py`
- `scripts/validate_p0_deep.py`
- `scripts/validate_p0_optimizations.py`

Fixture generation scripts and `fixtures/**` remain held pending redistribution
and ownership decisions.

## 8. `docs: add controlled-deployment RC documentation`

- `README.md`
- `CHANGELOG.md`
- tracked and new public `docs/*.md` listed in the worktree audit
- `reports/LumenFin_Worktree_Audit.md`
- `reports/LumenFin_Commit_Plan.md`
- `reports/LICENSE_RECOMMENDATION.md`
- `reports/LumenFin_Final_Release_Report.md`
- `reports/Joint_Compatibility_Report.md`
- `Release_Checklist.md`

Ignored historical root reports and local interview notes are excluded.

## 9. `chore: prepare v0.1.0-rc.1`

- `pyproject.toml`

Verify API/package version equals `0.1.0rc1`. This commit does not create a tag.

## Explicitly unstaged

- `.vscode/settings.json`
- real SEC-derived fixtures
- uncertain one-off scripts
- `.env`, DB/Milvus state, outputs, test artifacts, logs and caches

## Required verification before any tag

```bash
python scripts/run_tests.py
python ../finagentbench-demo/scripts/validate_cross_repo.py --profile ci
python ../finagentbench-demo/scripts/run_offline_demo.py
python ../finagentbench-demo/scripts/run_rc_validation.py
docker build --no-cache -t lumenfin:0.1.0-rc.1 .
```

Remote push, tag creation and GitHub Release remain explicitly unauthorized.
