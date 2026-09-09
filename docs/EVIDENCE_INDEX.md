# Evidence index (historical, do not delete)

Homepage READMEs keep the product story short. **Original reports remain.**
This index only points at them. Do **not** merge rows into one accuracy number.
Do **not** rerun FinanceBench confirmation-50 or LEDGER public_holdout to tune
the product.

## Same-product line

| Artifact | What it is | What it is not |
|----------|------------|----------------|
| `python scripts/run_portfolio_demo.py` | Offline A/B/C on the product graph | FinAgentBench market ranking |
| FinAgentBench `validate_cross_repo.py --profile ci` | Frozen FinRun contract vs pinned evaluator | Live product accuracy |
| UI bench drawer | Clean export → inject `999999%` → gate locates | 14/14 or 11/11 as this-page QA score |

## Dated unit / infra snapshots

Kept in README history and release reports. Treat as **dated**, not HEAD.

| Gate | Result | Primary source |
|------|--------|----------------|
| LumenFin 2026-09-08 R5 isolated venv | **1061** passed, 2 skipped (~209s) | [PHASED_IMPROVEMENT_LOG.md](PHASED_IMPROVEMENT_LOG.md) R5 |
| NVIDIA FY2025 excerpt upload gold | 81.453 billion USD OI; SHA256 `7f85d2c3…b68208` | `tests/fixtures/sec/nvda_fy2025_operating_income_gold.json` |
| LumenFin RC-tag regression (`v0.1.0-rc.3` Linux image) | 495 passed, 2 skipped | [PRODUCTION_LIMITATIONS.md](PRODUCTION_LIMITATIONS.md), [PORTFOLIO_RELEASE_REPORT.md](PORTFOLIO_RELEASE_REPORT.md) |
| LumenFin 2026-08-13 post-rc4 snapshot | 512 passed, 3 skipped | same |
| FinAgentBench unit regression (dated) | 149 passed | FinAgentBench reports |
| Queue/worker multi-process Docker | PASS `20260804T095357Z` | [QUEUE_WORKER_INTEGRATION.md](QUEUE_WORKER_INTEGRATION.md) |
| Worker-kill reclaim | no human redelivery | same |
| Tenant leakage | 0 | [MULTI_TENANCY_BOUNDARY.md](MULTI_TENANCY_BOUNDARY.md) |
| Provider fault Docker | PASS `docker_20260804T100817Z` | [PROVIDER_RESILIENCE.md](PROVIDER_RESILIENCE.md) |
| FinAgentBench completed-case mean | 92.97 informational under pin `v0.1.0-rc.1` | not the published rc3 evaluator score |
| Core mutation (local / evaluator pin) | 4/4 | DEMO_GUIDE + FinAgentBench mutations |
| Evaluator pin `v0.1.0-rc.3` | PASS schema 1.0; core 4/4; extended 7/7 | CI `finrun-contract` |
| Evaluator `v0.1.0-rc.4` | fail-closed compatibility lane | CI matrix; does not replace rc3 pin |
| Native BM25 + Qwen3 synthetic | Top-1/MRR 1.0/1.0 on that canary | **not** FinanceBench |

HEAD test counts belong in [PHASED_IMPROVEMENT_LOG.md](PHASED_IMPROVEMENT_LOG.md)
and GitHub Actions `ci.yml` on the current commit.

## Retrieval canaries (sealed)

| Gate | Result | Doc |
|------|--------|-----|
| FinanceBench confirmation-50 | consumed Hit@10 **0.62**; Phase 4 `NOT_RUN` | [FINANCEBENCH_EVAL.md](FINANCEBENCH_EVAL.md) |
| LEDGER public-dev | sealed / stopped | [FINANCEBENCH_NEXT_PHASE.md](FINANCEBENCH_NEXT_PHASE.md) |
| LEDGER public_holdout E2E v2 | **35/100** strict verified; Wilson 95% CI in source doc; dataset-specific | [LEDGER_PUBLIC_HOLDOUT_E2E.md](LEDGER_PUBLIC_HOLDOUT_E2E.md) |
| LEDGER public_holdout index dry-run | document-only; not a score | [LEDGER_PUBLIC_HOLDOUT_INDEX.md](LEDGER_PUBLIC_HOLDOUT_INDEX.md) |
| Structured-citation shadow | execution passed, citation quality failed | [STRUCTURED_ANSWER.md](STRUCTURED_ANSWER.md) |

## Product-dev set (Phase 4)

Hand gold catalog `data/eval_product_dev/catalog_v1.json` (36 items; train/dev/test
12/16/8). Offline ablation on **dev only**. Test split frozen unscored.
**Not** an 80%/95% claim. Log: [PHASED_IMPROVEMENT_LOG.md](PHASED_IMPROVEMENT_LOG.md).

## Interview / autumn recruiting

[AUTUMN_RECRUITING_EVIDENCE.md](AUTUMN_RECRUITING_EVIDENCE.md) ·
[AUTUMN_RECRUITING_DEMO_SCRIPT.md](AUTUMN_RECRUITING_DEMO_SCRIPT.md)

## Frozen hashes (do not rewrite)

| Object | Hash |
|--------|------|
| FinAgentBench `case_lumenfin_diligence` | `b0c9e003d7049ab06bf4a0b5cc9c8acf714d8fd89cf5ae1b16e9825da0dcf5fe` |

More hashes: Phase 0 section of [PHASED_IMPROVEMENT_LOG.md](PHASED_IMPROVEMENT_LOG.md).
