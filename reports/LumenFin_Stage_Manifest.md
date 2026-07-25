# LumenFin Final Stage Manifest

Date: 2026-07-25

## Summary

```text
Files to stage:
  - src/lumenfin/** (runtime + grounding + claims + RAG)
  - start_api.py, run_demo.py
  - mcp_layer/**
  - tests/** + tests/fixtures/sec/**
  - CI/Docker/deps/lock/config
  - supported scripts (repo_paths, fetch_sec_fixtures, rag worker, builders)
  - docs + README + notices + release checklist/changelog
  - reports/current|history + archived_audits
  - fixtures/stress/MANIFEST.json (portable root)

Files to archive:
  - tools/archived_audits/**

Files intentionally ignored:
  - .env, outputs/, data/, venv, caches
  - fixtures/e2e_real/, .local-fixtures/
  - root generated LumenFin_*.md

Files requiring manual review (NOT staged):
  - docs/portfolio/INTERVIEW_NOTES.md
  - reports/LumenFin_Cleanup_Plan.md
  - reports/LumenFin_Repository_Inventory.md
  - fixtures/stress/apple_msft_fy2025_table.pdf
  - fixtures/stress/apple_msft_fy2025_table_zh.pdf
  - fixtures/stress/tsmc_fy2025_table.pdf
```
