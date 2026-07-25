# LumenFin Commit Plan (Internal RC Candidate)

Status: staging plan only — **do not auto-commit**.

## Rules

- No `git add .` / `git add -A`
- Explicit paths only
- Never stage `.env`, `outputs/`, `data/*.db`, `.venv/`, `fixtures/e2e_real/`, root `LumenFin_*.md`
- SEC policy: commit minimized extracts + manifested derived PDFs only
- No public `LICENSE` file in this internal release

## Secret handling

Local `.env` contains real provider keys and is gitignored / never tracked.
Owner should rotate keys out-of-band if they may have been exposed in chat/logs.

## Commit groups

| Commit | Purpose | Exact files | Validation before commit |
|--------|---------|-------------|--------------------------|
| 1. `feat(runtime): isolate agent systems by request` | Request-scoped runtime | `src/lumenfin/service.py`, `src/lumenfin/graph.py`, `src/lumenfin/state.py`, `src/lumenfin/config.py`, related API/start entrypoints | `python -m unittest tests.test_service_concurrency tests.test_system -v` |
| 2. `feat(grounding): add issuer financial grounding and verified claims` | Grounding/claims | `src/lumenfin/claims.py`, `src/lumenfin/document_entity.py`, `src/lumenfin/sec_html.py`, `src/lumenfin/ticker_resolve.py`, `src/lumenfin/fundamentals.py`, `src/lumenfin/sec_fundamentals.py`, `src/lumenfin/metrics_schema.py`, `src/lumenfin/provider_retry.py`, agents/critic/reporting edits | grounding + claim binding tests |
| 3. `feat(rag): add production indexing and retrieval resilience` | RAG | `src/lumenfin/rag/**`, `scripts/run_rag_index_worker.py`, embedding scripts | RAG unit tests |
| 4. `feat(mcp): harden adapters and servers` | MCP | `mcp_layer/**` | `tests.test_mcp_adapters` |
| 5. `test(reliability): add concurrency and fail-closed regression coverage` | Tests | `tests/**` (exclude `test_artifacts`), `tests/fixtures/sec/**` | `python scripts/run_tests.py` |
| 6. `ci: enforce FinAgentBench compatibility gate` | CI/deps/docker | `.github/workflows/test.yml`, `pyproject.toml`, `requirements.txt`, `requirements-lock.txt`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `.dockerignore`, `.gitignore` | CI pin note for FAB tag; compose secrets required |
| 7. `docs: curate architecture and controlled deployment guidance` | Docs | `README.md`, `docs/**`, `CHANGELOG.md`, `Release_Checklist.md`, `THIRD_PARTY_NOTICES.md`, `mcp_layer/README.md` | no broken interview links; VALIDATION_COMMANDS present |
| 8. `chore(repo): archive historical reports and clean generated files` | Archive | `reports/current/**`, `reports/history/**`, `tools/archived_audits/**` | archive headers present; no active script imports archived tools |
| 9. `chore(release): prepare internal v0.1.0rc1 candidate` | Release evidence | refreshed `reports/current/*` after offline green | offline + cross-repo green |

## Do not stage

- `fixtures/e2e_real/**` (full/large SEC downloads)
- `fixtures/stress/*.pdf` regenerated synthetics unless demo packaging requires them
- `reports/LumenFin_Cleanup_Plan.md`, `reports/LumenFin_Repository_Inventory.md` (process docs; optional history)
- `.vscode/settings.json` unless team-shared intentionally
- Any path matching secret/cache ignores

## Post-stage checks (every group)

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached --name-status
```
