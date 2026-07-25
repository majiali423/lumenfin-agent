# Arbitrary-company ticker resolution

## Why

Alias tables (`COMPANY_HINTS` / `KNOWN_ALIASES`) cannot cover every issuer.
Live SEC/Yahoo calls need a **ticker**. This module bridges:

`company label or bare ticker` → `exchange symbol` via:

1. curated `DEFAULT_TICKER_MAP` (fast path, non-US aliases)
2. SEC public `company_tickers.json` (ticker token + conservative title match)
3. explicit `ticker:` / `(SYM)` in the query (existing rules, no spray)

## Wiring

| Call site | Behavior |
|-----------|----------|
| `ticker_resolve.py` | directory cache + resolve/enrich |
| `sec_fundamentals.resolve_cik` | CIK via shared directory |
| `tools.derive_target_symbols` | symbols for retrieval |
| `planning._merge_query_companies` | enrich LLM/rule names; pick up bare tickers |

## Limits

- US-listed focus (SEC file). HK/CN still rely on curated map or uploads.
- Title match is conservative (exact normalized / unique substring) to avoid wrong peers.
- Offline: fail soft; unresolved names keep working for document-only paths.
