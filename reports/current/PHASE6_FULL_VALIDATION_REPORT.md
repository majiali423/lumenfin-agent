# Phase 6 full validation report

Validation date: 2026-08-12  
Scope: current uncommitted LumenFin `0.1.0rc3` and FinAgentBench `0.1.0rc4`
worktrees  
Result: **PASS for controlled local RC validation**

> Phase 7 later raised package versions to LumenFin `0.1.0rc3` /
> FinAgentBench `0.1.0rc4` and adjusted Compose stop-grace periods. Fresh
> image rebuild and gate re-run evidence is recorded after those fixes.

This report records the current worktrees, not the historical HEAD commits.
Both trees remain intentionally dirty pending the separately approved Phase 7
commit review. No commit, push, tag, release, collection deletion, or volume
deletion was performed.

## Full suites and cross-repository contract

| Gate | Result |
|------|--------|
| LumenFin full Linux image suite | **495 passed, 2 skipped** (`scripts/run_tests.py`) |
| FinAgentBench full suite | **149 passed** |
| Cross-repository FinRun gate | **PASS**, schema `1.0`, score `100.0` |
| Core reliability mutations | **4/4 detected** |
| Extended provenance/period mutations | **7/7 detected** |
| Total negative controls | **11/11 detected** |
| Documentation links | **9/9 release documents resolved** |
| MIT package metadata | **PASS** in both repositories |
| `git diff --check` | **PASS** in both repositories |

The first LumenFin container invocation mounted the test workspace read-only
and produced 98 `EROFS` setup errors because tests create ignored temporary
databases under `test_artifacts/`. The corrected UID-10001 invocation used a
writable test mount and passed all 495 tests; there were no assertion failures
in the read-only attempt.

## Image and Compose

| Check | Result |
|-------|--------|
| Compose parse | **PASS** |
| Fresh application image build | **PASS** |
| Image | `lumenfin-agent:0.1.0rc3` |
| Image ID | `sha256:be17b80997621c03970a7160643970ac776a9e2ee967ebdf94d0555fe3f99ced` |
| Runtime identity | UID/GID `10001:10001` |
| Python / package | Python `3.12.13`; LumenFin `0.1.0rc3` |
| Retrieval packages | PyMilvus `3.0.0`; Jieba `0.42.1` |
| Deep readiness | PostgreSQL, Redis, and Milvus **healthy** |
| Ready collection | `lumenfin_chunks_v4_bm25` |

The first migration attempt intentionally used `--no-deps` while PostgreSQL
was stopped and confirmed the migrator fails loudly after its bounded wait.
Starting dependencies to healthy state and rerunning the documented migration
step applied migrations successfully. This was a harness-order correction, not
a migration defect.

## Retrieval and provider gates

| Gate | Result |
|------|--------|
| First application search | dense `2` hits; native BM25 `2` hits |
| Native BM25 corpus | **5/5** BM25 and **5/5** hybrid; R@3 `1.0`, MRR `1.0` |
| Offline RAG corpus | **4/4**; R@3 `1.0`, MRR `1.0`, citation coverage `100%` |
| Qwen3 hard negatives | Top-1 `1.0`, MRR `1.0`, nDCG@5 `0.971` |
| Qwen3 resilience | `0` fallbacks; telemetry **PASS** |
| Production synthetic upload | dense `1`, BM25 `1`, keyword `1`; Qwen3 one attempt |
| Production retrieval mode | `hybrid_dense_bm25_rrf+qwen3_rerank` |
| Retrieval degradation | `false`; rerank fallback count `0` |

Only repository-owned synthetic text was used for live provider calls. No
user-uploaded, recruitment, or real filing document was sent externally. The
company-scoped synthetic Apple run exercised the complete retrieval/rerank
path and then returned `incomplete_data`, correctly failing closed because the
fabricated upload was insufficient for a verified financial conclusion.

Timeout, 429 retry, malformed response, missing-key fallback, concurrency, and
redaction paths are included in the passing LumenFin full suite. The live
provider gate itself had no fallback and therefore did not manufacture a
provider outage.

## Security, backup, and shutdown

| Check | Result |
|-------|--------|
| Missing API key | HTTP `401` |
| Public config | PostgreSQL backend only; no DB path/URL |
| Literal configured secrets in Compose logs | `0` |
| Credential-bearing URLs in logs | `0` |
| Unredacted secret assignments in logs | `0` |
| Backup | `backups/production-20260812-190216` |
| Backup hashes/archive integrity | **PASS** |
| Backup automatic writer recovery | **PASS**, all services healthy |
| Final graceful stop | API, both workers, PostgreSQL, Redis, etcd, MinIO, Milvus all exit `0` |
| Post-stop secret rescan | `0` literal secret and credential-URL matches |

The backup and all existing Docker volumes, Milvus collections, uploaded
synthetic probes, and historical artifacts were retained.

## Boundary for Phase 7

Phase 6 validates the current local closure but does not make the worktrees
release artifacts. Phase 7 still requires a deliberate file-by-file commit
review, fresh commit(s), clean-tree rerun as appropriate, remote tag
inspection, and explicit approval before any commit, push, tag, or release.

## Phase 7 revalidation after version and runtime fixes

| Check | Result |
|-------|--------|
| LumenFin package / tag plan | `0.1.0rc3` / `v0.1.0-rc.3` |
| FinAgentBench package / tag plan | `0.1.0rc4` / `v0.1.0-rc.4` |
| FinAgentBench producer pin | LumenFin `v0.1.0-rc.3` |
| Compose stop grace | API/worker `150s`; index worker `210s`; Milvus `60s` |
| Backup unpause cleanup | continues remaining unpauses before failing |
| Image rebuild | **PASS** after pinning `setuptools==78.1.1` and splitting Docker dependency layers |
| Full Linux suite | **495 passed, 2 skipped** |
| FinAgentBench suite | **149 passed** |
| Cross-repo FinRun gate | **PASS**, score `100`, mutations `11/11` |
| Compose ready + first search | **PASS** on `lumenfin_chunks_v4_bm25` |
| Graceful stop | all services exit `0` |
