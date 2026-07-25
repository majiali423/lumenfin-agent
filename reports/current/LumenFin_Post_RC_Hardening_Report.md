# LumenFin + FinAgentBench Post-RC Hardening Report

Generated: 2026-07-25

Scope: production concurrency isolation, fail-closed evaluation, mutation CI,
and cross-repository reproducibility. No Agent, claim rule, benchmark threshold,
or `min_score` was added/relaxed.

---

## 1. Changes

| Issue | Root Cause | Code Change | Validation |
|-------|------------|-------------|------------|
| Request state could cross-contaminate | `LumenFinAnalysisService` cached one `LumenFinAgentSystem`; `run()` clears shared reasoning memory while graph/checkpointer/audit are mutable | `_system_for()` now creates a request-scoped system; provider/RAG infrastructure remains shared behind an initialization lock; graph constructor accepts shared RAG resources | Concurrent NVIDIA/Apple test asserts isolated systems, memories, checkpoints, entities, reports, verified claims, evidence and audit traces |
| Empty evaluation could pass | `unit_currency_consistency` and `temporal_consistency` returned 100 when `checked == 0`; empty evidence coverage had no shared fail-closed path | Reused `empty_check_result()` for unit/currency, temporal and evidence coverage when `require_checkable_metrics=true`; numeric and evidence consistency remain fail-closed | New empty numeric/evidence/unit/temporal tests; FinAgentBench full suite PASS |
| Mutation detection was documented but not enforced | Four mutations existed in an ad-hoc correctness script; CI regression suite did not require all four | Added deterministic `benchmarks/mutations/suite.json`, `run_mutation_suite.py`, unit test, CI step and report artifact upload | wrong number/entity and missing citation/risk all detected; detection rate 1.0 |
| Cross-repository runs depended on one workstation | Historical scripts embedded `C:\a_project\...`; no portable validation entry; LumenFin CI did not invoke FinAgentBench | Added sibling/env repository discovery; removed Python hardcoded workspace paths; added offline `validate_cross_repo.py`; LumenFin CI checks out/pins the benchmark branch and runs the gate | Portable gate exported sample state with current LumenFin exporter, FinAgentBench gate PASS, mutation gate PASS |
| Dependency resolution was floating | CI/Docker installed unconstrained direct dependencies; `requirements.txt` differed from `pyproject.toml` | Added pip-compiled `requirements-lock.txt`; compatibility requirements file delegates to lock; Docker/CI install lock then package `--no-deps` | Lock generated from `pyproject.toml`; regression and live RC completed on locked project environment |

### Concurrency boundary after the change

```text
HTTP request / queue job
  → new LumenFinAgentSystem
      → private SessionMemory
      → private ReasoningMemory
      → private LangGraph InMemorySaver
      → private AgentRuntime / audit state
  → shared provider clients and shared RAG store/indexer
  → persisted workflow checkpoint keyed by thread_id
```

The shared RAG store is infrastructure, not per-request FinanceState. Reusing
one in-process Milvus client avoids opening competing Lite connections to the
same file. Document deduplication remains repository-backed.

---

## 2. Production Risk Status

| Risk | Before | After |
|------|--------|-------|
| Concurrency isolation | One mutable system for all requests | Request-scoped graph/memory/checkpointer; concurrent issuer isolation regression |
| Empty evaluation | Some zero-check metrics scored 100 | Required checkability returns score 0 / not passed |
| Mutation regression | 4/4 observed manually | 4/4 enforced by unit test and GitHub Actions |
| Reproducibility | Local absolute paths, floating dependencies, manual linked gate | Relative/env discovery, lockfile, one offline cross-repo command, LumenFin CI → FinAgentBench |

---

## 3. Validation

### LumenFin

- Full offline regression: **263 tests PASS**
- New concurrent NVIDIA / Apple isolation: **PASS**
- Existing HITL restart and RAG indexing lifecycle tests: **PASS**

### FinAgentBench

- Full unit suite: **72 tests PASS**
- Empty-check fail-closed focused suite: **PASS**
- Mutation gate: **4/4 detected**
- Correctness validation: wrong revenue, missing citation, missing risk and wrong
  company all detected
- Cross-repository sample FinRun gate: **PASS**

### RC validation after hardening

- Offline gates: **PASS**
- Live RC pack: **8/8 PASS**
- Completed-case mean FAB score: **92.97** (informational; thresholds unchanged)
- Fail-closed OpenAI and sparse upload-only cases: **PASS**
- Final process exit: **0**

Mutation report:
`finagentbench-demo/outputs/mutation_detection_report.{json,md}`

Cross-repository validation:
`finagentbench-demo/outputs/cross_repo_validation/validation_summary.json`

---

## 4. Final Assessment

### 4.1 Production risks closed

1. Per-request reasoning, audit, checkpoint and graph state no longer share one
   mutable Agent system.
2. Required reliability checks cannot receive a perfect score solely because
   there is nothing to check.
3. Mutation sensitivity is now a CI invariant, not a report claim.
4. A third party can clone sibling repositories (or set root environment
   variables) and run one offline linked gate without the original drive layout.
5. CI and Docker dependency resolution use a committed lock.

### 4.2 Risks still present

- **Milvus Lite remains single-process local infrastructure.** Request state is
  isolated, but high-throughput multi-worker production should use Milvus
  service or a single queued indexing worker.
- **SQLite/checkpoint deployment is not a multi-tenant HA store.**
- **External API/model availability** (SEC, Yahoo, DeepSeek, embeddings) still
  affects live runs and requires operational monitoring.
- Shared injected provider test doubles/clients must themselves be thread-safe;
  production-created Agent clients should remain stateless or isolated.
- The cross-repository workflow currently references the benchmark branch; a
  release should pin a commit SHA/tag after both repositories are published.
- Live-source structured citations differ from filing page (`#pN`) citations.
- Real-company RC coverage is meaningful but finite.

### 4.3 Production-oriented Agent standard

**Yes — for a production-oriented research Agent / controlled deployment.**

The system now has request-state isolation, explicit fail-closed semantics,
deterministic mutation gates, locked dependencies, portable linked validation,
and a post-change live RC pass.

It is **not yet an internet-scale, multi-worker HA service**. That standard
would additionally require managed Milvus, a production checkpoint/database
topology, queue backpressure, load testing, provider SLO monitoring and pinned
cross-repository release tags.

---

## Reproduction

```powershell
# Both repositories cloned as siblings:
cd finagentbench-demo
python scripts\validate_cross_repo.py
```

Alternative roots:

```powershell
$env:LUMENFIN_ROOT = "path\to\lumenfin-agent"
$env:FINAGENTBENCH_DIR = "path\to\finagentbench-demo"
python scripts\validate_cross_repo.py
```
