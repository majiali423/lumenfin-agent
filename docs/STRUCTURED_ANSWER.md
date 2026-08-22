# Structured answer and citation protocol

Schema version: **`1.0`**. This is a machine-readable contract, **not** a product
accuracy claim. It does not retune retrieval and does not authorize opening
LEDGER `public_holdout`.

## Why this exists

Production retrieval already mints a stable `chunk_id`
(`{document_id}:p{page}:c{n}` / `:f{n}`). That ID is stored in Milvus metadata
and PostgreSQL `rag_chunks`, and it remains on `state.rag_evidence` hits.
Claim–Evidence Binding previously dropped it and kept only a display
`citation` string (`filename#pN`). FinRun therefore could not export real
chunk IDs, and LEDGER citation support (~0%/2% on the sealed public-dev
canary) could not read production evidence identities.

`HybridEvidenceRetriever.build_source_documents()` still exports page-level
display citations only. That helper is part of the sealed LEDGER ranking
source hash; this protocol does not change it. Structured citations are
taken from `rag_evidence` → verified `EvidenceRef.chunk_id`.

## Schema

```json
{
  "answer": "string",
  "citations": ["stable_chunk_id"],
  "structured_answer_schema_version": "1.0"
}
```

Peripheral fields (status, claims, evidence, `citation_source`) may exist
alongside this object. The two fields below have unique meaning:

| Field | Meaning |
|-------|---------|
| `answer` | User-facing final answer / report body (string). |
| `citations` | Ordered, de-duplicated list of **verified** stable chunk IDs that support that answer. |

`citation_source` is provenance, not a second citation list. Canonical write
values:

- `structured` — IDs taken from verified evidence objects / schema `citations`
- `legacy_structured` — IDs taken from the older LEDGER JSON field `cited_chunk_ids`
- `unavailable` — no reliable chunk IDs (including `incomplete_data` and fundamentals-only runs)

`legacy_text` is a read-only alias for unpublished drafts; writers emit
`legacy_structured`. There is **no** prose-regex citation path. Display
markers such as `[1]` are converted only through an explicit index→chunk map.

On validation failure FinRun/API do **not** keep a cleaned `structured`
object. They degrade to `citation_source=unavailable` with
`citation_validation=failed` and `citation_path=validation_failed`.

## Validation rules

Fail closed when any of the following is true:

- `answer` is not a string
- `citations` is not a string array, or contains empty / null / numeric IDs
- schema version is unknown
- a citation is missing from the current-run allowlist
- a citation is unverified
- a citation is from another tenant or session
- a citation is from a stale repair attempt
- a factual verified answer has RAG-backed evidence but zero citations

Allowed:

- `incomplete_data` / fail-loud gap with empty `citations`
- verified fundamentals-only answers with empty `citations` and
  `citation_source=unavailable` (do **not** invent chunk IDs)

`incomplete_data` is judged from `workflow_status` / `fatal_data_gap` and from
whether verified numeric/growth claims exist. The validator does not NLP-scan
report prose for ratios.

Order is first-seen order from verified evidence refs. Duplicates keep the
first position. Display markers such as `[1]` may be converted **only** through
a program-owned index→`chunk_id` map. Conversion failure emits no citation.

## Compatibility

- Existing `final_report` remains the prose field for old API clients.
- Optional API fields `answer`, `citations`, and
  `structured_answer_schema_version` are an atomic triple: all present and
  valid, or omitted (`answer=null`, empty citations, no schema version).
  `final_report` remains the prose field for old clients.
- FinRun envelope `schema_version` (`FINRUN_SCHEMA_VERSION`) is independent of
  `structured_answer_schema_version` even when both currently equal `1.0`.
- Historical FinRun artifacts and sealed LEDGER aggregates are not rewritten.

## FinRun mapping

Exporter re-validates the structured object against current-run evidence.
Illegal citations are not exported as `structured`. Provenance records
`citation_validation` and `citation_path`. Evidence entries may still carry
display `citation` strings. `chunk_id` is added on evidence rows when known.

## LEDGER reading

Evaluator prefers `citations` + schema `1.0`. The existing eval JSON field
`cited_chunk_ids` remains readable as `legacy_structured`. Missing structured
IDs are `unavailable`, not guessed. This protocol does not change sealed
public-dev metrics and is not an accuracy improvement.

## Failure behavior

Validators raise short errors. Messages must not include API keys, provider
payloads, tenant secrets, or full filing text.

## Synthetic contract canary

Offline, deterministic, synthetic-only. It proves the production citation
path (`chunk_document` → binding → structured answer → API triple → FinRun
→ LEDGER evaluator) fail-closes on illegal IDs. It is **not** product
accuracy, RAG recall, FinanceBench, or a LEDGER benchmark.

```powershell
python scripts/run_structured_citation_canary.py --output-dir outputs/structured_citation_canary_v1
```

The CLI refuses `public_holdout`, `--allow-remote`, and non-empty output
overwrite. Raw artifacts stay gitignored under
`outputs/structured_citation_canary_v1/`. Slim tracked record:
[`../data/eval_rag/structured_citation_canary_result.json`](../data/eval_rag/structured_citation_canary_result.json)
(executed at `6cc08c4`, `config_hash`
`6f85a617a16446afc17b940919bc57c10b397b588279466aa824e93e8536f2fa`,
`cases_failed=0`, `remote_request_count=0`). That file is not a
FinanceBench or LEDGER aggregate and is not a product-accuracy claim.

## LEDGER public/dev structured-citation shadow (recorded)

Recorded **exposed public/dev** `sealed_candidate_replay_shadow` at
`fc77288…`. Slim tracked ledger:
[`../data/eval_rag/ledger_structured_citation_shadow_result.json`](../data/eval_rag/ledger_structured_citation_shadow_result.json).
Raw official files stay gitignored under
`outputs/ledger_structured_citation_shadow_v1/`. The raw JSON has no
`status` field; the ledger records `seal_status=RECORDED_COMPLETE` from
exit 0 and 50/50 completed cases. Execution gate passed.
Structured-citation quality gate failed: 22/50 structured answers, 18/29
unknown citations, and 11 valid citations that did not form a
gold-supported claim. This is not held-out, not product accuracy, not
live retrieval, not a LEDGER benchmark, and not rc5. Do not retune
prompt or RAG from this exposed result. Possible follow-up audits
(free-form chunk IDs, evaluator strictness, candidate/gold mismatch,
claim-citation binding) are hypotheses only.

Frozen config:
[`../data/eval_rag/structured_citation_shadow_config.json`](../data/eval_rag/structured_citation_shadow_config.json).
Candidate-cache identity:
[`../data/eval_rag/structured_citation_shadow_cache_manifest.json`](../data/eval_rag/structured_citation_shadow_cache_manifest.json).
CLI: `scripts/run_ledger_structured_citation_shadow.py`.
Runtime embedding and reranker stay disabled. Gold values never enter the
generator prompt. The accepted v2 preflight (`f3179e05…` at `f69f133…`)
is `SUPERSEDED_BEFORE_SHADOW`. Do not rerun or resume this shadow. The v1
official preflight artifact is `INCOMPLETE_PREFLIGHT_AUDIT_SCHEMA`
(sha256 `755a7f60…`). Current config hash is `54f6e300…`.
