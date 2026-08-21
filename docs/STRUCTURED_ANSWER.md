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
`outputs/structured_citation_canary_v1/`. A slim tracked record may be sealed
at `data/eval_rag/structured_citation_canary_result.json` after one clean
worktree run. That file is not a FinanceBench or LEDGER aggregate.
