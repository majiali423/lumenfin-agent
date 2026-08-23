# Autumn recruiting evidence

Interview-facing summary of LumenFin as a controlled research-agent
project. Numbers below are already sealed. Do not rerun consumed
evaluations to refresh them.

## 1. Architecture and personal contribution

LumenFin is an evidence-grounded financial research agent:
`query → plan → retrieve → analyze → critic/repair → claim bind → synthesize`.
Public citations are stable retrieval `chunk_id`s (schema `1.0`). Generator
facing evidence is an ephemeral Top-10 alias window
(`citation_alias_protocol.v1`, `E01`…`E10`). Invalid aliases fail closed.
API `answer` / `citations` / schema version are an atomic triple.

This work added the alias contract, default-deny evaluation authorization,
an independent 8-case live-model synthetic compliance canary, and the
`0.1.0rc5` packaging/evidence close.

## 2. FinanceBench retrieval (page-level, not product accuracy)

Recorded confirmation-50 (consumed; do not rerun):

| Metric | Value |
|--------|-------|
| Hit@5 | 0.50 |
| Hit@10 | 0.62 |
| MRR | 0.2955 |
| nDCG@10 | 0.3461 |

These are **page-level retrieval** scores. They are not answer accuracy,
not financial Q&A accuracy, and not a product claim. `public_holdout`
was not opened. Phase 4 remains `NOT_RUN`.

## 3. FinAgentBench mutation

Required CI pin FinAgentBench `v0.1.0-rc.3`: core mutation **4/4**.
Compatibility lane against published `v0.1.0-rc.4`: extended **7/7**.
Neither lane uses FinAgentBench `master`.

## 4. Offline structured-citation synthetic canary

`scripts/run_structured_citation_canary.py` is offline and deterministic.
It proves the production citation path fail-closes on illegal IDs. It is
not product accuracy and not a LEDGER/FinanceBench score.

## 5. Live-model synthetic alias compliance

Independent suite `synthetic_remote_alias_compliance_canary` (8 fictional
cases). Claim name only:
**live-model synthetic alias protocol compliance**.

| Gate | Result |
|------|--------|
| protocol_gate_passed | true |
| synthetic_evidence_gate_passed | true |
| cases | 8/8 |
| provider_errors | 0 |
| remote calls | 8 logical / 8 recorded / 8 HTTP attempts |
| config hash | `da4cdc4ef515be5fbc95b67cf39809b031564d4e2f80bd85419a341af6d94445` |
| execution commit | `030bf725131c8323aaa5cfc4de946fcc254809d6` |

Tracked ledger:
[`../data/eval_rag/synthetic_alias_compliance_result.json`](../data/eval_rag/synthetic_alias_compliance_result.json).

## 6. Why the old LEDGER public/dev support metric is invalid

Sealed exposed public/dev shadow: 22/50 structured answers, 18 unknown
citations. Recorded `supported_claims=0` is **not** a valid support rate:
official scoring received empty qrels. A later read-only diagnosis found
window mismatch (Top-20 / Top-10). There is **no repaired public/dev
score**. Do not rerun or rescore. `public_holdout` stays closed.

## 7. Reproducible identity

- LumenFin package: `0.1.0rc5`
- Tag: `v0.1.0-rc.5`
- Canary config: `da4cdc4e…`
- Dataset: `0d982240…`
- Implementation ancestor: `f91c474…`
- Execution HEAD: `030bf725…`

## 8. Do not claim

- Product or financial-answer accuracy
- Held-out product evaluation
- A repaired LEDGER public/dev score
- That confirmation-50 Hit@5 is chatbot accuracy
- That this canary selected a production model or tuned prompts

## 9. Resume bullets

- Built a fail-closed financial research agent with planner–critic–repair,
  verified `chunk_id` citations, and an ephemeral E01–E10 alias window.
- Isolated evaluation from production: default-deny authorization,
  one-shot preflight/remote budgets, and no retune on exposed sets.
- Closed a live-model 8-case synthetic alias-compliance canary (8/8,
  protocol + synthetic evidence gates) without opening `public_holdout`.

## 10. 60-second talk

I built LumenFin as a controlled research agent, not a chatbot demo.
The model never sees stable chunk IDs; it can only cite E01–E10, and
the program maps those aliases or fail-closes. I kept evaluation
governance separate from product claims: FinanceBench numbers are
page-level retrieval, the old LEDGER support=0 is invalid because
qrels were unbound, and I did not rerun consumed public/dev. The rc5
close is an 8-case DeepSeek canary on fictional memos that only
proves alias-protocol compliance, then the grant closes.
