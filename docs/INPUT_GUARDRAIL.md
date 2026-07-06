# PDF Prompt-Injection Guardrail

Uploaded PDFs are untrusted input. Before RAG indexing or LLM consumption, `input_guardrail` scans page text for adversarial instruction patterns.

## Graph placement

```text
START -> input_guardrail -> query_planner -> ...
                |
                +-- blocked -> END
```

## Modes

| Env | Default | Behavior |
|-----|---------|----------|
| `MAS_INPUT_GUARDRAIL_ENABLED` | `true` | Toggle guardrail node |
| `MAS_INPUT_GUARDRAIL_MODE` | `sanitize` | Redact matches; `block` halts on critical patterns |

## Detection coverage

English patterns:

- Ignore / disregard previous instructions
- Role override (`You are now ...`)
- Policy override / prompt exfiltration
- `System:` / `Assistant:` role markers and XML instruction tags

Chinese patterns (stored as Unicode escapes in source for encoding safety):

- ignore prior instructions (忽略…指令)
- role override (你现在是)
- policy override (无视系统/安全/规则)

Matched spans are replaced with `[REDACTED_INJECTION]` and recorded in `state.input_guardrail_findings`.

## Blocked response

When `MAS_INPUT_GUARDRAIL_MODE=block` and a critical pattern is found:

```json
{
  "workflow_status": "blocked_by_guardrail",
  "final_report": "Analysis halted: uploaded PDF content matched critical prompt-injection patterns..."
}
```

## Summary

> PDFs are untrusted prompts. A dedicated guardrail node scans before RAG/LLM, redacts injection spans by default, and can hard-block critical hits with audit evidence. CJK patterns use Unicode-safe regex literals in code.
