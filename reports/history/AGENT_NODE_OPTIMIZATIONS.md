# Agent Node Cost / Correctness Fixes

Status: Historical
Superseded by: `../current/LumenFin_Final_Release_Report.md`
Purpose: Engineering evolution and implementation-plan evidence

Engineering plan for five known inefficiencies. Implemented in this change set.

## Decisions

| Item | Decision | Rationale |
|------|----------|-----------|
| Supervisor LLM | **Remove** chat call; template `task_brief` from `query_plan` | Parsed dims/questions were unused; planner already owns structure |
| Company fallback | Rules-only `extract_companies_from_query(..., llm_client=None)` | Avoid second LLM company extract |
| Profile LLM | `MAS_PROFILE_LLM_MAX_ATTEMPTS` default **1** (clamp 0–3) | Profiles are narrative, not AST truth |
| Psychologist LLM | Skip when quotes are weak/placeholder | Avoid confident themes from empty excerpts |
| Critic report checks | **Drop from `run_critic_checks`** | `report_sections` empty pre-synthesizer; synthesizer templates already include disclaimer/sources |
| Repair → retrieval | Only for `missing_structured_data` / `low_retrieval_confidence` | Stop full re-fetch loops for soft gaps |

## Config

- `MAS_PROFILE_LLM_MAX_ATTEMPTS` (default `1`)

## Non-goals (this pass)

- Post-synthesizer critic pass (optional later)
- Function calling for planner
- Changing AST / fail-closed gates
