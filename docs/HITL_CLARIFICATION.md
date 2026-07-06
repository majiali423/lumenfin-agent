# Human-in-the-loop Clarification

When Query Planner detects missing **company** or **time range**, the graph pauses before Supervisor.

## Flow

```text
query_planner
  |- missing fields and no clarification -> await_clarification -> END
  `- clarified or complete -> supervisor -> ...
```

## API

### 1. Start analysis (may pause)

```http
POST /api/v1/analyze
{
  "query": "Analyze supply chain risk and R&D spend.",
  "thread_id": "session-001"
}
```

Response when clarification is needed:

```json
{
  "workflow_status": "needs_clarification",
  "clarification_questions": [
    "Which company should be analyzed (e.g. Apple, Microsoft)?",
    "What time range or fiscal year (e.g. FY2025)?"
  ],
  "final_report": "",
  "thread_id": "session-001"
}
```

### 2. Resume from checkpoint

```http
POST /api/v1/clarify
{
  "thread_id": "session-001",
  "clarification": {
    "company": "Apple",
    "time_range": "FY2025"
  }
}
```

Uses LangGraph `update_state` + `invoke(None)` to continue from `query_planner` with merged context.

## Session storage

HITL checkpoints are persisted in SQLite (`workflow_checkpoints` table on the same DB as jobs):

| Column | Purpose |
|--------|---------|
| `thread_id` | Resume key for `/clarify` |
| `state_json` | Serialized LangGraph workflow state |
| `workflow_status` | e.g. `needs_clarification`, `completed` |
| `clarification_questions` | Questions shown in UI |
| `last_node` | Graph node used to bootstrap resume |
| `created_at` / `updated_at` | Audit timestamps |

`LumenFinAnalysisService` keeps one in-process `LumenFinAgentSystem`, but **checkpoint data survives API restarts**. After restart, `/clarify` loads the thread from SQLite and seeds LangGraph via `update_state`.

## Summary

> Planner detects underspecified tasks, returns structured questions, and resumes from the same `thread_id`. Checkpoints are SQLite-backed so a local demo survives process restarts without Redis/Postgres.
