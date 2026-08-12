from __future__ import annotations

from ..input_guardrail import guard_documents
from ..state import FinanceState


class InputGuardrailMixin:
    def input_guardrail(self, state: FinanceState) -> FinanceState:
        documents = state.get("document_contexts", [])
        if not self.input_guardrail_enabled or not documents:
            empty = guard_documents([], mode=self.input_guardrail_mode)
            summary = empty.to_dict()
            update: FinanceState = {
                "input_guardrail_findings": summary["findings"],
                "input_guardrail_summary": summary,
            }
            update.update(self._record("input_guardrail", "ok", "No uploaded documents to scan.", state))
            return update

        with self._track_step("input_guardrail") as timer:
            result = guard_documents(documents, mode=self.input_guardrail_mode)
            summary = result.to_dict()
            if not result.allowed:
                detail = result.blocked_reason or "Uploaded document blocked by input guardrail."
                update = {
                    "document_contexts": result.sanitized_documents,
                    "input_guardrail_findings": summary["findings"],
                    "input_guardrail_summary": summary,
                    "workflow_status": "blocked_by_guardrail",
                    "final_report": (
                        "Analysis halted: uploaded PDF content matched critical prompt-injection patterns. "
                        "Please remove adversarial instructions from the source document and retry."
                    ),
                }
                update.update(self._record("input_guardrail", "blocked", detail, state, timer.metrics()))
                self.session_memory.save({**state, **update})
                return update

            critical_count = summary.get("critical_count", 0)
            finding_count = summary.get("finding_count", 0)
            if finding_count:
                detail = (
                    f"Sanitized {finding_count} prompt-injection pattern(s) "
                    f"({critical_count} critical) across {len(documents)} document(s)."
                )
                status = "sanitized"
            else:
                detail = f"Scanned {len(documents)} uploaded document(s); no injection patterns detected."
                status = "ok"

            update = {
                "document_contexts": result.sanitized_documents,
                "input_guardrail_findings": summary["findings"],
                "input_guardrail_summary": summary,
            }
            update.update(self._record("input_guardrail", status, detail, state, timer.metrics()))
        self.session_memory.save({**state, **update})
        return update
