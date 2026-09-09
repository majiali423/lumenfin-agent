from __future__ import annotations

from ..bounded_repair import run_bounded_repair
from ..state import FinanceState


class BoundedRepairMixin:
    def bounded_repair(self, state: FinanceState) -> FinanceState:
        with self._track_step("bounded_repair") as timer:
            deadline = float(getattr(self, "bounded_repair_deadline_seconds", 2.0) or 2.0)
            update = run_bounded_repair(
                dict(state),
                allow_sample_data=bool(self.allow_sample_data),
                deadline_seconds=deadline,
            )
            detail = (
                f"Bounded repair steps={len(update.get('bounded_repair_trace') or [])}; "
                f"fatal={update.get('fatal_data_gap')}; "
                f"tools={[item.get('tool') for item in update.get('bounded_repair_trace') or [] if item.get('tool')]}."
            )
            update.update(self._record("bounded_repair", "ok", detail, state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update
