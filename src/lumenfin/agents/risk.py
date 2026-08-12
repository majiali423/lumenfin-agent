from __future__ import annotations

from ..parallel import map_in_parallel
from ..state import FinanceState
from ..tools import analyze_sentiment_deep


class RiskMixin:
    def psychologist(self, state: FinanceState) -> FinanceState:
        with self._track_step("psychologist") as timer:
            company_items = list(state["retrieved_docs"].items())
            sentiment_results = map_in_parallel(
                lambda item: (
                    item[0],
                    analyze_sentiment_deep(item[1].get("earnings_call_quotes", []), llm_client=self.llm_client),
                ),
                company_items,
                max_workers=self.company_parallelism,
            )
            sentiment_analysis = {company: sentiment for company, sentiment in sentiment_results}

            detail_parts = []
            for company, sentiment in sentiment_analysis.items():
                tone = sentiment.get("label", "unknown")
                conf = sentiment.get("confidence_score", "N/A")
                themes = sentiment.get("key_themes", [])
                detail_parts.append(f"{company}: {tone} (confidence:{conf}/10, themes: {', '.join(themes[:2])})")

            update = {"sentiment_analysis": sentiment_analysis}
            update.update(self._record(
                "psychologist",
                "ok",
                f"Deep sentiment intelligence extracted (parallel fan-out): {'; '.join(detail_parts)}",
                state,
                timer.metrics(),
            ))
            self.session_memory.save({**state, **update})
            return update
