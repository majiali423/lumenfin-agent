from __future__ import annotations

import json

from ..critic_checks import run_critic_checks
from ..critic_repair import classify_critic_violations, compliance_messages
from ..repair_policies import RETRIEVAL_WORTHY_CODES
from ..state import FinanceState


class CriticMixin:
    def critic(self, state: FinanceState) -> FinanceState:
        with self._track_step("critic") as timer:
            violations = run_critic_checks(state)
            findings = compliance_messages(violations)
            risk_scores: dict[str, dict[str, float]] = {}

            for company in state["companies"]:
                supply_chain = state.get("retrieved_docs", {}).get(company, {}).get("supply_chain", {})
                risk_level = supply_chain.get("risk_level", "low")
                sentiment = state.get("sentiment_analysis", {}).get(company, {})
                metrics = state.get("financial_metrics", {}).get(company, {})

                scores: dict[str, float] = {}
                ebitda_m = metrics.get("ebitda_margin", 0.15)
                scores["financial_risk"] = round(max(1.5, min(9.5, 9.0 - ebitda_m * 15)), 1)
                base_op = {"low": 2.5, "medium": 5.5, "high": 8.0}.get(risk_level, 5.0)
                if sentiment.get("risk_flags"):
                    base_op += len(sentiment["risk_flags"]) * 0.6
                scores["operational_risk"] = round(min(9.5, base_op), 1)
                market_base = 5.0
                if sentiment.get("caution_hits", 0) > sentiment.get("positive_hits", 0):
                    market_base += 1.8
                range_pos = metrics.get("range_position", 0.5)
                if range_pos > 0.8: market_base += 1.2
                elif range_pos < 0.2: market_base -= 1.0
                scores["market_risk"] = round(max(1.0, min(9.5, market_base)), 1)
                scores["regulatory_risk"] = 3.5
                scores["supply_chain_risk"] = round({"low": 2.0, "medium": 5.0, "high": 8.0}.get(risk_level, 5.0), 1)
                risk_scores[company] = scores

            avg_risk = sum(sum(s.values()) for s in risk_scores.values()) / max(len(risk_scores) * 5, 1)
            compliance_summary = self.llm_client.chat(
                system_prompt=(
                    "You are a financial compliance audit expert. Provide a 2-3 sentence audit opinion in English. "
                    "Address: (1) Data completeness assessment, (2) Risk exposure evaluation, "
                    "(3) Specific compliance recommendations. Be factual and actionable."
                ),
                user_prompt=(
                    f"Companies: {state['companies']}\n"
                    f"Data completeness: {'Complete' if not findings else 'Gaps detected'}\n"
                    f"Risk scores: {json.dumps(risk_scores, ensure_ascii=False)}\n"
                    f"Average risk score: {avg_risk:.1f}/10\n"
                    f"Issues: {findings if findings else 'None'}"
                ),
                temperature=0.1, max_tokens=220,
            )

            update: FinanceState = {
                "compliance_findings": findings,
                "compliance_violations": [item.to_dict() for item in violations],
                "compliance_summary": compliance_summary,
                "risk_scores": risk_scores,
            }
            if violations:
                update["critic_repair_target"] = classify_critic_violations(violations)
            status = "needs_fix" if findings else "ok"
            detail = (f"Risk architecture mapped: composite score {avg_risk:.1f}/10. "
                      f"{len(findings)} compliance issues identified." if findings else
                      f"All compliance checks passed. Composite risk score: {avg_risk:.1f}/10.")
            update.update(self._record("critic", status, detail, state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update

    # ═══════════════════════════════════════════════════════════════
    # REPAIR — Evaluator-router-retry prototype
    # ═══════════════════════════════════════════════════════════════
    def repair(self, state: FinanceState) -> FinanceState:
        with self._track_step("repair") as timer:
            iterations = state.get("critic_iterations", 0) + 1
            target = state.get("critic_repair_target", "quant")
            codes = {
                str(item.get("code") or "")
                for item in (state.get("compliance_violations") or [])
                if isinstance(item, dict)
            }
            # Never fan out a full retrieval loop for soft/report/unknown gaps.
            if target == "retrieval" and not codes.intersection(RETRIEVAL_WORTHY_CODES):
                if "missing_quantitative_results" in codes:
                    target = "quant"
                elif "missing_sentiment_analysis" in codes:
                    target = "psychologist"
                else:
                    target = "quant"
            detail = (
                f"Router-retry iteration {iterations}/{state.get('critic_max_iterations', 2)}: "
                f"re-running '{target}' to address {len(state.get('compliance_findings', []))} critic finding(s)."
            )
            update: FinanceState = {
                "critic_iterations": iterations,
                "critic_repair_target": target,
            }
            update.update(self._record("repair", "ok", detail, state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update

    # ═══════════════════════════════════════════════════════════════
    # APPENDIX_REPLAN — Supplementary retrieval (not live-provider retry)
    # ═══════════════════════════════════════════════════════════════
    def appendix_replan(self, state: FinanceState) -> FinanceState:
        """Enable one supplementary retrieval pass for appendix / evidence gaps.

        This node is intentionally NOT a SEC/Yahoo provider retry. It flips
        `appendix_search_done` so the next retrieval can include sample appendix
        fields and richer evidence; after two attempts it enters degraded mode.
        """
        with self._track_step("appendix_replan") as timer:
            retries = state.get("retries", 0) + 1
            degraded_mode = retries >= 2
            appendix_search_done = not degraded_mode
            detail = (
                "appendix_replan / supplementary_retrieval: enabling targeted appendix retrieval."
                if not degraded_mode
                else (
                    "appendix_replan exhausted after multiple supplementary_retrieval attempts; "
                    "entering degraded mode and generating report with acknowledged data gaps."
                )
            )
            update: FinanceState = {
                "retries": retries,
                "appendix_search_done": appendix_search_done,
                "degraded_mode": degraded_mode,
                "replan_reason": None,
            }
            update.update(self._record("appendix_replan", "ok", detail, state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update

    # Backward-compatible alias used by older call sites / docs.
    replanner = appendix_replan
