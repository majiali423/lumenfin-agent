from __future__ import annotations

from ..clarification import merge_clarification_into_query
from ..planning import build_query_plan
from ..skills import get_skill_specs
from ..state import FinanceState
from ..tools import (
    canonicalize_companies,
    derive_target_symbols,
    extract_companies_from_query,
)


class PlannerMixin:
    def query_planner(self, state: FinanceState) -> FinanceState:
        with self._track_step("query_planner") as timer:
            effective_query = state["query"]
            if state.get("user_clarification"):
                effective_query = merge_clarification_into_query(effective_query, state["user_clarification"])
            query_plan = build_query_plan(
                effective_query,
                document_contexts=state.get("document_contexts", []),
                llm_client=self.llm_client,
                user_clarification=state.get("user_clarification"),
            )
            required_skills = query_plan.required_skills
            update: FinanceState = {
                "query": query_plan.normalized_query,
                "query_plan": query_plan.to_dict(),
                "required_skills": required_skills,
                "skill_specs": get_skill_specs(required_skills),
                "missing_fields": query_plan.missing_fields,
                "clarification_questions": query_plan.clarification_questions,
                "workflow_status": "running",
            }
            detail = (
                f"Query planned as {query_plan.intent}; companies={query_plan.companies or 'unresolved'}; "
                f"dimensions={', '.join(query_plan.analysis_dimensions)}; skills={', '.join(required_skills)}."
            )
            if query_plan.missing_fields:
                detail += f" Missing fields: {', '.join(query_plan.missing_fields)}."
            status = "needs_clarification" if query_plan.missing_fields else "ok"
            update.update(
                self._record("query_planner", status, detail, state, timer.metrics())
            )
        self.session_memory.save({**state, **update})
        return update

    def await_clarification(self, state: FinanceState) -> FinanceState:
        with self._track_step("await_clarification") as timer:
            questions = state.get("clarification_questions", [])
            detail = (
                "Human-in-the-loop pause: awaiting user clarification before supervisor stage. "
                f"Questions: {' | '.join(questions) if questions else 'n/a'}"
            )
            update: FinanceState = {
                "workflow_status": "needs_clarification",
                "final_report": "",
            }
            update.update(
                self._record("await_clarification", "paused", detail, state, timer.metrics())
            )
        self.session_memory.save({**state, **update})
        return update

    def supervisor(self, state: FinanceState) -> FinanceState:
        with self._track_step("supervisor") as timer:
            query_plan = state.get("query_plan", {})
            planned_companies = list(query_plan.get("companies", []))
            company_scope = str(query_plan.get("company_scope") or "")
            intent = str(query_plan.get("intent") or "").lower()
            # Rules-only fallback: planner already ran LLM company extract when needed.
            companies = planned_companies or extract_companies_from_query(
                state["query"],
                document_contexts=state.get("document_contexts", []),
                llm_client=None,
            )
            # Upload expansion: issuer companies only (primary entity), never all body mentions.
            # Skip expansion when planner already scoped to query-only (non-compare).
            expand_uploads = not (
                planned_companies
                and company_scope == "query"
                and intent
                not in {
                    "compare",
                    "peer",
                    "comparison",
                    "comparative_financial_diligence",
                }
            )
            if expand_uploads or not companies:
                for doc in state.get("document_contexts", []):
                    issuers = doc.get("issuer_companies") or doc.get("detected_companies") or []
                    for company in issuers:
                        if company not in companies:
                            companies.append(company)
            companies = canonicalize_companies(companies)

            plan = [
                "Phase 1 — Data Acquisition: uploaded filings, SEC/Yahoo fundamentals, real-time market data",
                "Phase 2 — Quantitative Engine: Five-dimensional metric computation (Profitability, Liquidity, Solvency, Efficiency, Valuation)",
                "Phase 3 — Sentiment Intelligence: NLP-based management tone analysis with confidence scoring and thematic extraction",
                "Phase 4 — Risk Architecture: Multi-dimensional risk assessment with correlation mapping and stress testing",
                "Phase 5 — Synthesis: SWOT decomposition, scenario modeling (Base/Bull/Bear), investment thesis generation, peer benchmarking",
            ]
            analysis_dimensions = list(query_plan.get("analysis_dimensions") or []) or [
                "Profitability",
                "Liquidity",
                "Solvency",
                "Efficiency",
                "Valuation",
            ]
            key_questions = [str(q) for q in (query_plan.get("key_questions") or []) if str(q).strip()]
            target_symbols = derive_target_symbols(companies, state["query"])

            # Template brief from query_plan — no unused supervisor LLM JSON call.
            company_label = ", ".join(companies) if companies else "target companies"
            dim_label = ", ".join(analysis_dimensions)
            task_brief = (
                f"Conduct diligence on {company_label} across {dim_label}, "
                f"including management sentiment, risk architecture, and an audit-ready report. "
                f"User query: {str(state.get('query') or '')[:240]}"
            )
            if key_questions:
                task_brief += f" Focus: {'; '.join(key_questions[:3])}."

            update: FinanceState = {
                "companies": companies,
                "target_symbols": target_symbols,
                "plan": plan,
                "task_brief": task_brief,
                "retrieved_docs": {},
                "market_snapshots": {},
                "market_data_status": {},
                "appendix_search_done": state.get("appendix_search_done", False),
                "retries": state.get("retries", 0),
                "degraded_mode": state.get("degraded_mode", False),
                "replan_reason": state.get("replan_reason"),
                "llm_backend": self.llm_client.backend_name,
            }
            detail = (
                f"Strategic orchestration initiated for {len(companies)} companies "
                f"(template brief from query_plan; no supervisor LLM). "
                f"Analysis dimensions: {dim_label}."
            )
            update.update(self._record("supervisor", "ok", detail, state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update
