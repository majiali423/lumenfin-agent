from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from .input_guardrail import GuardrailMode, guard_documents
from .clarification import merge_clarification_into_query
from .critic_repair import classify_critic_repair_target
from .knowledge_store import KnowledgeStore
from .llm import BaseLLMClient
from .market_data import MarketDataClient
from .memory import ReasoningMemory, SessionMemory
from .observability import StepTimer, merge_telemetry
from .parallel import map_in_parallel
from .planning import build_query_plan
from .rag.hybrid_retriever import HybridEvidenceRetriever
from .skills import get_skill_specs
from .state import FinanceState
from .tools import (
    analyze_sentiment_deep,
    build_chart_data,
    calculate_derived_ratios,
    derive_target_symbols,
    extract_companies_from_query,
    generate_scenario_analysis,
    parse_with_fallback,
    resolve_safe_formula,
    retrieve_company_payload,
    safe_execute_formula,
    summarize_document_context,
    validate_report,
)


class AgentRuntime:
    def __init__(
        self,
        session_memory: SessionMemory,
        knowledge_memory: KnowledgeStore,
        reasoning_memory: ReasoningMemory,
        llm_client: BaseLLMClient,
        market_data_client: MarketDataClient,
        hybrid_retriever: HybridEvidenceRetriever | None = None,
        rag_enabled: bool = True,
        company_parallelism: int = 4,
        input_guardrail_enabled: bool = True,
        input_guardrail_mode: GuardrailMode = "sanitize",
        tool_backend: str = "local",
    ) -> None:
        self.session_memory = session_memory
        self.knowledge_memory = knowledge_memory
        self.reasoning_memory = reasoning_memory
        self.llm_client = llm_client
        self.market_data_client = market_data_client
        self.hybrid_retriever = hybrid_retriever
        self.rag_enabled = rag_enabled
        self.company_parallelism = max(1, company_parallelism)
        self.input_guardrail_enabled = input_guardrail_enabled
        self.input_guardrail_mode = input_guardrail_mode if input_guardrail_mode in {"sanitize", "block"} else "sanitize"
        self.tool_backend = tool_backend if tool_backend in {"local", "mcp"} else "local"

    def _record(
        self,
        step: str,
        status: str,
        detail: str,
        state: FinanceState,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = self.reasoning_memory.record(step=step, status=status, detail=detail, **(metrics or {}))
        telemetry = merge_telemetry(state.get("run_telemetry"), event)
        return {
            "audit_log": self.reasoning_memory.export(),
            "reasoning_memory": [
                f"{item['step']}::{item['status']}::{item['detail']}" for item in self.reasoning_memory.export()
            ],
            "run_telemetry": telemetry,
        }

    @contextmanager
    def _track_step(self, step: str) -> Iterator[StepTimer]:
        self.llm_client.mark_usage_start()
        yield StepTimer(step=step, llm_client=self.llm_client)

    # ═══════════════════════════════════════════════════════════════
    # INPUT GUARDRAIL — PDF prompt-injection defense
    # ═══════════════════════════════════════════════════════════════
    def input_guardrail(self, state: FinanceState) -> FinanceState:
        documents = state.get("document_contexts", [])
        if not self.input_guardrail_enabled or not documents:
            update: FinanceState = {
                "input_guardrail_findings": [],
                "input_guardrail_summary": {"allowed": True, "mode": self.input_guardrail_mode, "finding_count": 0},
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

    # ═══════════════════════════════════════════════════════════════
    # QUERY PLANNER — Intent, entities, and skill routing
    # ═══════════════════════════════════════════════════════════════
    def query_planner(self, state: FinanceState) -> FinanceState:
        with self._track_step("query_planner") as timer:
            effective_query = state["query"]
            if state.get("user_clarification"):
                effective_query = merge_clarification_into_query(effective_query, state["user_clarification"])
            query_plan = build_query_plan(
                effective_query,
                document_contexts=state.get("document_contexts", []),
                llm_client=self.llm_client,
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
            status = "needs_clarification" if query_plan.missing_fields and not state.get("user_clarification") else "ok"
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
            companies = planned_companies or extract_companies_from_query(
                state["query"],
                document_contexts=state.get("document_contexts", []),
                llm_client=self.llm_client,
            )
            for doc in state.get("document_contexts", []):
                for company in doc.get("detected_companies", []):
                    if company not in companies:
                        companies.append(company)

            plan = [
                "Phase 1 — Data Acquisition: PDF financial reports, real-time market data, sample database fusion",
                "Phase 2 — Quantitative Engine: Five-dimensional metric computation (Profitability, Liquidity, Solvency, Efficiency, Valuation)",
                "Phase 3 — Sentiment Intelligence: NLP-based management tone analysis with confidence scoring and thematic extraction",
                "Phase 4 — Risk Architecture: Multi-dimensional risk assessment with correlation mapping and stress testing",
                "Phase 5 — Synthesis: SWOT decomposition, scenario modeling (Base/Bull/Bear), investment thesis generation, peer benchmarking",
            ]
            requested_dimensions = list(query_plan.get("analysis_dimensions", []))
            target_symbols = derive_target_symbols(companies, state["query"])

            llm_brief = self.llm_client.chat(
                system_prompt=(
                    "You are the Supervisory Agent in an enterprise multi-agent financial analysis system. "
                    "Given the user query and identified companies, produce a structured analysis directive. "
                    "Return JSON: {\"task_brief\": \"...\", \"analysis_dimensions\": [\"dim1\",\"dim2\"], "
                    "\"key_questions\": [\"Q1\",\"Q2\"], \"risk_appetite\": \"conservative|moderate|aggressive\", "
                    "\"industry_context\": \"brief industry dynamics note\"}"
                ),
                user_prompt=(
                    f"Query: {state['query']}\nCompanies: {companies}\n"
                    f"Structured query plan: {json.dumps(query_plan, ensure_ascii=False)}"
                ),
                temperature=0.1,
                max_tokens=300,
            )

            task_brief = f"Conduct a five-dimensional deep financial analysis of {', '.join(companies)}, with management sentiment assessment, risk architecture mapping, and investment-grade report synthesis."
            analysis_dimensions: list[str] = requested_dimensions
            key_questions: list[str] = []
            industry_context = ""
            try:
                parsed = parse_with_fallback(llm_brief)
                task_brief = parsed.get("task_brief", task_brief)
                analysis_dimensions = parsed.get("analysis_dimensions", analysis_dimensions)
                key_questions = parsed.get("key_questions", [])
                industry_context = parsed.get("industry_context", "")
            except (json.JSONDecodeError, KeyError):
                pass

            update: FinanceState = {
                "companies": companies,
                "target_symbols": target_symbols,
                "plan": plan,
                "task_brief": task_brief,
                "retrieved_docs": {},
                "market_snapshots": {},
                "appendix_search_done": state.get("appendix_search_done", False),
                "retries": state.get("retries", 0),
                "degraded_mode": state.get("degraded_mode", False),
                "replan_reason": state.get("replan_reason"),
                "llm_backend": self.llm_client.backend_name,
            }
            detail = (f"Strategic orchestration initiated for {len(companies)} companies. "
                      f"Analysis dimensions: {', '.join(analysis_dimensions) if analysis_dimensions else 'Profitability/Liquidity/Solvency/Efficiency/Valuation'}. "
                      f"Industry context: {industry_context[:80] if industry_context else 'Cross-sector comparison'}.")
            update.update(self._record("supervisor", "ok", detail, state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update

    # ═══════════════════════════════════════════════════════════════
    # RETRIEVAL — Data Acquisition & Enrichment
    # ═══════════════════════════════════════════════════════════════
    def _retrieve_company_bundle(
        self,
        *,
        company: str,
        state: FinanceState,
        retrieval_query: str,
        document_contexts: list[dict[str, Any]],
        session_id: str,
        include_appendix: bool,
    ) -> dict[str, Any]:
        rag_hits: list[dict[str, Any]] = []
        if self.rag_enabled and self.hybrid_retriever and document_contexts:
            rag_hits = self.hybrid_retriever.retrieve_for_company(
                query=retrieval_query,
                company=company,
                session_id=session_id,
                document_contexts=document_contexts,
            )

        if rag_hits:
            document_summary = {
                "source_documents": self.hybrid_retriever.build_source_documents(rag_hits),
                "metric_hints": summarize_document_context(document_contexts, company)["metric_hints"],
            }
        else:
            document_summary = summarize_document_context(document_contexts, company)

        payload = retrieve_company_payload(
            company,
            include_appendix=include_appendix,
            document_contexts=document_contexts,
        )
        try:
            live_market = self.market_data_client.fetch_company_snapshot(
                company,
                state.get("target_symbols", {}).get(company),
            )
        except Exception as exc:
            live_market = {
                "provider": getattr(self.market_data_client, "provider", "unknown"),
                "symbol": state.get("target_symbols", {}).get(company, company),
                "company": company,
                "current_price": None,
                "monthly_return": None,
                "market_cap": None,
                "trailing_pe": None,
                "currency": None,
                "sector": None,
                "industry": None,
                "fifty_two_week_high": None,
                "fifty_two_week_low": None,
                "error": str(exc),
            }
        payload["live_market"] = live_market
        payload["source_documents"] = document_summary["source_documents"]
        if document_summary["metric_hints"]:
            payload["market_data"].update({
                "revenue_2025": document_summary["metric_hints"].get("revenue", payload["market_data"].get("revenue_2025")),
                "ebitda_2025": document_summary["metric_hints"].get("ebitda", payload["market_data"].get("ebitda_2025")),
                "r_and_d_2025": document_summary["metric_hints"].get("r_and_d", payload["market_data"].get("r_and_d_2025")),
            })
        if payload["source_documents"]:
            payload["earnings_call_quotes"] = payload["earnings_call_quotes"] or [
                doc["excerpt"][:300] for doc in payload["source_documents"] if doc.get("excerpt")
            ]
        if payload["source_documents"] and payload["supply_chain"]["risk_level"] == "unknown":
            excerpt = " ".join(doc.get("excerpt", "") for doc in payload["source_documents"]).lower()
            payload["supply_chain"]["risk_level"] = "medium" if "risk" in excerpt else "low"

        profile_prompt = (
            f"Provide a concise ~150-word enterprise profile for {company} covering: "
            f"(1) Core business segments and revenue mix, (2) Competitive moat and market position, "
            f"(3) Key strategic initiatives (R&D, M&A, expansion), (4) Recent material events. "
            f"Output in Chinese, factual and professional tone."
        )
        try:
            profile = self.llm_client.chat(
                system_prompt="You are an equity research analyst. Write factual, professional company profiles.",
                user_prompt=profile_prompt,
                temperature=0.2,
                max_tokens=280,
            )
        except Exception:
            profile = f"Profile generation pending for {company}."

        self.knowledge_memory.ingest_company_document(company, payload)
        return {
            "company": company,
            "payload": payload,
            "market_snapshot": live_market,
            "profile": profile,
            "rag_hits": rag_hits,
        }

    def retrieval(self, state: FinanceState) -> FinanceState:
        with self._track_step("retrieval") as timer:
            include_appendix = state.get("appendix_search_done", False)
            document_contexts = state.get("document_contexts", [])
            rag_index_stats = dict(state.get("rag_index_stats", {}))
            session_id = state.get("thread_id", "default-session")
            retrieval_query = state["query"]
            query_plan = state.get("query_plan", {})
            if query_plan.get("analysis_dimensions"):
                retrieval_query = (
                    f"{state['query']} | focus: {', '.join(query_plan['analysis_dimensions'])}"
                )

            if self.rag_enabled and self.hybrid_retriever and document_contexts and not rag_index_stats:
                rag_index_stats = self.hybrid_retriever.rag_store.index_documents(
                    document_contexts,
                    session_id=session_id,
                )

            bundles = map_in_parallel(
                lambda company: self._retrieve_company_bundle(
                    company=company,
                    state=state,
                    retrieval_query=retrieval_query,
                    document_contexts=document_contexts,
                    session_id=session_id,
                    include_appendix=include_appendix,
                ),
                state["companies"],
                max_workers=self.company_parallelism,
            )

            retrieved_docs: dict[str, dict[str, Any]] = {}
            market_snapshots: dict[str, dict[str, Any]] = {}
            company_profiles: dict[str, str] = {}
            rag_evidence: dict[str, list[dict[str, Any]]] = {}
            for bundle in bundles:
                company = bundle["company"]
                retrieved_docs[company] = bundle["payload"]
                market_snapshots[company] = bundle["market_snapshot"]
                company_profiles[company] = bundle["profile"]
                if bundle["rag_hits"]:
                    rag_evidence[company] = bundle["rag_hits"]

            needs_appendix = any(
                "appendix" not in p and not p.get("source_documents") and not market_snapshots.get(c)
                for c, p in retrieved_docs.items()
            )
            replan_reason = "Appendix data gap detected; switching to targeted supplementary retrieval." if needs_appendix else None

            update: FinanceState = {
                "retrieved_docs": retrieved_docs,
                "market_snapshots": market_snapshots,
                "knowledge_snapshot": self.knowledge_memory.snapshot(),
                "replan_reason": replan_reason,
                "company_profiles": company_profiles,
                "rag_evidence": rag_evidence,
                "rag_index_stats": rag_index_stats,
            }
            rag_chunks = sum(len(hits) for hits in rag_evidence.values())
            detail = (
                "Data fusion complete: real-time market data, PDF document parsing, "
                f"and LLM-generated corporate profiles for {len(state['companies'])} entities integrated "
                f"(parallel fan-out, workers={min(self.company_parallelism, len(state['companies']))})."
            )
            if rag_chunks:
                detail += (
                    f" Hybrid Milvus RAG retrieved {rag_chunks} evidence chunks "
                    f"(vector + keyword RRF, indexed {rag_index_stats.get('chunks_indexed', 0)} chunks)."
                )
            update.update(self._record("retrieval", "needs_replan" if replan_reason else "ok", detail, state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update

    # ═══════════════════════════════════════════════════════════════
    # QUANTITATIVE ANALYST — Metric Computation & Scenario Modeling
    # ═══════════════════════════════════════════════════════════════
    def _compute_company_quant(
        self,
        company: str,
        payload: dict[str, Any],
        state: FinanceState,
    ) -> dict[str, Any]:
        market = payload["market_data"]
        live_market = state.get("market_snapshots", {}).get(company, {})
        metrics: dict[str, float] = {}

        base_data: dict[str, float] = {}
        if market.get("revenue_2025"):
            base_data["revenue"] = market["revenue_2025"]
        if market.get("ebitda_2025"):
            base_data["ebitda"] = market["ebitda_2025"]
        if market.get("r_and_d_2025"):
            base_data["r_and_d"] = market["r_and_d_2025"]
        if market.get("operating_income_2025"):
            base_data["operating_income"] = market["operating_income_2025"]

        if len(base_data) >= 3:
            for formula, key in [
                ("ebitda / revenue", "ebitda_margin"),
                ("r_and_d / revenue", "r_and_d_intensity"),
                ("operating_income / revenue", "operating_margin"),
            ]:
                try:
                    if all(v in base_data for v in ["revenue"]):
                        if key == "r_and_d_intensity" and "r_and_d" not in base_data:
                            continue
                        if key == "operating_margin" and "operating_income" not in base_data:
                            continue
                        if key == "ebitda_margin" and "ebitda" not in base_data:
                            continue
                        metrics[key] = resolve_safe_formula(
                            formula,
                            base_data,
                            backend=self.tool_backend,
                        )
                except (KeyError, ValueError):
                    pass

        derived = calculate_derived_ratios(market)
        for key, value in derived.items():
            metrics.setdefault(key, value)

        cap = live_market.get("market_cap")
        if cap is not None:
            metrics["market_cap_billion"] = round(float(cap) / 1_000_000_000, 4)
        ret = live_market.get("monthly_return")
        if ret is not None:
            metrics["monthly_return"] = float(ret)
        cp = live_market.get("current_price")
        if cp is not None:
            metrics["current_price"] = float(cp)
        pe = live_market.get("trailing_pe")
        if pe is not None:
            metrics["pe_ratio"] = float(pe)
        high = live_market.get("fifty_two_week_high")
        low = live_market.get("fifty_two_week_low")
        price = live_market.get("current_price")
        if all(v is not None for v in (high, low, price)) and float(high) != float(low):
            metrics["range_position"] = round((float(price) - float(low)) / (float(high) - float(low)), 4)

        if not metrics:
            return {
                "company": company,
                "replan_reason": f"{company}: insufficient data for quantitative computation.",
            }
        return {
            "company": company,
            "metrics": metrics,
            "scenario": generate_scenario_analysis(metrics, company),
        }

    def quantitative_analyst(self, state: FinanceState) -> FinanceState:
        with self._track_step("quant") as timer:
            if state.get("replan_reason"):
                update: FinanceState = {"financial_metrics": {}, "replan_reason": state["replan_reason"]}
                update.update(self._record("quant", "blocked", state["replan_reason"], state, timer.metrics()))
                return update

            company_items = list(state["retrieved_docs"].items())
            quant_results = map_in_parallel(
                lambda item: self._compute_company_quant(item[0], item[1], state),
                company_items,
                max_workers=self.company_parallelism,
            )

            financial_metrics: dict[str, dict[str, float]] = {}
            scenario_analyses: dict[str, dict[str, Any]] = {}
            for result in quant_results:
                if result.get("replan_reason"):
                    update = {"replan_reason": result["replan_reason"]}
                    update.update(self._record("quant", "needs_replan", f"{result['company']} missing computable metrics.", state, timer.metrics()))
                    return update
                company = result["company"]
                financial_metrics[company] = result["metrics"]
                scenario_analyses[company] = result["scenario"]

            peer_comparison_text = ""
            try:
                metrics_summary = json.dumps(financial_metrics, ensure_ascii=False)
                peer_comparison_text = self.llm_client.chat(
                    system_prompt=(
                        "You are a senior quantitative analyst. Based on the provided metrics, "
                        "write a 2-3 sentence peer comparison in Chinese. Structure: "
                        "(1) Which company leads on profitability and why, "
                        "(2) Which leads on innovation efficiency, "
                        "(3) Key competitive dynamics revealed by the data."
                    ),
                    user_prompt=f"Company metrics: {metrics_summary}",
                    temperature=0.2,
                    max_tokens=220,
                )
            except Exception:
                peer_comparison_text = "Peer comparison based on available data."

            reasoning = (
                f"Quantitative engine computed {sum(len(m) for m in financial_metrics.values())} metrics "
                f"across {len(financial_metrics)} companies (parallel fan-out). "
                f"Key insight: {peer_comparison_text[:120]}..."
            )
            update = {
                "financial_metrics": financial_metrics,
                "replan_reason": None,
                "tool_backend": self.tool_backend,
                "peer_comparison": {
                    "summary": peer_comparison_text,
                    "metrics": financial_metrics,
                    "scenarios": scenario_analyses,
                },
            }
            update.update(self._record("quant", "ok", reasoning, state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update

    # ═══════════════════════════════════════════════════════════════
    # PSYCHOLOGIST — Management Sentiment Intelligence
    # ═══════════════════════════════════════════════════════════════
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

    # ═══════════════════════════════════════════════════════════════
    # CRITIC — Risk Architecture & Compliance Audit
    # ═══════════════════════════════════════════════════════════════
    def critic(self, state: FinanceState) -> FinanceState:
        with self._track_step("critic") as timer:
            findings: list[str] = []
            risk_scores: dict[str, dict[str, float]] = {}

            for company in state["companies"]:
                if company not in state.get("financial_metrics", {}):
                    findings.append(f"{company}: missing quantitative results.")
                if company not in state.get("sentiment_analysis", {}):
                    findings.append(f"{company}: missing sentiment analysis.")

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

            report_stub = "\n".join(state.get("report_sections", []))
            if report_stub:
                findings.extend(validate_report(report_stub))

            avg_risk = sum(sum(s.values()) for s in risk_scores.values()) / max(len(risk_scores) * 5, 1)
            compliance_summary = self.llm_client.chat(
                system_prompt=(
                    "You are a financial compliance audit expert. Provide a 2-3 sentence audit opinion in Chinese. "
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
                "compliance_summary": compliance_summary,
                "risk_scores": risk_scores,
            }
            if findings:
                update["critic_repair_target"] = classify_critic_repair_target(findings)
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
            target = state.get("critic_repair_target", "retrieval")
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
    # REPLANNER — Resilience & Fallback
    # ═══════════════════════════════════════════════════════════════
    def replanner(self, state: FinanceState) -> FinanceState:
        with self._track_step("replanner") as timer:
            retries = state.get("retries", 0) + 1
            degraded_mode = retries >= 2
            appendix_search_done = not degraded_mode
            detail = ("Re-planning triggered: switching to targeted appendix retrieval strategy." if not degraded_mode
                      else "Degraded mode activated after multiple attempts. Generating report with acknowledged data gaps.")
            update: FinanceState = {
                "retries": retries, "appendix_search_done": appendix_search_done,
                "degraded_mode": degraded_mode, "replan_reason": None if degraded_mode else None,
            }
            update.update(self._record("replanner", "ok", detail, state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update

    # ═══════════════════════════════════════════════════════════════
    # SYNTHESIZER — Investment-Grade Report Assembly
    # ═══════════════════════════════════════════════════════════════
    def synthesizer(self, state: FinanceState) -> FinanceState:
        with self._track_step("synthesizer") as timer:
            return self._synthesize_report(state, timer)

    def _synthesize_report(self, state: FinanceState, timer: StepTimer) -> FinanceState:
        doc_context = ""
        rag_citation_lines: list[str] = []
        if state.get("rag_evidence"):
            for company, hits in state["rag_evidence"].items():
                for hit in hits[:3]:
                    rag_citation_lines.append(
                        f"- [{company}] {hit.get('citation')} ({hit.get('retrieval_method')}): "
                        f"{hit.get('text', '')[:240]}"
                    )
        if rag_citation_lines:
            doc_context = "\nMilvus hybrid RAG evidence (with citations):\n" + "\n".join(rag_citation_lines)
        elif state.get("document_contexts"):
            excerpts = [d["excerpt"][:600] for d in state["document_contexts"] if d.get("excerpt")]
            if excerpts:
                doc_context = "\nUploaded PDF excerpts:\n" + "\n---\n".join(excerpts)

        has_metrics = any(state.get("financial_metrics", {}).values())
        knowledge_hint = ""
        if not has_metrics and not doc_context:
            knowledge_hint = (
                "\nNote: Limited structured data available. Leverage your public knowledge of these companies "
                "to provide insightful analysis. Do not simply state 'insufficient data'."
            )

        profile_lines = [f"{c}: {state.get('company_profiles', {}).get(c, '')}" for c in state["companies"]]
        profile_context = "\n".join(profile_lines)
        metrics_context = json.dumps(state.get("financial_metrics", {}), ensure_ascii=False)
        sentiment_context = json.dumps(state.get("sentiment_analysis", {}), ensure_ascii=False)
        risk_context = json.dumps(state.get("risk_scores", {}), ensure_ascii=False)
        peer_context = state.get("peer_comparison", {}).get("summary", "")

        # ── Executive Summary (Enhanced) ──
        llm_summary = self.llm_client.chat(
            system_prompt=(
                "You are the Director of Research at an institutional investment firm. "
                "Write a 4-5 sentence executive summary in Chinese that demonstrates rigorous analytical reasoning. "
                "Structure: (1) Top-line finding with specific metric evidence, "
                "(2) Risk-return profile characterization, "
                "(3) Key competitive insight, "
                "(4) Actionable investment implication. "
                "Use specific numbers from the data. Be decisive and insightful."
            ),
            user_prompt=(
                f"Mission: {state.get('task_brief', '')}\n"
                f"Company Profiles:\n{profile_context}\n"
                f"Financial Metrics: {metrics_context}\n"
                f"Sentiment Analysis: {sentiment_context}\n"
                f"Risk Scores: {risk_context}\n"
                f"Peer Comparison: {peer_context}\n"
                f"{doc_context}{knowledge_hint}"
            ),
            temperature=0.2, max_tokens=500,
        )

        # ── Report Construction ──
        sections: list[str] = []
        S = sections.append  # shorthand

        S("# LumenFin Intelligence Report")
        S("")
        S("**Report Type:** Investment-Grade Research | **Classification:** AI-Generated, For Reference Only")
        S("")
        S(f"## 1. Executive Summary")
        S(f"{llm_summary}")
        S("")
        S(f"## 2. Analytical Framework & Methodology")
        S("")
        S("This report employs a **six-layer analytical architecture** powered by a LangGraph-based multi-agent system:")
        S("")
        S("| Layer | Agent | Methodology | Output |")
        S("|-------|-------|-------------|--------|")
        S("| L1 | Supervisor | Strategic task decomposition, dimension identification | Analysis blueprint |")
        S("| L2 | Retrieval | Hybrid Milvus RAG (vector + keyword RRF) + structured sample DB + market snapshots | Enriched evidence payloads |")
        S("| L3 | Quantitative Analyst | AST-safe expression engine + derived ratio computation | Five-dimension metrics |")
        S("| L4 | Psychologist | NLP deep sentiment analysis with confidence scoring | Tone intelligence |")
        S("| L5 | Critic | Multi-factor risk scoring + compliance validation | Risk architecture |")
        S("| L6 | Synthesizer | Structured reasoning, scenario modeling, evidence mapping | Investment report |")
        S("")

        S(f"## 3. Company Profiles & Business Overview")
        for company in state["companies"]:
            profile = state.get("company_profiles", {}).get(company, "Profile not available.")
            S(f"### {company}")
            S(f"{profile}")
            S("")

        S(f"## 4. Financial Performance Analysis")
        S("")
        S("*The following metrics were computed using an AST-safe expression evaluator with industry benchmarking.*")
        S("")

        for company in state["companies"]:
            metrics = state.get("financial_metrics", {}).get(company, {})
            sentiment = state.get("sentiment_analysis", {}).get(company, {})
            risk_level = state["retrieved_docs"][company]["supply_chain"]["risk_level"]
            risk_data = state.get("risk_scores", {}).get(company, {})
            live_market = state.get("market_snapshots", {}).get(company, {})

            S(f"### {company}")
            S("")

            if metrics:
                S(f"**Key Financial Indicators**")
                S("")
                S("| Metric | Value | Benchmark | Assessment | Rationale |")
                S("|--------|-------|-----------|------------|-----------|")

                def add_row(metric_key, label, benchmark, value=None):
                    v = value if value is not None else metrics.get(metric_key)
                    if v is None: return
                    if metric_key in ("ebitda_margin", "r_and_d_intensity", "operating_margin", "estimated_net_margin", "estimated_fcf_margin"):
                        grade = "Strong" if v > 0.25 else ("Adequate" if v > 0.12 else "Weak")
                        rationale = (f"EBITDA of {v:.1%} indicates robust operational efficiency" if "EBITDA" in label else
                                     f"R&D spend at {v:.1%} of revenue" if "R&D" in label else
                                     f"At {v:.1%}, reflects effective cost management" if "Margin" in label else
                                     f"FCF generation at {v:.1%} of revenue")
                        S(f"| {label} | {v:.2%} | {benchmark} | {grade} | {rationale} |")
                    elif metric_key == "pe_ratio":
                        S(f"| {label} | {v:.2f}x | {benchmark} | — | Market-implied valuation multiple |")
                    elif metric_key == "monthly_return":
                        direction = "Upward momentum" if v > 0 else "Downward pressure"
                        S(f"| {label} | {v:.2%} | {benchmark} | — | {direction} |")
                    elif metric_key == "range_position":
                        position = "Near highs" if v > 0.7 else ("Near lows" if v < 0.3 else "Mid-range")
                        S(f"| {label} | {v:.1%} | {benchmark} | — | 52-week {position} |")

                add_row("ebitda_margin", "EBITDA Margin", ">25%")
                add_row("operating_margin", "Operating Margin", ">20%")
                add_row("estimated_net_margin", "Est. Net Margin", ">15%")
                add_row("estimated_fcf_margin", "Est. FCF Yield", ">10%")
                add_row("r_and_d_intensity", "R&D Intensity", "5-15%")
                add_row("pe_ratio", "P/E (TTM)", "Industry avg")
                add_row("monthly_return", "Monthly Return", "—")
                add_row("range_position", "52W Range Position", "—")
                S("")

                # Reasoning chain
                S(f"**Analytical Reasoning Chain**")
                S("")
                ebitda_m = metrics.get("ebitda_margin", 0)
                rd_i = metrics.get("r_and_d_intensity", 0)
                reasoning_lines = []
                reasoning_lines.append(f"1. **Profitability Assessment**: {company} achieves an EBITDA margin of {ebitda_m:.1%}. "
                    f"{'This significantly exceeds the 25% industry benchmark, indicating strong pricing power and operational leverage.' if ebitda_m > 0.25 else 'This is below the 25% threshold, suggesting room for operational efficiency improvement.'}")
                reasoning_lines.append(f"2. **Innovation Capacity**: R&D intensity of {rd_i:.1%} "
                    f"{'demonstrates commitment to sustaining competitive advantage through innovation.' if rd_i > 0.06 else 'may constrain long-term innovation trajectory relative to peers.'}")
                reasoning_lines.append(f"3. **Risk Integration**: Supply chain risk is rated '{risk_level}'. "
                    f"{'This represents a manageable operational risk factor.' if risk_level == 'low' else 'This requires active monitoring and mitigation strategies.' if risk_level == 'medium' else 'This is a material risk factor that warrants hedging or diversification.'}")

                if sentiment.get("confidence_score"):
                    cs = sentiment["confidence_score"]
                    reasoning_lines.append(f"4. **Management Credibility**: Leadership confidence score of {cs}/10 "
                        f"{'indicates high conviction in strategic direction.' if cs >= 7 else 'suggests measured caution in outlook.' if cs >= 5 else 'warrants further scrutiny of narrative consistency.'}")
                S("\n".join(reasoning_lines))
                S("")
            else:
                S("*[Degraded Analysis] Insufficient structured data for quantitative assessment.*")
                S("")

            # Sentiment
            S(f"**Management Sentiment Profile**")
            S(f"- Overall Tone: **{sentiment.get('label', 'N/A').capitalize()}**")
            if sentiment.get("confidence_score"):
                S(f"- Conviction Level: {sentiment['confidence_score']}/10")
            if sentiment.get("key_themes"):
                S(f"- Strategic Themes: {' | '.join(sentiment['key_themes'])}")
            if sentiment.get("strategic_priority"):
                S(f"- Forward Priority: {sentiment['strategic_priority']}")
            if sentiment.get("risk_flags"):
                S(f"- Flagged Risks: {' | '.join(sentiment['risk_flags'])}")
            S("")

            # Risk
            if risk_data:
                S(f"**Risk Exposure Matrix**")
                S("")
                S("| Dimension | Score (1-10) | Level |")
                S("|-----------|-------------|-------|")
                dim_labels = {"financial_risk": "Financial", "operational_risk": "Operational",
                              "market_risk": "Market", "regulatory_risk": "Regulatory",
                              "supply_chain_risk": "Supply Chain"}
                for dim, label in dim_labels.items():
                    score = risk_data.get(dim, 5.0)
                    level = "Low Risk" if score < 3.5 else ("Moderate" if score < 6.5 else "Elevated")
                    S(f"| {label} | {score:.1f} | {level} |")
                S("")

        # ── Industry & Macro Context ──
        S("## 5. Industry Dynamics & Macroeconomic Context")
        S("")
        S("*The following context integrates LLM knowledge with structured data analysis to provide a comprehensive operating environment assessment.*")
        S("")
        for company in state["companies"]:
            metrics = state.get("financial_metrics", {}).get(company, {})
            risk = state.get("retrieved_docs", {}).get(company, {}).get("supply_chain", {})
            S(f"### {company} — Operating Environment")
            S("")
            ebitda_m = metrics.get("ebitda_margin", 0)
            rd_i = metrics.get("r_and_d_intensity", 0)
            risk_level = risk.get("risk_level", "unknown")
            S(f"- **Sector Position**: {' Market leader with significant pricing power' if ebitda_m > 0.25 else ' Competitive player with margin expansion potential'}")
            S(f"- **Innovation Trajectory**: {' Heavy R&D investment (' + str(round(rd_i*100,1)) + '% of revenue) supports technology leadership in core markets' if rd_i > 0.06 else ' Moderate R&D intensity may require strategic increases to maintain competitive parity'}")
            S(f"- **Supply Chain Resilience**: {' Well-diversified supply base with multiple contingency options' if risk_level == 'low' else ' Moderate concentration risk requiring active monitoring and dual-sourcing strategies' if risk_level == 'medium' else ' Significant concentration exposure necessitating strategic inventory buffers and alternative supplier development'}")
            S(f"- **Regulatory Landscape**: Technology sector faces evolving antitrust, data privacy, and AI governance frameworks across major jurisdictions")
            S(f"- **Macro Sensitivity**: {' Lower cyclicality due to diversified revenue streams and recurring service income' if ebitda_m > 0.30 else ' Moderate exposure to consumer and enterprise spending cycles'}")
            S("")

        # ── SWOT ──
        S("## 6. Strategic Analysis (SWOT)")
        S("")
        swot: dict[str, dict[str, str]] = {}
        for company in state["companies"]:
            metrics = state.get("financial_metrics", {}).get(company, {})
            sentiment = state.get("sentiment_analysis", {}).get(company, {})
            risk = state.get("retrieved_docs", {}).get(company, {}).get("supply_chain", {})
            ebitda_m = metrics.get("ebitda_margin", 0)
            rd_i = metrics.get("r_and_d_intensity", 0)
            tone = sentiment.get("label", "neutral")

            strengths = ["Strong financial fundamentals with healthy margin profile"] if ebitda_m > 0.20 else ["Adequate baseline profitability"]
            if rd_i > 0.05: strengths.append("Above-peer R&D investment sustaining innovation pipeline")
            if tone == "bullish": strengths.append("Management demonstrates high conviction and strategic clarity")

            weaknesses = []
            if ebitda_m < 0.15: weaknesses.append("Margin profile below industry leadership threshold")
            if rd_i < 0.03: weaknesses.append("R&D intensity may lag innovation requirements")
            if risk.get("risk_level") != "low": weaknesses.append(f"Supply chain risk exposure at '{risk.get('risk_level')}' level")

            opportunities = ["Technology-driven productivity gains and digital transformation", "Emerging market expansion with favorable demographic trends"]
            threats = ["Macroeconomic uncertainty including monetary policy shifts", "Intensifying competitive dynamics and potential disruption", "Evolving regulatory landscape across jurisdictions"]
            if risk.get("risk_level") == "high": threats.append("Concentrated supply chain presents operational vulnerability")

            swot[company] = {
                "strengths": "; ".join(strengths) + ".",
                "weaknesses": "; ".join(weaknesses) + "." if weaknesses else "No material weaknesses identified at current data resolution.",
                "opportunities": "; ".join(opportunities) + ".",
                "threats": "; ".join(threats) + ".",
            }

            S(f"### {company}")
            S("")
            S("| Quadrant | Assessment |")
            S("|----------|------------|")
            S(f"| Strengths | {swot[company]['strengths']} |")
            S(f"| Weaknesses | {swot[company]['weaknesses']} |")
            S(f"| Opportunities | {swot[company]['opportunities']} |")
            S(f"| Threats | {swot[company]['threats']} |")
            S("")

        # ── Scenario Analysis ──
        S("## 7. Scenario Analysis & Forward Projections")
        S("")
        S("*The following scenarios are derived from current financial metrics and industry dynamics. They represent analytical projections, not forecasts.*")
        S("")
        for company in state["companies"]:
            metrics = state.get("financial_metrics", {}).get(company, {})
            scenario = generate_scenario_analysis(metrics, company)
            S(f"### {company}")
            S("")
            S("| Scenario | Revenue Growth | Probability | Key Narrative |")
            S("|----------|---------------|-------------|---------------|")
            for case_name in ["base_case", "bull_case", "bear_case"]:
                c = scenario[case_name]
                label = {"base_case": "Base Case", "bull_case": "Bull Case", "bear_case": "Bear Case"}[case_name]
                S(f"| {label} | {c['revenue_growth']} | {c['probability']} | {c['narrative']} |")
            S("")

        # ── Investment Thesis ──
        S("## 8. Investment Thesis & Positioning")
        S("")
        investment_thesis: dict[str, dict[str, str]] = {}
        for company in state["companies"]:
            metrics = state.get("financial_metrics", {}).get(company, {})
            sentiment = state.get("sentiment_analysis", {}).get(company, {})
            ebitda_m = metrics.get("ebitda_margin", 0)
            tone = sentiment.get("label", "neutral")

            if ebitda_m > 0.25 and tone == "bullish":
                bull = (f"Strong profitability (EBITDA margin {ebitda_m:.1%}) combined with confident management guidance "
                        f"suggests earnings visibility above consensus. Recommend overweight position with disciplined entry on pullbacks.")
                bear = (f"Premium valuation may limit near-term upside. Key downside risks include competitive disruption "
                        f"and macro-driven multiple compression. Position size should account for these tail risks.")
            elif ebitda_m > 0.15:
                bull = (f"Solid financial foundation with manageable risk profile. Suitable as core portfolio holding "
                        f"for medium-to-long-term investors seeking quality compounders.")
                bear = (f"Limited near-term catalysts for re-rating. Margin improvement trajectory may be gradual. "
                        f"Consider pairing with higher-growth names for portfolio balance.")
            else:
                bull = (f"Potential value unlock if operational turnaround materializes. Current metrics may understate "
                        f"recovery optionality. Tactical opportunity for risk-tolerant investors.")
                bear = (f"Weak profitability metrics suggest structural challenges. Recommend awaiting definitive "
                        f"evidence of business improvement before committing capital.")

            investment_thesis[company] = {"bull_case": bull, "bear_case": bear}
            S(f"### {company}")
            S(f"- **Investment Rationale (Bull Case):** {bull}")
            S(f"- **Risk Considerations (Bear Case):** {bear}")
            S("")

        # ── Peer Comparison ──
        if state.get("peer_comparison", {}).get("summary"):
            S("## 9. Competitive Landscape & Peer Benchmarking")
            S("")
            S(state["peer_comparison"]["summary"])
            S("")

        # ── Compliance ──
        S("## 10. Compliance Review & Data Integrity")
        S("")
        if state.get("compliance_summary"):
            S(f"**Audit Opinion:** {state['compliance_summary']}")
            S("")
        if state.get("compliance_findings"):
            S("**Identified Issues:**")
            for item in state["compliance_findings"]:
                S(f"- {item}")
            if state.get("critic_iterations", 0) >= state.get("critic_max_iterations", 2):
                S("")
                S(
                    f"*Evaluator-optimizer loop exhausted after {state['critic_iterations']} iteration(s); "
                    "report generated with acknowledged compliance gaps.*"
                )
        else:
            S("All core compliance and data integrity checks passed. No material gaps detected.")
        S("")

        # ── Methodology & Disclaimer ──
        S("## 11. Methodology, Data Sources & Disclaimer")
        S("")
        S("**Analytical Methods:** AST-safe expression evaluator for numerical computation; LLM-based deep semantic analysis for sentiment extraction; multi-factor risk scoring model with evidence-based calibration; LangGraph-directed multi-agent orchestration with checkpoint-based state persistence.")
        S("")
        S("**Data Sources:** Uploaded PDF financial documents (PyMuPDF parsing + Milvus Lite vector index); hybrid retrieval with citation-backed evidence chunks; real-time market data via financial data provider APIs; curated sample financial database; LLM knowledge base for industry context and company profiles.")
        S("")
        S("**Disclaimer:** This report is generated by an AI-powered multi-agent system for research and demonstration purposes only. It does not constitute investment advice, a solicitation, or a recommendation to buy or sell any security. All investment decisions involve risk and should be made in consultation with qualified financial professionals. Past performance and AI-generated projections are not indicative of future results.")

        final_report = "\n".join(sections)

        # ── Chart Data ──
        chart_data = build_chart_data(
            companies=state["companies"],
            financial_metrics=state.get("financial_metrics", {}),
            sentiment_analysis=state.get("sentiment_analysis", {}),
            risk_scores=state.get("risk_scores", {}),
            audit_log=state.get("audit_log", []),
        )

        update: FinanceState = {
            "report_sections": sections,
            "executive_summary": llm_summary,
            "final_report": final_report,
            "llm_backend": self.llm_client.backend_name,
            "swot_analysis": swot,
            "investment_thesis": investment_thesis,
            "chart_data": chart_data,
            "workflow_status": "completed",
        }
        update.update(self._record("synthesizer", "ok",
            "Investment-grade report assembled: SWOT, Scenario Analysis (Base/Bull/Bear), Investment Thesis, Peer Benchmarking, Compliance Review, and structured Chart Data.",
            state, timer.metrics()))
        self.session_memory.save({**state, **update})
        return update
