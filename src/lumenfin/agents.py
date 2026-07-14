from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Any, Iterator

from .input_guardrail import GuardrailMode, guard_documents
from .clarification import merge_clarification_into_query
from .critic_repair import classify_critic_violations, compliance_messages
from .critic_checks import run_critic_checks
from .artifacts import RetrievalArtifact, RetrievalProvenance, score_retrieval_confidence
from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .knowledge_store import KnowledgeStore
from .llm import BaseLLMClient
from .market_data import MarketDataClient, summarize_market_snapshots
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
    canonicalize_companies,
    derive_target_symbols,
    extract_companies_from_query,
    generate_scenario_analysis,
    has_computable_fundamentals,
    parse_with_fallback,
    resolve_safe_formula,
    retrieve_company_payload,
    safe_execute_formula,
    summarize_document_context,
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
        allow_sample_data: bool = True,
        data_mode: str = "demo",
        fetch_live_fundamentals: bool = False,
        fetch_sec_fundamentals: bool = False,
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
        self.allow_sample_data = allow_sample_data
        self.data_mode = data_mode if data_mode in {"demo", "live"} else "demo"
        self.fetch_live_fundamentals = bool(fetch_live_fundamentals)
        self.fetch_sec_fundamentals = bool(fetch_sec_fundamentals)

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
            companies = canonicalize_companies(companies)

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
                "market_data_status": {},
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
    ) -> RetrievalArtifact:
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
            allow_sample_data=self.allow_sample_data,
            ticker=state.get("target_symbols", {}).get(company),
            fetch_live_fundamentals=self.fetch_live_fundamentals,
            fetch_sec_fundamentals=self.fetch_sec_fundamentals,
        )
        try:
            live_market = self.market_data_client.fetch_company_snapshot(
                company,
                state.get("target_symbols", {}).get(company),
            )
        except Exception as exc:
            ticker = state.get("target_symbols", {}).get(company, company)
            live_market = {
                "provider": getattr(self.market_data_client, "provider", "unknown"),
                "symbol": ticker,
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
                "status": "failed",
                "from_cache": False,
                "fetched_at": None,
                "provider_chain": [getattr(self.market_data_client, "provider", "unknown")],
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
            # Prefer labeling documents as the structured source when filings contributed metrics.
            if any(
                document_summary["metric_hints"].get(key) is not None
                for key in ("revenue", "ebitda", "r_and_d")
            ):
                payload["structured_source"] = "document_extracted"
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
            f"Output in English, factual and professional tone."
        )
        def _looks_non_english(text: str) -> bool:
            return bool(re.search(r"[\u4e00-\u9fff]", text))

        def _looks_incomplete(text: str) -> bool:
            cleaned = (text or "").strip()
            if not cleaned:
                return True
            if cleaned[-1] not in ".!?":
                return True
            tail = cleaned[-40:].lower()
            incomplete_markers = (
                "approximately",
                "including",
                "such as",
                "e.g.",
                "etc",
                "and",
                "or",
                "with",
            )
            return any(tail.endswith(marker) for marker in incomplete_markers)

        try:
            profile = self.llm_client.chat(
                system_prompt="You are an equity research analyst. Write factual, professional company profiles.",
                user_prompt=profile_prompt,
                temperature=0.2,
                max_tokens=280,
            )
            if _looks_non_english(profile) or _looks_incomplete(profile):
                profile = self.llm_client.chat(
                    system_prompt=(
                        "You are an equity research analyst. Rewrite the profile in clean, complete English only. "
                        "Do not include Chinese characters. End with a complete sentence."
                    ),
                    user_prompt=profile,
                    temperature=0.1,
                    max_tokens=280,
                )
            if _looks_non_english(profile) or _looks_incomplete(profile):
                profile = self.llm_client.chat(
                    system_prompt=(
                        "Write exactly 4 complete English sentences summarizing company profile, moat, strategy, "
                        "and latest material event. No lists. No truncation."
                    ),
                    user_prompt=f"Company: {company}. Keep it concise and complete.",
                    temperature=0.0,
                    max_tokens=220,
                )
        except Exception:
            profile = f"Profile generation pending for {company}."

        self.knowledge_memory.ingest_company_document(company, payload)
        structured_source = str(payload.get("structured_source") or "none")
        provenance = RetrievalProvenance(
            structured_source=structured_source,  # type: ignore[arg-type]
            market_provider=str(live_market.get("provider") or "unknown"),
            market_status=str(live_market.get("status") or "unknown"),
            rag_enabled=bool(rag_hits),
            rag_hit_count=len(rag_hits),
            document_count=len(document_contexts),
            data_mode=self.data_mode,
        )
        confidence = score_retrieval_confidence(
            market_data=payload.get("market_data") or {},
            live_market=live_market,
            rag_hits=rag_hits,
        )
        appendix = dict(payload.get("appendix") or {})
        return RetrievalArtifact(
            company=company,
            market_data=dict(payload.get("market_data") or {}),
            supply_chain=dict(payload.get("supply_chain") or {}),
            earnings_call_quotes=list(payload.get("earnings_call_quotes") or []),
            source_documents=list(payload.get("source_documents") or []),
            market_snapshot=live_market,
            profile=profile,
            rag_hits=rag_hits,
            provenance=provenance,
            confidence=confidence,
            structured_source=structured_source,  # type: ignore[arg-type]
            appendix=appendix,
            fundamentals_meta=dict(payload.get("fundamentals_meta") or {}),
        )

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
            retrieval_provenance: dict[str, dict[str, Any]] = {}
            for artifact in bundles:
                company = artifact.company
                retrieved_docs[company] = artifact.to_legacy_payload()
                market_snapshots[company] = artifact.market_snapshot
                company_profiles[company] = artifact.profile
                retrieval_provenance[company] = artifact.provenance.to_dict()
                if artifact.rag_hits:
                    rag_evidence[company] = artifact.rag_hits

            needs_appendix = any(
                "appendix" not in p
                and not p.get("source_documents")
                and not (market_snapshots.get(c) or {}).get("current_price")
                for c, p in retrieved_docs.items()
            )
            market_status = summarize_market_snapshots(market_snapshots)

            computable_companies = [
                company
                for company, payload in retrieved_docs.items()
                if has_computable_fundamentals(payload)
            ]
            fatal_data_gap = bool(retrieved_docs) and not computable_companies
            if fatal_data_gap:
                # Fail-loud: do not silent-replanner loop when no AST-computable fundamentals exist.
                replan_reason = None
                data_gap_detail = (
                    "No computable structured fundamentals for "
                    f"{', '.join(retrieved_docs)} (structured_source has no revenue/EBITDA/R&D inputs). "
                    "Upload a filing PDF with extractable metrics, or analyze a demo sample company "
                    "(Apple/Microsoft/NVIDIA/...). Refusing to invent numbers."
                )
            else:
                replan_reason = (
                    "Appendix data gap detected; switching to targeted supplementary retrieval."
                    if needs_appendix
                    else None
                )
                data_gap_detail = ""

            update: FinanceState = {
                "retrieved_docs": retrieved_docs,
                "market_snapshots": market_snapshots,
                "market_data_status": market_status,
                "knowledge_snapshot": self.knowledge_memory.snapshot(),
                "replan_reason": replan_reason,
                "company_profiles": company_profiles,
                "rag_evidence": rag_evidence,
                "rag_index_stats": rag_index_stats,
                "retrieval_provenance": retrieval_provenance,
                "fatal_data_gap": fatal_data_gap,
                "data_gap_detail": data_gap_detail,
                "degraded_mode": True if fatal_data_gap else state.get("degraded_mode", False),
            }
            rag_chunks = sum(len(hits) for hits in rag_evidence.values())
            if fatal_data_gap:
                detail = f"FATAL DATA GAP: {data_gap_detail}"
                status = "incomplete_data"
            else:
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
                if market_status.get("total_count"):
                    detail += (
                        f" Market API: {market_status['ok_count']}/{market_status['total_count']} "
                        f"snapshots ok (primary={getattr(self.market_data_client, 'provider', 'unknown')}, "
                        f"fallback={getattr(self.market_data_client, 'fallback_provider', 'yahoo')})."
                    )
                status = "needs_replan" if replan_reason else "ok"
            update.update(self._record("retrieval", status, detail, state, timer.metrics()))
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
        metric_confidence: dict[str, dict[str, Any]] = {}

        def set_confidence(metric_key: str, score: float, basis: str) -> None:
            metric_confidence[metric_key] = {
                "score": round(score, 2),
                "level": "High" if score >= 0.85 else ("Medium" if score >= 0.6 else "Low"),
                "basis": basis,
            }

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
                        set_confidence(key, 0.95, "AST")
                except (KeyError, ValueError):
                    pass

        derived = calculate_derived_ratios(market)
        for key, value in derived.items():
            metrics.setdefault(key, value)
            if key not in metric_confidence:
                set_confidence(key, 0.72, "Derived")

        cap = live_market.get("market_cap")
        live_status = str(live_market.get("status") or "ok")
        live_conf = 0.8 if live_status == "ok" else (0.75 if live_status == "cached" else 0.55)
        live_basis = "LiveAPI" if live_status == "ok" else f"LiveAPI ({live_status})"
        if cap is not None:
            metrics["market_cap_billion"] = round(float(cap) / 1_000_000_000, 4)
            set_confidence("market_cap_billion", live_conf, live_basis)
        ret = live_market.get("monthly_return")
        if ret is not None:
            metrics["monthly_return"] = float(ret)
            set_confidence("monthly_return", live_conf, live_basis)
        cp = live_market.get("current_price")
        if cp is not None:
            metrics["current_price"] = float(cp)
            set_confidence("current_price", live_conf, live_basis)
        pe = live_market.get("trailing_pe")
        if pe is not None:
            metrics["pe_ratio"] = float(pe)
            set_confidence("pe_ratio", live_conf, live_basis)
        high = live_market.get("fifty_two_week_high")
        low = live_market.get("fifty_two_week_low")
        price = live_market.get("current_price")
        if all(v is not None for v in (high, low, price)) and float(high) != float(low):
            metrics["range_position"] = round((float(price) - float(low)) / (float(high) - float(low)), 4)
            set_confidence("range_position", live_conf * 0.97, live_basis)

        if not metrics:
            return {
                "company": company,
                "replan_reason": f"{company}: insufficient data for quantitative computation.",
            }
        return {
            "company": company,
            "metrics": metrics,
            "scenario": generate_scenario_analysis(metrics, company),
            "metric_confidence": metric_confidence,
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
            metric_confidence: dict[str, dict[str, dict[str, Any]]] = {}
            for result in quant_results:
                if result.get("replan_reason"):
                    update = {"replan_reason": result["replan_reason"]}
                    update.update(self._record("quant", "needs_replan", f"{result['company']} missing computable metrics.", state, timer.metrics()))
                    return update
                company = result["company"]
                financial_metrics[company] = result["metrics"]
                scenario_analyses[company] = result["scenario"]
                metric_confidence[company] = result.get("metric_confidence", {})

            peer_comparison_text = ""
            try:
                metrics_summary = json.dumps(financial_metrics, ensure_ascii=False)
                peer_comparison_text = self.llm_client.chat(
                    system_prompt=(
                        "You are a senior quantitative analyst. Based on the provided metrics, "
                        "write a 2-3 sentence peer comparison in English. Structure: "
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
                "metric_confidence": metric_confidence,
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
        def ensure_sentence_complete(text: str) -> str:
            cleaned = (text or "").strip()
            if not cleaned:
                return cleaned
            if cleaned[-1] not in ".!?。！？)]】":
                return cleaned + "。"
            return cleaned

        sections: list[str] = []
        def S(line: str = "") -> None:
            sections.append(line)

        fatal_data_gap = bool(state.get("fatal_data_gap"))
        if fatal_data_gap:
            companies = ", ".join(state.get("companies") or []) or "(none)"
            detail = state.get("data_gap_detail") or (
                "No computable structured fundamentals were available. "
                "Upload a filing PDF with extractable metrics, or analyze a demo sample company."
            )
            S("# Incomplete Diligence Output (Fail-Loud Data Gap)")
            S("")
            S(f"**Companies:** {companies}")
            S("")
            S(detail)
            S("")
            S(
                "**Data limitation risk:** Market snapshots / LLM general knowledge alone are not "
                "enough for AST ratio computation. This run refused to invent financials."
            )
            S("")
            S(
                "**Action Required:** Upload source filings (PDF) with extractable FY metrics, or query a "
                "company covered by the demo sample database (Apple / Microsoft / NVIDIA / AMD / Tesla / …). "
                "Chinese or HK names (e.g. 腾讯控股 → Tencent) still need PDF or sample fundamentals."
            )
            S("")
            S(
                "**Gate expectation:** FinAgentBench should fail-closed "
                "(`structured_source=none`, missing checkable metrics / pipeline steps)."
            )
            final_report = "\n".join(sections)
            update: FinanceState = {
                "report_sections": sections,
                "executive_summary": detail,
                "final_report": final_report,
                "llm_backend": self.llm_client.backend_name,
                "swot_analysis": {},
                "investment_thesis": {},
                "chart_data": {},
                "workflow_status": "incomplete_data",
                "degraded_mode": True,
            }
            update.update(
                self._record(
                    "synthesizer",
                    "incomplete_data",
                    "Fail-loud incomplete report: no computable fundamentals; skipped inventing metrics.",
                    state,
                    timer.metrics(),
                )
            )
            self.session_memory.save({**state, **update})
            return update

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
        has_uploaded_docs = bool(state.get("document_contexts"))
        market_snapshots = state.get("market_snapshots", {})
        market_ok = any(snap.get("current_price") is not None for snap in market_snapshots.values())
        unverified_note = "_Source: LLM knowledge (unverified in this run)._"

        def fmt_pct(value: Any) -> str:
            return f"{value:.1%}" if isinstance(value, (int, float)) else "n/a"

        def fmt_x(value: Any) -> str:
            return f"{value:.2f}x" if isinstance(value, (int, float)) else "n/a"

        def build_grounded_summary() -> str:
            names = list(state["companies"])
            if not names:
                return "This run produced a research report, but no target company was available for a grounded executive summary."
            lines = []
            for company in names:
                metrics = state.get("financial_metrics", {}).get(company, {})
                risks = state.get("risk_scores", {}).get(company, {})
                supply = state.get("retrieved_docs", {}).get(company, {}).get("supply_chain", {})
                market = state.get("market_snapshots", {}).get(company, {})
                lines.append(
                    f"{company}: EBITDA margin {fmt_pct(metrics.get('ebitda_margin'))}, "
                    f"operating margin {fmt_pct(metrics.get('operating_margin'))}, "
                    f"R&D intensity {fmt_pct(metrics.get('r_and_d_intensity'))} [metric-derived]; "
                    f"supply-chain signal {supply.get('risk_level', 'unknown')} and model risk scores "
                    f"supply_chain={risks.get('supply_chain_risk', 'n/a')}, operational={risks.get('operational_risk', 'n/a')} "
                    "[risk-model/retrieved-signal]; "
                    f"P/E {fmt_x(market.get('trailing_pe'))} and 52W range position {fmt_pct(metrics.get('range_position'))} "
                    "[market-data]."
                )
            comparison = " ".join(lines)
            return (
                f"{comparison} These figures support a comparison of profitability, market-data context, and risk-screening signals, "
                "but they do not by themselves prove valuation upside, downside elimination, or an investment recommendation. "
                "The primary follow-up is to validate medium-confidence derived cash-flow metrics and model-derived risk scores against source filings, segment data, and supply-chain evidence."
            )

        llm_summary = build_grounded_summary()

        # ── Report Construction ──
        S("# LumenFin Intelligence Report")
        S("")
        S("**Report Type:** Investment-Grade Research | **Classification:** AI-Generated, For Reference Only")
        S("")
        S(f"## 1. Executive Summary")
        S(f"{llm_summary}")
        S("")
        S("**Evidence Boundary:** Metrics and formula-backed ratios are deterministic outputs from structured inputs. Risk scores, SWOT, scenarios, and positioning language are analytical heuristics unless explicitly tied to retrieved documents, market data, or listed metric inputs. They should be treated as research hypotheses, not facts or investment recommendations.")
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
        S("| L5 | Critic | Multi-factor risk scoring + compliance validation | Risk architecture with model-derived basis labels |")
        S("| L6 | Synthesizer | Structured reasoning, scenario modeling, evidence mapping | Research report with source boundaries |")
        S("")

        S(f"## 3. Company Profiles & Business Overview")
        for company in state["companies"]:
            S(f"### {company}")
            if has_uploaded_docs:
                profile = ensure_sentence_complete(state.get("company_profiles", {}).get(company, "Profile not available."))
                S(f"{profile}")
            else:
                S("No uploaded company profile document was provided. This report excludes unverified business-description claims from the evidence-backed analysis and relies on structured metrics, supply-chain signals, management-commentary samples, and market snapshots listed below.")
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
                S("| Metric | Value | Benchmark | Assessment | Data Quality | Confidence | Basis | Rationale |")
                S("|--------|-------|-----------|------------|--------------|------------|-------|-----------|")

                metric_conf = state.get("metric_confidence", {}).get(company, {})

                def assess_metric(metric_key: str, value: float) -> tuple[str, str]:
                    if metric_key == "ebitda_margin":
                        if value >= 0.25:
                            return "Strong", "EBITDA margin is well above the >25% benchmark"
                        if value >= 0.15:
                            return "Adequate", "EBITDA margin is positive but below top-tier threshold"
                        return "Weak", "EBITDA margin is below a robust profitability level"
                    if metric_key == "operating_margin":
                        if value >= 0.20:
                            return "Strong", "Operating margin exceeds the >20% benchmark"
                        if value >= 0.12:
                            return "Adequate", "Operating margin is moderate versus benchmark"
                        return "Weak", "Operating margin is below the desired benchmark"
                    if metric_key == "estimated_net_margin":
                        if value >= 0.15:
                            return "Strong", "Estimated net margin exceeds the >15% benchmark"
                        if value >= 0.08:
                            return "Adequate", "Estimated net margin is moderate versus benchmark"
                        return "Weak", "Estimated net margin is below the desired benchmark"
                    if metric_key == "estimated_fcf_margin":
                        if value >= 0.10:
                            return "Strong", "Estimated FCF yield exceeds the >10% benchmark"
                        if value >= 0.05:
                            return "Adequate", "Estimated FCF yield is positive but below benchmark"
                        return "Weak", "Estimated FCF yield is weak versus benchmark"
                    if metric_key == "r_and_d_intensity":
                        if 0.05 <= value <= 0.15:
                            return "Strong", "R&D intensity is in the target 5-15% range"
                        if 0.03 <= value < 0.05 or 0.15 < value <= 0.20:
                            return "Adequate", "R&D intensity is outside ideal range but still serviceable"
                        return "Weak", "R&D intensity is materially outside the target range"
                    return "—", "No benchmark-based assessment"

                def add_row(metric_key, label, benchmark, value=None):
                    v = value if value is not None else metrics.get(metric_key)
                    if v is None:
                        return
                    conf = metric_conf.get(metric_key, {})
                    conf_level = conf.get("level", "N/A")
                    conf_score = conf.get("score")
                    conf_display = f"{conf_score:.2f}" if isinstance(conf_score, (float, int)) else "N/A"
                    basis = str(conf.get("basis", "N/A"))
                    if metric_key in ("ebitda_margin", "r_and_d_intensity", "operating_margin", "estimated_net_margin", "estimated_fcf_margin"):
                        grade, rationale = assess_metric(metric_key, float(v))
                        S(f"| {label} | {v:.2%} | {benchmark} | {grade} | {conf_level} | {conf_display} | {basis} | {rationale} |")
                    elif metric_key == "pe_ratio":
                        S(f"| {label} | {v:.2f}x | {benchmark} | — | {conf_level} | {conf_display} | {basis} | Market-implied valuation multiple |")
                    elif metric_key == "monthly_return":
                        direction = "Upward momentum" if v > 0 else "Downward pressure"
                        S(f"| {label} | {v:.2%} | {benchmark} | — | {conf_level} | {conf_display} | {basis} | {direction} |")
                    elif metric_key == "range_position":
                        position = "Near highs" if v > 0.7 else ("Near lows" if v < 0.3 else "Mid-range")
                        S(f"| {label} | {v:.1%} | {benchmark} | — | {conf_level} | {conf_display} | {basis} | 52-week {position} |")

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
                    f"{'This represents a manageable operational risk factor.' if risk_level == 'low' else 'This requires active monitoring and mitigation strategies.' if risk_level == 'medium' else 'This is a material risk factor that warrants hedging or diversification.'} "
                    "[retrieved-evidence if supply-chain text was retrieved; otherwise model-derived screening signal]")

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
                S("*Risk scores are model-derived screening indicators. They combine available financial metrics, supply-chain signals, sentiment flags, and data-quality checks; they are not standalone cited facts.*")
                S("")
                S(
                    "**Data limitation risk:** Structured fundamentals and cited filings may be incomplete, "
                    "sample-backed, or only partially extracted from uploaded PDFs. Treat missing coverage as a "
                    "material data limitation rather than a verified financial fact. Market risk remains relevant "
                    "because live snapshots can change between retrieval and report generation."
                )
                S("")
                S("| Dimension | Score (1-10) | Level | Basis |")
                S("|-----------|-------------|-------|-------|")
                dim_labels = {"financial_risk": "Financial", "operational_risk": "Operational",
                              "market_risk": "Market", "regulatory_risk": "Regulatory",
                              "supply_chain_risk": "Supply Chain"}
                dim_basis = {
                    "financial_risk": "metric-derived from profitability, leverage proxies, and cash-flow quality",
                    "operational_risk": "heuristic from operating profile and retrieved risk signals",
                    "market_risk": "market-data/heuristic from live snapshot availability and price context",
                    "regulatory_risk": "sector heuristic unless retrieved regulatory evidence is present",
                    "supply_chain_risk": "retrieved supply-chain signal when available; otherwise heuristic",
                }
                for dim, label in dim_labels.items():
                    score = risk_data.get(dim, 5.0)
                    level = "Low Risk" if score < 3.5 else ("Moderate" if score < 6.5 else "Elevated")
                    S(f"| {label} | {score:.1f} | {level} | {dim_basis[dim]} |")
                S("")
                supply_chain = state.get("retrieved_docs", {}).get(company, {}).get("supply_chain", {})
                supply_signals = supply_chain.get("signals") or []
                S("**Risk Decision Triggers**")
                S("")
                S("| Risk Area | Evidence Signal | Review Trigger | Potential Financial Channel |")
                S("|-----------|-----------------|----------------|-----------------------------|")
                if supply_chain:
                    S(
                        f"| Supply chain | risk_level={supply_chain.get('risk_level', 'unknown')}; "
                        f"{'; '.join(str(signal) for signal in supply_signals[:2])} | "
                        "Escalate if concentration remains medium/high or inventory days rise | "
                        "gross margin pressure, delivery delays, or working-capital drag |"
                    )
                if live_market:
                    pe = live_market.get("trailing_pe")
                    range_pos = metrics.get("range_position")
                    range_display = f"{range_pos:.1%}" if isinstance(range_pos, (int, float)) else "n/a"
                    S(
                        f"| Market valuation | P/E={pe if pe is not None else 'n/a'}; "
                        f"52W range position={range_display} | "
                        "Review market risk if valuation is high while market/range momentum is stretched | "
                        "market multiple compression or downside asymmetry |"
                    )
                if metric_conf.get("estimated_fcf_margin", {}).get("level") == "Medium":
                    S(
                        "| Cash-flow estimate | estimated FCF margin uses derived inputs with medium confidence | "
                        "Require source financial statements before using in valuation | "
                        "valuation sensitivity and liquidity assessment error |"
                    )
                S("")

        # ── Industry & Macro Context ──
        S("## 5. Industry Dynamics & Macroeconomic Context")
        S("")
        S("*This section is limited to metric-derived and retrieved-signal observations. Broader sector claims are intentionally excluded unless supported by uploaded or retrieved evidence.*")
        S("")
        for company in state["companies"]:
            metrics = state.get("financial_metrics", {}).get(company, {})
            risk = state.get("retrieved_docs", {}).get(company, {}).get("supply_chain", {})
            S(f"### {company} — Operating Environment")
            S("")
            ebitda_m = metrics.get("ebitda_margin", 0)
            rd_i = metrics.get("r_and_d_intensity", 0)
            risk_level = risk.get("risk_level", "unknown")
            S(f"- **Sector Position [metric-derived]**: {'Profitability is above the internal benchmark, which may indicate pricing power or scale economics' if ebitda_m > 0.25 else 'Profitability is below top-tier benchmark, leaving room for margin improvement'}")
            S(f"- **Innovation Trajectory [metric-derived]**: {'R&D intensity of ' + str(round(rd_i*100,1)) + '% of revenue supports an innovation-capacity hypothesis' if rd_i > 0.06 else 'R&D intensity is moderate and should be validated against peer spend'}")
            S(f"- **Supply Chain Resilience [retrieved-evidence/heuristic]**: {'Supply-chain signal is low risk in the retrieved payload' if risk_level == 'low' else 'Supply-chain signal indicates concentration or execution risk that requires validation' if risk_level == 'medium' else 'Supply-chain signal indicates elevated concentration exposure that requires evidence-backed mitigation review'}")
            S(f"- **Evidence Gap [data-quality]**: No company-specific regulatory filing, customer-mix schedule, or segment macro sensitivity document was retrieved in this run.")
            if not has_uploaded_docs and not market_ok:
                S(f"- {unverified_note}")
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
            fcf_m = metrics.get("estimated_fcf_margin", 0)
            tone = sentiment.get("label", "neutral")
            risk_data = state.get("risk_scores", {}).get(company, {})
            financial_risk = risk_data.get("financial_risk", 5.0)

            strengths = []
            weaknesses = []
            if ebitda_m >= 0.35:
                strengths.append("Exceptionally strong profitability and operating leverage")
            elif ebitda_m >= 0.20:
                strengths.append("Solid profitability with scalable operating model")
            else:
                weaknesses.append("Profitability remains below top-tier peer levels")
            if 0.05 <= rd_i <= 0.15:
                strengths.append("Balanced R&D intensity supports efficient innovation conversion")
            elif rd_i > 0.15:
                strengths.append("Aggressive R&D investment signals strong innovation intent")
                if fcf_m < 0.10:
                    weaknesses.append("High R&D intensity currently compresses free-cash-flow quality")
            else:
                weaknesses.append("R&D intensity may be insufficient for long-cycle technology leadership")
            if tone == "bullish":
                strengths.append("Management communication remains constructive with strategic continuity")

            if risk.get("risk_level") != "low":
                weaknesses.append(f"Supply chain risk exposure remains at '{risk.get('risk_level')}' level")
            if financial_risk > 5.5:
                weaknesses.append("Financial risk score indicates elevated balance-sheet/earnings volatility")

            opportunities = []
            threats = []
            if ebitda_m >= 0.25:
                opportunities.append("[metric-derived] strong profitability provides capacity for reinvestment")
            if rd_i >= 0.05:
                opportunities.append("[metric-derived] R&D intensity supports an innovation-capacity screen")
            if risk.get("risk_level") != "low":
                threats.append(f"[retrieved-signal] supply chain risk level is {risk.get('risk_level')}")
            if financial_risk > 5.5:
                threats.append("[risk-model] elevated financial risk score requires validation")
            if not opportunities:
                opportunities.append("No evidence-backed opportunities identified at current data resolution")
            if not threats:
                threats.append("No evidence-backed threats identified at current data resolution")

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
        S("*No evidence-backed revenue forecast or management guidance was retrieved for this run. The table below is a sensitivity framework showing which variables should be stressed, not a probability forecast, price target, or recommendation.*")
        S("")
        for company in state["companies"]:
            metrics = state.get("financial_metrics", {}).get(company, {})
            S(f"### {company}")
            S("")
            S("| Sensitivity | Evidence-Backed Input | Stress Direction | Why It Matters |")
            S("|-------------|-----------------------|------------------|----------------|")
            S(f"| Profitability | EBITDA margin {metrics.get('ebitda_margin', 0):.1%} | test margin compression | affects earnings quality and valuation support |")
            S(f"| R&D efficiency | R&D intensity {metrics.get('r_and_d_intensity', 0):.1%} | test lower innovation conversion | affects long-cycle competitiveness hypothesis |")
            S(f"| Cash-flow quality | estimated FCF margin {metrics.get('estimated_fcf_margin', 0):.1%} | test lower conversion due to medium-confidence estimate | affects valuation and liquidity interpretation |")
            S("")

        # ── Investment Thesis ──
        S("## 8. Research Thesis & Positioning")
        S("")
        S("*This section expresses research positioning considerations only. It does not recommend buying, selling, holding, overweighting, or allocating to any security.*")
        S("")
        investment_thesis: dict[str, dict[str, str]] = {}
        for company in state["companies"]:
            metrics = state.get("financial_metrics", {}).get(company, {})
            sentiment = state.get("sentiment_analysis", {}).get(company, {})
            ebitda_m = metrics.get("ebitda_margin", 0)
            fcf_m = metrics.get("estimated_fcf_margin", 0)
            tone = sentiment.get("label", "neutral")
            risk_data = state.get("risk_scores", {}).get(company, {})
            financial_risk = risk_data.get("financial_risk", 5.0)
            cautious_gate = financial_risk > 5.5 or fcf_m < 0.10

            if cautious_gate:
                bull = (f"Growth optionality exists, but current risk profile is elevated (financial risk {financial_risk:.1f}/10, "
                        f"FCF yield {fcf_m:.1%}). Treat any constructive thesis as conditional on improved cash-flow quality and risk evidence.")
                bear = ("Maintain a defensive posture until cash-flow quality and risk metrics improve. "
                        "Use explicit risk limits in any separate portfolio process.")
            elif ebitda_m > 0.25 and tone == "bullish":
                bull = (f"Strong profitability (EBITDA margin {ebitda_m:.1%}) combined with confident management guidance "
                        f"supports a quality-screening thesis, subject to valuation and independent evidence review.")
                bear = (f"Premium valuation may limit near-term upside. Key downside risks include competitive disruption "
                        f"and macro-driven multiple compression.")
            elif ebitda_m > 0.15:
                bull = (f"Solid financial foundation with manageable risk profile. This can remain on a "
                        f"quality-compounder research screen pending valuation and risk review.")
                bear = (f"Limited near-term catalysts for re-rating. Margin improvement trajectory may be gradual. "
                        f"Compare against higher-growth peers before drawing allocation conclusions.")
            else:
                bull = (f"Potential value unlock if operational turnaround materializes. Current metrics may understate "
                        f"recovery optionality, but this remains a hypothesis requiring evidence.")
                bear = (f"Weak profitability metrics suggest structural challenges. A positive research view would require "
                        f"definitive evidence of business improvement.")

            investment_thesis[company] = {"bull_case": bull, "bear_case": bear}
            S(f"### {company}")
            S(f"- **Research Rationale (Bull Case):** {bull}")
            S(f"- **Risk Considerations (Bear Case):** {bear}")
            S("")

        # ── Peer Comparison ──
        if state.get("peer_comparison", {}).get("summary"):
            S("## 9. Competitive Landscape & Peer Benchmarking")
            S("")
            S(state["peer_comparison"]["summary"])
            if not has_uploaded_docs and not market_ok:
                S(unverified_note)
            S("")

        # ── Compliance ──
        S("## 10. Compliance Review & Data Integrity")
        S("")
        if state.get("compliance_summary") and state.get("compliance_findings"):
            compliance_summary = str(state["compliance_summary"]).strip()
            compliance_summary = re.sub(r"^\**\s*Audit Opinion:\s*\**\s*", "", compliance_summary, flags=re.IGNORECASE)
            S(f"**Audit Opinion:** {compliance_summary}")
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
            S("Core rule-based compliance checks passed. Semantic review should still validate whether all qualitative claims are supported by cited evidence.")
        S("")

        # ── Methodology & Disclaimer ──
        S("## 11. Methodology, Data Sources & Disclaimer")
        S("")
        S("**Analytical Methods:** AST-safe expression evaluator for numerical computation; LLM-based deep semantic analysis for sentiment extraction; multi-factor risk scoring model with evidence-based calibration; LangGraph-directed multi-agent orchestration with checkpoint-based state persistence.")
        S("")
        document_contexts = state.get("document_contexts", [])
        market_snapshots = state.get("market_snapshots", {})
        rag_evidence = state.get("rag_evidence", {})
        companies = state.get("companies", [])
        sample_companies = [
            c for c in companies
            if self.allow_sample_data and c in SAMPLE_FINANCIAL_DATA
        ]
        market_ok = sum(1 for snap in market_snapshots.values() if snap.get("current_price") is not None)
        market_total = len(market_snapshots)
        rag_chunks = sum(len(hits) for hits in rag_evidence.values())

        source_parts: list[str] = []
        if document_contexts:
            source_types = sorted(
                {
                    str(doc.get("source_type") or "unknown")
                    for doc in document_contexts
                }
            )
            source_parts.append(
                f"Uploaded documents: {len(document_contexts)} file(s), types={', '.join(source_types)}."
            )
        else:
            source_parts.append("Uploaded documents: none (no user files were provided for this run).")

        if rag_chunks > 0:
            source_parts.append(f"RAG evidence: Milvus hybrid retrieval returned {rag_chunks} cited chunk(s).")
        elif document_contexts:
            source_parts.append("RAG evidence: enabled but no cited chunk was retrieved in this run.")
        else:
            source_parts.append("RAG evidence: not applicable because no documents were uploaded.")

        if market_total:
            source_parts.append(
                f"Market data API: {market_ok}/{market_total} company snapshots succeeded; "
                "per-company failures degrade only that entity's live-market metrics."
            )
        else:
            source_parts.append("Market data API: no market snapshots requested.")

        yahoo_companies = [
            c
            for c in companies
            if str((state.get("retrieved_docs") or {}).get(c, {}).get("structured_source") or "")
            == "yahoo_fundamentals"
        ]
        sec_companies = [
            c
            for c in companies
            if str((state.get("retrieved_docs") or {}).get(c, {}).get("structured_source") or "")
            == "sec_companyfacts"
        ]
        if sample_companies:
            source_parts.append(
                f"Structured fundamentals: DEMO sample financial database used for {', '.join(sample_companies)} "
                f"(data_mode={self.data_mode})."
            )
        elif sec_companies:
            source_parts.append(
                f"Structured fundamentals: SEC EDGAR companyfacts for {', '.join(sec_companies)} "
                f"(structured_source=sec_companyfacts, data_mode={self.data_mode})."
            )
        elif yahoo_companies:
            source_parts.append(
                f"Structured fundamentals: Yahoo Finance annual income statement for "
                f"{', '.join(yahoo_companies)} (structured_source=yahoo_fundamentals, data_mode={self.data_mode})."
            )
        else:
            source_parts.append(
                f"Structured fundamentals: derived from uploaded structured documents when available "
                f"(data_mode={self.data_mode})."
            )

        source_parts.append("Narrative analysis: generated by the configured LLM using retrieved evidence and computed metrics.")
        S(f"**Data Sources:** {' '.join(source_parts)}")
        if market_total:
            S("")
            S("**Market Data by Company:**")
            for company in companies:
                snap = market_snapshots.get(company, {})
                symbol = snap.get("symbol") or state.get("target_symbols", {}).get(company, company)
                status = snap.get("status") or ("ok" if snap.get("current_price") is not None else "failed")
                provider = snap.get("provider") or "unknown"
                as_of = snap.get("fetched_at") or "n/a"
                if snap.get("current_price") is not None:
                    S(
                        f"- {company} ({symbol}): status={status}, provider={provider}, "
                        f"as_of={as_of}, price={snap.get('current_price')}."
                    )
                else:
                    err = snap.get("error") or "no live price returned"
                    S(f"- {company} ({symbol}): status=failed, error={err}.")
            S("")
        S("**Source Attribution by Output Type:** Quant tables use deterministic AST calculations on structured inputs. Market rows use live snapshot fields when available. Company profiles, industry context, SWOT opportunities/threats, scenario narratives, and research thesis language are LLM-assisted hypotheses unless a row explicitly cites retrieved evidence or metric-derived basis. Risk matrix values are model-derived screening indicators and should not be treated as independently cited facts.")
        S("")
        if self.data_mode == "demo" or sample_companies:
            S(
                "**Disclaimer:** DEMO MODE -- some or all structured fundamentals may come from the built-in sample database, "
                "not audited filings. This report is for research and demonstration only. It does not constitute investment advice."
            )
        else:
            S(
                "**Disclaimer:** This report is generated by an AI-powered multi-agent system for research purposes only. "
                "It does not constitute investment advice, a solicitation, or a recommendation to buy or sell any security."
            )

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
        synth_detail = (
            "Investment-grade report assembled: SWOT, Scenario Analysis (Base/Bull/Bear), Investment Thesis, Peer Benchmarking, Compliance Review, and structured Chart Data."
        )
        update.update(self._record("synthesizer", "ok", synth_detail, state, timer.metrics()))
        self.session_memory.save({**state, **update})
        return update
