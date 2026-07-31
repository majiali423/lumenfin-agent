from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Any, Iterator

from .input_guardrail import GuardrailMode, guard_documents, sanitize_retrieval_hits
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
from .rag.dedupe import dedupe_cross_company_rag_hits
from .rag.telemetry import summarize_rag_telemetry
from .repair_policies import RETRIEVAL_WORTHY_CODES
from .reporting import (
    build_analyst_executive_summary,
    effective_report_output_format,
    filter_claims_for_brief,
    format_comparison_capsule,
    format_next_actions,
    format_peer_metric_matrix,
    format_period_alignment_notice,
    format_rag_citation_section,
    humanize_citation,
    is_low_signal_claim,
    requested_fiscal_year_from_state,
)
from .claims import (
    binding_summary,
    build_claims,
    claim_to_dict,
    filter_verified,
    format_verified_claims_ledger,
    verified_by_entity,
)
from .skills import get_skill_specs
from .state import FinanceState
from .metrics_schema import get_fundamental, set_fundamental
from .fundamentals import is_plausible_revenue_billion_usd
from .documents import is_trusted_ast_amount, normalize_metric_hints_to_billion_usd
from .tools import (
    analyze_sentiment_deep,
    build_chart_data,
    build_coverage_matrix,
    calculate_derived_ratios,
    canonicalize_companies,
    classify_quant_status,
    derive_target_symbols,
    extract_companies_from_query,
    generate_scenario_analysis,
    has_computable_fundamentals,
    is_partial_compare_gap,
    non_comparable_companies,
    resolve_safe_formula,
    retrieve_company_payload,
    safe_execute_formula,
    summarize_document_context,
    has_supply_chain_signal,
)


_PEER_COMPARISON_LEAK_MARKERS = (
    "we need to",
    "let's draft",
    "the instruction says",
    "the user asked",
    "we must ",
    "i need to",
)


def _single_company_peer_summary(company: str) -> str:
    return (
        f"Peer comparison is unavailable because only {company} has "
        "comparable structured ratio metrics in this run."
    )


def _peer_comparison_is_safe(text: str) -> bool:
    cleaned = (text or "").strip()
    lowered = cleaned.casefold()
    if not cleaned or len(cleaned) > 1200:
        return False
    if any(marker in lowered for marker in _PEER_COMPARISON_LEAK_MARKERS):
        return False
    return cleaned.endswith((".", "!", "?"))


def _peer_comparison_fallback(companies: list[str]) -> str:
    names = ", ".join(companies)
    return (
        f"Structured peer metrics are available for {names}. "
        "See the Executive Summary comparison capsule and Peer Metric Matrix for verified ratios; "
        "no free-form peer narrative is invented beyond those figures."
    )


def _deterministic_peer_comparison(comparable_metrics: dict[str, dict[str, Any]]) -> str:
    """Rule-based peer blurb from AST metrics only (no LLM; no invented margins/returns)."""
    companies = list(comparable_metrics.keys())
    if len(companies) < 2:
        return _single_company_peer_summary(companies[0]) if companies else (
            "No quantitative metrics were available for peer comparison."
        )
    capsule = format_comparison_capsule(
        {"companies": companies, "financial_metrics": comparable_metrics}
    )
    bullets = [line for line in capsule if line.startswith("- ")]
    if not bullets:
        return _peer_comparison_fallback(companies)
    return (
        "Peer comparison is limited to verified structured ratios "
        "(see Executive Summary capsule and Peer Metric Matrix):\n"
        + "\n".join(bullets)
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
        rag_index_mode: str = "sync_on_run",
        company_parallelism: int = 4,
        profile_llm_max_attempts: int = 1,
        input_guardrail_enabled: bool = True,
        input_guardrail_mode: GuardrailMode = "sanitize",
        rag_sanitize_hits: bool = True,
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
        self.rag_index_mode = rag_index_mode if rag_index_mode in {"sync_on_run", "async_on_upload"} else "sync_on_run"
        self.company_parallelism = max(1, company_parallelism)
        self.profile_llm_max_attempts = max(0, min(3, int(profile_llm_max_attempts)))
        self.input_guardrail_enabled = input_guardrail_enabled
        self.input_guardrail_mode = input_guardrail_mode if input_guardrail_mode in {"sanitize", "block"} else "sanitize"
        self.rag_sanitize_hits = bool(rag_sanitize_hits)
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
        rag_meta: dict[str, Any] = {
            "degraded": False,
            "degrade_reason": "",
            "mode": "",
            "vector_hits": 0,
            "keyword_hits": 0,
        }
        if self.rag_enabled and self.hybrid_retriever and document_contexts:
            source_document_ids = list(state.get("rag_document_ids") or [])
            use_stored = self.rag_index_mode == "async_on_upload" and bool(source_document_ids)
            rag_hits, rag_meta = self.hybrid_retriever.retrieve_for_company_with_meta(
                query=retrieval_query,
                company=company,
                session_id=session_id,
                document_contexts=document_contexts,
                tenant_id=state.get("rag_tenant_id") if use_stored else None,
                source_document_ids=source_document_ids if use_stored else None,
                use_stored_chunks=use_stored,
            )
            if self.rag_sanitize_hits and rag_hits:
                rag_hits, sanitize_findings = sanitize_retrieval_hits(rag_hits)
                rag_meta["sanitized_finding_count"] = len(sanitize_findings)
            else:
                rag_meta["sanitized_finding_count"] = 0

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
            allow_sample_data=self.allow_sample_data
            and not bool((state.get("query_plan") or {}).get("prefer_uploaded_only")),
            ticker=state.get("target_symbols", {}).get(company),
            fetch_live_fundamentals=self.fetch_live_fundamentals
            and not bool((state.get("query_plan") or {}).get("prefer_uploaded_only")),
            fetch_sec_fundamentals=self.fetch_sec_fundamentals
            and not bool((state.get("query_plan") or {}).get("prefer_uploaded_only")),
            prefer_uploaded_only=bool((state.get("query_plan") or {}).get("prefer_uploaded_only")),
            prefer_fiscal_year=requested_fiscal_year_from_state(state),
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
            doc_text = "\n".join(
                str(doc.get("text") or doc.get("excerpt") or "")
                for doc in document_contexts
                if isinstance(doc, dict)
            )
            hint_meta: dict[str, dict] = {}
            for doc in document_contexts:
                if not isinstance(doc, dict):
                    continue
                if isinstance(doc.get("metric_hint_meta"), dict):
                    hint_meta.update(doc["metric_hint_meta"])
                per_co = (doc.get("per_company_metric_hint_meta") or {}).get(company)
                if isinstance(per_co, dict):
                    hint_meta.update(per_co)
            normalized_hints = normalize_metric_hints_to_billion_usd(
                dict(document_summary["metric_hints"]),
                text=doc_text,
                hint_meta=hint_meta or None,
            )
            payload.setdefault("document_observations", {})
            payload["document_observations"]["metric_hints"] = dict(normalized_hints)
            payload["document_observations"]["metric_hint_meta"] = dict(hint_meta)
            payload.setdefault("fundamental_provenance", {})
            applied_abs = False
            for key in ("revenue", "ebitda", "r_and_d", "operating_income"):
                meta = hint_meta.get(key) or {}
                if not is_trusted_ast_amount(meta):
                    continue
                value = meta.get("normalized_value", normalized_hints.get(key))
                if value is None:
                    continue
                if key == "revenue" and not is_plausible_revenue_billion_usd(float(value)):
                    continue
                set_fundamental(payload["market_data"], key, float(value))
                payload["fundamental_provenance"][key] = {
                    "source": "document_extracted",
                    "confidence": meta.get("confidence"),
                    "normalization_source": meta.get("normalization_source"),
                    "period": meta.get("period_hint"),
                }
                applied_abs = True
            # Prefer document label only when upload alone provided the AST spine.
            # Issuer SEC/Yahoo gap-fill must keep sec_companyfacts / yahoo_fundamentals.
            if applied_abs and any(
                key in (payload.get("fundamental_provenance") or {})
                for key in ("revenue", "ebitda", "r_and_d")
            ):
                meta = payload.get("fundamentals_meta") or {}
                if not meta.get("live_fallback_used"):
                    payload["structured_source"] = "document_extracted"
        if payload["source_documents"]:
            payload["earnings_call_quotes"] = payload["earnings_call_quotes"] or [
                doc["excerpt"][:300] for doc in payload["source_documents"] if doc.get("excerpt")
            ]
        if payload["source_documents"] and payload["supply_chain"]["risk_level"] == "unknown":
            excerpt = " ".join(doc.get("excerpt", "") for doc in payload["source_documents"])
            payload["supply_chain"]["risk_level"] = "medium" if has_supply_chain_signal(excerpt) else "low"

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

        max_attempts = self.profile_llm_max_attempts
        if max_attempts <= 0:
            profile = f"Profile generation skipped for {company}."
        else:
            try:
                profile = self.llm_client.chat(
                    system_prompt="You are an equity research analyst. Write factual, professional company profiles.",
                    user_prompt=profile_prompt,
                    temperature=0.2,
                    max_tokens=280,
                )
                attempts_used = 1
                if attempts_used < max_attempts and (
                    _looks_non_english(profile) or _looks_incomplete(profile)
                ):
                    profile = self.llm_client.chat(
                        system_prompt=(
                            "You are an equity research analyst. Rewrite the profile in clean, complete English only. "
                            "Do not include Chinese characters. End with a complete sentence."
                        ),
                        user_prompt=profile,
                        temperature=0.1,
                        max_tokens=280,
                    )
                    attempts_used += 1
                if attempts_used < max_attempts and (
                    _looks_non_english(profile) or _looks_incomplete(profile)
                ):
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
            rag_enabled=bool(self.rag_enabled and self.hybrid_retriever),
            rag_hit_count=len(rag_hits),
            document_count=len(document_contexts),
            data_mode=self.data_mode,
            rag_degraded=bool(rag_meta.get("degraded")),
            rag_degrade_reason=str(rag_meta.get("degrade_reason") or ""),
            rag_mode=str(rag_meta.get("mode") or ""),
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
            provider_errors=list(payload.get("provider_errors") or []),
            rag_meta=dict(rag_meta),
        )

    def retrieval(self, state: FinanceState) -> FinanceState:
        with self._track_step("retrieval") as timer:
            include_appendix = state.get("appendix_search_done", False)
            document_contexts = state.get("document_contexts", [])
            rag_index_stats = dict(state.get("rag_index_stats", {}))
            session_id = state.get("thread_id", "default-session")
            retrieval_query = state["query"]
            query_plan = state.get("query_plan", {})
            if query_plan.get("retrieval_query"):
                retrieval_query = str(query_plan["retrieval_query"])
            elif query_plan.get("analysis_dimensions"):
                retrieval_query = (
                    f"{state['query']} | focus: {', '.join(query_plan['analysis_dimensions'])}"
                )

            if (
                self.rag_enabled
                and self.hybrid_retriever
                and getattr(self.hybrid_retriever, "rag_store", None)
                and document_contexts
                and not rag_index_stats
                and self.rag_index_mode == "sync_on_run"
            ):
                rag_index_stats = self.hybrid_retriever.rag_store.index_documents(
                    document_contexts,
                    session_id=session_id,
                )
            elif self.rag_index_mode == "async_on_upload" and rag_index_stats:
                # Already indexed at upload time; preserve stats and do not re-embed.
                rag_index_stats = {
                    **rag_index_stats,
                    "mode": rag_index_stats.get("mode") or "async_on_upload",
                    "search_only": True,
                }

            # Warm query embedding once before parallel per-company search.
            if (
                self.rag_enabled
                and self.hybrid_retriever
                and getattr(self.hybrid_retriever, "rag_store", None)
                and document_contexts
            ):
                try:
                    self.hybrid_retriever.rag_store.prime_query_embedding(retrieval_query)
                except Exception:
                    # Per-company retrieve will degrade to keyword-only if configured.
                    pass

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
            rag_degraded_companies: list[str] = []
            company_rag_metas: list[dict[str, Any]] = []
            sanitized_finding_count = 0
            for artifact in bundles:
                company = artifact.company
                retrieved_docs[company] = artifact.to_legacy_payload()
                market_snapshots[company] = artifact.market_snapshot
                company_profiles[company] = artifact.profile
                retrieval_provenance[company] = artifact.provenance.to_dict()
                company_rag_metas.append(dict(artifact.rag_meta or {}))
                sanitized_finding_count += int((artifact.rag_meta or {}).get("sanitized_finding_count") or 0)
                if artifact.provenance.rag_degraded:
                    rag_degraded_companies.append(company)
                if artifact.rag_hits:
                    rag_evidence[company] = artifact.rag_hits

            rag_evidence = dedupe_cross_company_rag_hits(rag_evidence)

            if rag_degraded_companies:
                rag_index_stats = {
                    **rag_index_stats,
                    "rag_degraded": True,
                    "degraded_companies": rag_degraded_companies,
                    "degrade_mode": "keyword_only",
                }
            # Capture query-embed timing from the shared store after prime/search.
            store = self.hybrid_retriever.rag_store if self.hybrid_retriever else None
            if store is not None:
                rag_index_stats = {
                    **rag_index_stats,
                    "embed_ms": float(rag_index_stats.get("embed_ms") or getattr(store, "last_embed_ms", 0.0) or 0.0),
                    "embed_chars": int(
                        rag_index_stats.get("embed_chars") or getattr(store, "last_embed_chars", 0) or 0
                    ),
                }
            rag_telemetry = summarize_rag_telemetry(
                rag_index_stats=rag_index_stats,
                company_metas=company_rag_metas,
                sanitized_finding_count=sanitized_finding_count,
            )
            rag_index_stats = {**rag_index_stats, **rag_telemetry}
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
            provider_errors: list[dict[str, Any]] = []
            for company, payload in retrieved_docs.items():
                for item in payload.get("provider_errors") or []:
                    entry = dict(item)
                    entry.setdefault("company", company)
                    provider_errors.append(entry)
            from .provider_retry import summarize_provider_errors

            provider_error_summary = summarize_provider_errors(provider_errors)

            fatal_data_gap = bool(retrieved_docs) and not computable_companies
            company_names = list(retrieved_docs.keys())
            coverage_matrix = build_coverage_matrix(company_names, retrieved_docs)
            partial_data_gap = is_partial_compare_gap(company_names, coverage_matrix)
            prefer_uploaded_only = bool(query_plan.get("prefer_uploaded_only"))
            source_resolution = {
                "prefer_uploaded_only": prefer_uploaded_only,
                "mode": "uploaded_only" if prefer_uploaded_only else ("hybrid" if document_contexts else "live_or_sample"),
                "companies": {},
            }
            for company, payload in retrieved_docs.items():
                meta = dict(payload.get("fundamentals_meta") or {})
                source = str(payload.get("structured_source") or "none")
                live_fallback = bool(meta.get("live_fallback_used")) or source in {
                    "sec_companyfacts",
                    "yahoo_fundamentals",
                    "sample_db",
                } and bool(document_contexts) and source != "document_extracted"
                source_resolution["companies"][company] = {
                    "structured_source": source,
                    "upload_present": bool(meta.get("upload_present")) or bool(document_contexts),
                    "upload_had_computable_metrics": bool(
                        meta.get("upload_had_computable_metrics")
                    ),
                    "live_fallback_used": bool(meta.get("live_fallback_used"))
                    or (live_fallback and source != "document_extracted"),
                    "fallback_reason": meta.get("fallback_reason") or "",
                    "grounding_layer": meta.get("grounding_layer") or "",
                    "sec_filled_keys": list(meta.get("sec_filled_keys") or []),
                }
            if fatal_data_gap:
                # Fail-loud: do not enter appendix_replan loop when no AST-computable fundamentals exist.
                replan_reason = None
                if prefer_uploaded_only:
                    action_hint = (
                        "You asked to use uploaded materials only. The upload lacked extractable "
                        "revenue/EBITDA/R&D, and live SEC/Yahoo/sample backfill was disabled. "
                        "Upload a filing/CSV with those metrics, or remove the upload-only wording "
                        "so the system may fill gaps from SEC/Yahoo."
                    )
                elif self.data_mode == "demo":
                    action_hint = (
                        "Upload a filing PDF with extractable metrics, or analyze a company covered by "
                        "the demo sample database. Refusing to invent numbers."
                    )
                else:
                    action_hint = (
                        "Upload source filings with extractable metrics, retry the live fundamentals provider, "
                        "or explicitly switch to DATA_MODE=demo for demonstrations. Refusing to invent numbers."
                    )
                data_gap_detail = (
                    "No computable structured fundamentals for "
                    f"{', '.join(retrieved_docs)} (structured_source has no revenue/EBITDA/R&D inputs). "
                    f"{action_hint}"
                )
                if provider_error_summary.get("count"):
                    data_gap_detail += (
                        f" Provider errors: transient={provider_error_summary['transient_count']}, "
                        f"truly_missing/unavailable={provider_error_summary['missing_count']}, "
                        f"other={provider_error_summary['other_count']} "
                        f"(by_class={provider_error_summary['by_class']})."
                    )
                    if provider_error_summary.get("has_transient"):
                        data_gap_detail += (
                            " Transient provider failures were observed after bounded retries; "
                            "this may recover on a later run."
                        )
            else:
                replan_reason = (
                    "Appendix / evidence gap detected; switching to supplementary_retrieval "
                    "(appendix_replan) for one targeted retrieval pass."
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
                "source_resolution": source_resolution,
                "fatal_data_gap": fatal_data_gap,
                "partial_data_gap": partial_data_gap,
                "data_gap_detail": data_gap_detail,
                "coverage_matrix": coverage_matrix,
                "non_comparable_companies": non_comparable_companies(company_names, coverage_matrix),
                "provider_errors": provider_errors,
                "provider_error_summary": provider_error_summary,
                "degraded_mode": True if fatal_data_gap else (partial_data_gap or state.get("degraded_mode", False)),
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
                if rag_index_stats.get("rag_degraded"):
                    degraded = ", ".join(rag_index_stats.get("degraded_companies") or []) or "unknown"
                    detail += (
                        f" RAG degraded to keyword-only for {degraded} "
                        "(vector/embedding failure after retries)."
                    )
                if market_status.get("total_count"):
                    detail += (
                        f" Market API: {market_status['ok_count']}/{market_status['total_count']} "
                        f"snapshots ok (primary={getattr(self.market_data_client, 'provider', 'unknown')}, "
                        f"fallback={getattr(self.market_data_client, 'fallback_provider', 'yahoo')})."
                    )
                status = "needs_replan" if replan_reason else "ok"
            update.update(self._record("retrieval", status, detail, state, timer.metrics()))
            telemetry = dict(update.get("run_telemetry") or {})
            telemetry["rag"] = rag_telemetry
            update["run_telemetry"] = telemetry
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
        revenue = get_fundamental(market, "revenue")
        ebitda = get_fundamental(market, "ebitda")
        r_and_d = get_fundamental(market, "r_and_d")
        operating_income = get_fundamental(market, "operating_income")
        if revenue is not None:
            base_data["revenue"] = revenue
        if ebitda is not None:
            base_data["ebitda"] = ebitda
        if r_and_d is not None:
            base_data["r_and_d"] = r_and_d
        if operating_income is not None:
            base_data["operating_income"] = operating_income

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
                "metrics": {},
                "quant_status": "uncomputable",
                "metric_confidence": {},
            }

        quant_status = classify_quant_status(metrics)
        return {
            "company": company,
            "metrics": metrics,
            "quant_status": quant_status,
            "scenario": generate_scenario_analysis(metrics, company) if quant_status == "ast_ok" else {},
            "metric_confidence": metric_confidence,
        }

    def quantitative_analyst(self, state: FinanceState) -> FinanceState:
        with self._track_step("quant") as timer:
            company_items = list(state["retrieved_docs"].items())
            quant_results = map_in_parallel(
                lambda item: self._compute_company_quant(item[0], item[1], state),
                company_items,
                max_workers=self.company_parallelism,
            )

            financial_metrics: dict[str, dict[str, float]] = {}
            scenario_analyses: dict[str, dict[str, Any]] = {}
            metric_confidence: dict[str, dict[str, dict[str, Any]]] = {}
            uncomputable: list[str] = []
            market_only: list[str] = []
            for result in quant_results:
                company = result["company"]
                status = str(result.get("quant_status") or classify_quant_status(result.get("metrics")))
                if status == "uncomputable":
                    uncomputable.append(company)
                    continue
                financial_metrics[company] = result.get("metrics") or {}
                scenario_analyses[company] = result.get("scenario") or {}
                metric_confidence[company] = result.get("metric_confidence") or {}
                if status == "market_only":
                    market_only.append(company)

            companies = list(state.get("companies") or [])
            coverage_matrix = build_coverage_matrix(companies, state.get("retrieved_docs") or {}, financial_metrics)
            partial_data_gap = is_partial_compare_gap(companies, coverage_matrix)
            skipped = non_comparable_companies(companies, coverage_matrix)

            comparable_metrics = {
                company: metrics
                for company, metrics in financial_metrics.items()
                if (coverage_matrix.get(company) or {}).get("comparable")
            }
            peer_comparison_text = ""
            if len(comparable_metrics) == 1:
                peer_comparison_text = _single_company_peer_summary(
                    next(iter(comparable_metrics))
                )
            elif comparable_metrics:
                peer_comparison_text = _deterministic_peer_comparison(comparable_metrics)
                if skipped:
                    peer_comparison_text += (
                        f" Non-comparable peers omitted: {', '.join(skipped)}."
                    )
            elif financial_metrics:
                peer_comparison_text = (
                    "Structured ratio comparison skipped: only market-level indicators were available "
                    f"for {', '.join(financial_metrics.keys())}."
                )
            else:
                peer_comparison_text = "No quantitative metrics were available for peer comparison."

            detail_parts = [
                f"Quantitative engine computed {sum(len(m) for m in financial_metrics.values())} metrics "
                f"across {len(financial_metrics)} companies (parallel fan-out)."
            ]
            if partial_data_gap:
                detail_parts.append(
                    f"Partial peer coverage: comparable={list(comparable_metrics.keys())}, "
                    f"non_comparable={skipped}."
                )
            if uncomputable:
                detail_parts.append(f"Uncomputable entities: {', '.join(uncomputable)}.")
            if market_only:
                detail_parts.append(f"Market-only entities: {', '.join(market_only)}.")
            detail_parts.append(f"Key insight: {peer_comparison_text[:120]}...")

            quant_status_label = "partial" if partial_data_gap else "ok"
            update: FinanceState = {
                "financial_metrics": financial_metrics,
                "metric_confidence": metric_confidence,
                "replan_reason": None,
                "partial_data_gap": partial_data_gap,
                "coverage_matrix": coverage_matrix,
                "non_comparable_companies": skipped,
                "degraded_mode": bool(state.get("degraded_mode")) or partial_data_gap,
                "tool_backend": self.tool_backend,
                "peer_comparison": {
                    "summary": peer_comparison_text,
                    "metrics": financial_metrics,
                    "scenarios": scenario_analyses,
                    "comparable_companies": list(comparable_metrics.keys()),
                    "non_comparable_companies": skipped,
                },
            }
            update.update(self._record("quant", quant_status_label, " ".join(detail_parts), state, timer.metrics()))
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

    # ═══════════════════════════════════════════════════════════════
    # CLAIM BINDER — structural Claim → Evidence verification
    # ═══════════════════════════════════════════════════════════════
    def claim_binder(self, state: FinanceState) -> FinanceState:
        with self._track_step("claim_binder") as timer:
            claims = build_claims(state)
            verified = filter_verified(claims)
            summary = binding_summary(claims)
            detail = (
                f"Built {summary['total_claims']} claims; verified={summary['verified_claims']}; "
                f"rejected={summary['rejected_claims']}; page_anchored={summary['page_anchored_verified']}; "
                f"bind_rate={summary['bind_rate']}."
            )
            update: FinanceState = {
                "claims": [claim_to_dict(c) for c in claims],
                "verified_claims": [claim_to_dict(c) for c in verified],
                "claim_binding": summary,
            }
            update.update(self._record("claim_binder", "ok", detail, state, timer.metrics()))
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
                "Upload a filing PDF with extractable metrics or retry the configured live fundamentals provider."
            )
            S("# Incomplete Diligence Output (Fail-Loud Data Gap)")
            S("")
            S(f"**Companies:** {companies}")
            S("")
            S("## 1. Executive Summary")
            S("")
            S(detail)
            S("")
            S(
                "**Evidence Boundary:** This run produced no AST-verifiable revenue/EBITDA/R&D inputs. "
                "Market snapshots and LLM general knowledge alone are not treated as structured fundamentals. "
                "No ratios, SWOT, or investment positioning were invented."
            )
            S("")
            S("## 3. Financial Performance Analysis")
            S("")
            S(
                "Not available — fail-closed. Structured fundamentals were missing or non-computable "
                f"for: {companies}."
            )
            S("")
            S("## 4. Risk")
            S("")
            S(
                "**Data limitation risk (high):** Without extractable FY metrics, quantitative risk scoring "
                "and peer margin comparison are withheld rather than estimated."
            )
            S("")
            S("## 6. Compliance Review & Data Integrity")
            S("")
            S(
                "Fail-closed compliance path: the synthesizer refused to fabricate checkable metrics. "
                f"Gate expectation: `structured_source=none` for {companies}."
            )
            provider_summary = state.get("provider_error_summary") or {}
            if provider_summary.get("count"):
                S("")
                S(
                    "**Provider error summary:** "
                    f"transient={provider_summary.get('transient_count', 0)}, "
                    f"truly_missing/unavailable={provider_summary.get('missing_count', 0)}, "
                    f"other={provider_summary.get('other_count', 0)}, "
                    f"by_class={provider_summary.get('by_class', {})}."
                )
            S("")
            S("## Appendix B. Methodology, Data Sources & Disclaimer")
            S("")
            if self.data_mode == "demo":
                S(
                    "**Action Required:** Upload source filings (PDF) with extractable FY metrics, or query a "
                    "company covered by the demo sample database."
                )
            else:
                S(
                    "**Action Required:** Upload source filings (PDF) with extractable FY metrics, or retry the "
                    "configured live fundamentals provider. To use local demo coverage, switch DATA_MODE=demo explicitly."
                )
            S("")
            for line in format_next_actions({**state, "workflow_status": "incomplete_data", "fatal_data_gap": True}):
                S(line)
            for line in format_rag_citation_section(state.get("rag_evidence")):
                S(line)
            if self.data_mode == "demo":
                S(
                    "**Disclaimer:** DEMO MODE — incomplete output. This is research/demo only and does not "
                    "constitute investment advice."
                )
            else:
                S(
                    "**Disclaimer:** This incomplete report is generated by an AI-powered multi-agent system for "
                    "research purposes only. It does not constitute investment advice, a solicitation, or a "
                    "recommendation to buy or sell any security."
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

        # Claim → Evidence: synthesizer may only assert from verified claims.
        from .claims import claims_from_state

        if not state.get("verified_claims") and not state.get("claims"):
            built = build_claims(state)
            state = {
                **state,
                "claims": [claim_to_dict(c) for c in built],
                "verified_claims": [claim_to_dict(c) for c in filter_verified(built)],
                "claim_binding": binding_summary(built),
            }
        all_claims = claims_from_state(state)
        verified_claims = filter_verified(all_claims)

        def cite_for(company: str, *, claim_type: str | None = None, metric_name: str | None = None) -> str:
            hits = verified_by_entity(
                verified_claims,
                company,
                claim_type=claim_type,  # type: ignore[arg-type]
                metric_name=metric_name,
            )
            if not hits:
                return ""
            return hits[0].primary_citation

        def build_grounded_summary() -> str:
            return build_analyst_executive_summary(
                state,
                verified_claims,
                brief=effective_report_output_format(state) != "research_report",
            )

        llm_summary = build_grounded_summary()

        output_format = effective_report_output_format(state)
        is_full = output_format == "research_report"
        is_table = output_format == "table_summary"
        # Brief/table: keep source, summary/ledger (except pure table), metrics, gaps, compliance, disclaimer.
        include_narrative_sections = is_full
        include_summary_and_ledger = not is_table
        ledger_claims = filter_claims_for_brief(verified_claims) if not is_full else verified_claims

        # ── Report Construction (analyst-first; audit details in appendices) ──
        S("# LumenFin Diligence Report")
        S("")
        if is_full:
            S(
                "**Report Type:** Diligence Screening Report (AI-assisted) | "
                "**Classification:** For internal research reference only"
            )
        elif is_table:
            S("**Report Type:** Table Summary | **Classification:** AI-Generated, For Reference Only")
            S("")
            S(f"**Report Mode:** `{output_format}` (explicit UI/API selection; keywords do not auto-trim).")
        else:
            S("**Report Type:** Brief Diligence | **Classification:** AI-Generated, For Reference Only")
            S("")
            S(f"**Report Mode:** `{output_format}` (explicit UI/API selection; keywords do not auto-trim).")
        S("")
        if include_summary_and_ledger:
            S("## 1. Executive Summary")
            S("")
            S(llm_summary)
            S("")
            S(
                "**Evidence Boundary:** Material numeric and investment assertions are limited to "
                "verified claims with bound sources. Incomplete or unbound inputs are treated as "
                "data limitations (including Computed/unverified ratios). Risk-model scores remain "
                "screening indicators even when bound to risk-model evidence."
            )
            S("")
        if state.get("partial_data_gap"):
            coverage = state.get("coverage_matrix") or {}
            comparable = [
                company for company in state.get("companies") or [] if (coverage.get(company) or {}).get("comparable")
            ]
            skipped = state.get("non_comparable_companies") or non_comparable_companies(
                list(state.get("companies") or []),
                coverage,
            )
            S("**Partial Peer Coverage Notice:**")
            S(
                f"- Comparable ratio set: {', '.join(comparable) if comparable else '(none)'}"
            )
            S(
                f"- Non-comparable peers: {', '.join(skipped) if skipped else '(none)'} "
                "(missing extractable revenue/EBITDA/R&D inputs)."
            )
            S("")
        for line in format_next_actions(state):
            S(line)

        # Period + source alignment (merged)
        S("## 2. Period & Source Alignment")
        S("")
        for line in format_period_alignment_notice(state):
            # Downgrade nested "## Period Alignment" to a plain label inside §2.
            if line.startswith("## Period Alignment"):
                S("### Period Alignment")
                continue
            S(line)
        source_resolution = state.get("source_resolution") or {}
        company_resolutions = source_resolution.get("companies") or {}
        fallback_rows = [
            (company, info)
            for company, info in company_resolutions.items()
            if info.get("live_fallback_used")
        ]
        if source_resolution.get("prefer_uploaded_only") or fallback_rows or state.get("document_contexts"):
            mode = str(source_resolution.get("mode") or "hybrid")
            S("### Source Resolution")
            S("")
            if source_resolution.get("prefer_uploaded_only"):
                S(
                    "**Mode: uploaded materials only.** Structured fundamentals were not backfilled "
                    "from SEC/Yahoo/sample even if the upload lacked computable metrics."
                )
            elif fallback_rows:
                S(
                    "**Mode: hybrid.** Uploads are preferred; when they lack extractable "
                    "revenue/EBITDA/R&D, live providers may fill the gap — listed below."
                )
            else:
                S(
                    f"**Mode: {mode}.** Per-company structured source is listed so document narrative "
                    "is not confused with SEC/Yahoo numbers."
                )
            S("")
            S("| Company | Fundamentals source | Upload had metrics? | Notes |")
            S("|---------|---------------------|---------------------|-------|")
            for company in state.get("companies") or []:
                info = company_resolutions.get(company) or {}
                source = str(
                    info.get("structured_source")
                    or (state.get("retrieved_docs") or {}).get(company, {}).get("structured_source")
                    or "none"
                )
                had_metrics = (
                    "yes"
                    if info.get("upload_had_computable_metrics")
                    else ("n/a" if not state.get("document_contexts") else "no")
                )
                note = str(info.get("fallback_reason") or "")
                if not note and source == "document_extracted":
                    note = "Numbers taken from uploaded materials."
                elif not note and info.get("live_fallback_used"):
                    note = f"Upload lacked metrics; used {source}."
                S(f"| {company} | {source} | {had_metrics} | {note or '—'} |")
            S("")

        # Profiles are LLM narrative noise in production; keep only in demo for orientation.
        if include_narrative_sections and has_uploaded_docs and self.data_mode == "demo":
            shown_profile = False
            for company in state["companies"]:
                profile = ensure_sentence_complete(
                    state.get("company_profiles", {}).get(company, "")
                )
                if not profile or "not available" in profile.lower() or "pending" in profile.lower():
                    continue
                if len(profile) > 420:
                    profile = profile[:419].rstrip() + "…"
                if not shown_profile:
                    S("## Company Profiles *(LLM profile — unverified)*")
                    S("")
                    shown_profile = True
                S(f"### {company}")
                S(profile)
                S("")

        S("## 3. Financial Performance Analysis")
        S("")
        S("*Ratios below are computed from structured fundamentals. Prefer verified rows for decision use.*")
        S("")
        for line in format_peer_metric_matrix(state):
            S(line)
        for company in state["companies"]:
            metrics = state.get("financial_metrics", {}).get(company, {})

            S(f"### {company}")
            S("")

            if metrics:
                S("**Key Financial Indicators**")
                S("")
                S(
                    "*Internal screen thresholds are LumenFin heuristics (not industry peer medians). "
                    "P/E is TTM live and may not match the FY window of the statement ratios.*"
                )
                S("")
                S("| Metric | Value | Internal screen | Vs screen | Status | Source |")
                S("|--------|-------|-----------------|-----------|--------|--------|")

                metric_conf = state.get("metric_confidence", {}).get(company, {})

                def assess_metric(metric_key: str, value: float) -> tuple[str, str]:
                    if metric_key == "ebitda_margin":
                        if value >= 0.25:
                            return "Above", "internal screen >25%"
                        if value >= 0.15:
                            return "Near", "internal screen >25%"
                        return "Below", "internal screen >25%"
                    if metric_key == "operating_margin":
                        if value >= 0.20:
                            return "Above", "internal screen >20%"
                        if value >= 0.12:
                            return "Near", "internal screen >20%"
                        return "Below", "internal screen >20%"
                    if metric_key == "estimated_net_margin":
                        if value >= 0.15:
                            return "Above", "internal screen >15%"
                        if value >= 0.08:
                            return "Near", "internal screen >15%"
                        return "Below", "internal screen >15%"
                    if metric_key == "estimated_fcf_margin":
                        if value >= 0.10:
                            return "Above", "internal screen >10%"
                        if value >= 0.05:
                            return "Near", "internal screen >10%"
                        return "Below", "internal screen >10%"
                    if metric_key == "r_and_d_intensity":
                        if 0.05 <= value <= 0.15:
                            return "In range", "internal screen 5-15%"
                        if 0.03 <= value < 0.05 or 0.15 < value <= 0.20:
                            return "Near", "internal screen 5-15%"
                        return "Outside", "internal screen 5-15%"
                    return "—", "—"

                def _status_label(basis: str) -> str:
                    b = (basis or "").lower()
                    if "verified" in b or b == "ast":
                        return "Verified"
                    if "live" in b:
                        return "Live market"
                    if "unverified" in b or "computed" in b:
                        return "Computed (unverified)"
                    return basis or "—"

                def add_row(metric_key, label, screen, value=None):
                    v = value if value is not None else metrics.get(metric_key)
                    if v is None:
                        return
                    verified_hit = verified_by_entity(
                        verified_claims, company, metric_name=metric_key
                    )
                    allow_computed = metric_key in (
                        "ebitda_margin",
                        "operating_margin",
                        "r_and_d_intensity",
                    )
                    if metric_key in ("ebitda_margin", "operating_margin", "r_and_d_intensity", "pe_ratio"):
                        if not verified_hit and not (allow_computed and isinstance(v, (int, float))):
                            return
                    conf = metric_conf.get(metric_key, {})
                    if verified_hit:
                        raw_basis = str(conf.get("basis", "Verified"))
                        citation = humanize_citation(verified_hit[0].primary_citation)
                    else:
                        raw_basis = "Computed (unverified)"
                        citation = "structured fundamentals (claim not bound)"
                    status = _status_label(raw_basis)
                    if metric_key in (
                        "ebitda_margin",
                        "r_and_d_intensity",
                        "operating_margin",
                        "estimated_net_margin",
                        "estimated_fcf_margin",
                    ):
                        if metric_key.startswith("estimated_") and not verified_hit:
                            return
                        vs_screen, _ = assess_metric(metric_key, float(v))
                        S(
                            f"| {label} | {v:.2%} | {screen} | {vs_screen} | {status} | {citation} |"
                        )
                    elif metric_key == "pe_ratio":
                        if not verified_hit:
                            return
                        S(
                            f"| {label} | {v:.2f}x | {screen} | — | {status} | {citation} |"
                        )

                add_row("ebitda_margin", "EBITDA Margin", ">25%")
                add_row("operating_margin", "Operating Margin", ">20%")
                add_row("r_and_d_intensity", "R&D Intensity", "5-15%")
                add_row("pe_ratio", "P/E (TTM, live)", "—")
                # Absolute fundamentals (once) for analyst context
                market = ((state.get("retrieved_docs") or {}).get(company) or {}).get("market_data") or {}
                abs_bits = []
                for key, label in (
                    ("revenue", "Revenue"),
                    ("operating_income", "Operating income"),
                    ("r_and_d", "R&D"),
                ):
                    hits = verified_by_entity(verified_claims, company, metric_name=key)
                    if hits:
                        abs_bits.append(hits[0].render_with_citation(humanize=True))
                        continue
                    raw_abs = get_fundamental(market, key)
                    if isinstance(raw_abs, (int, float)):
                        abs_bits.append(
                            f"{company} {label} is {float(raw_abs):.2f} billion USD "
                            f"(structured fundamentals; claim not bound)."
                        )
                if abs_bits:
                    S("")
                    S("**Key absolute figures**")
                    for bit in abs_bits[:3]:
                        S(f"- {bit}")
                S("")
            else:
                S(
                    "*[Partial Coverage] Insufficient structured data for ratio comparison. "
                    "Market-only or risk-screening context may still appear below.*"
                )
                if company in (state.get("non_comparable_companies") or []):
                    source = state.get("retrieved_docs", {}).get(company, {}).get("structured_source", "none")
                    S(f"*Structured source for {company}: {source}. Peer margin comparison skipped.*")
                S("")

        # ── Risk (dedicated section; screening scores labeled honestly) ──
        swot: dict[str, dict[str, str]] = {}
        investment_thesis: dict[str, dict[str, str]] = {}
        any_risk = False
        for company in state["companies"]:
            if verified_by_entity(verified_claims, company, claim_type="risk_conclusion") or state.get(
                "risk_scores", {}
            ).get(company):
                any_risk = True
                break
        if any_risk:
            S("## 4. Risk")
            S("")
            S(
                "*Screening scores (model-derived; not a 10-K Item 1A extract). "
                "Use as diligence flags, not as independently audited risk conclusions.*"
            )
            S("")
            for company in state["companies"]:
                risk_claims = verified_by_entity(verified_claims, company, claim_type="risk_conclusion")
                risk_data = state.get("risk_scores", {}).get(company, {}) or {}
                if not risk_claims and not risk_data:
                    continue
                S(f"### {company} — Risk Screening Matrix")
                S("")
                S("| Dimension | Screening score (1-10) | Level | Source |")
                S("|-----------|------------------------|-------|--------|")
                dim_labels = {
                    "financial_risk": "Financial",
                    "operational_risk": "Operational",
                    "market_risk": "Market",
                    "regulatory_risk": "Regulatory",
                    "supply_chain_risk": "Supply Chain",
                }
                unknown_supply = False
                for dim, label in dim_labels.items():
                    hits = verified_by_entity(verified_claims, company, metric_name=dim)
                    if dim == "supply_chain_risk" and not hits:
                        hits = [c for c in risk_claims if c.metric_name == "supply_chain_risk"]
                    if not hits:
                        continue
                    claim = hits[0]
                    if dim == "supply_chain_risk" and str(claim.value).lower() in {
                        "unknown",
                        "n/a",
                        "none",
                    }:
                        unknown_supply = True
                        continue
                    score = risk_data.get(dim, claim.value if isinstance(claim.value, (int, float)) else 5.0)
                    if not isinstance(score, (int, float)):
                        score = 5.0
                    level = "Low" if score < 3.5 else ("Moderate" if score < 6.5 else "Elevated")
                    S(
                        f"| {label} | {score:.1f} | {level} | "
                        f"{humanize_citation(claim.primary_citation)} |"
                    )
                S("")
                if unknown_supply:
                    S(
                        "*Supply-chain screen: no clear filing signal in this run "
                        "(not shown as a Moderate/Elevated score).*"
                    )
                    S("")
                material_risk = [
                    c
                    for c in risk_claims
                    if not (
                        c.metric_name == "supply_chain_risk"
                        and str(c.value).lower() in {"unknown", "n/a", "none"}
                    )
                ]
                if material_risk:
                    S("**Screening conclusions**")
                    S("")
                    for claim in material_risk:
                        if not is_full and is_low_signal_claim(claim):
                            continue
                        S(f"- {claim.render_with_citation(humanize=True)}")
                    S("")

        # ── Research thesis (verified investment claims only) ──
        if include_narrative_sections:
            S("## 5. Research Thesis & Positioning")
            S("")
            S(
                "*Not a buy/sell recommendation. Emitted only from verified investment conclusions "
                "backed by verified numeric + risk evidence.*"
            )
            S("")
            for company in state["companies"]:
                inv = verified_by_entity(verified_claims, company, claim_type="investment_conclusion")
                S(f"### {company}")
                if inv:
                    bull = inv[0].render_with_citation(humanize=True)
                    risk_lines = [
                        c
                        for c in verified_by_entity(
                            verified_claims, company, claim_type="risk_conclusion"
                        )
                        if not (
                            c.metric_name == "supply_chain_risk"
                            and str(c.value).lower() in {"unknown", "n/a", "none"}
                        )
                    ]
                    # Prefer scored dimensions for the bear line over supply-chain noise.
                    scored = [
                        c
                        for c in risk_lines
                        if c.metric_name in {"financial_risk", "operational_risk", "market_risk"}
                    ]
                    bear_claim = scored[0] if scored else (risk_lines[0] if risk_lines else None)
                    bear = (
                        bear_claim.render_with_citation(humanize=True)
                        if bear_claim
                        else "See Risk screening section; no separate unverified bear narrative is invented."
                    )
                    investment_thesis[company] = {"bull_case": bull, "bear_case": bear}
                    S(f"- **Bull case (screening):** {bull}")
                    S(f"- **Bear / risk case (screening):** {bear}")
                else:
                    rejected = [
                        c
                        for c in all_claims
                        if c.entity == company
                        and c.claim_type == "investment_conclusion"
                        and c.verification == "rejected"
                    ]
                    msg = rejected[0].statement if rejected else (
                        f"{company}: investment conclusion withheld — missing verified claims."
                    )
                    investment_thesis[company] = {"bull_case": msg, "bear_case": msg}
                    S(f"- {msg}")
                S("")

        # ── Compliance ──
        S("## 6. Compliance Review & Data Integrity")
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
            S(
                "Core compliance checks passed. Material assertions are limited to verified claims "
                "with bound evidence."
            )
        S("")

        # ── Appendices ──
        if include_summary_and_ledger:
            for line in format_verified_claims_ledger(ledger_claims):
                S(line)
            binding = state.get("claim_binding") or binding_summary(all_claims)
            S(
                f"*Binding stats: verified={binding.get('verified_claims', 0)}/"
                f"{binding.get('total_claims', 0)} "
                f"(bind_rate={binding.get('bind_rate', 0)}, "
                f"page_anchored={binding.get('page_anchored_verified', 0)}).*"
            )
            S("")

        S("## Appendix B. Methodology, Data Sources & Disclaimer")
        S("")
        S(
            "**Methods:** Deterministic ratio engine on structured fundamentals; claim→evidence binder; "
            "optional LLM screening for sentiment/profile; multi-factor risk screening scores."
        )
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

        resolution = state.get("source_resolution") or {}
        if resolution.get("prefer_uploaded_only"):
            source_parts.append(
                "Source policy: prefer_uploaded_only=true (SEC/Yahoo/sample backfill disabled for this run)."
            )
        else:
            fallback_companies = [
                name
                for name, info in (resolution.get("companies") or {}).items()
                if info.get("live_fallback_used")
            ]
            if fallback_companies:
                source_parts.append(
                    "Live/sample backfill after sparse upload for: "
                    + ", ".join(fallback_companies)
                    + " (see Period & Source Alignment)."
                )

        source_parts.append("Narrative analysis: generated by the configured LLM using retrieved evidence and computed metrics.")
        S(f"**Data Sources:** {' '.join(source_parts)}")
        S("")
        for line in format_rag_citation_section(rag_evidence):
            S(line)
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
        S(
            "**Source Attribution:** Quant tables use deterministic calculations on structured inputs. "
            "Market rows use live snapshots when available. Company profiles and thesis language are "
            "LLM-assisted unless a row cites bound evidence. Risk matrix values are model-derived "
            "screening indicators and should not be treated as independently audited facts."
        )
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
            f"Report assembled from verified claims only "
            f"(mode={output_format}; verified={len(verified_claims)}/{len(all_claims)}; "
            f"bind_rate={(state.get('claim_binding') or {}).get('bind_rate', 0)})."
        )
        update.update(self._record("synthesizer", "ok", synth_detail, state, timer.metrics()))
        self.session_memory.save({**state, **update})
        return update
