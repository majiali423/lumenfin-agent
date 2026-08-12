from __future__ import annotations

import re
from typing import Any

from ..artifacts import RetrievalArtifact, RetrievalProvenance, score_retrieval_confidence
from ..documents import is_trusted_ast_amount, normalize_metric_hints_to_billion_usd
from ..fundamentals import is_plausible_revenue_billion_usd
from ..input_guardrail import sanitize_retrieval_hits
from ..market_data import summarize_market_snapshots
from ..metrics_schema import set_fundamental
from ..parallel import map_in_parallel
from ..rag.dedupe import dedupe_cross_company_rag_hits
from ..rag.telemetry import summarize_rag_telemetry
from ..reporting import requested_fiscal_year_from_state
from ..state import FinanceState
from ..tools import (
    build_coverage_matrix,
    has_computable_fundamentals,
    has_supply_chain_signal,
    is_partial_compare_gap,
    non_comparable_companies,
    retrieve_company_payload,
    summarize_document_context,
)


class RetrievalMixin:
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
            "bm25_hits": 0,
            "keyword_hits": 0,
            "lexical_fallback_hits": 0,
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
                    "period": meta.get("period"),
                    "period_type": meta.get("period_type") or meta.get("period_hint"),
                    "period_source": meta.get("period_source"),
                    "period_alignment": meta.get("period_alignment"),
                    "citation": meta.get("citation"),
                    "source_record_id": (
                        meta.get("source_record_id") or meta.get("provider_record_id")
                    ),
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
            from ..provider_retry import summarize_provider_errors

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
