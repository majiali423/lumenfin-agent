from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .constants import (
    DEFAULT_BM25_RRF_WEIGHT,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    INDEX_SCOPES,
    SCHEMA_VERSION,
)
from .frozen import (
    FrozenConfig,
    enforce_confirmation_lock,
    is_confirmation_split,
    provenance,
    resolved_embedding_model,
)
from .ingestion import (
    build_ingestion_report,
    cohort_case_ids,
    fallback_doc_names,
    load_fallback_documents,
    question_uses_zero_chunk,
)
from .metrics import hit_at_k, mean_reciprocal_rank, ndcg_at_k, recall_at_k
from .prepare import prepare_financebench_eval
from .qrels import gold_pages_for, map_chunks_to_qrels, retrieved_page_keys
from .reporting import (
    aggregate_case_metrics,
    breakdowns,
    compare_modes,
    completed_case_ids,
    environment_payload,
    percentile,
    read_jsonl,
    render_markdown,
    write_json,
    write_jsonl,
)
from .retrieval import (
    build_eval_store,
    resolve_embedding_dimension,
    resolve_modes,
    retrieve_for_mode,
)
from .schema import FinanceBenchQuestion
from .split import experiment_governance, forbid_test_split_tuning, questions_for_split
from .taxonomy import classify_case, classify_failure


PAGE_K = (1, 3, 5, 10)
CHUNK_K = (5, 10, 20)


def _rank_metrics(
    retrieved: list[object],
    relevant: set[object],
    *,
    k_values: tuple[int, ...],
    ndcg_k: tuple[int, ...],
) -> dict[str, Any]:
    first = next((index for index, item in enumerate(retrieved, start=1) if item in relevant), 0)
    return {
        "hit_at": {str(k): round(hit_at_k(retrieved, relevant, k=k), 4) for k in k_values},
        "recall_at": {str(k): round(recall_at_k(retrieved, relevant, k=k), 4) for k in k_values},
        "mrr": round(mean_reciprocal_rank(retrieved, relevant), 4),
        "ndcg_at": {str(k): round(ndcg_at_k(retrieved, relevant, k=k), 4) for k in ndcg_k},
        "first_relevant_rank": first,
        "gold_count": len(relevant),
        "retrieved_count": len(retrieved),
    }


def _safe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Machine-readable hits without document body or secrets."""
    safe: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        safe.append(
            {
                "rank": rank,
                "chunk_id": hit.get("chunk_id"),
                "document_id": hit.get("document_id"),
                "filename": hit.get("filename"),
                "page": hit.get("page"),
                "citation": hit.get("citation"),
                "score": hit.get("fusion_score", hit.get("rerank_score", hit.get("score"))),
                "retrieval_method": hit.get("retrieval_method"),
            }
        )
    return safe


def score_retrieval_case(
    question: FinanceBenchQuestion,
    *,
    chunks: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    mode: str,
    meta: dict[str, Any],
    top_k: int,
    ingestion_failure: bool = False,
) -> dict[str, Any]:
    qrels = map_chunks_to_qrels(question, chunks)
    page_retrieved = retrieved_page_keys(hits)
    gold_pages = {page.key for page in gold_pages_for(question)}
    chunk_retrieved = [str(hit.get("chunk_id") or "") for hit in hits]
    gold_chunks = set(qrels.gold_chunk_ids)
    labels = classify_case(question)
    failure = classify_failure(
        retrieved_pages=page_retrieved,
        gold_pages=gold_pages,
        top_k=top_k,
        empty=not hits,
        provider_error=str(meta.get("error_type") or ""),
        degraded=bool(meta.get("degraded")),
        ingestion_failure=ingestion_failure,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": question.case_id,
        "financebench_id": question.financebench_id,
        "mode": mode,
        "status": "ok" if not meta.get("error_type") else "error",
        "company": question.company,
        "doc_name": question.doc_name,
        "labels": labels,
        "single_gold_page": qrels.single_gold_page,
        "page_provenance_ok": qrels.page_provenance_ok,
        "qrel_notes": list(qrels.notes),
        "span_mapped_count": qrels.span_mapped_count,
        "span_unmapped_count": qrels.span_unmapped_count,
        "page": _rank_metrics(page_retrieved, gold_pages, k_values=PAGE_K, ndcg_k=(5, 10)),
        "chunk": _rank_metrics(chunk_retrieved, gold_chunks, k_values=CHUNK_K, ndcg_k=(10,)),
        "chunk_deprecated": True,
        "chunk_semantics": "union of page-derived and span-overlap chunk ids",
        "page_chunk": _rank_metrics(
            chunk_retrieved, set(qrels.page_chunk_ids), k_values=CHUNK_K, ndcg_k=(10,)
        ),
        "span_chunk": _rank_metrics(
            chunk_retrieved, set(qrels.span_chunk_ids), k_values=CHUNK_K, ndcg_k=(10,)
        ),
        "citations": [item["citation"] for item in _safe_hits(hits) if item.get("citation")],
        "hits": _safe_hits(hits),
        "failure_class": failure,
        "latency_ms": meta.get("latency_ms", 0.0),
        "degraded": bool(meta.get("degraded")),
        "rerank_fallback": bool(meta.get("rerank_fallback")),
        "rerank_tokens": int(meta.get("rerank_tokens") or 0),
        "retrieval_methods": list(meta.get("retrieval_methods") or []),
        "effective_mode": meta.get("mode", mode),
        "error_type": meta.get("error_type") or "",
        "ingestion_failure": ingestion_failure,
    }


def _company_documents(
    question: FinanceBenchQuestion,
    parsed_documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    names = {question.doc_name, *(span.evidence_doc_name for span in question.evidence)}
    return [parsed_documents[name] for name in names if name in parsed_documents]


def _all_chunks_for_question(
    question: FinanceBenchQuestion,
    chunks_by_doc: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    names = {question.doc_name, *(span.evidence_doc_name for span in question.evidence)}
    chunks: list[dict[str, Any]] = []
    for name in names:
        chunks.extend(chunks_by_doc.get(name) or [])
    return chunks


def _chunks_for_scoring(
    question: FinanceBenchQuestion,
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    index_scope: str,
) -> list[dict[str, Any]]:
    if index_scope == "corpus":
        chunks: list[dict[str, Any]] = []
        for items in chunks_by_doc.values():
            chunks.extend(items)
        return chunks
    return _all_chunks_for_question(question, chunks_by_doc)


def _write_mode_bundle(
    *,
    out: Path,
    prepared: dict[str, Any],
    repo_root: Path,
    split: str,
    mode: str,
    index_scope: str,
    embedding_provider: str,
    embedding_dimension: int,
    allow_remote: bool,
    top_k: int,
    bm25_rrf_weight: float,
    collection_name: str,
    resume: bool,
    limit: int | None,
    skipped_completed: int,
    all_rows: list[dict[str, Any]],
    latencies: list[float],
    indexing_ms: float,
    embed_calls: int,
    rerank_calls: int,
    rerank_fallbacks: int,
    degraded: int,
    provider_errors: int,
    rerank_tokens: int,
    ingestion_report: dict[str, Any] | None = None,
    cohorts: dict[str, Any] | None = None,
    frozen: FrozenConfig | None = None,
) -> dict[str, Any]:
    summary = aggregate_case_metrics(all_rows)
    failure_counts: dict[str, int] = {}
    for row in all_rows:
        key = str(row.get("failure_class") or "unknown")
        failure_counts[key] = failure_counts.get(key, 0) + 1
    governance = experiment_governance(split, index_scope)
    env = environment_payload(
        repo_root=repo_root,
        dataset_hash=str(prepared["manifest"]["dataset_hash"]),
        split_manifest_hash=str(prepared["manifest"]["split_manifest_hash"]),
        embedding_provider=embedding_provider,
        embedding_model=(
            frozen.embedding_model
            if frozen
            else resolved_embedding_model(embedding_provider)
        ),
        rerank_provider="qwen3" if mode == "hybrid-qwen3" else "none",
        rerank_model=(
            frozen.rerank_model if frozen and mode == "hybrid-qwen3" else ("qwen3-rerank" if mode == "hybrid-qwen3" else "")
        ),
        chunk_size=DEFAULT_CHUNK_CHARS,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        collection_name=collection_name,
        bm25_rrf_weight=bm25_rrf_weight,
        top_k=top_k,
        mode=mode,
        split=split,
        remote_calls_enabled=allow_remote,
        extra={
            "limit": limit,
            "resume": resume,
            "skipped_completed": skipped_completed,
            "index_scope": index_scope,
            "embedding_dimension": embedding_dimension,
            **governance,
            **(provenance(frozen) if frozen else {}),
        },
    )
    system = {
        "indexing_ms": indexing_ms,
        "query_p50_ms": round(percentile(latencies, 0.50), 2) if latencies else 0.0,
        "query_p95_ms": round(percentile(latencies, 0.95), 2) if latencies else 0.0,
        "embedding_calls": embed_calls,
        "rerank_calls": rerank_calls,
        "rerank_fallback_rate": round(rerank_fallbacks / max(1, rerank_calls), 4) if rerank_calls else 0.0,
        "degraded_retrieval_count": degraded,
        "provider_error_count": provider_errors,
        "token_usage": rerank_tokens,
        "success_count": summary.get("succeeded", 0),
        "failure_count": summary.get("failed", 0),
    }
    results = {
        "schema_version": SCHEMA_VERSION,
        "status": "recorded",
        "environment": env,
        "summary": summary,
        "breakdowns": breakdowns(all_rows),
        "failures": failure_counts,
        "system": system,
        "synthetic_gate_disclaimer": (
            "Existing 10-case Qwen3 numbers in docs/QWEN3_RERANK.md are synthetic hard-negative "
            "gates, not FinanceBench accuracy."
        ),
        "split_status": governance["split_status"],
        "experiment_role": governance["experiment_role"],
        "held_out_status": (
            "confirmation_unseen" if governance["held_out"] else "exposed_test"
        ),
        "ingestion": ingestion_report or {},
        "cohorts": cohorts or {},
        **(provenance(frozen) if frozen else {}),
    }
    write_json(out / "environment.json", env)
    write_json(out / "results.json", results)
    (out / "results.md").write_text(render_markdown(results), encoding="utf-8")
    write_json(out / "manifest.json", prepared["manifest"])
    return results


def run_retrieval_eval(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    split: str = "dev",
    mode: str = "hybrid",
    top_k: int = DEFAULT_TOP_K,
    embedding_provider: str = "deterministic",
    embedding_dimension: int = 384,
    allow_remote: bool = False,
    resume: bool = False,
    limit: int | None = None,
    tuning: bool = False,
    collection_name: str = "financebench_eval",
    bm25_rrf_weight: float = DEFAULT_BM25_RRF_WEIGHT,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    expected_questions: int | None = None,
    require_pdfs: bool = False,
    keep_index: bool = False,
    milvus_uri: str = "",
    index_scope: str = "company",
    frozen_config_path: str | Path | None = None,
    confirm_held_out: bool = False,
    require_clean_worktree: bool = True,
    verify_dataset_hash: bool = True,
) -> dict[str, Any]:
    if index_scope not in INDEX_SCOPES:
        raise ValueError(f"unsupported index scope {index_scope!r}")
    frozen = None
    if is_confirmation_split(split):
        frozen = enforce_confirmation_lock(
            split=split,
            mode=mode,
            index_scope=index_scope,
            embedding_provider=embedding_provider,
            embedding_dimension=embedding_dimension,
            top_k=top_k,
            bm25_rrf_weight=bm25_rrf_weight,
            rerank_candidates=rerank_candidates,
            chunk_size=DEFAULT_CHUNK_CHARS,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            query_rewriting=False,
            limit=limit,
            frozen_config_path=frozen_config_path,
            confirm_held_out=confirm_held_out,
            repo_root=repo_root,
            dataset_dir=dataset_dir,
            require_clean_worktree=require_clean_worktree,
            verify_dataset_hash=verify_dataset_hash,
        )
    forbid_test_split_tuning(split, tuning=tuning)
    modes = resolve_modes(mode)
    embedding_dimension = resolve_embedding_dimension(embedding_provider, embedding_dimension)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    prepared = prepare_financebench_eval(
        source_dir=dataset_dir,
        output_dir=out_root / "_prepared",
        expected_questions=expected_questions,
        require_pdfs=require_pdfs,
        parse_pdfs=True,
        max_chunk_chars=DEFAULT_CHUNK_CHARS,
        overlap_chars=DEFAULT_CHUNK_OVERLAP,
    )
    questions = questions_for_split(prepared["questions"], prepared["assignment"], split)
    if limit is not None:
        questions = questions[: max(0, int(limit))]

    index_root = Path(milvus_uri).parent if milvus_uri else out_root / f"index-{uuid4().hex[:8]}"
    index_root.mkdir(parents=True, exist_ok=True)
    uri = milvus_uri or str(index_root / "eval.db")
    store_mode = "hybrid-qwen3" if "hybrid-qwen3" in modes else modes[0]
    store = build_eval_store(
        uri=uri,
        embedding_provider=embedding_provider,
        embedding_dimension=embedding_dimension,
        collection_name=collection_name,
        allow_remote=allow_remote,
        mode=store_mode,
        embedding_model=frozen.embedding_model if frozen else "",
    )
    session_id = "financebench-eval"
    parsed_documents: dict[str, dict[str, Any]] = prepared["parsed_documents"]
    chunks_by_doc: dict[str, list[dict[str, Any]]] = prepared["chunks_by_doc"]
    fallback_records = load_fallback_documents(dataset_dir)
    fallback_names = fallback_doc_names(fallback_records)
    ingestion_report = build_ingestion_report(
        questions=list(prepared["questions"]),
        parsed_documents=parsed_documents,
        chunks_by_doc=chunks_by_doc,
        fallback_records=fallback_records,
        missing_pdfs=list((prepared.get("manifest") or {}).get("missing_pdfs") or []),
    )
    zero_chunk_names = set(ingestion_report.get("zero_chunk_documents") or [])
    prepared["manifest"]["ingestion"] = ingestion_report
    prepared["manifest"]["split_status"] = experiment_governance(split, index_scope)
    if frozen:
        prepared["manifest"].update(provenance(frozen))
    corpus_docs = list(parsed_documents.values())
    embed_calls = 0
    indexing_ms = 0.0
    per_mode_results: dict[str, dict[str, Any]] = {}
    try:
        index_started = time.perf_counter()
        if parsed_documents:
            print(
                f"indexing documents={len(parsed_documents)} "
                f"(DashScope embeddings; this can take a while)",
                flush=True,
            )
            for doc_index, document in enumerate(corpus_docs, start=1):
                stats = store.index_documents([document], session_id=session_id)
                embed_calls += int(stats.get("embed_calls") or 1)
                if doc_index == 1 or doc_index == len(corpus_docs) or doc_index % 5 == 0:
                    print(
                        f"indexed {doc_index}/{len(corpus_docs)} "
                        f"{document.get('document_id') or document.get('filename')} "
                        f"chunks={stats.get('chunks_indexed', 0)}",
                        flush=True,
                    )
        indexing_ms = round((time.perf_counter() - index_started) * 1000, 2)
        print(
            f"indexed documents={len(parsed_documents)} chunks={sum(len(v) for v in chunks_by_doc.values())} "
            f"ms={indexing_ms} scope={index_scope}",
            flush=True,
        )

        for current_mode in modes:
            mode_dir = out_root if len(modes) == 1 else out_root / current_mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            per_case_path = mode_dir / "per_case.jsonl"
            failures_path = mode_dir / "failures.jsonl"
            if not resume:
                for stale in (per_case_path, failures_path):
                    if stale.exists():
                        stale.unlink()
            already = completed_case_ids(per_case_path, mode=current_mode) if resume else set()
            latencies: list[float] = []
            rerank_calls = 0
            rerank_fallbacks = 0
            degraded = 0
            provider_errors = 0
            rerank_tokens = 0
            mode_embed_calls = embed_calls
            for index, question in enumerate(questions, start=1):
                if question.case_id in already:
                    continue
                document_contexts = (
                    corpus_docs
                    if index_scope == "corpus"
                    else _company_documents(question, parsed_documents)
                )
                hits, meta = retrieve_for_mode(
                    mode=current_mode,
                    store=store,
                    query=question.question,
                    company=question.company,
                    session_id=session_id,
                    document_contexts=document_contexts,
                    top_k=top_k,
                    rerank_candidates=rerank_candidates,
                    bm25_rrf_weight=bm25_rrf_weight,
                    allow_remote=allow_remote,
                    embedding_provider=embedding_provider,
                    index_scope=index_scope,
                    rerank_model=frozen.rerank_model if frozen else None,
                    rerank_instruct=frozen.rerank_instruct if frozen else None,
                )
                if current_mode in {"dense", "hybrid", "hybrid-qwen3"}:
                    mode_embed_calls += 1
                latencies.append(float(meta.get("latency_ms") or 0.0))
                rerank_calls += int(meta.get("rerank_calls") or 0)
                rerank_fallbacks += int(bool(meta.get("rerank_fallback")))
                degraded += int(bool(meta.get("degraded")))
                provider_errors += int(bool(meta.get("error_type")))
                rerank_tokens += int(meta.get("rerank_tokens") or 0)
                row = score_retrieval_case(
                    question,
                    chunks=_chunks_for_scoring(question, chunks_by_doc, index_scope),
                    hits=hits,
                    mode=current_mode,
                    meta=meta,
                    top_k=top_k,
                    ingestion_failure=question_uses_zero_chunk(question, zero_chunk_names),
                )
                write_jsonl(per_case_path, [row], append=True)
                if row["failure_class"] not in {"hit", "rank_gt_1", "degraded_hit"}:
                    write_jsonl(
                        failures_path,
                        [
                            {
                                "case_id": row["case_id"],
                                "mode": current_mode,
                                "failure_class": row["failure_class"],
                                "company": row["company"],
                                "doc_name": row["doc_name"],
                                "labels": row["labels"],
                                "error_type": row["error_type"],
                            }
                        ],
                        append=True,
                    )
                if index == 1 or index == len(questions) or index % 5 == 0:
                    print(
                        f"[{current_mode}] {index}/{len(questions)} {question.case_id} "
                        f"{row['failure_class']}",
                        flush=True,
                    )
            all_rows = [row for row in read_jsonl(per_case_path) if row.get("mode") == current_mode]
            if limit is not None:
                wanted = {item.case_id for item in questions}
                all_rows = [row for row in all_rows if row.get("case_id") in wanted]
            cohorts = {
                name: aggregate_case_metrics(
                    [row for row in all_rows if row.get("case_id") in ids]
                )
                for name, ids in (
                    (
                        "all",
                        cohort_case_ids(questions, fallback_names=fallback_names, cohort="all"),
                    ),
                    (
                        "real_pdf",
                        cohort_case_ids(questions, fallback_names=fallback_names, cohort="real_pdf"),
                    ),
                    (
                        "fallback",
                        cohort_case_ids(questions, fallback_names=fallback_names, cohort="fallback"),
                    ),
                )
            }
            per_mode_results[current_mode] = _write_mode_bundle(
                out=mode_dir,
                prepared=prepared,
                repo_root=Path(repo_root),
                split=split,
                mode=current_mode,
                index_scope=index_scope,
                embedding_provider=embedding_provider,
                embedding_dimension=embedding_dimension,
                allow_remote=allow_remote,
                top_k=top_k,
                bm25_rrf_weight=bm25_rrf_weight,
                collection_name=collection_name,
                resume=resume,
                limit=limit,
                skipped_completed=len(already),
                all_rows=all_rows,
                latencies=latencies,
                indexing_ms=indexing_ms,
                embed_calls=mode_embed_calls,
                rerank_calls=rerank_calls,
                rerank_fallbacks=rerank_fallbacks,
                degraded=degraded,
                provider_errors=provider_errors,
                rerank_tokens=rerank_tokens,
                ingestion_report=ingestion_report,
                cohorts=cohorts,
                frozen=frozen,
            )
    finally:
        store.close()
        if not keep_index and not milvus_uri:
            shutil.rmtree(index_root, ignore_errors=True)

    if len(modes) == 1:
        return per_mode_results[modes[0]]

    per_case_by_mode = {
        name: read_jsonl(out_root / name / "per_case.jsonl") for name in modes
    }
    governance = experiment_governance(split, index_scope)
    comparison = compare_modes(per_case_by_mode)
    write_json(out_root / "ablation.json", comparison)
    write_jsonl(out_root / "rank_movements.jsonl", comparison.get("movements") or [])
    combined = {
        "schema_version": SCHEMA_VERSION,
        "status": "recorded",
        "mode": "all",
        "index_scope": index_scope,
        "split": split,
        "split_status": governance["split_status"],
        "experiment_role": governance["experiment_role"],
        "held_out_status": (
            "confirmation_unseen" if governance["held_out"] else "exposed_test"
        ),
        "ingestion": ingestion_report,
        "modes": per_mode_results,
        "ablation": comparison,
        "synthetic_gate_disclaimer": (
            "Existing 10-case Qwen3 numbers in docs/QWEN3_RERANK.md are synthetic hard-negative "
            "gates, not FinanceBench accuracy."
        ),
        "product_accuracy_claim": False,
        **(provenance(frozen) if frozen else {}),
    }
    write_json(out_root / "results.json", combined)
    return combined
