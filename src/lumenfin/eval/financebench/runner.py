"""FinanceBench retrieval evaluation runner (offline by default)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lumenfin.documents import parse_pdf_document
from lumenfin.eval.financebench.constants import (
    CHUNK_K_VALUES,
    DEFAULT_BM25_RRF_WEIGHT,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    EVAL_COLLECTION,
    EVAL_COMPANY_TAG,
    EVAL_SESSION_ID,
    PAGE_K_VALUES,
    REMOTE_MODES,
    RETRIEVAL_MODES,
    SCHEMA_VERSION,
    SPLIT_SALT,
)
from lumenfin.eval.financebench.fetch_pdfs import fetch_pdfs
from lumenfin.eval.financebench.loader import (
    load_financebench_dataset,
    resolve_financebench_source,
    resolve_pdf_path,
)
from lumenfin.eval.financebench.prepare import prepare_artifacts, write_chunk_qrels
from lumenfin.eval.financebench.qrels import map_chunks_to_qrels
from lumenfin.eval.financebench.reporting import (
    aggregate_case_metrics,
    breakdowns,
    compare_mode_dirs,
    completed_case_ids,
    environment_payload,
    percentile,
    sha256_file,
    write_json,
    write_jsonl,
    write_markdown_report,
)
from lumenfin.eval.financebench.retrieval import (
    RemoteEvalBlocked,
    build_eval_store,
    iter_indexed_chunks,
    retrieve_for_mode,
)
from lumenfin.eval.financebench.scoring import score_retrieval_case
from lumenfin.eval.financebench.split import assign_splits, forbid_test_split_tuning, questions_for_split

log = logging.getLogger("lumenfin.eval.financebench.runner")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def eval_document_from_pdf(pdf_path: Path, *, doc_name: str, company: str) -> dict[str, Any]:
    """Parse a PDF and tag it for the shared eval corpus (eval-only metadata)."""
    document = parse_pdf_document(pdf_path)
    document["document_id"] = doc_name
    document["source_document_id"] = doc_name
    document["filename"] = f"{doc_name}.pdf"
    # Shared tag so bm25/dense/hybrid all search the same 84-doc collection
    # through production retriever APIs. Production isolation is unchanged.
    document["issuer_companies"] = [EVAL_COMPANY_TAG]
    document["detected_companies"] = [EVAL_COMPANY_TAG, company] if company else [EVAL_COMPANY_TAG]
    document["primary_company"] = EVAL_COMPANY_TAG
    return document


def _load_index_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"indexed_docs": [], "skipped_missing_pdf": [], "skipped_empty_parse": []}
    return json.loads(path.read_text(encoding="utf-8"))


def index_corpus(
    store: Any,
    questions: Sequence[Any],
    documents: dict[str, Any],
    pdf_dir: Path,
    *,
    manifest_path: Path,
    limit_docs: int | None = None,
    selected_only: bool = False,
) -> dict[str, Any]:
    unique: list[str] = []
    seen: set[str] = set()
    for question in questions:
        if question.doc_name not in seen:
            seen.add(question.doc_name)
            unique.append(question.doc_name)
    if not selected_only:
        unique = sorted(documents)
    if limit_docs is not None:
        unique = unique[:limit_docs]
    manifest = _load_index_manifest(manifest_path)
    already = set(manifest.get("indexed_docs") or [])
    indexed = list(already)
    skipped_missing: list[str] = list(manifest.get("skipped_missing_pdf") or [])
    skipped_empty: list[str] = list(manifest.get("skipped_empty_parse") or [])
    started = time.perf_counter()
    for doc_name in unique:
        if doc_name in already:
            continue
        info = documents.get(doc_name)
        pdf = resolve_pdf_path(pdf_dir, doc_name, info)
        if pdf is None:
            skipped_missing.append(doc_name)
            continue
        company = ""
        for question in questions:
            if question.doc_name == doc_name:
                company = question.company
                break
        document = eval_document_from_pdf(pdf, doc_name=doc_name, company=company)
        stats = store.index_documents([document], session_id=EVAL_SESSION_ID)
        n_chunks = int(stats.get("chunks_indexed") or 0)
        if n_chunks <= 0:
            skipped_empty.append(doc_name)
        else:
            indexed.append(doc_name)
            already.add(doc_name)
            log.info("indexed %s chunks=%s", doc_name, n_chunks)
        write_json(
            manifest_path,
            {
                "indexed_docs": indexed,
                "skipped_missing_pdf": skipped_missing,
                "skipped_empty_parse": skipped_empty,
            },
        )
    return {
        "unique_docs": len(unique),
        "indexed_docs": len(indexed),
        "skipped_missing_pdf": skipped_missing,
        "skipped_empty_parse": skipped_empty,
        "indexing_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _write_mode_report(
    *,
    out_dir: Path,
    mode: str,
    split: str,
    cases: list[dict[str, Any]],
    env: dict[str, Any],
    index_meta: dict[str, Any],
) -> dict[str, Any]:
    summary = aggregate_case_metrics(cases)
    latencies = [
        float((item.get("retrieval") or {}).get("latency_ms") or 0.0) for item in cases
    ]
    failures: dict[str, int] = {}
    rerank_fallback = 0
    rerank_calls = 0
    for item in cases:
        klass = str(item.get("failure_class") or "unknown")
        failures[klass] = failures.get(klass, 0) + 1
        retrieval = item.get("retrieval") or {}
        if retrieval.get("rerank_fallback"):
            rerank_fallback += 1
        if retrieval.get("rerank_provider") or mode == "hybrid-qwen3":
            rerank_calls += 1
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUN",
        "mode": mode,
        "split": split,
        "split_salt": SPLIT_SALT,
        "n_cases": len(cases),
        "k_values": {"page": list(PAGE_K_VALUES), "chunk": list(CHUNK_K_VALUES)},
        "summary": summary,
        "breakdowns": breakdowns(cases),
        "failure_classes": failures,
        "index": index_meta,
        "system": {
            "indexing_ms": index_meta.get("indexing_ms", "NOT_RUN"),
            "query_p50_ms": round(percentile(latencies, 0.5), 2) if latencies else "NOT_RUN",
            "query_p95_ms": round(percentile(latencies, 0.95), 2) if latencies else "NOT_RUN",
            "rerank_calls": rerank_calls,
            "rerank_fallback_rate": round(rerank_fallback / len(cases), 4) if cases else 0.0,
        },
        "environment": env,
        "notes": (
            "Held-out test metrics are frozen-config only. Do not tune on test. "
            "Existing 4/5/10 synthetic gates are not this eval. "
            "Phase 4 e2e answer eval is NOT_RUN."
        ),
    }
    write_json(out_dir / "metrics.json", report)
    write_markdown_report(out_dir / "report.md", report)
    write_json(out_dir / "environment.json", env)
    return report


def run_one_mode(
    *,
    mode: str,
    questions: Sequence[Any],
    store: Any,
    chunks: list[dict[str, Any]],
    out_dir: Path,
    allow_remote: bool,
    embedding_provider: str,
    resume: bool,
    top_k: int,
    split: str,
    env: dict[str, Any],
    index_meta: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "per_case.jsonl"
    done = completed_case_ids(results_path, mode=mode) if resume else set()
    cases: list[dict[str, Any]] = []
    if resume and results_path.is_file():
        from lumenfin.eval.financebench.reporting import read_jsonl

        cases.extend(item for item in read_jsonl(results_path) if item.get("mode") == mode)
    pending = [question for question in questions if question.case_id not in done]
    for question in pending:
        qrels = map_chunks_to_qrels(question, chunks)
        hits, meta = retrieve_for_mode(
            mode=mode,
            store=store,
            query=question.question,
            top_k=top_k,
            allow_remote=allow_remote,
            embedding_provider=embedding_provider,
        )
        row = score_retrieval_case(
            question=question,
            qrels=qrels,
            hits=hits,
            mode=mode,
            retrieval_meta=meta,
            top_k=top_k,
        )
        cases.append(row)
        write_jsonl(results_path, [row], append=True)
        log.info(
            "eval %s %s page_hit@5=%s first_page=%s fallback=%s",
            mode,
            question.case_id,
            row["page"]["hit_at"]["5"],
            row["page"]["first_relevant_rank"],
            bool((row.get("retrieval") or {}).get("rerank_fallback")),
        )
    return _write_mode_report(
        out_dir=out_dir,
        mode=mode,
        split=split,
        cases=cases,
        env={**env, "mode": mode},
        index_meta=index_meta,
    )


def run_eval(
    *,
    dataset_root: Path | None,
    out_dir: Path,
    modes: Sequence[str],
    split: str,
    allow_remote: bool,
    limit: int | None,
    resume: bool,
    tune: bool,
    persist_dir: Path | None,
    fetch: bool,
    index_limit_docs: int | None,
    index_scope: str,
    embedding_provider: str | None,
    expected_questions: int | None = 150,
) -> dict[str, Any]:
    forbid_test_split_tuning(split, tuning=tune)
    for mode in modes:
        if mode in REMOTE_MODES and not allow_remote:
            raise RemoteEvalBlocked(
                f"mode {mode} needs DashScope rerank; pass --allow-remote (offline default)"
            )
    provider = (embedding_provider or os.getenv("MAS_EMBEDDING_PROVIDER", "deterministic")).strip()
    embedding_dimension = int(os.getenv("MAS_EMBEDDING_DIMENSION", "384"))
    if persist_dir is None:
        persist_dir = out_dir / "index"
    persist_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = resolve_financebench_source(dataset_root)
    questions, documents, paths = load_financebench_dataset(
        source,
        expected_questions=expected_questions,
        require_pdfs=False,
    )
    assignment = assign_splits(questions)
    selected = questions_for_split(questions, assignment, split)
    selected.sort(key=lambda item: item.case_id)
    if limit is not None:
        selected = selected[:limit]

    prepared_dir = out_dir / "prepared"
    prepare_artifacts(
        dataset_root=source,
        out_dir=prepared_dir,
        expected_questions=expected_questions,
        require_pdfs=False,
    )
    if fetch:
        fetch_pdfs(dataset_root=source, pdf_dir=paths.pdf_dir)

    uri = str(persist_dir / "financebench.db")
    # Indexing uses the strictest requested mode so dashscope/qwen3 flags apply once.
    gate_mode = "hybrid-qwen3" if "hybrid-qwen3" in modes else modes[0]
    store = build_eval_store(
        uri=uri,
        embedding_provider=provider,
        embedding_dimension=embedding_dimension,
        collection_name=EVAL_COLLECTION,
        allow_remote=allow_remote,
        mode=gate_mode,
    )
    index_questions = selected if index_scope == "selected" else questions
    index_meta = index_corpus(
        store,
        index_questions,
        documents,
        paths.pdf_dir,
        manifest_path=persist_dir / "index_manifest.json",
        limit_docs=index_limit_docs,
        selected_only=index_scope == "selected",
    )
    chunks = iter_indexed_chunks(store, session_id=EVAL_SESSION_ID)
    write_chunk_qrels(prepared_dir, selected, chunks)

    split_path = prepared_dir / "split_manifest.json"
    dataset_hash = sha256_file(paths.questions_path)
    split_hash = sha256_file(split_path) if split_path.is_file() else ""
    env = environment_payload(
        repo_root=_repo_root(),
        dataset_hash=dataset_hash,
        split_manifest_hash=split_hash,
        embedding_provider=provider,
        embedding_model=os.getenv("DASHSCOPE_EMBEDDING_MODEL", ""),
        rerank_provider="qwen3" if "hybrid-qwen3" in modes else "none",
        rerank_model=os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank"),
        chunk_size=DEFAULT_CHUNK_CHARS,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        collection_name=EVAL_COLLECTION,
        bm25_rrf_weight=DEFAULT_BM25_RRF_WEIGHT,
        top_k=max(DEFAULT_TOP_K, max(CHUNK_K_VALUES)),
        mode=",".join(modes),
        split=split,
        remote_calls_enabled=allow_remote,
        extra={
            "eval_company_tag": EVAL_COMPANY_TAG,
            "eval_session_id": EVAL_SESSION_ID,
            "index_scope": index_scope,
            "n_indexed_chunks": len(chunks),
        },
    )
    write_json(out_dir / "environment.json", env)

    reports: dict[str, Any] = {}
    top_k = max(DEFAULT_TOP_K, max(CHUNK_K_VALUES))
    mode_dirs: list[Path] = []
    for mode in modes:
        mode_dir = out_dir / mode if len(modes) > 1 else out_dir
        reports[mode] = run_one_mode(
            mode=mode,
            questions=selected,
            store=store,
            chunks=chunks,
            out_dir=mode_dir,
            allow_remote=allow_remote,
            embedding_provider=provider,
            resume=resume,
            top_k=top_k,
            split=split,
            env=env,
            index_meta=index_meta,
        )
        mode_dirs.append(mode_dir)
    comparison = None
    if len(mode_dirs) > 1:
        comparison = compare_mode_dirs(mode_dirs)
        write_json(out_dir / "compare_modes.json", comparison)
    try:
        store.close()
    except Exception:
        pass
    return {"modes": reports, "compare": comparison, "index": index_meta}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinanceBench retrieval eval (offline default)")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/financebench_eval"))
    parser.add_argument("--mode", choices=(*RETRIEVAL_MODES, "all"), default="bm25")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Dev-only; refused on --split test")
    parser.add_argument("--persist-dir", type=Path, default=None)
    parser.add_argument("--fetch-pdfs", action="store_true")
    parser.add_argument("--index-limit-docs", type=int, default=None)
    parser.add_argument(
        "--index-scope",
        choices=("corpus", "selected"),
        default="corpus",
        help="corpus indexes all PDFs; selected indexes only docs for the chosen questions (smoke)",
    )
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--expected-questions", type=int, default=150)
    parser.add_argument("--compare-dirs", nargs="*", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.compare_dirs:
        payload = compare_mode_dirs([Path(item) for item in args.compare_dirs])
        out = Path(args.out_dir) / "compare_modes.json"
        write_json(out, payload)
        print(json.dumps({"compare": str(out), "n_modes": len(payload.get("modes", []))}))
        return 0
    modes: tuple[str, ...] = RETRIEVAL_MODES if args.mode == "all" else (args.mode,)
    payload = run_eval(
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        modes=modes,
        split=args.split,
        allow_remote=bool(args.allow_remote),
        limit=args.limit,
        resume=bool(args.resume),
        tune=bool(args.tune),
        persist_dir=args.persist_dir,
        fetch=bool(args.fetch_pdfs),
        index_limit_docs=args.index_limit_docs,
        index_scope=args.index_scope,
        embedding_provider=args.embedding_provider,
        expected_questions=args.expected_questions or None,
    )
    printable = {
        "index": {
            "indexed_docs": payload["index"]["indexed_docs"],
            "skipped_missing_pdf": len(payload["index"]["skipped_missing_pdf"]),
        },
        "modes": {
            mode: {
                "n_cases": report.get("n_cases"),
                "page": (report.get("summary") or {}).get("page"),
                "rerank_fallback_rate": (report.get("system") or {}).get("rerank_fallback_rate"),
            }
            for mode, report in payload["modes"].items()
        },
    }
    print(json.dumps(printable))
    return 0
