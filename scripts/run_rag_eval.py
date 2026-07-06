#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.embeddings import DeterministicEmbeddingProvider
from lumenfin.rag.hybrid_retriever import HybridEvidenceRetriever
from lumenfin.rag.metrics import evaluate_retrieval_case, summarize_eval_results
from lumenfin.rag.milvus_store import MilvusRAGStore
from lumenfin.stdio import configure_stdio_utf8


def _relevant_terms(case: dict) -> list[str]:
    return case.get("relevant_terms") or case.get("must_include_any") or []


def main() -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Run hybrid RAG retrieval evaluation.")
    parser.add_argument("--cases", default=str(ROOT / "data" / "eval_rag" / "rag_cases.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    results = []

    tmp_dir = ROOT / "test_artifacts" / f"rag-eval-{uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    uri = str(tmp_dir / "rag_eval.db")
    store = MilvusRAGStore(uri, DeterministicEmbeddingProvider(), collection_name="rag_eval")
    retriever = HybridEvidenceRetriever(store, top_k=args.top_k)
    try:
        for case in cases:
            document = case["document"]
            store.index_documents([document], session_id=case["session_id"])
            hits = retriever.retrieve_for_company(
                query=case["query"],
                company=case["company"],
                session_id=case["session_id"],
                document_contexts=[document],
            )
            eval_result = evaluate_retrieval_case(
                case_id=case.get("id", case["session_id"]),
                company=case["company"],
                query=case["query"],
                document=document,
                retrieved=hits,
                relevant_terms=_relevant_terms(case),
                k_values=case.get("k_values"),
            )
            results.append(eval_result)

            status = "PASS" if eval_result.passed else "FAIL"
            recall_bits = " ".join(
                f"R@{k}={eval_result.recall_at_k.get(k, 0.0):.2f}"
                for k in sorted(eval_result.recall_at_k)
            )
            cite_bits = " ".join(
                f"CiteR@{k}={eval_result.citation_recall_at_k.get(k, 0.0):.2f}"
                for k in sorted(eval_result.citation_recall_at_k)
            )
            print(f"[{status}] {eval_result.company} :: {eval_result.query}")
            print(
                f"  {recall_bits} | MRR={eval_result.mrr:.2f} | "
                f"citation_cov={eval_result.citation_coverage:.0%} | {cite_bits} | "
                f"groundedness={eval_result.groundedness:.2f}"
            )
            if not eval_result.passed:
                print(f"  relevant_chunks={eval_result.relevant_chunk_count} top_citations={eval_result.top_citations}")
    finally:
        store.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    summary = summarize_eval_results(results)
    print("\nRAG eval summary")
    print(f"  cases: {summary['passed']}/{summary['cases']} passed ({summary['pass_rate']:.0%})")
    print(f"  mean MRR: {summary['mean_mrr']:.3f}")
    for key, value in sorted(summary.items()):
        if key.startswith("mean_recall_at_") or key.startswith("mean_citation_recall_at_"):
            print(f"  {key}: {value:.3f}")
    print(f"  mean citation coverage: {summary['mean_citation_coverage']:.0%}")
    print(f"  mean groundedness: {summary['mean_groundedness']:.3f}")

    if args.json_out:
        payload = {
            "summary": summary,
            "cases": [result.to_dict() for result in results],
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote metrics to {out_path}")

    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
