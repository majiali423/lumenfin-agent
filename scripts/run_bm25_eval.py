#!/usr/bin/env python3
"""Compare dense, native BM25, and dense+BM25 RRF on a fixed corpus."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from tempfile import gettempdir
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


def _evaluate(mode: str, case: dict, hits: list[dict]) -> object:
    return evaluate_retrieval_case(
        case_id=f"{case['id']}:{mode}",
        company=case["company"],
        query=case["query"],
        document=case["document"],
        retrieved=hits,
        relevant_terms=case["relevant_terms"],
        k_values=case.get("k_values"),
    )


def main() -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Evaluate native Milvus BM25 retrieval.")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "data" / "eval_rag" / "bm25_cases.json"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Require BM25 and hybrid to pass every case with full recall at 3.",
    )
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    root = Path(gettempdir()) / f"lumenfin-bm25-eval-{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    store = MilvusRAGStore(
        str(root / "eval.db"),
        DeterministicEmbeddingProvider(),
        collection_name="bm25_eval",
    )
    results: dict[str, list] = {"dense": [], "bm25": [], "hybrid": []}
    try:
        for case in cases:
            document = case["document"]
            session_id = case["session_id"]
            store.index_documents([document], session_id=session_id)
            dense_hits = store.vector_search(
                case["query"],
                session_id=session_id,
                companies=[case["company"]],
                top_k=args.top_k,
            )
            bm25_hits = store.bm25_search(
                case["query"],
                session_id=session_id,
                companies=[case["company"]],
                top_k=args.top_k,
            )
            hybrid = HybridEvidenceRetriever(store, top_k=args.top_k)
            hybrid_hits = hybrid.retrieve_for_company(
                query=case["query"],
                company=case["company"],
                session_id=session_id,
                document_contexts=[document],
            )
            for mode, hits in (
                ("dense", dense_hits),
                ("bm25", bm25_hits),
                ("hybrid", hybrid_hits),
            ):
                result = _evaluate(mode, case, hits)
                results[mode].append(result)
                print(
                    f"[{mode:6}] {case['id']}: pass={result.passed} "
                    f"R@3={result.recall_at_k.get(3, 0.0):.2f} MRR={result.mrr:.2f}"
                )
    finally:
        store.close()
        shutil.rmtree(root, ignore_errors=True)

    summaries = {mode: summarize_eval_results(items) for mode, items in results.items()}
    print("\nBM25 comparison summary")
    for mode in ("dense", "bm25", "hybrid"):
        summary = summaries[mode]
        print(
            f"  {mode:6}: pass={summary['passed']}/{summary['cases']} "
            f"R@3={summary.get('mean_recall_at_3', 0.0):.3f} "
            f"MRR={summary.get('mean_mrr', 0.0):.3f}"
        )

    passed = all(
        summaries[mode].get("pass_rate") == 1.0
        and summaries[mode].get("mean_recall_at_3") == 1.0
        for mode in ("bm25", "hybrid")
    )
    payload = {
        "passed": passed,
        "summaries": summaries,
        "cases": {
            mode: [item.to_dict() for item in items]
            for mode, items in results.items()
        },
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    if args.gate and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
