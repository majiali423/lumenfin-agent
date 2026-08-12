#!/usr/bin/env python3
"""Compare candidate order, lexical rerank, and optional qwen3 rerank."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.rerank import LexicalReranker, build_reranker
from lumenfin.stdio import configure_stdio_utf8


def _dcg(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def _score_case(case: dict[str, Any], ranked: list[dict[str, Any]], *, k: int) -> dict[str, Any]:
    grade_by_id = {str(item["id"]): int(item.get("relevance") or 0) for item in case["candidates"]}
    ranked_grades = [grade_by_id.get(str(hit.get("chunk_id") or ""), 0) for hit in ranked[:k]]
    ideal = sorted(grade_by_id.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    ndcg = _dcg(ranked_grades) / ideal_dcg if ideal_dcg else 1.0
    reciprocal_rank = 0.0
    for rank, grade in enumerate(ranked_grades, start=1):
        if grade >= 2:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "id": case["id"],
        "top_id": str(ranked[0].get("chunk_id") or "") if ranked else "",
        "top1_correct": bool(ranked_grades and ranked_grades[0] >= 2),
        "mrr": round(reciprocal_rank, 4),
        "ndcg_at_k": round(ndcg, 4),
    }


def _summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    if not count:
        return {"cases": 0, "top1_accuracy": 0.0, "mean_mrr": 0.0, "mean_ndcg_at_k": 0.0}
    return {
        "cases": count,
        "top1_accuracy": round(sum(bool(item["top1_correct"]) for item in items) / count, 4),
        "mean_mrr": round(sum(float(item["mrr"]) for item in items) / count, 4),
        "mean_ndcg_at_k": round(sum(float(item["ndcg_at_k"]) for item in items) / count, 4),
    }


def _hits(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item["id"],
            "text": item["text"],
            "chunk_type": "narrative",
            "score": 0.0,
            "retrieval_method": "eval_candidate",
        }
        for item in case["candidates"]
    ]


def _telemetry_item(case_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Return non-secret provider telemetry suitable for an evaluation artifact."""
    return {
        "case_id": case_id,
        "requested_provider": str(
            meta.get("rerank_requested_provider") or meta.get("rerank_provider") or ""
        ),
        "actual_provider": str(meta.get("rerank_provider") or ""),
        "model": str(meta.get("rerank_model") or ""),
        "latency_ms": round(float(meta.get("rerank_latency_ms") or 0.0), 2),
        "attempts": int(meta.get("rerank_attempts") or 0),
        "tokens": int(meta.get("rerank_tokens") or 0),
        "fallback": bool(meta.get("rerank_fallback")),
        "error_type": str(meta.get("rerank_error_type") or ""),
        # Record completeness without persisting the provider's request identifier.
        "request_id_present": bool(meta.get("rerank_request_id")),
    }


def _qwen3_telemetry_complete(items: list[dict[str, Any]]) -> bool:
    return bool(items) and all(
        item["requested_provider"] == "dashscope"
        and item["actual_provider"] == "dashscope"
        and item["model"] == "qwen3-rerank"
        and item["latency_ms"] > 0.0
        and item["attempts"] >= 1
        and item["tokens"] > 0
        and not item["fallback"]
        and not item["error_type"]
        and item["request_id_present"]
        for item in items
    )


def main() -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Run hard-negative rerank evaluation.")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "data" / "eval_rag" / "rerank_cases.json"),
    )
    parser.add_argument("--provider", choices=("lexical", "qwen3"), default="lexical")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    if args.provider == "qwen3" and not args.allow_remote:
        parser.error("qwen3 evaluation requires explicit --allow-remote data-egress approval")

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        parser.error("rerank cases must be a non-empty JSON list")
    top_k = max(1, int(args.top_k))
    lexical = LexicalReranker()
    selected = lexical
    result_order = ["candidate", "lexical"]
    if args.provider == "qwen3":
        selected = build_reranker(
            args.provider,
            model=os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank"),
            base_url=os.getenv("DASHSCOPE_RERANK_BASE_URL", ""),
            instruct=os.getenv(
                "MAS_RAG_RERANK_INSTRUCT",
                "Given a financial due diligence query, retrieve passages that directly answer it. "
                "Prefer the correct company, reporting period, metric, scope, and filing context over "
                "merely topical passages.",
            ),
            timeout_seconds=float(os.getenv("MAS_RAG_RERANK_TIMEOUT_SECONDS", "12")),
            max_attempts=int(os.getenv("MAS_RAG_RERANK_MAX_ATTEMPTS", "2")),
            backoff_seconds=float(os.getenv("MAS_RAG_RERANK_BACKOFF_SECONDS", "0.25")),
            max_inflight=int(os.getenv("MAS_RAG_RERANK_MAX_INFLIGHT_PER_PROCESS", "2")),
            max_document_chars=int(os.getenv("MAS_RAG_RERANK_MAX_DOCUMENT_CHARS", "4000")),
        )
        result_order.append("qwen3")

    results: dict[str, list[dict[str, Any]]] = {name: [] for name in result_order}
    provider_metas: list[dict[str, Any]] = []
    telemetry: list[dict[str, Any]] = []
    for case in cases:
        candidates = _hits(case)
        lexical_hits, _ = lexical.rerank(case["query"], candidates, top_k=top_k)
        selected_hits, selected_meta = selected.rerank(case["query"], candidates, top_k=top_k)
        provider_metas.append(selected_meta)
        telemetry.append(_telemetry_item(str(case["id"]), selected_meta))
        results["candidate"].append(_score_case(case, candidates, k=top_k))
        results["lexical"].append(_score_case(case, lexical_hits, k=top_k))
        if args.provider == "qwen3":
            results["qwen3"].append(_score_case(case, selected_hits, k=top_k))

    summaries = {name: _summarize(items) for name, items in results.items()}
    print("Rerank hard-negative comparison")
    for name in result_order:
        summary = summaries[name]
        print(
            f"  {name:9} top1={summary['top1_accuracy']:.3f} "
            f"MRR={summary['mean_mrr']:.3f} nDCG@{top_k}={summary['mean_ndcg_at_k']:.3f}"
        )

    fallbacks = sum(bool(meta.get("rerank_fallback")) for meta in provider_metas)
    selected_summary = summaries[args.provider]
    lexical_summary = summaries["lexical"]
    telemetry_complete = (
        _qwen3_telemetry_complete(telemetry) if args.provider == "qwen3" else True
    )
    passed = (
        fallbacks == 0
        and telemetry_complete
        and selected_summary["top1_accuracy"] + 1e-9 >= lexical_summary["top1_accuracy"]
        and selected_summary["mean_mrr"] + 1e-9 >= lexical_summary["mean_mrr"]
        and selected_summary["mean_ndcg_at_k"] + 1e-9 >= lexical_summary["mean_ndcg_at_k"]
    )
    payload = {
        "provider": args.provider,
        "passed": passed,
        "fallbacks": fallbacks,
        "telemetry_complete": telemetry_complete,
        "telemetry": telemetry,
        "summaries": summaries,
        "cases": results,
    }
    print(
        f"  fallbacks={fallbacks} telemetry={'PASS' if telemetry_complete else 'FAIL'} "
        f"gate={'PASS' if passed else 'FAIL'}"
    )
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
