#!/usr/bin/env python3
"""Demo: real PDF + real DashScope RAG ranking (RRF vs lexical rerank).

Usage:
  python scripts/demo_rag_ranking.py
  python scripts/demo_rag_ranking.py --pdf fixtures/nvidia_fy2025_earnings_excerpt.pdf --company NVIDIA
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.config import AppConfig
from lumenfin.document_ingest import parse_upload_documents
from lumenfin.rag.factory import build_hybrid_retriever, build_rag_store
from lumenfin.rag.hybrid_retriever import reciprocal_rank_fusion
from lumenfin.rag.profiles import apply_showcase_rag_env
from lumenfin.rag.rerank import rerank_hits
from lumenfin.stdio import configure_stdio_utf8


def _clip(text: str, n: int = 140) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= n else cleaned[: n - 1] + "…"


def _print_rank_table(title: str, hits: list[dict], *, score_keys: list[str]) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)
    if not hits:
        print("(empty)")
        return
    for i, hit in enumerate(hits, start=1):
        scores = " | ".join(f"{k}={hit.get(k)}" for k in score_keys if hit.get(k) is not None)
        print(
            f"#{i:02d}  {hit.get('citation') or hit.get('filename')}  "
            f"type={hit.get('chunk_type')}  method={hit.get('retrieval_method')}"
        )
        if scores:
            print(f"     {scores}")
        print(f"     {_clip(str(hit.get('text') or ''), 160)}")


def main() -> int:
    configure_stdio_utf8()
    apply_showcase_rag_env(overwrite=False)

    parser = argparse.ArgumentParser(description="Show real RAG ranking on a PDF.")
    parser.add_argument(
        "--pdf",
        default=str(ROOT / "fixtures" / "nvidia_fy2025_earnings_excerpt.pdf"),
        help="Path to a real PDF fixture.",
    )
    parser.add_argument("--company", default="NVIDIA")
    parser.add_argument(
        "--query",
        default="NVIDIA FY2025 data center revenue, GPU demand, and supply chain risk",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument(
        "--json-out",
        default=str(ROOT / "outputs" / "rag_ranking_demo.json"),
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    # Isolated Lite DB for this demo (avoid clashing with API/demo locks / dims).
    demo_root = ROOT / "test_artifacts" / f"rag-rank-demo-{uuid4().hex[:8]}"
    demo_root.mkdir(parents=True, exist_ok=True)
    milvus_uri = str(demo_root / "milvus_dashscope.db")

    config = AppConfig.from_env()
    from dataclasses import replace

    config = replace(
        config,
        milvus_uri=milvus_uri,
        milvus_collection="rag_rank_demo",
        milvus_isolate=False,
        rag_index_mode="async_on_upload",
        rag_rerank_enabled=True,
        rag_rerank_candidates=args.candidates,
        rag_top_k=args.top_k,
        rag_tenant_id="rank-demo",
    )

    print("PDF:", pdf_path)
    print("Company:", args.company)
    print("Query:", args.query)
    print("Embedding:", config.embedding_provider, "dim=", config.embedding_dimension)
    print("Milvus:", milvus_uri)

    if config.embedding_provider not in {"dashscope", "aliyun", "alibaba"}:
        print(
            "WARN: embedding_provider is not dashscope — set MAS_EMBEDDING_PROVIDER=dashscope in .env",
            file=sys.stderr,
        )

    contexts = parse_upload_documents(pdf_path)
    if not contexts:
        print("No document contexts parsed from PDF.", file=sys.stderr)
        return 1
    for ctx in contexts:
        if args.company not in (ctx.get("detected_companies") or []):
            ctx["detected_companies"] = list(dict.fromkeys([*(ctx.get("detected_companies") or []), args.company]))

    store = build_rag_store(config)
    if store is None:
        print("RAG store unavailable (MAS_RAG_ENABLED?).", file=sys.stderr)
        return 1
    # Index via store session path using parsed contexts (same chunks as hybrid).
    index_stats = store.index_documents(contexts, session_id="rank-demo-session")
    print("Index stats:", index_stats)

    retriever = build_hybrid_retriever(config, rag_store=store)
    candidate_k = max(args.candidates, args.top_k)

    # --- Stage breakdown (same pieces hybrid uses) ---
    keyword_hits = []
    from lumenfin.rag import hybrid_retriever as hr

    keyword_hits = hr._keyword_search(
        contexts,
        company=args.company,
        query=args.query,
        top_k=candidate_k,
    )
    vector_hits = store.vector_search(
        args.query,
        session_id="rank-demo-session",
        companies=[args.company],
        top_k=candidate_k,
    )
    fused = reciprocal_rank_fusion([vector_hits, keyword_hits])[:candidate_k]
    reranked = rerank_hits(args.query, fused, top_k=args.top_k)

    # Also run the public API for parity.
    final_hits, meta = retriever.retrieve_for_company_with_meta(
        query=args.query,
        company=args.company,
        session_id="rank-demo-session",
        document_contexts=contexts,
    )

    _print_rank_table(
        f"1) Keyword-only top-{len(keyword_hits)}",
        keyword_hits,
        score_keys=["score"],
    )
    _print_rank_table(
        f"2) Vector (DashScope) top-{len(vector_hits)}",
        vector_hits,
        score_keys=["score"],
    )
    _print_rank_table(
        f"3) Hybrid RRF fused top-{len(fused)} (before lexical rerank)",
        fused,
        score_keys=["fusion_score", "score"],
    )
    _print_rank_table(
        f"4) After lexical rerank → top-{len(reranked)}",
        reranked,
        score_keys=["rerank_score", "fusion_score"],
    )
    _print_rank_table(
        f"5) Public HybridEvidenceRetriever result (meta={meta.get('mode')})",
        final_hits,
        score_keys=["rerank_score", "fusion_score", "score"],
    )

    # Rank movement: RRF position → rerank position
    print()
    print("=" * 88)
    print("Rank movement (RRF → lexical rerank)")
    print("=" * 88)
    rrf_pos = {
        (h.get("citation") or h.get("chunk_id") or i): i + 1 for i, h in enumerate(fused)
    }
    for new_i, hit in enumerate(reranked, start=1):
        key = hit.get("citation") or hit.get("chunk_id") or new_i
        old = rrf_pos.get(key, "?")
        arrow = "→" if old == new_i else f"#{old} → #{new_i}"
        print(f"  {arrow:12}  {hit.get('citation')}  rerank={hit.get('rerank_score')}")

    out = {
        "pdf": str(pdf_path),
        "company": args.company,
        "query": args.query,
        "embedding_provider": config.embedding_provider,
        "embedding_dimension": config.embedding_dimension,
        "index_stats": index_stats,
        "meta": meta,
        "keyword": keyword_hits,
        "vector": vector_hits,
        "rrf": fused,
        "reranked": reranked,
        "final": final_hits,
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print()
    print("Wrote:", out_path)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
