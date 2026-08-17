#!/usr/bin/env python3
"""Exposed test-100 candidate-depth diagnostic. Does not run confirmation/dev."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.env_bootstrap import bootstrap_dotenv, describe_credential_sources
from lumenfin.eval.financebench.candidate_depth import (
    CandidateDepthError,
    InvalidEmptyRetrievalError,
    LOCKED_EMBEDDING_DIMENSION,
    LOCKED_EMBEDDING_MODEL,
    LOCKED_EMBEDDING_PROVIDER,
    LOCKED_INDEX_SCOPE,
    LOCKED_SPLIT,
    run_candidate_depth_diagnostic,
    validate_candidate_depth_request,
)
from lumenfin.eval.financebench.index_inspect import IndexIncompatibleError
from lumenfin.eval.financebench.index_session import (
    IndexSessionError,
    LOCKED_OUTPUT_DIRNAME,
    LOCKED_PREFLIGHT_OUTPUT_DIRNAME,
    SOURCE_INDEX_SESSION_ID,
)
from lumenfin.eval.financebench.retrieval import RemoteEvalBlocked
from lumenfin.eval.financebench.split import SplitError
from lumenfin.stdio import configure_stdio_utf8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "FinanceBench exposed test-100 candidate-depth diagnostic. "
            "Locked to split=test, company scope, candidate_k=50, "
            "text-embedding-v4/1024, RRF dense 1.0 / BM25 1.1, "
            f"session_id={SOURCE_INDEX_SESSION_ID}. "
            "Not held-out. Not confirmation-50. Does not call Qwen3."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT / "data" / "external" / "financebench-src"),
    )
    parser.add_argument("--split", default=LOCKED_SPLIT)
    parser.add_argument(
        "--confirm-exposed-diagnostic",
        action="store_true",
        help="Required. Acknowledges test-100 is exposed and this is not held-out.",
    )
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Copy and verify the historical index without query embeddings or scoring.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            f"outputs/{LOCKED_PREFLIGHT_OUTPUT_DIRNAME}/ for --preflight-only, "
            f"otherwise outputs/{LOCKED_OUTPUT_DIRNAME}/."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    if not output_dir:
        if args.preflight_only:
            output_dir = str(ROOT / "outputs" / LOCKED_PREFLIGHT_OUTPUT_DIRNAME)
        else:
            output_dir = str(ROOT / "outputs" / LOCKED_OUTPUT_DIRNAME)
    try:
        validate_candidate_depth_request(
            split=args.split,
            confirm_exposed_diagnostic=bool(args.confirm_exposed_diagnostic),
            allow_remote=bool(args.allow_remote),
            embedding_provider=LOCKED_EMBEDDING_PROVIDER,
            embedding_dimension=LOCKED_EMBEDDING_DIMENSION,
            index_scope=LOCKED_INDEX_SCOPE,
            embedding_model=LOCKED_EMBEDDING_MODEL,
            session_id=SOURCE_INDEX_SESSION_ID,
            preflight_only=bool(args.preflight_only),
            output_dir=output_dir,
            repo_root=ROOT,
        )
        if not args.preflight_only:
            for report in describe_credential_sources(root=ROOT):
                if report.key == "DASHSCOPE_API_KEY":
                    print(f"DASHSCOPE_API_KEY source={report.source}", flush=True)
        results = run_candidate_depth_diagnostic(
            dataset_dir=args.dataset_dir,
            output_dir=output_dir,
            repo_root=ROOT,
            split=args.split,
            confirm_exposed_diagnostic=bool(args.confirm_exposed_diagnostic),
            allow_remote=bool(args.allow_remote),
            embedding_provider=LOCKED_EMBEDDING_PROVIDER,
            embedding_dimension=LOCKED_EMBEDDING_DIMENSION,
            index_scope=LOCKED_INDEX_SCOPE,
            embedding_model=LOCKED_EMBEDDING_MODEL,
            session_id=SOURCE_INDEX_SESSION_ID,
            require_clean_worktree=True,
            preflight_only=bool(args.preflight_only),
        )
    except InvalidEmptyRetrievalError as exc:
        print(f"blocked: {exc}")
        print(f"query_embedding_calls={exc.query_embedding_calls}", flush=True)
        return 2
    except (CandidateDepthError, SplitError, RemoteEvalBlocked, IndexIncompatibleError, IndexSessionError) as exc:
        print(f"blocked: {exc}")
        return 2

    if results.get("status") == "PREFLIGHT_OK":
        print("[candidate-depth] PREFLIGHT_OK", flush=True)
        print(f"Wrote {Path(output_dir) / 'preflight.json'}", flush=True)
        return 0

    summary = results.get("summary") or {}
    hybrid = (summary.get("modes") or {}).get("hybrid_rrf") or {}
    print(
        f"[candidate-depth] split={results.get('split')} cases={summary.get('cases', 0)} "
        f"Hit@10={hybrid.get('page_hit_at_10', 'NOT_RUN')} "
        f"Hit@50={hybrid.get('page_hit_at_50', 'NOT_RUN')} "
        f"role={results.get('experiment_role')}",
        flush=True,
    )
    print(f"Wrote {Path(output_dir) / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
