#!/usr/bin/env python3
"""Exposed test-100 candidate-pool / Qwen3 paired ablation. Not confirmation/dev."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.env_bootstrap import bootstrap_dotenv, describe_credential_sources
from lumenfin.eval.financebench.candidate_pool_ablation import (
    ABLATION_OUTPUT_DIRNAME,
    ABLATION_PREFLIGHT_DIRNAME,
    AblationError,
    InvalidEmptyRetrievalError,
    LOCKED_EMBEDDING_PROVIDER,
    LOCKED_SPLIT,
    SOURCE_INDEX_SESSION_ID,
    run_candidate_pool_ablation,
    validate_ablation_request,
)
from lumenfin.eval.financebench.index_inspect import IndexIncompatibleError
from lumenfin.eval.financebench.index_session import IndexSessionError
from lumenfin.eval.financebench.retrieval import RemoteEvalBlocked
from lumenfin.eval.financebench.split import SplitError
from lumenfin.stdio import configure_stdio_utf8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "FinanceBench exposed test-100 candidate-pool / Qwen3 paired ablation. "
            "Locked arms A/B/C, split=test, company scope, session_id="
            f"{SOURCE_INDEX_SESSION_ID}. Not held-out. Not confirmation-50."
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
        help="Copy and verify the historical index without embeddings, rerank, or scoring.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run in the same output directory when hashes match.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Real CLI scoring must use outputs/"
            f"{ABLATION_OUTPUT_DIRNAME}/; preflight must use outputs/"
            f"{ABLATION_PREFLIGHT_DIRNAME}/. Other paths are rejected."
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
            output_dir = str(ROOT / "outputs" / ABLATION_PREFLIGHT_DIRNAME)
        else:
            output_dir = str(ROOT / "outputs" / ABLATION_OUTPUT_DIRNAME)
    try:
        validate_ablation_request(
            split=args.split,
            confirm_exposed_diagnostic=bool(args.confirm_exposed_diagnostic),
            allow_remote=bool(args.allow_remote),
            embedding_provider=LOCKED_EMBEDDING_PROVIDER,
            session_id=SOURCE_INDEX_SESSION_ID,
            preflight_only=bool(args.preflight_only),
            resume=bool(args.resume),
            output_dir=output_dir,
            repo_root=ROOT,
            enforce_locked_output_dir=True,
        )
        if not args.preflight_only:
            for report in describe_credential_sources(root=ROOT, keys=("DASHSCOPE_API_KEY",)):
                print(f"{report.key} source={report.source}", flush=True)
        results = run_candidate_pool_ablation(
            dataset_dir=args.dataset_dir,
            output_dir=output_dir,
            repo_root=ROOT,
            split=args.split,
            confirm_exposed_diagnostic=bool(args.confirm_exposed_diagnostic),
            allow_remote=bool(args.allow_remote),
            embedding_provider=LOCKED_EMBEDDING_PROVIDER,
            session_id=SOURCE_INDEX_SESSION_ID,
            require_clean_worktree=True,
            preflight_only=bool(args.preflight_only),
            resume=bool(args.resume),
            enforce_locked_output_dir=True,
        )
    except InvalidEmptyRetrievalError as exc:
        print(f"blocked: {exc}")
        print(f"query_embedding_calls={exc.query_embedding_calls}", flush=True)
        return 2
    except (AblationError, SplitError, RemoteEvalBlocked, IndexIncompatibleError, IndexSessionError) as exc:
        print(f"blocked: {exc}")
        return 2

    if results.get("status") == "PREFLIGHT_OK":
        print("[candidate-pool-ablation] PREFLIGHT_OK", flush=True)
        print(f"Wrote {Path(output_dir) / 'preflight.json'}", flush=True)
        return 0

    summary = results.get("summary") or {}
    arms = summary.get("arms") or {}
    hybrid = (arms.get("C") or {})
    print(
        f"[candidate-pool-ablation] split={results.get('split')} cases={summary.get('cases', 0)} "
        f"C_Hit@10={hybrid.get('page_hit_at_10', 'NOT_RUN')} "
        f"role={results.get('experiment_role')}",
        flush=True,
    )
    print(f"Wrote {Path(output_dir) / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
