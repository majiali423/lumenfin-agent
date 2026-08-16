#!/usr/bin/env python3
"""Run FinanceBench retrieval ablations. Default is offline; remote needs --allow-remote."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.env_bootstrap import bootstrap_dotenv, describe_credential_sources
from lumenfin.eval.financebench.constants import CLI_MODES, CLI_SPLITS, INDEX_SCOPES, RETRIEVAL_MODES
from lumenfin.eval.financebench.frozen import FrozenConfigError
from lumenfin.eval.financebench.reporting import compare_modes, read_jsonl, write_json, write_jsonl
from lumenfin.eval.financebench.retrieval import RemoteEvalBlocked
from lumenfin.eval.financebench.runner import run_retrieval_eval
from lumenfin.eval.financebench.split import SplitError
from lumenfin.stdio import configure_stdio_utf8


def _print_mode_summary(mode: str, split: str, results: dict) -> None:
    summary = results.get("summary") or {}
    page = summary.get("page") or {}
    print(
        f"[{mode}] split={split} cases={summary.get('cases', 0)} "
        f"Hit@5={page.get('hit_at_5', 'NOT_RUN')} MRR={page.get('mrr', 'NOT_RUN')} "
        f"nDCG@10={page.get('ndcg_at_10', 'NOT_RUN')}",
        flush=True,
    )


def main() -> int:
    configure_stdio_utf8()
    bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
    parser = argparse.ArgumentParser(description="FinanceBench retrieval evaluation.")
    parser.add_argument(
        "--dataset-dir",
        default=str(ROOT / "data" / "external" / "financebench-src"),
    )
    parser.add_argument("--split", choices=CLI_SPLITS, default="dev")
    parser.add_argument("--mode", choices=CLI_MODES, default="hybrid")
    parser.add_argument("--index-scope", choices=INDEX_SCOPES, default="company")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--embedding-provider", default="deterministic")
    parser.add_argument("--embedding-dimension", type=int, default=0)
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-index", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Opt into threshold/prompt fitting. Forbidden on test/all.",
    )
    parser.add_argument(
        "--compare-dirs",
        nargs="*",
        default=[],
        help="Optional per-mode result directories for rank-movement ablation.",
    )
    parser.add_argument(
        "--frozen-config",
        default="",
        help="Path to the git-tracked confirmation frozen config JSON.",
    )
    parser.add_argument(
        "--confirm-held-out",
        action="store_true",
        help="Required to execute confirmation/dev. Does not retune. One-shot after approval.",
    )
    args = parser.parse_args()
    for report in describe_credential_sources(root=ROOT):
        if report.key == "DASHSCOPE_API_KEY":
            print(f"DASHSCOPE_API_KEY source={report.source}", flush=True)

    if args.mode == "all":
        output_dir = Path(args.output_dir or ROOT / "outputs" / "financebench_eval")
    else:
        output_dir = Path(args.output_dir or ROOT / "outputs" / "financebench_eval" / args.mode)
    try:
        results = run_retrieval_eval(
            dataset_dir=args.dataset_dir,
            output_dir=output_dir,
            repo_root=ROOT,
            split=args.split,
            mode=args.mode,
            index_scope=args.index_scope,
            top_k=args.top_k,
            embedding_provider=args.embedding_provider,
            embedding_dimension=args.embedding_dimension,
            allow_remote=args.allow_remote,
            resume=args.resume,
            keep_index=args.keep_index,
            limit=args.limit or None,
            tuning=args.tune,
            expected_questions=None,
            require_pdfs=True,
            frozen_config_path=args.frozen_config or None,
            confirm_held_out=bool(args.confirm_held_out),
        )
    except FrozenConfigError as exc:
        print(f"frozen config error: {exc}")
        return 2
    except RemoteEvalBlocked as exc:
        print(f"blocked: {exc}")
        return 2
    except SplitError as exc:
        print(f"split error: {exc}")
        return 2

    if args.mode == "all":
        for mode_name, mode_results in (results.get("modes") or {}).items():
            _print_mode_summary(mode_name, args.split, mode_results)
        ablation = results.get("ablation") or {}
        print(
            f"ablation improved={ablation.get('improved')} "
            f"degraded={ablation.get('degraded')} "
            f"never_retrieved={ablation.get('never_retrieved')}",
            flush=True,
        )
    else:
        _print_mode_summary(args.mode, args.split, results)

    json_out = Path(args.json_out) if args.json_out else output_dir / "results.json"
    if args.json_out:
        write_json(json_out, results)
        print(f"Wrote {json_out}")
    if args.markdown_out:
        md_out = Path(args.markdown_out)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        source_md = output_dir / "results.md"
        if not source_md.is_file() and args.mode == "all":
            source_md = output_dir / RETRIEVAL_MODES[-1] / "results.md"
        md_out.write_text(source_md.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote {md_out}")

    compare_dirs = list(args.compare_dirs)
    if not compare_dirs and args.mode == "all":
        compare_dirs = [str(output_dir / name) for name in RETRIEVAL_MODES]
    if compare_dirs:
        per_mode = {}
        for directory in compare_dirs:
            path = Path(directory)
            rows = read_jsonl(path / "per_case.jsonl")
            if not rows:
                continue
            mode = str(rows[0].get("mode") or path.name)
            per_mode[mode] = rows
        comparison = compare_modes(per_mode)
        write_json(output_dir / "ablation.json", comparison)
        write_jsonl(output_dir / "rank_movements.jsonl", comparison.get("movements") or [])
        print(
            f"ablation improved={comparison.get('improved')} "
            f"degraded={comparison.get('degraded')} "
            f"never_retrieved={comparison.get('never_retrieved')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
