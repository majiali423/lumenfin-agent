#!/usr/bin/env python3
"""Run unit tests with noisy third-party loggers suppressed."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Pin CI RAG profile before any lumenfin import / dotenv load from discover().
from lumenfin.rag.profiles import apply_ci_rag_env

apply_ci_rag_env()
os.environ.setdefault("APP_ENV", "test")


def configure_quiet_test_logging() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    for logger_name in (
        "grpc",
        "grpc._server",
        "milvus_lite",
        "milvus_lite.server_manager",
        "pymilvus",
        "faiss",
        "faiss.loader",
        "httpx",
        "lumenfin.api",
    ):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


# Fast required CI: no FinAgentBench install. Joint scoring is a separate gate.
FAST_MODULES = (
    "tests.test_html_sanitize",
    "tests.test_frontend_bench_drawer",
    "tests.test_phase5_demo_ui",
    "tests.test_phase4_task_spec",
    "tests.test_query_answer_focus",
    "tests.test_company_upload_mismatch",
    "tests.test_phase3_boundaries",
    "tests.test_graph_routing",
    "tests.test_offline_env",
    "tests.test_phase6_docs",
    "tests.test_package_import",
    "tests.test_bounded_repair_policy",
)

# Needs a published v3 product scorer (FINAGENTBENCH_DIR / PRODUCT_REF). Not --fast.
JOINT_MODULE_FILES = frozenset(
    {
        "test_product_dev_scoring.py",
        "test_product_quality_loop.py",
        "test_upload_product_loop.py",
        "test_finrun_adapter_parity.py",
        "test_eval_contract_runner.py",
    }
)
JOINT_MODULES = tuple(f"tests.{name[:-3]}" for name in sorted(JOINT_MODULE_FILES))


def _load_offline_suite(loader: unittest.TestLoader, *, skip_joint: bool) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    tests_root = ROOT / "tests"
    for path in sorted(tests_root.rglob("test_*.py")):
        if "support" in path.parts:
            continue
        if skip_joint and path.name in JOINT_MODULE_FILES:
            continue
        rel = path.relative_to(tests_root).with_suffix("")
        module = "tests." + ".".join(rel.parts)
        suite.addTests(loader.loadTestsFromName(module))
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LumenFin unit tests (quiet Milvus/gRPC logs).")
    parser.add_argument("--integration", action="store_true", help="Also run live API integration tests.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Required PR slice: sanitize, frontend, routing, phase 3–6, offline env.",
    )
    parser.add_argument(
        "--skip-joint",
        action="store_true",
        help="Full solo offline suite without v3 product-scorer tests.",
    )
    parser.add_argument(
        "--joint-only",
        action="store_true",
        help="Only tests that require FinAgentBench visible_supported_claims v3.",
    )
    args = parser.parse_args()
    if args.fast and (args.skip_joint or args.joint_only):
        parser.error("--fast cannot be combined with --skip-joint or --joint-only")
    if args.skip_joint and args.joint_only:
        parser.error("--skip-joint cannot be combined with --joint-only")

    configure_quiet_test_logging()
    if args.integration:
        os.environ["RUN_INTEGRATION_TESTS"] = "1"

    loader = unittest.TestLoader()
    if args.fast:
        suite = unittest.TestSuite()
        for name in FAST_MODULES:
            suite.addTests(loader.loadTestsFromName(name))
    elif args.joint_only:
        suite = unittest.TestSuite()
        for name in JOINT_MODULES:
            suite.addTests(loader.loadTestsFromName(name))
    else:
        suite = _load_offline_suite(loader, skip_joint=args.skip_joint)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
