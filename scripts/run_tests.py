#!/usr/bin/env python3
"""Run unit tests with noisy third-party loggers suppressed."""
from __future__ import annotations

import argparse
import logging
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LumenFin unit tests (quiet Milvus/gRPC logs).")
    parser.add_argument("--integration", action="store_true", help="Also run live API integration tests.")
    args = parser.parse_args()

    configure_quiet_test_logging()
    if args.integration:
        import os

        os.environ["RUN_INTEGRATION_TESTS"] = "1"

    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
