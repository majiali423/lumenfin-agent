#!/usr/bin/env python3
"""Start the real LumenFin API/UI with deterministic, network-free providers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _apply_offline_demo_env(*, host: str, port: int) -> None:
    from scripts.offline_env import apply_offline_env

    output_dir = ROOT / "outputs" / "offline_demo_api"
    apply_offline_env(
        extra={
            "MAS_MILVUS_URI": str(output_dir / "milvus.db"),
            "MAS_MILVUS_COLLECTION": "lumenfin_offline_demo",
            "MAS_MILVUS_ISOLATE": "true",
            "MAS_DB_PATH": str(output_dir / "lumenfin.db"),
            "MAS_OUTPUT_DIR": str(output_dir),
            "MAS_UPLOAD_DIR": str(output_dir / "uploads"),
            "MAS_HOST": host,
            "MAS_PORT": str(port),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the deterministic offline LumenFin demo API and UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    _apply_offline_demo_env(host=args.host, port=args.port)

    import uvicorn

    from lumenfin.api.app import create_app
    from lumenfin.config import AppConfig
    from lumenfin.llm import LocalFallbackLLMClient
    from scripts.run_portfolio_demo import OfflineMarketDataClient

    config = AppConfig.from_env()
    app = create_app(
        config,
        llm_client=LocalFallbackLLMClient(),
        market_data_client=OfflineMarketDataClient(),
    )
    print(
        f"OFFLINE_DEMO_READY url=http://{args.host}:{args.port}/ "
        "providers=local-fallback,deterministic,offline-fixture",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
