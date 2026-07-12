#!/usr/bin/env python3
"""Run a deterministic offline portfolio demo."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin import LumenFinAgentSystem
from lumenfin.config import AppConfig
from lumenfin.evaluation import evaluate_run_state
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.market_data import DEFAULT_TICKER_MAP
from lumenfin.reporting import export_run_artifacts
from lumenfin.stdio import configure_stdio_utf8


OFFLINE_MARKET_FIXTURES: dict[str, dict[str, Any]] = {
    "Apple": {
        "current_price": 212.4,
        "monthly_return": 0.031,
        "market_cap": 3_250_000_000_000,
        "trailing_pe": 31.2,
        "currency": "USD",
        "sector": "Technology",
        "industry": "Consumer Electronics",
    },
    "Microsoft": {
        "current_price": 465.7,
        "monthly_return": 0.024,
        "market_cap": 3_460_000_000_000,
        "trailing_pe": 35.4,
        "currency": "USD",
        "sector": "Technology",
        "industry": "Software - Infrastructure",
    },
    "Tesla": {
        "current_price": 248.3,
        "monthly_return": -0.018,
        "market_cap": 790_000_000_000,
        "trailing_pe": 62.8,
        "currency": "USD",
        "sector": "Consumer Cyclical",
        "industry": "Auto Manufacturers",
    },
}


class OfflineMarketDataClient:
    provider = "offline-fixture"

    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict[str, Any]:
        ticker = symbol or DEFAULT_TICKER_MAP.get(company, company)
        fixture = dict(OFFLINE_MARKET_FIXTURES.get(company, {}))
        fixture.update(
            {
                "provider": self.provider,
                "symbol": ticker,
                "company": company,
                "status": "ok" if fixture else "failed",
                "from_cache": False,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "provider_chain": [self.provider],
            }
        )
        if not fixture.get("current_price"):
            fixture.update(
                {
                    "current_price": None,
                    "monthly_return": None,
                    "market_cap": None,
                    "trailing_pe": None,
                    "currency": None,
                    "sector": None,
                    "industry": None,
                    "fifty_two_week_high": None,
                    "fifty_two_week_low": None,
                    "error": "offline fixture not available",
                }
            )
        return fixture


def build_offline_config(output_dir: Path) -> AppConfig:
    config = AppConfig.from_env()
    data_dir = output_dir / "_demo_data"
    return replace(
        config,
        output_dir=output_dir,
        upload_dir=output_dir / "_uploads",
        db_path=data_dir / "lumenfin_demo.db",
        database_url=f"sqlite:///{(data_dir / 'lumenfin_demo.db').as_posix()}",
        market_data_provider="offline-fixture",
        market_data_fallback="offline-fixture",
        alphavantage_api_key=None,
        rag_enabled=False,
        milvus_uri=str(data_dir / "milvus_demo.db"),
        tool_backend="local",
    )


def main() -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Run an offline LumenFin portfolio demo.")
    parser.add_argument(
        "--query",
        default="Compare Apple, Microsoft, and Tesla FY2025 profitability, R&D intensity, liquidity, and supply-chain risk.",
    )
    parser.add_argument("--thread-id", default="portfolio-demo")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "portfolio_demo"))
    parser.add_argument("--write", action="store_true", help="Export report/state/audit/manifest artifacts.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    app = LumenFinAgentSystem(
        llm_client=LocalFallbackLLMClient(),
        app_config=build_offline_config(output_dir),
        market_data_client=OfflineMarketDataClient(),
    )
    result = app.run(args.query, thread_id=args.thread_id)
    evaluation = evaluate_run_state(result).to_dict()
    artifacts = (
        export_run_artifacts(
            result,
            output_dir,
            args.thread_id,
            llm_backend=result.get("llm_backend"),
            rag_enabled=False,
            market_provider="offline-fixture",
        )
        if args.write
        else {}
    )
    summary = {
        "thread_id": result.get("thread_id"),
        "workflow_status": result.get("workflow_status"),
        "companies": result.get("companies", []),
        "llm_backend": result.get("llm_backend"),
        "evaluator_score": evaluation["score"],
        "evaluator_grade": evaluation["grade"],
        "audit_steps": [event.get("step") for event in result.get("audit_log", [])],
        "artifacts": artifacts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.get("workflow_status") == "completed" and evaluation["score"] >= 85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
