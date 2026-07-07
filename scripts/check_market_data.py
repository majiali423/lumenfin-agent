from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.config import AppConfig
from lumenfin.market_data import MarketDataClient


def main() -> int:
    cfg = AppConfig.from_env()
    print(f"provider={cfg.market_data_provider}")
    print(f"fallback={cfg.market_data_fallback}")
    print(f"av_key_set={bool(cfg.alphavantage_api_key)}")
    client = MarketDataClient(
        provider=cfg.market_data_provider,
        alphavantage_api_key=cfg.alphavantage_api_key,
        fallback_provider=cfg.market_data_fallback,
        cache_ttl_seconds=cfg.market_cache_ttl_seconds,
    )
    print(f"chain={client._provider_chain()}")
    ok = 0
    for company, symbol in [("NVIDIA", "NVDA"), ("AMD", "AMD"), ("Apple", "AAPL")]:
        snap = client.fetch_company_snapshot(company, symbol)
        status = snap.get("status")
        if snap.get("current_price") is not None:
            ok += 1
        err = snap.get("error")
        err_preview = (err[:100] + "...") if err and len(err) > 100 else err
        print(
            f"{company} ({symbol}): status={status} provider={snap.get('provider')} "
            f"price={snap.get('current_price')} as_of={snap.get('fetched_at')} error={err_preview}"
        )
    print(f"summary: {ok}/3 snapshots with price")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
