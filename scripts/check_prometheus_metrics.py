#!/usr/bin/env python3
"""Check the LumenFin Prometheus endpoint without sending an analysis request."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request


REQUIRED = (
    "lumenfin_http_requests_total",
    "lumenfin_workflow_runs_total",
    "lumenfin_node_duration_seconds",
    "lumenfin_provider_calls_total",
    "lumenfin_rag_queries_total",
    "lumenfin_queue_depth",
    "lumenfin_worker_jobs_total",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check LumenFin /metrics.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    url = args.base_url.rstrip("/") + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310 - explicit local URL
            payload = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"METRICS_UNREACHABLE url={url} error_type={type(exc).__name__}")
        return 1
    missing = [name for name in REQUIRED if name not in payload]
    if missing:
        print("METRICS_INCOMPLETE missing=" + ",".join(missing))
        return 2
    print("METRICS_OK")
    print("Prometheus targets: http://127.0.0.1:9090/targets")
    print("Grafana dashboard:  http://127.0.0.1:3000/d/lumenfin-ops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
