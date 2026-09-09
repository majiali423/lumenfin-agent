from __future__ import annotations

import time
from typing import Any


def poll_job(client: Any, job_id: str, *, timeout: float = 90.0) -> dict:
    """Poll ``GET /api/v1/jobs/{id}`` until completed or failed.

    This helper must stay free of FinAgentBench imports so skip-joint tests
    can reuse it without the product scorer.
    """
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        if response.status_code != 200:
            time.sleep(0.25)
            continue
        last = response.json()
        if last.get("status") in {"completed", "failed"}:
            return last
        time.sleep(0.25)
    raise TimeoutError(f"job {job_id} did not finish: {last}")
