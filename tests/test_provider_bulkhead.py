"""Bulkhead acquire/release and provider_busy classification."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.provider_resilience import (
    ProviderBusyError,
    ProviderCallContext,
    acquire_provider_slot,
    classify_provider_exception,
    get_provider_semaphore,
)


class BulkheadTestCase(unittest.TestCase):
    def test_release_on_exception_path_no_permit_leak(self) -> None:
        name = "test-bulkhead-leak"
        # Unique max_inflight avoids colliding with other tests' semaphores.
        max_inflight = 1
        sem = get_provider_semaphore(name, max_inflight=max_inflight)
        # Drain any leftover (should be free).
        self.assertTrue(sem.acquire(blocking=False))
        sem.release()

        ctx = ProviderCallContext.create(deadline_seconds=5)
        release = acquire_provider_slot(
            name,
            max_inflight=max_inflight,
            context=ctx,
            acquire_timeout_seconds=1.0,
        )
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            release()

        # Must be able to acquire again immediately.
        self.assertTrue(sem.acquire(timeout=0.2))
        sem.release()

    def test_provider_busy_not_retryable_http(self) -> None:
        name = "test-bulkhead-busy"
        max_inflight = 1
        ctx_holder = ProviderCallContext.create(deadline_seconds=5)
        first = acquire_provider_slot(
            name,
            max_inflight=max_inflight,
            context=ctx_holder,
            acquire_timeout_seconds=1.0,
        )
        try:
            ctx2 = ProviderCallContext.create(deadline_seconds=5)
            with self.assertRaises(ProviderBusyError) as raised:
                acquire_provider_slot(
                    name,
                    max_inflight=max_inflight,
                    context=ctx2,
                    acquire_timeout_seconds=0.05,
                )
            self.assertEqual(classify_provider_exception(raised.exception), "provider_busy")
        finally:
            first()


if __name__ == "__main__":
    unittest.main()
