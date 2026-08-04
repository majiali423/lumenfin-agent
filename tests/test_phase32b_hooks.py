from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path

from lumenfin.integration_hooks import (
    maybe_barrier_after_checkpoint_read,
    maybe_pause_after_index_claim,
)


class Phase32BHooksTestCase(unittest.TestCase):
    def test_hooks_noop_without_env(self) -> None:
        os.environ.pop("MAS_INTEGRATION_CHECKPOINT_BARRIER_DIR", None)
        os.environ.pop("MAS_INTEGRATION_INDEX_PAUSE_DIR", None)
        maybe_barrier_after_checkpoint_read("t1", 0)
        maybe_pause_after_index_claim(
            document_id="d1",
            tenant_id="tenant",
            index_owner="owner",
            index_attempt=1,
        )

    def test_barrier_waits_only_when_armed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            barrier = Path(tmp)
            os.environ["MAS_INTEGRATION_CHECKPOINT_BARRIER_DIR"] = str(barrier)
            os.environ["MAS_WORKER_ID"] = "hook-test"
            maybe_barrier_after_checkpoint_read("t1", 0)  # unarmed no-op
            self.assertFalse(any(barrier.glob("ready-*")))

            (barrier / "armed").write_text("1", encoding="utf-8")

            def _release() -> None:
                import time

                time.sleep(0.2)
                (barrier / "release").write_text("1", encoding="utf-8")

            thread = threading.Thread(target=_release)
            thread.start()
            maybe_barrier_after_checkpoint_read("t1", 1)
            thread.join(timeout=5)
            self.assertTrue((barrier / "ready-hook-test").exists())
            os.environ.pop("MAS_INTEGRATION_CHECKPOINT_BARRIER_DIR", None)
            os.environ.pop("MAS_WORKER_ID", None)


if __name__ == "__main__":
    unittest.main()
