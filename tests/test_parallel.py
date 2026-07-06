from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.parallel import map_in_parallel


class ParallelMapTestCase(unittest.TestCase):
    def test_preserves_order(self) -> None:
        results = map_in_parallel(lambda value: value * 2, [1, 2, 3, 4], max_workers=2)
        self.assertEqual(results, [2, 4, 6, 8])

    def test_runs_tasks_concurrently(self) -> None:
        active = 0
        peak = 0
        lock = threading.Lock()

        def slow_square(value: int) -> int:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return value * value

        started = time.perf_counter()
        results = map_in_parallel(slow_square, [1, 2, 3, 4], max_workers=4)
        elapsed = time.perf_counter() - started

        self.assertEqual(results, [1, 4, 9, 16])
        self.assertGreaterEqual(peak, 2)
        self.assertLess(elapsed, 0.18)


if __name__ == "__main__":
    unittest.main()
