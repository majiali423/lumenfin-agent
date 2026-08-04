from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from lumenfin.queueing import RedisQueueManager
from redis.exceptions import TimeoutError as RedisTimeoutError


class RedisQueueResilienceTestCase(unittest.TestCase):
    def test_dequeue_returns_none_on_socket_timeout(self) -> None:
        queue = RedisQueueManager("redis://localhost:6379/0", "q-test")
        mock_client = MagicMock()
        mock_client.blpop.side_effect = RedisTimeoutError("Timeout reading from socket")
        with patch.object(queue, "connection", return_value=mock_client):
            self.assertIsNone(queue.dequeue(timeout_seconds=1))
        # Connection is dropped so the next call can reconnect.
        self.assertIsNone(queue._client)


if __name__ == "__main__":
    unittest.main()
