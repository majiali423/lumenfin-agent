from __future__ import annotations

import json
import time
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from lumenfin.queueing import RedisQueueManager
from redis.exceptions import TimeoutError as RedisTimeoutError


class _ListRedis:
    """Minimal Redis stand-in that executes the reliable-queue Lua scripts in Python."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}

    def from_url(self, *_args: Any, **_kwargs: Any) -> "_ListRedis":
        return self

    def rpush(self, key: str, value: str) -> int:
        self.lists.setdefault(key, []).append(value)
        return len(self.lists[key])

    def lpop(self, key: str) -> str | None:
        items = self.lists.get(key) or []
        if not items:
            return None
        return items.pop(0)

    def llen(self, key: str) -> int:
        return len(self.lists.get(key) or [])

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self.lists.get(key) or []
        if end == -1:
            end = len(items) - 1
        if end < start or start >= len(items):
            return []
        return list(items[start : end + 1])

    def lrem(self, key: str, count: int, value: str) -> int:
        items = self.lists.get(key) or []
        removed = 0
        if count == 0:
            keep = [item for item in items if item != value]
            removed = len(items) - len(keep)
            self.lists[key] = keep
            return removed
        new_items: list[str] = []
        for item in items:
            if removed < abs(count) and item == value:
                removed += 1
                continue
            new_items.append(item)
        self.lists[key] = new_items
        return removed

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.lists:
                deleted += 1
                del self.lists[key]
        return deleted

    def blpop(self, key: str, timeout: int = 0) -> tuple[str, str] | None:
        del timeout
        value = self.lpop(key)
        if value is None:
            return None
        return key, value

    def eval(self, script: str, numkeys: int, *args: Any) -> Any:
        keys = list(args[:numkeys])
        argv = list(args[numkeys:])
        if "LPOP" in script and "RPUSH" in script and "reserved_by" in script:
            return self._reserve(keys[0], keys[1], int(argv[0]), str(argv[1]))
        if "LREM" in script and numkeys == 1 and "dead" not in script.lower():
            # ACK script only touches processing
            return self._ack(keys[0], str(argv[0]))
        if "dead_letter" in script and "reclaimed_stale_processing" not in script:
            return self._retry(keys[0], keys[1], keys[2], str(argv[0]), str(argv[1]), int(argv[2]), int(argv[3]))
        if "reclaimed_stale_processing" in script:
            return self._reclaim(keys[0], keys[1], keys[2], int(argv[0]), int(argv[1]), int(argv[2]))
        raise AssertionError("unexpected lua script")

    def _reserve(self, pending: str, processing: str, now: int, worker_id: str) -> str | bool:
        raw = self.lpop(pending)
        if raw is None:
            return False
        msg = json.loads(raw)
        msg["attempt"] = int(msg.get("attempt") or 0) + 1
        msg["reserved_at"] = now
        msg["reserved_by"] = worker_id
        encoded = json.dumps(msg)
        self.rpush(processing, encoded)
        return encoded

    def _ack(self, processing: str, message_id: str) -> int:
        for item in list(self.lrange(processing, 0, -1)):
            msg = json.loads(item)
            if str(msg.get("message_id")) == message_id:
                return self.lrem(processing, 1, item)
        return 0

    def _retry(
        self,
        processing: str,
        pending: str,
        dead: str,
        message_id: str,
        error_text: str,
        max_attempts: int,
        now: int,
    ) -> str:
        for item in list(self.lrange(processing, 0, -1)):
            msg = json.loads(item)
            if str(msg.get("message_id")) != message_id:
                continue
            self.lrem(processing, 1, item)
            msg["last_error"] = error_text
            msg["reserved_at"] = None
            msg["reserved_by"] = None
            attempt = int(msg.get("attempt") or 0)
            if attempt >= max_attempts:
                msg["failed_at"] = now
                self.rpush(dead, json.dumps(msg))
                return json.dumps({"action": "dead_letter", "attempt": attempt})
            self.rpush(pending, json.dumps(msg))
            return json.dumps({"action": "requeued", "attempt": attempt})
        return json.dumps({"action": "missing", "attempt": 0})

    def _reclaim(
        self,
        processing: str,
        pending: str,
        dead: str,
        now: int,
        idle: int,
        max_attempts: int,
    ) -> str:
        reclaimed = 0
        dead_count = 0
        for item in list(self.lrange(processing, 0, -1)):
            msg = json.loads(item)
            reserved_at = int(msg.get("reserved_at") or 0)
            if reserved_at > 0 and (now - reserved_at) >= idle:
                if self.lrem(processing, 1, item) <= 0:
                    continue
                msg["last_error"] = "reclaimed_stale_processing"
                msg["reserved_at"] = None
                msg["reserved_by"] = None
                attempt = int(msg.get("attempt") or 0)
                if attempt >= max_attempts:
                    msg["failed_at"] = now
                    self.rpush(dead, json.dumps(msg))
                    dead_count += 1
                else:
                    self.rpush(pending, json.dumps(msg))
                    reclaimed += 1
        return json.dumps({"reclaimed": reclaimed, "dead_lettered": dead_count})


class RedisQueueResilienceTestCase(unittest.TestCase):
    def test_dequeue_returns_none_on_socket_timeout(self) -> None:
        queue = RedisQueueManager("redis://localhost:6379/0", "q-test")
        mock_client = MagicMock()
        mock_client.blpop.side_effect = RedisTimeoutError("Timeout reading from socket")
        with patch.object(queue, "connection", return_value=mock_client):
            self.assertIsNone(queue.dequeue(timeout_seconds=1))
        self.assertIsNone(queue._client)

    def test_reserve_ack_retry_reclaim_and_dead_letter(self) -> None:
        backend = _ListRedis()
        queue = RedisQueueManager(
            "redis://fake/0",
            "q-reliable",
            max_attempts=2,
            reclaim_idle_seconds=1,
        )
        with patch("lumenfin.queueing.Redis.from_url", return_value=backend):
            message_id = queue.enqueue({"document_id": "doc-1"})
            depths = queue.depths()
            self.assertEqual(depths["pending"], 1)
            self.assertEqual(depths["processing"], 0)

            reserved = queue.reserve(timeout_seconds=1, worker_id="worker-a")
            assert reserved is not None
            self.assertEqual(reserved.message_id, message_id)
            self.assertEqual(reserved.attempt, 1)
            self.assertEqual(queue.depths()["pending"], 0)
            self.assertEqual(queue.depths()["processing"], 1)

            first_ack = queue.ack(message_id, "worker-a")
            second_ack = queue.ack(message_id, "worker-a")
            self.assertTrue(first_ack)
            self.assertFalse(second_ack)
            self.assertEqual(queue.depths()["processing"], 0)

            message_id = queue.enqueue({"document_id": "doc-fail"})
            reserved = queue.reserve(timeout_seconds=1, worker_id="worker-a")
            assert reserved is not None
            result = queue.retry(reserved.message_id, "worker-a", "boom")
            self.assertEqual(result.action, "requeued")
            self.assertEqual(queue.depths()["pending"], 1)

            reserved = queue.reserve(timeout_seconds=1, worker_id="worker-b")
            assert reserved is not None
            self.assertEqual(reserved.attempt, 2)
            result = queue.retry(reserved.message_id, "worker-b", "boom-again")
            self.assertEqual(result.action, "dead_letter")
            depths = queue.depths()
            self.assertEqual(depths["pending"], 0)
            self.assertEqual(depths["processing"], 0)
            self.assertEqual(depths["dead_letter"], 1)
            letter = queue.list_dead_letters(limit=1)[0]
            self.assertEqual(letter["message_id"], reserved.message_id)
            self.assertEqual(letter["attempt"], 2)
            self.assertEqual(letter["last_error"], "boom-again")
            self.assertIn("failed_at", letter)

            message_id = queue.enqueue({"document_id": "doc-stale"})
            reserved = queue.reserve(timeout_seconds=1, worker_id="worker-a")
            assert reserved is not None
            # Force reserved_at into the past for reclaim idle.
            processing = backend.lists[queue.processing_key]
            msg = json.loads(processing[0])
            msg["reserved_at"] = int(time.time()) - 10
            processing[0] = json.dumps(msg)
            reclaimed = queue.reclaim_stale()
            self.assertEqual(reclaimed["reclaimed"], 1)
            self.assertEqual(queue.depths()["pending"], 1)
            self.assertEqual(queue.depths()["processing"], 0)


if __name__ == "__main__":
    unittest.main()
