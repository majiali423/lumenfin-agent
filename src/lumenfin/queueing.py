from __future__ import annotations

import json
from dataclasses import dataclass, field

from redis import Redis
from redis.exceptions import RedisError, TimeoutError as RedisTimeoutError


@dataclass
class RedisQueueManager:
    redis_url: str
    queue_name: str
    _client: Redis | None = field(default=None, init=False, repr=False)

    def connection(self) -> Redis:
        if self._client is None:
            # socket_timeout must exceed BLPOP timeout or idle waits raise TimeoutError.
            self._client = Redis.from_url(
                self.redis_url,
                socket_timeout=None,
                socket_connect_timeout=5,
                health_check_interval=30,
            )
        return self._client

    def enqueue(self, payload: dict) -> None:
        self.connection().rpush(self.queue_name, json.dumps(payload, ensure_ascii=False))

    def dequeue(self, timeout_seconds: int = 5) -> dict | None:
        try:
            result = self.connection().blpop(self.queue_name, timeout=timeout_seconds)
        except (RedisTimeoutError, TimeoutError, ConnectionError, OSError, RedisError):
            # Transient network/socket issues must not kill long-running workers.
            self._client = None
            return None
        if not result:
            return None
        _, raw_payload = result
        return json.loads(raw_payload.decode("utf-8"))
