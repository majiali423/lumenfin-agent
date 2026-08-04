"""Redis job queues with reliable reserve / ACK / retry / reclaim.

Queue layout (per logical queue name ``Q``)::

    Q:pending       LIST of JSON envelopes
    Q:processing    LIST of JSON envelopes
    Q:dead-letter   LIST of JSON envelopes

Reserve uses a Lua script so LPOP(pending)+metadata+RPUSH(processing) is atomic.
Blocking workers poll that script with a short sleep (no destructive BLPOP).

Legacy ``dequeue()`` remains for compatibility but is destructive and deprecated
for long-running workers; prefer ``reserve`` + ``ack`` / ``retry``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from redis import Redis
from redis.exceptions import RedisError, TimeoutError as RedisTimeoutError

RetryAction = Literal["requeued", "dead_letter", "missing"]


@dataclass(frozen=True)
class ReservedMessage:
    message_id: str
    payload: dict[str, Any]
    attempt: int
    created_at: int
    reserved_at: int
    reserved_by: str
    last_error: str | None
    raw: str


@dataclass(frozen=True)
class RetryResult:
    action: RetryAction
    message_id: str
    attempt: int
    last_error: str | None = None


_RESERVE_LUA = """
local pending = KEYS[1]
local processing = KEYS[2]
local now = tonumber(ARGV[1])
local worker_id = ARGV[2]
local raw = redis.call('LPOP', pending)
if not raw then
  return false
end
local ok, msg = pcall(cjson.decode, raw)
if not ok then
  msg = {message_id=tostring(raw), payload={}, attempt=0, created_at=now}
end
msg['attempt'] = tonumber(msg['attempt'] or 0) + 1
msg['reserved_at'] = now
msg['reserved_by'] = worker_id
if msg['message_id'] == nil or msg['message_id'] == '' then
  msg['message_id'] = tostring(now) .. '-' .. tostring(math.random(100000, 999999))
end
if msg['payload'] == nil then
  msg['payload'] = {}
end
if msg['created_at'] == nil then
  msg['created_at'] = now
end
local encoded = cjson.encode(msg)
redis.call('RPUSH', processing, encoded)
return encoded
"""

_ACK_LUA = """
local processing = KEYS[1]
local message_id = ARGV[1]
local items = redis.call('LRANGE', processing, 0, -1)
for i, item in ipairs(items) do
  local ok, msg = pcall(cjson.decode, item)
  if ok and tostring(msg['message_id']) == message_id then
    redis.call('LREM', processing, 1, item)
    return 1
  end
end
return 0
"""

_RETRY_LUA = """
local processing = KEYS[1]
local pending = KEYS[2]
local dead = KEYS[3]
local message_id = ARGV[1]
local error_text = ARGV[2]
local max_attempts = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local items = redis.call('LRANGE', processing, 0, -1)
for i, item in ipairs(items) do
  local ok, msg = pcall(cjson.decode, item)
  if ok and tostring(msg['message_id']) == message_id then
    redis.call('LREM', processing, 1, item)
    msg['last_error'] = error_text
    msg['reserved_at'] = false
    msg['reserved_by'] = false
    local attempt = tonumber(msg['attempt'] or 0)
    if attempt >= max_attempts then
      msg['failed_at'] = now
      redis.call('RPUSH', dead, cjson.encode(msg))
      return cjson.encode({action='dead_letter', attempt=attempt})
    end
    redis.call('RPUSH', pending, cjson.encode(msg))
    return cjson.encode({action='requeued', attempt=attempt})
  end
end
return cjson.encode({action='missing', attempt=0})
"""

_RECLAIM_LUA = """
local processing = KEYS[1]
local pending = KEYS[2]
local dead = KEYS[3]
local now = tonumber(ARGV[1])
local idle = tonumber(ARGV[2])
local max_attempts = tonumber(ARGV[3])
local reclaimed = 0
local dead_count = 0
local items = redis.call('LRANGE', processing, 0, -1)
for i, item in ipairs(items) do
  local ok, msg = pcall(cjson.decode, item)
  if ok then
    local reserved_at = tonumber(msg['reserved_at'] or 0)
    if reserved_at == nil then
      reserved_at = 0
    end
    if reserved_at > 0 and (now - reserved_at) >= idle then
      local removed = redis.call('LREM', processing, 1, item)
      if removed > 0 then
        -- Keep attempt as-is; the next reserve() increments it.
        -- Reclaim only recovers the message so another worker can deliver it.
        msg['last_error'] = 'reclaimed_stale_processing'
        msg['reserved_at'] = false
        msg['reserved_by'] = false
        local attempt = tonumber(msg['attempt'] or 0)
        if attempt >= max_attempts then
          msg['failed_at'] = now
          redis.call('RPUSH', dead, cjson.encode(msg))
          dead_count = dead_count + 1
        else
          redis.call('RPUSH', pending, cjson.encode(msg))
          reclaimed = reclaimed + 1
        end
      end
    end
  end
end
return cjson.encode({reclaimed=reclaimed, dead_lettered=dead_count})
"""


@dataclass
class RedisQueueManager:
    redis_url: str
    queue_name: str
    max_attempts: int = 3
    reclaim_idle_seconds: int = 10
    retry_backoff_seconds: float = 1.0
    _client: Redis | None = field(default=None, init=False, repr=False)

    @property
    def pending_key(self) -> str:
        return f"{self.queue_name}:pending"

    @property
    def processing_key(self) -> str:
        return f"{self.queue_name}:processing"

    @property
    def dead_letter_key(self) -> str:
        return f"{self.queue_name}:dead-letter"

    def connection(self) -> Redis:
        if self._client is None:
            self._client = Redis.from_url(
                self.redis_url,
                socket_timeout=None,
                socket_connect_timeout=5,
                health_check_interval=30,
            )
        return self._client

    def reset_connection(self) -> None:
        self._client = None

    def depths(self) -> dict[str, int]:
        conn = self.connection()
        return {
            "pending": int(conn.llen(self.pending_key) or 0),
            "processing": int(conn.llen(self.processing_key) or 0),
            "dead_letter": int(conn.llen(self.dead_letter_key) or 0),
            # Legacy single-list depth (pre-reliable migrations).
            "legacy": int(conn.llen(self.queue_name) or 0),
        }

    def purge(self) -> dict[str, int]:
        before = self.depths()
        conn = self.connection()
        conn.delete(self.pending_key, self.processing_key, self.dead_letter_key, self.queue_name)
        return before

    def enqueue(self, payload: dict[str, Any]) -> str:
        """Enqueue payload onto the pending list. Returns message_id."""
        now = int(time.time())
        message_id = uuid4().hex
        envelope = {
            "message_id": message_id,
            "payload": payload,
            "attempt": 0,
            "created_at": now,
            "reserved_at": None,
            "reserved_by": None,
            "last_error": None,
        }
        self.connection().rpush(self.pending_key, json.dumps(envelope, ensure_ascii=False))
        return message_id

    def migrate_legacy_messages(self) -> int:
        """Move any pre-reliable plain payloads from ``queue_name`` into pending."""
        moved = 0
        try:
            conn = self.connection()
        except (RedisTimeoutError, TimeoutError, ConnectionError, OSError, RedisError):
            self.reset_connection()
            return 0
        while True:
            try:
                raw = conn.lpop(self.queue_name)
            except (RedisTimeoutError, TimeoutError, ConnectionError, OSError, RedisError):
                self.reset_connection()
                break
            if raw is None:
                break
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {"raw": text}
            if isinstance(parsed, dict) and "payload" in parsed and "message_id" in parsed:
                conn.rpush(self.pending_key, text)
            else:
                self.enqueue(parsed if isinstance(parsed, dict) else {"raw": text})
            moved += 1
        return moved

    def try_reserve(self, worker_id: str) -> ReservedMessage | None:
        now = int(time.time())
        try:
            raw = self.connection().eval(
                _RESERVE_LUA,
                2,
                self.pending_key,
                self.processing_key,
                now,
                worker_id,
            )
        except (RedisTimeoutError, TimeoutError, ConnectionError, OSError, RedisError):
            self.reset_connection()
            return None
        if not raw:
            return None
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        data = json.loads(text)
        return ReservedMessage(
            message_id=str(data["message_id"]),
            payload=dict(data.get("payload") or {}),
            attempt=int(data.get("attempt") or 0),
            created_at=int(data.get("created_at") or now),
            reserved_at=int(data.get("reserved_at") or now),
            reserved_by=str(data.get("reserved_by") or worker_id),
            last_error=data.get("last_error"),
            raw=text,
        )

    def reserve(self, timeout_seconds: int, worker_id: str) -> ReservedMessage | None:
        """Block-ish reserve: poll atomic Lua until timeout."""
        deadline = time.monotonic() + max(0, timeout_seconds)
        while True:
            reserved = self.try_reserve(worker_id)
            if reserved is not None:
                return reserved
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def ack(self, message_id: str, worker_id: str | None = None) -> bool:
        """Remove message from processing. Idempotent: missing id returns False."""
        del worker_id  # ownership is enforced at business layer; ACK matches message_id.
        try:
            removed = self.connection().eval(_ACK_LUA, 1, self.processing_key, message_id)
        except (RedisTimeoutError, TimeoutError, ConnectionError, OSError, RedisError):
            self.reset_connection()
            raise
        return int(removed or 0) == 1

    def retry(self, message_id: str, worker_id: str, error: str) -> RetryResult:
        del worker_id
        now = int(time.time())
        raw = self.connection().eval(
            _RETRY_LUA,
            3,
            self.processing_key,
            self.pending_key,
            self.dead_letter_key,
            message_id,
            error[:2000],
            int(self.max_attempts),
            now,
        )
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        data = json.loads(text)
        return RetryResult(
            action=str(data.get("action") or "missing"),  # type: ignore[arg-type]
            message_id=message_id,
            attempt=int(data.get("attempt") or 0),
            last_error=error,
        )

    def reclaim_stale(self) -> dict[str, int]:
        now = int(time.time())
        try:
            raw = self.connection().eval(
                _RECLAIM_LUA,
                3,
                self.processing_key,
                self.pending_key,
                self.dead_letter_key,
                now,
                int(self.reclaim_idle_seconds),
                int(self.max_attempts),
            )
        except (RedisTimeoutError, TimeoutError, ConnectionError, OSError, RedisError):
            self.reset_connection()
            return {"reclaimed": 0, "dead_lettered": 0}
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        data = json.loads(text)
        return {
            "reclaimed": int(data.get("reclaimed") or 0),
            "dead_lettered": int(data.get("dead_lettered") or 0),
        }

    def list_dead_letters(self, *, limit: int = 50) -> list[dict[str, Any]]:
        raw_items = self.connection().lrange(self.dead_letter_key, 0, max(0, limit - 1))
        out: list[dict[str, Any]] = []
        for raw in raw_items or []:
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            try:
                out.append(json.loads(text))
            except json.JSONDecodeError:
                out.append({"raw": text})
        return out

    def dequeue(self, timeout_seconds: int = 5) -> dict | None:
        """Deprecated destructive consume (legacy BLPOP on bare queue name).

        Prefer ``reserve`` + ``ack``. Kept for older callers/tests.
        """
        try:
            result = self.connection().blpop(self.queue_name, timeout=timeout_seconds)
        except (RedisTimeoutError, TimeoutError, ConnectionError, OSError, RedisError):
            self.reset_connection()
            return None
        if not result:
            # Also try pending for transitional deployments.
            reserved = self.reserve(timeout_seconds=0, worker_id="legacy-dequeue")
            if reserved is None:
                return None
            self.ack(reserved.message_id)
            return reserved.payload
        _, raw_payload = result
        parsed = json.loads(raw_payload.decode("utf-8"))
        if isinstance(parsed, dict) and "payload" in parsed and "message_id" in parsed:
            return dict(parsed.get("payload") or {})
        return parsed
