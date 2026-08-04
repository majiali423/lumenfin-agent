from __future__ import annotations

from typing import Any

from redis import Redis


def client(redis_url: str) -> Redis:
    return Redis.from_url(redis_url)


def queue_depth(redis_url: str, queue_name: str) -> int:
    return int(client(redis_url).llen(queue_name) or 0)


def enqueue_index_job(
    redis_url: str,
    queue_name: str,
    *,
    document_id: str,
    tenant_id: str,
    count: int = 1,
) -> None:
    import json

    conn = client(redis_url)
    payload = json.dumps(
        {"type": "rag_index", "document_id": document_id, "tenant_id": tenant_id},
        ensure_ascii=False,
    )
    for _ in range(count):
        conn.rpush(queue_name, payload)


def purge_queue(redis_url: str, queue_name: str) -> int:
    conn = client(redis_url)
    depth = int(conn.llen(queue_name) or 0)
    conn.delete(queue_name)
    return depth


def observe_queue(redis_url: str, queue_name: str) -> dict[str, Any]:
    return {"queue_name": queue_name, "depth": queue_depth(redis_url, queue_name)}
