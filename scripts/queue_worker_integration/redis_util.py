from __future__ import annotations

from typing import Any

from lumenfin.queueing import RedisQueueManager


def manager(redis_url: str, queue_name: str, **kwargs: Any) -> RedisQueueManager:
    return RedisQueueManager(redis_url, queue_name, **kwargs)


def queue_depth(redis_url: str, queue_name: str) -> int:
    depths = manager(redis_url, queue_name).depths()
    return int(depths["pending"] + depths["processing"] + depths["legacy"])


def enqueue_index_job(
    redis_url: str,
    queue_name: str,
    *,
    document_id: str,
    tenant_id: str,
    count: int = 1,
) -> list[str]:
    queue = manager(redis_url, queue_name)
    ids: list[str] = []
    for _ in range(count):
        ids.append(
            queue.enqueue(
                {
                    "type": "rag_index",
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                }
            )
        )
    return ids


def purge_queue(redis_url: str, queue_name: str) -> dict[str, int]:
    return manager(redis_url, queue_name).purge()


def observe_queue(redis_url: str, queue_name: str) -> dict[str, Any]:
    depths = manager(redis_url, queue_name).depths()
    return {
        "queue_name": queue_name,
        "depth": int(depths["pending"] + depths["processing"] + depths["legacy"]),
        **depths,
    }
