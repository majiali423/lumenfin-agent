from __future__ import annotations

import logging
import sys
import time
from uuid import uuid4

from fastapi import Request

from .stdio import configure_stdio_utf8


def configure_logging() -> None:
    configure_stdio_utf8()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


async def request_logging_middleware(request: Request, call_next):
    logger = logging.getLogger("lumenfin.api")
    request_id = uuid4().hex[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Request-Id"] = request_id
    return response
