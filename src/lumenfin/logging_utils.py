from __future__ import annotations

import logging
import re
import sys
import time
from uuid import uuid4

from fastapi import Request

from .stdio import configure_stdio_utf8


_REDACTION_INSTALLED = False
_SECRET_QUERY_PATTERN = re.compile(r"(?i)([?&](?:api[_-]?key|apikey|token|access_token|secret)=)[^&\s]+")
_SECRET_HEADER_PATTERN = re.compile(r"(?i)\b(authorization:\s*bearer\s+)[^\s,;]+")


def redact_secrets(value: object) -> object:
    text = value if isinstance(value, str) else str(value)
    if not _SECRET_QUERY_PATTERN.search(text) and not _SECRET_HEADER_PATTERN.search(text):
        return value
    redacted = _SECRET_QUERY_PATTERN.sub(r"\1[REDACTED]", text)
    redacted = _SECRET_HEADER_PATTERN.sub(r"\1[REDACTED]", redacted)
    return redacted


def install_secret_redaction_filter() -> None:
    global _REDACTION_INSTALLED
    if _REDACTION_INSTALLED:
        return
    original_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = original_factory(*args, **kwargs)
        record.msg = redact_secrets(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact_secrets(arg) for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {key: redact_secrets(val) for key, val in record.args.items()}
        return record

    logging.setLogRecordFactory(record_factory)
    _REDACTION_INSTALLED = True


def configure_logging() -> None:
    configure_stdio_utf8()
    install_secret_redaction_filter()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    # Milvus Lite / gRPC noise: AllocTimestamp is unimplemented in Lite and not actionable.
    for noisy in ("grpc", "grpc._server", "pymilvus", "milvus_lite"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


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
