"""Shared upload limits, chunked reads, and rollback cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

ALLOWED_UPLOAD_SUFFIXES = {
    ".pdf",
    ".md",
    ".txt",
    ".csv",
    ".xlsx",
    ".json",
    ".htm",
    ".html",
}
CHUNK_SIZE = 64 * 1024


class UploadRejected(ValueError):
    http_status = 400


class UploadLimitExceeded(UploadRejected):
    http_status = 413


class UnsupportedUploadType(UploadRejected):
    http_status = 400


def http_status_for_upload_error(exc: BaseException) -> int:
    return int(getattr(exc, "http_status", 400) or 400)


def validate_upload_filename(filename: str) -> str:
    name = Path(filename or "document.pdf").name or "document.pdf"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise UnsupportedUploadType(
            f"Unsupported upload type for '{name}'. Allowed: {sorted(ALLOWED_UPLOAD_SUFFIXES)}"
        )
    return name


async def read_uploads_chunked(
    files: Sequence,
    *,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> list[tuple[str, bytes]]:
    """Read FastAPI UploadFile objects with count/size limits applied while streaming."""
    if len(files) > max_files:
        raise UploadLimitExceeded(
            f"Too many uploads: {len(files)} files exceeds limit of {max_files}."
        )
    payloads: list[tuple[str, bytes]] = []
    total = 0
    for upload in files:
        filename = validate_upload_filename(getattr(upload, "filename", None) or "document.pdf")
        chunks: list[bytes] = []
        size = 0
        while True:
            read = getattr(upload, "read")
            chunk = await read(CHUNK_SIZE)
            if not chunk:
                break
            size += len(chunk)
            total += len(chunk)
            if size > max_file_bytes:
                await _close_quietly(upload)
                raise UploadLimitExceeded(
                    f"Upload '{filename}' is {size} bytes; max is {max_file_bytes}."
                )
            if total > max_total_bytes:
                await _close_quietly(upload)
                raise UploadLimitExceeded(
                    f"Total upload size {total} bytes exceeds limit of {max_total_bytes}."
                )
            chunks.append(chunk)
        await _close_quietly(upload)
        payloads.append((filename, b"".join(chunks)))
    return payloads


def save_upload_payloads(
    files: Iterable[tuple[str, bytes]],
    *,
    upload_dir: Path,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> list[str]:
    items = list(files)
    if len(items) > max_files:
        raise UploadLimitExceeded(
            f"Too many uploads: {len(items)} files exceeds limit of {max_files}."
        )
    total = 0
    for filename, content in items:
        validate_upload_filename(filename)
        size = len(content)
        total += size
        if size > max_file_bytes:
            raise UploadLimitExceeded(
                f"Upload '{filename}' is {size} bytes; max is {max_file_bytes}."
            )
        if total > max_total_bytes:
            raise UploadLimitExceeded(
                f"Total upload size {total} bytes exceeds limit of {max_total_bytes}."
            )
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    try:
        for filename, content in items:
            safe_name = validate_upload_filename(filename)
            target = upload_dir / f"{uuid4().hex[:8]}_{safe_name}"
            target.write_bytes(content)
            saved.append(str(target))
    except Exception:
        cleanup_saved_uploads(saved)
        raise
    return saved


def cleanup_saved_uploads(paths: Iterable[str]) -> None:
    for raw in paths:
        path = Path(raw)
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            continue


async def _close_quietly(upload: object) -> None:
    close = getattr(upload, "close", None)
    if close is None:
        return
    try:
        result = close()
        if hasattr(result, "__await__"):
            await result
    except Exception:
        return
