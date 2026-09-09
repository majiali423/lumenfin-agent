"""Chunked upload limits, rollback cleanup, and consistent 4xx across routes."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from lumenfin.api.app import create_app
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.uploads import (
    CHUNK_SIZE,
    UploadLimitExceeded,
    read_uploads_chunked,
    save_upload_payloads,
)
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("APP_ENV", "test")


class _FakeUpload:
    def __init__(self, filename: str, payload: bytes) -> None:
        self.filename = filename
        self._payload = payload
        self._offset = 0
        self.reads = 0
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        self.reads += 1
        if size is None or size < 0:
            chunk = self._payload[self._offset :]
            self._offset = len(self._payload)
            return chunk
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    async def close(self) -> None:
        self.closed = True


class UploadLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "test_artifacts" / f"phase1-upload-{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.root / "uploads"

    def test_chunked_read_stops_before_buffering_the_whole_file(self) -> None:
        payload = b"x" * (CHUNK_SIZE + 32)
        upload = _FakeUpload("big.md", payload)

        async def _run() -> None:
            await read_uploads_chunked(
                [upload],
                max_files=1,
                max_file_bytes=64,
                max_total_bytes=64,
            )

        with self.assertRaises(UploadLimitExceeded):
            import asyncio

            asyncio.run(_run())
        self.assertTrue(upload.closed)
        self.assertEqual(upload.reads, 1)
        self.assertLess(upload._offset, len(payload))

    def test_partial_save_rolls_back_written_files(self) -> None:
        original = Path.write_bytes
        calls = {"n": 0}

        def flaky(self: Path, data: bytes) -> int:
            calls["n"] += 1
            if calls["n"] >= 2:
                raise OSError("disk full")
            return original(self, data)

        with patch.object(Path, "write_bytes", flaky):
            with self.assertRaises(OSError):
                save_upload_payloads(
                    [("ok.md", b"one"), ("later.md", b"two")],
                    upload_dir=self.upload_dir,
                    max_files=5,
                    max_file_bytes=1024,
                    max_total_bytes=1024,
                )
        self.assertEqual(list(self.upload_dir.glob("*")), [])

    def test_three_upload_routes_return_the_same_4xx(self) -> None:
        config = replace(
            build_test_config(self.root),
            max_upload_bytes=16,
            max_upload_files=1,
            max_upload_total_bytes=16,
            rag_enabled=False,
        )
        app = create_app(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        oversized = ("big.md", b"x" * 64, "text/markdown")
        extra = ("second.md", b"ok", "text/markdown")
        routes = (
            "/api/v1/analyze-upload",
            "/api/v1/documents/index",
            "/api/v1/jobs/upload",
        )
        with TestClient(app) as client:
            for route in routes:
                with self.subTest(route=route, kind="too-large"):
                    data = {"query": "Analyze Apple"} if "analyze" in route or "jobs" in route else {}
                    response = client.post(route, data=data, files={"files": oversized})
                    self.assertEqual(response.status_code, 413, response.text)
                with self.subTest(route=route, kind="too-many"):
                    data = {"query": "Analyze Apple"} if "analyze" in route or "jobs" in route else {}
                    response = client.post(
                        route,
                        data=data,
                        files=[("files", ("a.md", b"ok", "text/markdown")), ("files", extra)],
                    )
                    self.assertEqual(response.status_code, 413, response.text)
                with self.subTest(route=route, kind="bad-type"):
                    data = {"query": "Analyze Apple"} if "analyze" in route or "jobs" in route else {}
                    response = client.post(
                        route,
                        data=data,
                        files={"files": ("notes.exe", b"not-allowed", "application/octet-stream")},
                    )
                    self.assertEqual(response.status_code, 400, response.text)
            leftover = list(self.upload_dir.glob("*")) if self.upload_dir.exists() else []
            self.assertEqual(leftover, [])


if __name__ == "__main__":
    unittest.main()
