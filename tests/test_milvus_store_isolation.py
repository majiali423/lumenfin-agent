from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.embeddings import DeterministicEmbeddingProvider
from lumenfin.rag.milvus_store import BM25_SPARSE_FIELD, MilvusRAGStore


def _store_with_client(client: MagicMock) -> MilvusRAGStore:
    store = object.__new__(MilvusRAGStore)
    store.collection_name = "financebench_eval"
    store.embedder = DeterministicEmbeddingProvider(dimension=8)
    store.bm25_enabled = True
    store.client = client
    return store


class MilvusStoreIsolationTests(unittest.TestCase):
    def test_other_filesystem_rename_still_uses_stdlib(self) -> None:
        import lumenfin.rag.milvus_store as milvus_store

        milvus_store._patch_windows_milvus_lite_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "a.txt"
            dst = Path(tmp) / "b.txt"
            src.write_text("keep", encoding="utf-8")
            os.rename(src, dst)
            self.assertTrue(dst.is_file())
            src.write_text("new", encoding="utf-8")
            if os.name == "nt":
                with self.assertRaises(OSError):
                    os.rename(src, dst)

    def test_import_does_not_patch_os_rename(self) -> None:
        import lumenfin.rag.milvus_store as milvus_store

        self.assertFalse(hasattr(os, "_lumenfin_overwrite_rename"))
        self.assertNotIn("milvus_store", inspect_module_name(os.rename))
        self.assertIsNot(os.rename, os.replace)
        milvus_store._patch_windows_milvus_lite_manifest()
        self.assertIsNot(os.rename, os.replace)
        if os.name == "nt":
            from milvus_lite.storage.manifest import Manifest

            self.assertTrue(getattr(Manifest.save, "_lumenfin_windows_save", False))
        import lumenfin.rag.milvus_store as milvus_store

        self.assertFalse(hasattr(os, "_lumenfin_overwrite_rename"))
        self.assertNotIn("milvus_store", inspect_module_name(os.rename))
        milvus_store._patch_windows_milvus_lite_manifest()
        self.assertNotIn("milvus_store", inspect_module_name(os.rename))
        self.assertIsNot(os.rename, os.replace)
        if os.name == "nt":
            from milvus_lite.storage.manifest import Manifest

            self.assertTrue(getattr(Manifest.save, "_lumenfin_windows_save", False))

    def test_index_exception_without_verified_index_fails(self) -> None:
        client = MagicMock()
        client.list_indexes.side_effect = RuntimeError("183 FileExistsError already exists")
        client.describe_index.side_effect = RuntimeError("already exists")
        client.search.side_effect = RuntimeError("index missing")
        store = _store_with_client(client)
        with self.assertRaises(RuntimeError) as ctx:
            store._validate_collection_indexes()
        self.assertIn("unverifiable", str(ctx.exception).lower())

    def test_index_create_exception_requires_verified_bm25_index(self) -> None:
        client = MagicMock()
        client.create_index.side_effect = OSError("WinError 183")
        client.list_indexes.return_value = []
        client.describe_index.side_effect = RuntimeError("missing")
        client.search.side_effect = RuntimeError("missing")
        store = _store_with_client(client)
        with self.assertRaises(RuntimeError):
            store._create_index_verified(MagicMock())
        client.list_indexes.return_value = [{"field_name": BM25_SPARSE_FIELD}]
        client.search.side_effect = None
        store._create_index_verified(MagicMock())
        client = MagicMock()
        client.list_indexes.return_value = [
            {"field_name": "vector"},
            {"field_name": BM25_SPARSE_FIELD},
        ]
        store = _store_with_client(client)
        store._validate_collection_indexes()

    def test_flush_failure_without_searchable_rows_fails(self) -> None:
        client = MagicMock()
        client.flush.side_effect = OSError("WinError 183")
        client.has_collection.return_value = True
        client.get_load_state.return_value = {"state": "Loaded"}
        client.query.side_effect = RuntimeError("not ready")
        client.search.side_effect = RuntimeError("not ready")
        store = _store_with_client(client)
        with self.assertRaises(RuntimeError) as ctx:
            store._wait_until_writes_visible()
        self.assertIn("not searchable", str(ctx.exception))

    def test_flush_failure_with_searchable_rows_continues(self) -> None:
        client = MagicMock()
        client.flush.side_effect = OSError("WinError 183")
        client.has_collection.return_value = True
        client.get_load_state.return_value = {"state": "Loaded"}
        client.query.return_value = [{"id": 1}]
        store = _store_with_client(client)
        store._wait_until_writes_visible()


def inspect_module_name(fn) -> str:
    return str(getattr(fn, "__module__", "") or "")


if __name__ == "__main__":
    unittest.main()
