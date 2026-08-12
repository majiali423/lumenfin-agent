from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.database import RagDocumentRepository
from lumenfin.rag.embeddings import DeterministicEmbeddingProvider
from lumenfin.rag.milvus_store import MilvusRAGStore
from lumenfin.rag.rebuild import build_rebuild_manifest, rebuild_vector_index


class RagVectorRebuildTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "test_artifacts" / f"rag-rebuild-{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "rag.db"
        self.repository = RagDocumentRepository(
            f"sqlite:///{self.db_path.as_posix()}", db_path=self.db_path
        )
        self.store = MilvusRAGStore(
            str(self.root / "target.db"),
            DeterministicEmbeddingProvider(),
            collection_name="rebuild_target",
        )

    def tearDown(self) -> None:
        self.store.close()
        self.repository.engine.dispose()
        shutil.rmtree(self.root, ignore_errors=True)

    def _chunk(self, *, document_id: str, token: str) -> dict:
        return {
            "chunk_id": f"{document_id}-chunk-1",
            "document_id": f"{document_id}:context",
            "filename": f"{document_id}.md",
            "page": 1,
            "text": f"{token} financial evidence for Apple.",
            "companies": ["Apple"],
            "chunk_type": "narrative",
            "char_count": len(f"{token} financial evidence for Apple."),
        }

    def _ready_document(self, *, document_id: str, chunk_count: int = 1) -> dict:
        chunk = self._chunk(document_id=document_id, token="REBUILD_SOURCE_TOKEN")
        self.repository.upsert_document(
            document_id=document_id,
            tenant_id="tenant-a",
            filename=f"{document_id}.md",
            content_hash=f"hash-{document_id}",
            index_status="ready",
            contexts=[],
            chunk_count=chunk_count,
            source_path=str(self.root / f"{document_id}.md"),
        )
        self.repository.replace_chunks(
            source_document_id=document_id,
            tenant_id="tenant-a",
            chunks=[chunk],
            content_hash=f"hash-{document_id}",
        )
        return chunk

    def _seed_target(self) -> None:
        self.store.index_chunks(
            [self._chunk(document_id="old-target", token="PRESERVE_TARGET_TOKEN")],
            tenant_id="tenant-old",
            source_document_id="old-target",
            content_hash="old-hash",
        )

    def test_rebuild_from_durable_chunks_and_verify_first_search(self) -> None:
        self._ready_document(document_id="source-doc")

        result = rebuild_vector_index(
            repository=self.repository,
            store=self.store,
            reset_collection=True,
        )

        self.assertEqual(result["documents_indexed"], 1)
        self.assertEqual(result["documents_verified"], 1)
        self.assertEqual(result["bm25_documents_verified"], 1)
        self.assertTrue(result["bm25_enabled"])
        self.assertEqual(result["chunks_indexed"], 1)
        hits = self.store.vector_search(
            "REBUILD_SOURCE_TOKEN",
            tenant_id="tenant-a",
            source_document_ids=["source-doc"],
        )
        self.assertTrue(hits)

    def test_preflight_count_mismatch_does_not_reset_target(self) -> None:
        self._seed_target()
        self._ready_document(document_id="broken-doc", chunk_count=2)

        with self.assertRaisesRegex(RuntimeError, "metadata chunk_count=2, durable chunks=1"):
            rebuild_vector_index(
                repository=self.repository,
                store=self.store,
                reset_collection=True,
            )

        preserved = self.store.vector_search(
            "PRESERVE_TARGET_TOKEN",
            tenant_id="tenant-old",
            source_document_ids=["old-target"],
        )
        self.assertTrue(preserved)

    def test_empty_manifest_refuses_reset(self) -> None:
        self._seed_target()
        self.assertEqual(build_rebuild_manifest(self.repository), [])

        with self.assertRaisesRegex(RuntimeError, "no ready documents"):
            rebuild_vector_index(
                repository=self.repository,
                store=self.store,
                reset_collection=True,
            )

        preserved = self.store.vector_search(
            "PRESERVE_TARGET_TOKEN",
            tenant_id="tenant-old",
            source_document_ids=["old-target"],
        )
        self.assertTrue(preserved)

    def test_tenant_scoped_rebuild_cannot_reset_shared_collection(self) -> None:
        self._seed_target()
        self._ready_document(document_id="tenant-doc")

        with self.assertRaisesRegex(ValueError, "tenant-scoped rebuild cannot reset"):
            rebuild_vector_index(
                repository=self.repository,
                store=self.store,
                tenant_id="tenant-a",
                reset_collection=True,
            )

        preserved = self.store.vector_search(
            "PRESERVE_TARGET_TOKEN",
            tenant_id="tenant-old",
            source_document_ids=["old-target"],
        )
        self.assertTrue(preserved)


if __name__ == "__main__":
    unittest.main()
