"""Hash tracked files from a Git revision without reading the working tree."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# Same ordered inputs as scripts/run_ledger_public_dev_ranking.py:_evaluator_source_sha256.
EVALUATOR_SOURCE_RELPATHS = (
    "scripts/run_ledger_public_dev_ranking.py",
    "scripts/run_ledger_public_dev_bm25_sharded.py",
    "src/lumenfin/eval/holdout/ledger.py",
    "src/lumenfin/eval/holdout/ledger_corpus.py",
    "src/lumenfin/eval/holdout/ledger_scoring.py",
    "src/lumenfin/eval/holdout/page_collapse.py",
    "src/lumenfin/eval/holdout/ranking.py",
    "src/lumenfin/eval/financebench/metrics.py",
    "src/lumenfin/eval/financebench/constants.py",
    "src/lumenfin/eval/financebench/retrieval.py",
    "src/lumenfin/rag/chunking.py",
    "src/lumenfin/rag/embeddings.py",
    "src/lumenfin/rag/hybrid_retriever.py",
    "src/lumenfin/rag/milvus_store.py",
    "src/lumenfin/rag/milvus_client.py",
)

HYBRID_ORCHESTRATOR_RELPATH = "scripts/run_ledger_public_dev_hybrid_stratified.py"


def git_blob(repo_root: Path, revision: str, relpath: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{revision}:{relpath}"],
        cwd=repo_root,
    )


def hash_posix_sources_at_revision(
    repo_root: Path,
    revision: str,
    relpaths: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    for rel in relpaths:
        blob = git_blob(repo_root, revision, rel)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        text = blob.decode("utf-8").replace("\r\n", "\n")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def hash_text_file_at_revision(repo_root: Path, revision: str, relpath: str) -> str:
    text = git_blob(repo_root, revision, relpath).decode("utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
