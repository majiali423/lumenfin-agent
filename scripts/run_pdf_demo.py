from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.config import AppConfig
from lumenfin.reporting import build_run_manifest
from lumenfin.service import LumenFinAnalysisService
from lumenfin.stdio import configure_stdio_utf8


def main() -> None:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Run LumenFin with uploaded PDF(s) for RAG testing.")
    parser.add_argument(
        "--pdf",
        action="append",
        required=True,
        help="Path to PDF file (repeat for multiple files).",
    )
    parser.add_argument(
        "--query",
        default=(
            "Based on the uploaded filing, extract FY2025 revenue drivers, R&D intensity, "
            "and explicit supply-chain risk disclosures for NVIDIA."
        ),
    )
    parser.add_argument("--thread-id", default="t5-pdf-rag")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    pdf_paths = [str((ROOT / p).resolve() if not Path(p).is_absolute() else Path(p)) for p in args.pdf]
    for path in pdf_paths:
        if not Path(path).exists():
            raise SystemExit(f"PDF not found: {path}")

    config = AppConfig.from_env()
    if args.output_dir:
        config = replace(config, output_dir=Path(args.output_dir))

    service = LumenFinAnalysisService(config)
    payload = service.analyze(
        query=args.query,
        thread_id=args.thread_id,
        export_artifacts=True,
        document_paths=pdf_paths,
    )
    result = payload["result"]
    artifacts = payload.get("artifacts", {})
    manifest = build_run_manifest(
        result,
        thread_id=args.thread_id,
        llm_backend=result.get("llm_backend"),
        artifact_paths=artifacts,
        embedding_provider=config.embedding_provider,
        rag_enabled=config.rag_enabled,
        market_provider=config.market_data_provider,
    )
    sources = manifest.get("data_sources", {})

    print(result.get("final_report", ""))
    print(f"\n[LLM] backend={result.get('llm_backend', 'unknown')}")
    print(f"[RAG] status={sources.get('rag', 'unknown')} pdf_uploaded={sources.get('pdf_uploaded')}")
    rag_chunks = sum(len(hits) for hits in (result.get("rag_evidence") or {}).values())
    print(f"[RAG] cited_chunks={rag_chunks} indexed={result.get('rag_index_stats', {}).get('chunks_indexed', 0)}")
    if artifacts.get("report_path"):
        print(f"[Artifacts] report={artifacts['report_path']}")
        print(f"[Artifacts] state={artifacts.get('state_path')}")
        print(f"[Artifacts] manifest={artifacts.get('manifest_path')}")


if __name__ == "__main__":
    main()
