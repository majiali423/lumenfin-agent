from __future__ import annotations

from pathlib import Path
from threading import RLock
from uuid import uuid4

from .checkpoint_store import WorkflowCheckpointRepository
from .config import AppConfig
from .database import JobRepository, RagDocumentRepository
from .data_ingest import structured_metrics_to_document_contexts
from .document_ingest import parse_upload_documents
from .graph import LumenFinAgentSystem
from .llm import BaseLLMClient, fork_llm_client
from .market_data import MarketDataClient
from .providers.registry import ProviderRegistry, build_provider_registry
from .queueing import RedisQueueManager
from .rag.factory import build_document_indexer, build_rag_store
from .rag.indexer import IndexReceipt, summarize_index_receipts
from .reporting import export_run_artifacts


class LumenFinAnalysisService:
    def __init__(
        self,
        config: AppConfig,
        *,
        llm_client: BaseLLMClient | None = None,
        market_data_client: MarketDataClient | None = None,
        checkpoint_repo: WorkflowCheckpointRepository | None = None,
    ) -> None:
        self.config = config
        self.repository = JobRepository(config.database_url, db_path=config.db_path)
        self.rag_repository = RagDocumentRepository(config.database_url, db_path=config.db_path)
        self.checkpoint_repo = checkpoint_repo or WorkflowCheckpointRepository.from_database_url(
            config.database_url,
            db_path=config.db_path,
        )
        self._llm_client = llm_client
        self._market_data_client = market_data_client
        self._providers: ProviderRegistry | None = None
        self._resource_lock = RLock()
        self._rag_store = None
        self._indexer = None

    @property
    def providers(self) -> ProviderRegistry:
        if self._providers is None:
            with self._resource_lock:
                if self._providers is None:
                    self._providers = build_provider_registry(
                        self.config,
                        llm_client=self._llm_client,
                        market_data_client=self._market_data_client,
                    )
        return self._providers

    def _build_system(self) -> LumenFinAgentSystem:
        provider_llm_client = self._llm_client or self.providers.llm.client
        llm_client = fork_llm_client(provider_llm_client)
        assert llm_client is not None
        market_data_client = self._market_data_client or self.providers.market_data.client
        rag_store, document_indexer = self._rag_resources()
        return LumenFinAgentSystem(
            llm_client=llm_client,
            app_config=self.config,
            market_data_client=market_data_client,
            rag_store=rag_store,
            document_indexer=document_indexer,
        )

    def _system_for(self, thread_id: str) -> LumenFinAgentSystem:
        # Execution state (memory, checkpointer, audit log, graph runtime) is
        # request scoped. Only provider clients and the RAG infrastructure are
        # shared; they do not contain per-run FinanceState.
        return self._build_system()

    def _rag_resources(self):
        if self._indexer is not None:
            return self._rag_store, self._indexer
        with self._resource_lock:
            if self._indexer is None:
                self._rag_store = build_rag_store(self.config)
                self._indexer = build_document_indexer(
                    self.config,
                    rag_store=self._rag_store,
                    repository=self.rag_repository,
                )
        return self._rag_store, self._indexer

    def _document_indexer(self, system: LumenFinAgentSystem | None = None):
        _, indexer = self._rag_resources()
        return indexer

    def _load_thread_state(
        self,
        system: LumenFinAgentSystem,
        thread_id: str,
        record: dict | None = None,
    ) -> dict | None:
        state = system.get_thread_state(thread_id)
        if state is not None:
            return state
        record = record or self.checkpoint_repo.get(thread_id)
        if record is None:
            return None
        return system.bootstrap_thread_from_store(thread_id, self.checkpoint_repo, record=record)

    def index_document_paths(
        self,
        document_paths: list[str],
        *,
        tenant_id: str | None = None,
        system: LumenFinAgentSystem | None = None,
    ) -> list[IndexReceipt]:
        indexer = self._document_indexer(system)
        return indexer.index_paths(document_paths, tenant_id=tenant_id or self.config.rag_tenant_id)

    def enqueue_document_paths(
        self,
        document_paths: list[str],
        *,
        tenant_id: str | None = None,
        system: LumenFinAgentSystem | None = None,
    ) -> list[IndexReceipt]:
        indexer = self._document_indexer(system)
        return indexer.enqueue_paths(document_paths, tenant_id=tenant_id or self.config.rag_tenant_id)

    def process_document_index(
        self,
        document_id: str,
        *,
        tenant_id: str | None = None,
        system: LumenFinAgentSystem | None = None,
    ) -> IndexReceipt:
        indexer = self._document_indexer(system)
        return indexer.process_pending(document_id, tenant_id=tenant_id or self.config.rag_tenant_id)

    def enqueue_index_job(self, document_id: str, *, tenant_id: str | None = None) -> bool:
        if not self.config.redis_url:
            return False
        queue = RedisQueueManager(self.config.redis_url, self.config.redis_index_queue_name)
        queue.enqueue(
            {
                "type": "rag_index",
                "document_id": document_id,
                "tenant_id": tenant_id or self.config.rag_tenant_id,
            }
        )
        return True

    def get_document_status(self, document_id: str, *, tenant_id: str | None = None) -> dict | None:
        return self.rag_repository.get_document(
            document_id,
            tenant_id=tenant_id or self.config.rag_tenant_id,
        )

    def analyze(
        self,
        query: str,
        thread_id: str | None = None,
        export_artifacts: bool = True,
        document_paths: list[str] | None = None,
        structured_metrics: dict[str, dict] | None = None,
        document_ids: list[str] | None = None,
        tenant_id: str | None = None,
        output_format: str | None = None,
    ) -> dict:
        actual_thread_id = thread_id or f"run-{uuid4().hex[:8]}"
        base_checkpoint = self.checkpoint_repo.get(actual_thread_id)
        expected_revision = int(base_checkpoint["revision"]) if base_checkpoint else 0
        system = self._system_for(actual_thread_id)
        tenant = (tenant_id or self.config.rag_tenant_id).strip() or "default"
        document_contexts: list[dict] = []
        rag_index_stats: dict = {}
        rag_document_ids: list[str] = list(document_ids or [])

        if document_ids:
            contexts, statuses = system.document_indexer.load_contexts_for_documents(
                document_ids,
                tenant_id=tenant,
                require_ready=self.config.rag_require_ready,
            )
            document_contexts.extend(contexts)
            rag_index_stats = {
                "mode": "async_on_upload",
                "chunks_indexed": sum(int(item.get("chunk_count") or 0) for item in statuses if item.get("index_status") == "ready"),
                "documents_indexed": sum(1 for item in statuses if item.get("index_status") == "ready"),
                "document_ids": list(document_ids),
                "statuses": statuses,
                "search_only": True,
            }

        if document_paths:
            if self.config.rag_index_mode == "async_on_upload" and self.config.rag_enabled:
                receipts = self.index_document_paths(document_paths, tenant_id=tenant, system=system)
                failed = [item for item in receipts if item["status"] == "failed"]
                if failed and self.config.rag_require_ready:
                    details = "; ".join(f"{item['filename']}: {item['error']}" for item in failed)
                    raise ValueError(f"Document indexing failed: {details}")
                for receipt in receipts:
                    if receipt["status"] in {"ready", "skipped_duplicate"}:
                        document_contexts.extend(receipt["contexts"])
                        rag_document_ids.append(receipt["document_id"])
                rag_index_stats = summarize_index_receipts(receipts)
                rag_index_stats["search_only"] = True
            else:
                for path in document_paths:
                    try:
                        document_contexts.extend(parse_upload_documents(Path(path)))
                    except ValueError as exc:
                        raise ValueError(f"Upload parse failed for {Path(path).name}: {exc}") from exc

        if structured_metrics:
            document_contexts.extend(structured_metrics_to_document_contexts(structured_metrics))

        result = system.run(
            query,
            thread_id=actual_thread_id,
            document_contexts=document_contexts,
            rag_index_stats=rag_index_stats or None,
            rag_document_ids=rag_document_ids or None,
            rag_tenant_id=tenant,
            output_format=output_format,
        )
        committed_checkpoint = self.checkpoint_repo.upsert(
            thread_id=actual_thread_id,
            query=query,
            state=system.get_thread_state(actual_thread_id) or result,
            llm_backend=result.get("llm_backend", system.llm_client.backend_name),
            expected_revision=expected_revision,
        )
        packaged = self._package_response(
            actual_thread_id,
            query,
            system,
            result,
            export_artifacts,
            checkpoint=committed_checkpoint,
        )
        if rag_index_stats:
            packaged["rag_index"] = {
                "document_ids": rag_document_ids,
                "stats": rag_index_stats,
                "tenant_id": tenant,
            }
        return packaged

    def clarify(
        self,
        thread_id: str,
        clarification: dict,
        export_artifacts: bool = True,
    ) -> dict:
        system = self._system_for(thread_id)
        record = self.checkpoint_repo.get(thread_id)
        prior = self._load_thread_state(system, thread_id, record=record)
        if prior is None:
            raise ValueError(f"No checkpoint found for thread_id={thread_id}")
        if prior.get("workflow_status") != "needs_clarification":
            raise ValueError(f"Thread {thread_id} is not awaiting clarification.")
        result = system.resume_with_clarification(thread_id, clarification)
        assert record is not None
        query = record.get("query", "")
        committed_checkpoint = self.checkpoint_repo.upsert(
            thread_id=thread_id,
            query=query,
            state=system.get_thread_state(thread_id) or result,
            llm_backend=result.get("llm_backend", system.llm_client.backend_name),
            expected_revision=int(record["revision"]),
        )
        return self._package_response(
            thread_id,
            query,
            system,
            result,
            export_artifacts,
            checkpoint=committed_checkpoint,
        )

    def get_checkpoint(self, thread_id: str) -> dict | None:
        return self.checkpoint_repo.get(thread_id)

    def _package_response(
        self,
        thread_id: str,
        query: str,
        system: LumenFinAgentSystem,
        result: dict,
        export_artifacts: bool,
        checkpoint: dict | None = None,
    ) -> dict:
        artifacts: dict[str, str] = {}
        workflow_status = result.get("workflow_status", "completed")
        if export_artifacts and workflow_status in {
            "completed",
            "incomplete_data",
            "needs_clarification",
            "blocked_by_guardrail",
        }:
            artifacts = export_run_artifacts(
                result=result,
                output_dir=self.config.output_dir,
                thread_id=thread_id,
                llm_backend=result.get("llm_backend", system.llm_client.backend_name),
                embedding_provider=self.config.embedding_provider,
                rag_enabled=self.config.rag_enabled,
                market_provider=self.config.market_data_provider,
            )
        if checkpoint is None:
            checkpoint = self.checkpoint_repo.get(thread_id)
        return {
            "thread_id": thread_id,
            "query": query or (checkpoint or {}).get("query", ""),
            "llm_backend": result.get("llm_backend", system.llm_client.backend_name),
            "workflow_status": workflow_status,
            "clarification_questions": result.get("clarification_questions", []),
            "checkpoint": checkpoint,
            "provider_health": self.providers.health_report(),
            "result": result,
            "artifacts": artifacts,
        }

    def submit_job(self, query: str, thread_id: str | None = None) -> dict:
        actual_thread_id = thread_id or f"run-{uuid4().hex[:8]}"
        job_id = f"job-{uuid4().hex[:10]}"
        self.repository.create_job(job_id=job_id, thread_id=actual_thread_id, query=query)
        return {"job_id": job_id, "thread_id": actual_thread_id, "status": "pending"}

    def enqueue_job(
        self,
        job_id: str,
        query: str,
        thread_id: str,
        export_artifacts: bool = True,
        document_paths: list[str] | None = None,
        output_format: str | None = None,
    ) -> bool:
        if not self.config.redis_url:
            return False
        queue = RedisQueueManager(self.config.redis_url, self.config.redis_queue_name)
        queue.enqueue(
            {
                "job_id": job_id,
                "query": query,
                "thread_id": thread_id,
                "export_artifacts": export_artifacts,
                "document_paths": document_paths or [],
                "output_format": output_format,
            }
        )
        return True

    def run_job(
        self,
        job_id: str,
        query: str,
        thread_id: str,
        export_artifacts: bool = True,
        document_paths: list[str] | None = None,
        output_format: str | None = None,
    ) -> None:
        self.repository.update_job_status(job_id=job_id, status="running")
        try:
            response = self.analyze(
                query=query,
                thread_id=thread_id,
                export_artifacts=export_artifacts,
                document_paths=document_paths,
                output_format=output_format,
            )
            self.repository.update_job_status(
                job_id=job_id,
                status="completed",
                llm_backend=response["llm_backend"],
                result=response["result"],
                artifacts=response["artifacts"],
            )
        except Exception as exc:
            self.repository.update_job_status(
                job_id=job_id,
                status="failed",
                error_message=str(exc),
            )
            raise

    def get_job(self, job_id: str) -> dict | None:
        return self.repository.get_job(job_id)

    def list_jobs(self, limit: int = 20) -> list[dict]:
        return self.repository.list_jobs(limit=limit)

    def save_uploaded_files(self, files: list[tuple[str, bytes]]) -> list[str]:
        # Keep in sync with document_ingest parsers (.xls is accepted by older clients but not parsed).
        allowed_suffixes = {".pdf", ".md", ".txt", ".csv", ".xlsx", ".json", ".htm", ".html"}
        if len(files) > self.config.max_upload_files:
            raise ValueError(
                f"Too many uploads: {len(files)} files exceeds limit of {self.config.max_upload_files}."
            )
        self.config.upload_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        for filename, content in files:
            suffix = Path(filename).suffix.lower()
            if suffix not in allowed_suffixes:
                raise ValueError(f"Unsupported upload type for '{filename}'. Allowed: {sorted(allowed_suffixes)}")
            if len(content) > self.config.max_upload_bytes:
                raise ValueError(
                    f"Upload '{filename}' is {len(content)} bytes; max is {self.config.max_upload_bytes}."
                )
            unique_name = f"{uuid4().hex[:8]}_{Path(filename).name}"
            target_path = self.config.upload_dir / unique_name
            target_path.write_bytes(content)
            saved_paths.append(str(target_path))
        return saved_paths
