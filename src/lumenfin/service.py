from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from .checkpoint_store import WorkflowCheckpointRepository
from .config import AppConfig
from .database import JobRepository
from .data_ingest import structured_metrics_to_document_contexts
from .document_ingest import parse_upload_documents
from .graph import LumenFinAgentSystem
from .llm import BaseLLMClient
from .market_data import MarketDataClient
from .providers.registry import ProviderRegistry, build_provider_registry
from .queueing import RedisQueueManager
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
        self.checkpoint_repo = checkpoint_repo or WorkflowCheckpointRepository.from_database_url(
            config.database_url,
            db_path=config.db_path,
        )
        self._llm_client = llm_client
        self._market_data_client = market_data_client
        self._system: LumenFinAgentSystem | None = None
        self._providers: ProviderRegistry | None = None

    @property
    def providers(self) -> ProviderRegistry:
        if self._providers is None:
            self._providers = build_provider_registry(
                self.config,
                llm_client=self._llm_client,
                market_data_client=self._market_data_client,
            )
        return self._providers

    def _build_system(self) -> LumenFinAgentSystem:
        llm_client = self._llm_client or self.providers.llm.client
        market_data_client = self._market_data_client or self.providers.market_data.client
        return LumenFinAgentSystem(
            llm_client=llm_client,
            app_config=self.config,
            market_data_client=market_data_client,
        )

    def _system_for(self, thread_id: str) -> LumenFinAgentSystem:
        if self._system is None:
            self._system = self._build_system()
        return self._system

    def _load_thread_state(self, system: LumenFinAgentSystem, thread_id: str) -> dict | None:
        state = system.get_thread_state(thread_id)
        if state is not None:
            return state
        record = self.checkpoint_repo.get(thread_id)
        if record is None:
            return None
        return system.bootstrap_thread_from_store(thread_id, self.checkpoint_repo)

    def analyze(
        self,
        query: str,
        thread_id: str | None = None,
        export_artifacts: bool = True,
        document_paths: list[str] | None = None,
        structured_metrics: dict[str, dict] | None = None,
    ) -> dict:
        actual_thread_id = thread_id or f"run-{uuid4().hex[:8]}"
        system = self._system_for(actual_thread_id)
        document_contexts: list[dict] = []
        for path in document_paths or []:
            document_contexts.extend(parse_upload_documents(Path(path)))
        if structured_metrics:
            document_contexts.extend(structured_metrics_to_document_contexts(structured_metrics))
        result = system.run(query, thread_id=actual_thread_id, document_contexts=document_contexts)
        self.checkpoint_repo.upsert(
            thread_id=actual_thread_id,
            query=query,
            state=system.get_thread_state(actual_thread_id) or result,
            llm_backend=result.get("llm_backend", system.llm_client.backend_name),
        )
        return self._package_response(actual_thread_id, query, system, result, export_artifacts)

    def clarify(
        self,
        thread_id: str,
        clarification: dict,
        export_artifacts: bool = True,
    ) -> dict:
        system = self._system_for(thread_id)
        prior = self._load_thread_state(system, thread_id)
        if prior is None:
            raise ValueError(f"No checkpoint found for thread_id={thread_id}")
        if prior.get("workflow_status") != "needs_clarification":
            raise ValueError(f"Thread {thread_id} is not awaiting clarification.")
        result = system.resume_with_clarification(thread_id, clarification)
        record = self.checkpoint_repo.get(thread_id) or {}
        query = record.get("query", "")
        self.checkpoint_repo.upsert(
            thread_id=thread_id,
            query=query,
            state=system.get_thread_state(thread_id) or result,
            llm_backend=result.get("llm_backend", system.llm_client.backend_name),
        )
        return self._package_response(thread_id, query, system, result, export_artifacts)

    def get_checkpoint(self, thread_id: str) -> dict | None:
        return self.checkpoint_repo.get(thread_id)

    def _package_response(
        self,
        thread_id: str,
        query: str,
        system: LumenFinAgentSystem,
        result: dict,
        export_artifacts: bool,
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
    ) -> None:
        self.repository.update_job_status(job_id=job_id, status="running")
        try:
            response = self.analyze(
                query=query,
                thread_id=thread_id,
                export_artifacts=export_artifacts,
                document_paths=document_paths,
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
        allowed_suffixes = {".pdf", ".md", ".txt", ".csv", ".xlsx", ".xls", ".json"}
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
