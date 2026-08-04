from __future__ import annotations

import os
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.responses import RedirectResponse

from ..config import AppConfig
from ..checkpoint_store import CheckpointConflictError
from ..llm import BaseLLMClient, shutdown_llm_http_clients
from ..logging_utils import configure_logging, request_logging_middleware
from ..market_data import MarketDataClient, probe_market_provider
from ..provider_resilience import close_shared_http_clients
from ..reporting import build_run_manifest, load_run_manifest
from ..service import LumenFinAnalysisService
from .auth import build_api_key_dependency
from .schemas import (
    AnalyzeDataRequest,
    AnalyzeRequest,
    AnalyzeResponse,
    ClarifyRequest,
    DocumentIndexResponse,
    DocumentStatusResponse,
    HealthResponse,
    JobResponse,
    ProviderProbeRequest,
    ProviderProbeResponse,
    SubmitJobRequest,
    SubmitJobResponse,
)


def _package_version() -> str:
    try:
        return metadata.version("lumenfin-agent")
    except metadata.PackageNotFoundError:
        return "0.1.0rc1"


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    yield
    shutdown_llm_http_clients()
    close_shared_http_clients()


def create_app(
    config: AppConfig | None = None,
    *,
    llm_client: BaseLLMClient | None = None,
    market_data_client: MarketDataClient | None = None,
) -> FastAPI:
    configure_logging()
    app_config = config or AppConfig.from_env()
    if app_config.requires_api_key() and not app_config.api_key:
        raise RuntimeError(
            "MAS_API_KEY is required when APP_ENV is not dev/test. "
            "Set MAS_API_KEY or use APP_ENV=dev for local demos."
        )
    service = LumenFinAnalysisService(
        app_config,
        llm_client=llm_client,
        market_data_client=market_data_client,
    )
    auth_dependency = build_api_key_dependency(
        app_config.api_key,
        require_key=app_config.requires_api_key(),
    )

    app = FastAPI(
        title="LumenFin API",
        version=_package_version(),
        description="Deployable multi-agent finance research and compliance API powered by LangGraph and DeepSeek.",
        lifespan=_app_lifespan,
    )
    app.middleware("http")(request_logging_middleware)

    static_dir = Path(__file__).resolve().parent.parent.parent.parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    from starlette.responses import Response

    @app.middleware("http")
    async def _cache_control(request, call_next):
        response: Response = await call_next(request)
        if request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/static/index.html")

    @app.middleware("http")
    async def _worker_identity_headers(request, call_next):
        response = await call_next(request)
        worker_id = (os.getenv("MAS_WORKER_ID") or "").strip()
        if worker_id:
            response.headers["X-Worker-Id"] = worker_id
            response.headers["X-Worker-Pid"] = str(os.getpid())
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        backend = "deepseek" if app_config.llm.api_key else "local-fallback"
        market_client = service.providers.market_data.client
        market_probe = probe_market_provider(market_client)
        worker_id = (os.getenv("MAS_WORKER_ID") or "").strip() or None
        return HealthResponse(
            status="ok",
            llm_backend=backend,
            llm_configured=bool(app_config.llm.api_key),
            market_provider=app_config.market_data_provider,
            market_provider_ok=bool(market_probe.get("ok")),
            embedding_provider=app_config.embedding_provider,
            rag_enabled=app_config.rag_enabled,
            pid=os.getpid() if app_config.app_env in {"dev", "test", "integration"} else None,
            worker_id=worker_id if app_config.app_env in {"dev", "test", "integration"} else None,
        )

    if app_config.app_env in {"test", "integration", "dev"}:

        @app.post("/api/v1/provider-resilience/probe", response_model=ProviderProbeResponse)
        def provider_resilience_probe(
            payload: ProviderProbeRequest,
            _: None = Depends(auth_dependency),
        ) -> ProviderProbeResponse:
            """Single logical LLM call for dual-API provider resilience harnesses."""
            import threading
            from uuid import uuid4

            from ..llm import DeepSeekChatClient, LocalFallbackLLMClient, ResilientLLMClient, fork_llm_client
            from ..provider_resilience import (
                ProviderCallContext,
                classify_provider_exception,
                summarize_provider_trace,
            )

            request_id = f"probe-{uuid4().hex[:12]}"
            thread_id = f"thread-{threading.get_ident()}"
            context = ProviderCallContext.create(
                request_id=request_id,
                thread_id=thread_id,
                deadline_seconds=float(app_config.analysis_deadline_seconds),
            )
            context.sleep = lambda _: None
            base = fork_llm_client(service.providers.llm.client)
            deepseek = base
            if isinstance(base, ResilientLLMClient):
                deepseek = base.primary
            if payload.max_attempts is not None and isinstance(deepseek, DeepSeekChatClient):
                from dataclasses import replace

                deepseek.settings = replace(
                    deepseek.settings, max_retries=max(1, int(payload.max_attempts))
                )
            if isinstance(deepseek, DeepSeekChatClient):
                deepseek.extra_headers["X-LumenFin-Scenario"] = payload.scenario
            bind = getattr(base, "bind_call_context", None)
            if callable(bind):
                bind(context)

            use_fallback = payload.scenario in {"always_503", "timeout"}
            client = (
                ResilientLLMClient(
                    primary=deepseek if isinstance(deepseek, DeepSeekChatClient) else base,
                    fallback=LocalFallbackLLMClient(),
                    allow_fallback=True,
                )
                if use_fallback
                else (deepseek if isinstance(deepseek, DeepSeekChatClient) else base)
            )
            if hasattr(client, "bind_call_context") and callable(client.bind_call_context):
                client.bind_call_context(context)

            ok = False
            degraded = False
            fallback = False
            attempts = 0
            error_class = None
            text = ""
            try:
                text = client.chat("You are a probe.", payload.prompt)
                ok = bool(text)
                degraded = bool(getattr(client, "degraded", False) or getattr(client, "used_fallback", False))
                fallback = bool(getattr(client, "used_fallback", False))
                attempts = int(
                    getattr(client, "primary_attempts", None)
                    or getattr(deepseek, "last_attempts", 0)
                    or 0
                )
            except Exception as exc:  # noqa: BLE001
                error_class = classify_provider_exception(exc)
                attempts = int(getattr(deepseek, "last_attempts", 0) or 0)
                ok = False

            trace = list(getattr(deepseek, "last_trace", None) or context.trace_sink or [])
            client_id = None
            if isinstance(deepseek, DeepSeekChatClient):
                shared = deepseek._client(deepseek._policy().httpx_timeout())
                client_id = f"{id(shared)}"

            return ProviderProbeResponse(
                ok=ok,
                degraded=degraded,
                fallback=fallback,
                attempts=attempts,
                error_class=error_class,
                text_preview=(text or "")[:120],
                request_id=request_id,
                thread_id=thread_id,
                worker_id=(os.getenv("MAS_WORKER_ID") or "").strip() or None,
                pid=os.getpid(),
                client_id=client_id,
                trace=trace,
                provider_call_summary=summarize_provider_trace(trace),
            )

    @app.get("/api/v1/config")
    def get_config(_: None = Depends(auth_dependency)) -> dict:
        return {
            "output_dir": str(app_config.output_dir),
            "upload_dir": str(app_config.upload_dir),
            "db_path": str(app_config.db_path),
            "database_url": app_config.database_url,
            "host": app_config.host,
            "port": app_config.port,
            "deepseek_model": app_config.llm.model,
            "deepseek_enabled": bool(app_config.llm.api_key),
            "api_key_enabled": bool(app_config.api_key),
            "redis_enabled": bool(app_config.redis_url),
            "neo4j_enabled": bool(app_config.neo4j_uri),
            "rag_enabled": app_config.rag_enabled,
            "rag_index_mode": app_config.rag_index_mode,
            "rag_tenant_id": app_config.rag_tenant_id,
            "rag_require_ready": app_config.rag_require_ready,
            "milvus_uri": app_config.milvus_uri,
            "embedding_provider": app_config.embedding_provider,
            "market_data_provider": app_config.market_data_provider,
        }

    def _compact_state(result: dict) -> dict:
        return {
            "run_id": result.get("run_id"),
            "thread_id": result.get("thread_id"),
            "companies": result.get("companies"),
            "workflow_status": result.get("workflow_status"),
            "degraded_mode": result.get("degraded_mode"),
            "data_mode": result.get("data_mode") or app_config.data_mode,
            "llm_backend": result.get("llm_backend"),
            "clarification_questions": result.get("clarification_questions", []),
        }

    def _to_response(payload: dict, *, include_state: bool = False) -> AnalyzeResponse:
        result = payload["result"]
        artifacts = payload.get("artifacts", {})
        run_manifest = load_run_manifest(artifacts) or build_run_manifest(
            result,
            thread_id=payload["thread_id"],
            llm_backend=payload.get("llm_backend"),
            artifact_paths=artifacts,
            embedding_provider=app_config.embedding_provider,
            rag_enabled=app_config.rag_enabled,
            market_provider=app_config.market_data_provider,
        )
        checkpoint = payload.get("checkpoint")
        if checkpoint and "state" in checkpoint:
            checkpoint = {
                "thread_id": checkpoint.get("thread_id"),
                "workflow_status": checkpoint.get("workflow_status"),
                "last_node": checkpoint.get("last_node"),
                "clarification_questions": checkpoint.get("clarification_questions"),
                "revision": checkpoint.get("revision"),
                "created_at": checkpoint.get("created_at"),
                "updated_at": checkpoint.get("updated_at"),
            }
        return AnalyzeResponse(
            thread_id=payload["thread_id"],
            llm_backend=payload["llm_backend"],
            workflow_status=payload.get("workflow_status", result.get("workflow_status", "completed")),
            clarification_questions=result.get("clarification_questions", []),
            final_report=result.get("final_report", ""),
            executive_summary=result.get("executive_summary"),
            compliance_summary=result.get("compliance_summary"),
            audit_log=result.get("audit_log", []),
            artifacts=artifacts,
            state=result if include_state else _compact_state(result),
            chart_data=result.get("chart_data"),
            run_telemetry=result.get("run_telemetry"),
            run_manifest=run_manifest,
            provider_health=payload.get("provider_health"),
            checkpoint=checkpoint,
            degraded=bool(payload.get("degraded")),
            provider_degraded=payload.get("provider_degraded"),
            provider_call_summary=payload.get("provider_call_summary"),
        )

    @app.post("/api/v1/analyze", response_model=AnalyzeResponse)
    def analyze(payload: AnalyzeRequest, _: None = Depends(auth_dependency)) -> AnalyzeResponse:
        try:
            response = service.analyze(
                query=payload.query,
                thread_id=payload.thread_id,
                export_artifacts=payload.export_artifacts,
                output_format=payload.output_format,
            )
        except CheckpointConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_response(response, include_state=payload.include_state)

    @app.post("/api/v1/clarify", response_model=AnalyzeResponse)
    def clarify(payload: ClarifyRequest, _: None = Depends(auth_dependency)) -> AnalyzeResponse:
        try:
            response = service.clarify(
                thread_id=payload.thread_id,
                clarification=payload.clarification,
                export_artifacts=payload.export_artifacts,
            )
        except CheckpointConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_response(response, include_state=payload.include_state)

    @app.post("/api/v1/analyze-data", response_model=AnalyzeResponse)
    def analyze_data(payload: AnalyzeDataRequest, _: None = Depends(auth_dependency)) -> AnalyzeResponse:
        try:
            response = service.analyze(
                query=payload.query,
                thread_id=payload.thread_id,
                export_artifacts=payload.export_artifacts,
                structured_metrics=payload.company_metrics,
                output_format=payload.output_format,
            )
        except CheckpointConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_response(response, include_state=payload.include_state)

    @app.post("/api/v1/analyze-upload", response_model=AnalyzeResponse)
    async def analyze_upload(
        query: str = Form(...),
        thread_id: str | None = Form(default=None),
        export_artifacts: bool = Form(default=True),
        include_state: bool = Form(default=False),
        output_format: str | None = Form(default=None),
        files: list[UploadFile] = File(...),
        _: None = Depends(auth_dependency),
    ) -> AnalyzeResponse:
        try:
            uploaded_files = [
                (upload.filename or "document.pdf", await upload.read()) for upload in files
            ]
            saved_paths = await run_in_threadpool(
                service.save_uploaded_files,
                uploaded_files,
            )
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        try:
            response = await run_in_threadpool(
                service.analyze,
                query=query,
                thread_id=thread_id,
                export_artifacts=export_artifacts,
                document_paths=saved_paths,
                output_format=output_format,
            )
        except CheckpointConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_response(response, include_state=include_state)

    @app.post("/api/v1/documents/index", response_model=DocumentIndexResponse)
    async def index_documents(
        background_tasks: BackgroundTasks,
        files: list[UploadFile] = File(...),
        tenant_id: str | None = Form(default=None),
        async_mode: bool = Form(default=False),
        _: None = Depends(auth_dependency),
    ) -> DocumentIndexResponse:
        try:
            uploaded_files = [
                (upload.filename or "document.pdf", await upload.read()) for upload in files
            ]
            saved_paths = await run_in_threadpool(
                service.save_uploaded_files,
                uploaded_files,
            )
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        effective_tenant = tenant_id or app_config.rag_tenant_id
        if async_mode:
            receipts = await run_in_threadpool(
                service.enqueue_document_paths,
                saved_paths,
                tenant_id=effective_tenant,
            )
            for item in receipts:
                if item["status"] != "pending":
                    continue
                queued = await run_in_threadpool(
                    service.enqueue_index_job,
                    item["document_id"],
                    tenant_id=effective_tenant,
                )
                if not queued:
                    background_tasks.add_task(
                        service.process_document_index,
                        item["document_id"],
                        tenant_id=effective_tenant,
                    )
        else:
            receipts = await run_in_threadpool(
                service.index_document_paths,
                saved_paths,
                tenant_id=effective_tenant,
            )
        return DocumentIndexResponse(
            tenant_id=effective_tenant,
            documents=[
                {
                    "document_id": item["document_id"],
                    "filename": item["filename"],
                    "content_hash": item["content_hash"],
                    "status": item["status"],
                    "chunk_count": item["chunk_count"],
                    "error": item["error"],
                    "embed_calls": item["embed_calls"],
                }
                for item in receipts
            ],
        )

    @app.post("/api/v1/documents/{document_id}/process", response_model=DocumentStatusResponse)
    def process_document(
        document_id: str,
        tenant_id: str | None = Query(default=None),
        _: None = Depends(auth_dependency),
    ) -> DocumentStatusResponse:
        receipt = service.process_document_index(document_id, tenant_id=tenant_id)
        if receipt["status"] == "failed" and receipt.get("error") == "document not found":
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        record = service.get_document_status(document_id, tenant_id=tenant_id) or {}
        return DocumentStatusResponse(
            document_id=receipt["document_id"],
            tenant_id=receipt["tenant_id"],
            filename=receipt["filename"] or str(record.get("filename") or ""),
            content_hash=receipt["content_hash"] or str(record.get("content_hash") or ""),
            index_status=str(record.get("index_status") or receipt["status"]),
            chunk_count=int(record.get("chunk_count") or receipt.get("chunk_count") or 0),
            error=record.get("error") or receipt.get("error"),
            indexed_at=record.get("indexed_at"),
        )

    @app.get("/api/v1/documents/{document_id}", response_model=DocumentStatusResponse)
    def get_document_status(
        document_id: str,
        tenant_id: str | None = Query(default=None),
        _: None = Depends(auth_dependency),
    ) -> DocumentStatusResponse:
        record = service.get_document_status(document_id, tenant_id=tenant_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        return DocumentStatusResponse(
            document_id=record["document_id"],
            tenant_id=record["tenant_id"],
            filename=record["filename"],
            content_hash=record["content_hash"],
            index_status=record["index_status"],
            chunk_count=int(record.get("chunk_count") or 0),
            error=record.get("error"),
            indexed_at=record.get("indexed_at"),
        )

    @app.post("/api/v1/jobs", response_model=SubmitJobResponse, status_code=202)
    def submit_job(
        payload: SubmitJobRequest,
        background_tasks: BackgroundTasks,
        _: None = Depends(auth_dependency),
    ) -> SubmitJobResponse:
        created = service.submit_job(query=payload.query, thread_id=payload.thread_id)
        queued = service.enqueue_job(
            created["job_id"],
            payload.query,
            created["thread_id"],
            payload.export_artifacts,
            output_format=payload.output_format,
        )
        if not queued:
            background_tasks.add_task(
                service.run_job,
                created["job_id"],
                payload.query,
                created["thread_id"],
                payload.export_artifacts,
                None,
                payload.output_format,
            )
        return SubmitJobResponse(**created, queue_backend="redis" if queued else "background-task")

    @app.post("/api/v1/jobs/upload", response_model=SubmitJobResponse, status_code=202)
    async def submit_upload_job(
        background_tasks: BackgroundTasks,
        query: str = Form(...),
        thread_id: str | None = Form(default=None),
        export_artifacts: bool = Form(default=True),
        output_format: str | None = Form(default=None),
        files: list[UploadFile] = File(...),
        _: None = Depends(auth_dependency),
    ) -> SubmitJobResponse:
        saved_paths = service.save_uploaded_files([(upload.filename or "document.pdf", await upload.read()) for upload in files])
        created = service.submit_job(query=query, thread_id=thread_id)
        queued = service.enqueue_job(
            created["job_id"],
            query,
            created["thread_id"],
            export_artifacts,
            document_paths=saved_paths,
            output_format=output_format,
        )
        if not queued:
            background_tasks.add_task(
                service.run_job,
                created["job_id"],
                query,
                created["thread_id"],
                export_artifacts,
                saved_paths,
                output_format,
            )
        return SubmitJobResponse(**created, queue_backend="redis" if queued else "background-task")

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str, _: None = Depends(auth_dependency)) -> JobResponse:
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return JobResponse(**job)

    @app.get("/api/v1/jobs", response_model=list[JobResponse])
    def list_jobs(
        limit: int = Query(default=20, ge=1, le=100),
        _: None = Depends(auth_dependency),
    ) -> list[JobResponse]:
        return [JobResponse(**job) for job in service.list_jobs(limit=limit)]

    return app


app = create_app()
