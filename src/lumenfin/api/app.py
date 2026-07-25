from __future__ import annotations

from importlib import metadata
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse

from ..config import AppConfig
from ..llm import BaseLLMClient
from ..logging_utils import configure_logging, request_logging_middleware
from ..market_data import MarketDataClient, probe_market_provider
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
    SubmitJobRequest,
    SubmitJobResponse,
)


def _package_version() -> str:
    try:
        return metadata.version("lumenfin-agent")
    except metadata.PackageNotFoundError:
        return "0.1.0rc1"


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

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        backend = "deepseek" if app_config.llm.api_key else "local-fallback"
        market_client = service.providers.market_data.client
        market_probe = probe_market_provider(market_client)
        return HealthResponse(
            status="ok",
            llm_backend=backend,
            llm_configured=bool(app_config.llm.api_key),
            market_provider=app_config.market_data_provider,
            market_provider_ok=bool(market_probe.get("ok")),
            embedding_provider=app_config.embedding_provider,
            rag_enabled=app_config.rag_enabled,
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
        )

    @app.post("/api/v1/analyze", response_model=AnalyzeResponse)
    def analyze(payload: AnalyzeRequest, _: None = Depends(auth_dependency)) -> AnalyzeResponse:
        try:
            response = service.analyze(
                query=payload.query,
                thread_id=payload.thread_id,
                export_artifacts=payload.export_artifacts,
            )
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
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_response(response, include_state=payload.include_state)

    @app.post("/api/v1/analyze-upload", response_model=AnalyzeResponse)
    async def analyze_upload(
        query: str = Form(...),
        thread_id: str | None = Form(default=None),
        export_artifacts: bool = Form(default=True),
        include_state: bool = Form(default=False),
        files: list[UploadFile] = File(...),
        _: None = Depends(auth_dependency),
    ) -> AnalyzeResponse:
        try:
            saved_paths = service.save_uploaded_files(
                [(upload.filename or "document.pdf", await upload.read()) for upload in files]
            )
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        try:
            response = service.analyze(
                query=query,
                thread_id=thread_id,
                export_artifacts=export_artifacts,
                document_paths=saved_paths,
            )
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
            saved_paths = service.save_uploaded_files(
                [(upload.filename or "document.pdf", await upload.read()) for upload in files]
            )
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        effective_tenant = tenant_id or app_config.rag_tenant_id
        if async_mode:
            receipts = service.enqueue_document_paths(saved_paths, tenant_id=effective_tenant)
            for item in receipts:
                if item["status"] != "pending":
                    continue
                queued = service.enqueue_index_job(item["document_id"], tenant_id=effective_tenant)
                if not queued:
                    background_tasks.add_task(
                        service.process_document_index,
                        item["document_id"],
                        tenant_id=effective_tenant,
                    )
        else:
            receipts = service.index_document_paths(saved_paths, tenant_id=effective_tenant)
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
            index_status=receipt["status"] if receipt["status"] != "skipped_duplicate" else "ready",
            chunk_count=int(receipt.get("chunk_count") or 0),
            error=receipt.get("error"),
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
        )
        if not queued:
            background_tasks.add_task(
                service.run_job,
                created["job_id"],
                payload.query,
                created["thread_id"],
                payload.export_artifacts,
            )
        return SubmitJobResponse(**created, queue_backend="redis" if queued else "background-task")

    @app.post("/api/v1/jobs/upload", response_model=SubmitJobResponse, status_code=202)
    async def submit_upload_job(
        background_tasks: BackgroundTasks,
        query: str = Form(...),
        thread_id: str | None = Form(default=None),
        export_artifacts: bool = Form(default=True),
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
        )
        if not queued:
            background_tasks.add_task(
                service.run_job,
                created["job_id"],
                query,
                created["thread_id"],
                export_artifacts,
                saved_paths,
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
