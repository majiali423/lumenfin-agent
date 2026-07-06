from __future__ import annotations

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
    HealthResponse,
    JobResponse,
    SubmitJobRequest,
    SubmitJobResponse,
)


def create_app(
    config: AppConfig | None = None,
    *,
    llm_client: BaseLLMClient | None = None,
    market_data_client: MarketDataClient | None = None,
) -> FastAPI:
    configure_logging()
    app_config = config or AppConfig.from_env()
    service = LumenFinAnalysisService(
        app_config,
        llm_client=llm_client,
        market_data_client=market_data_client,
    )
    auth_dependency = build_api_key_dependency(app_config.api_key)

    app = FastAPI(
        title="LumenFin API",
        version="0.3.0",
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
            "milvus_uri": app_config.milvus_uri,
            "embedding_provider": app_config.embedding_provider,
            "market_data_provider": app_config.market_data_provider,
        }

    def _to_response(payload: dict) -> AnalyzeResponse:
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
            state=result,
            chart_data=result.get("chart_data"),
            run_telemetry=result.get("run_telemetry"),
            run_manifest=run_manifest,
            provider_health=payload.get("provider_health"),
            checkpoint=checkpoint,
        )

    @app.post("/api/v1/analyze", response_model=AnalyzeResponse)
    def analyze(payload: AnalyzeRequest, _: None = Depends(auth_dependency)) -> AnalyzeResponse:
        response = service.analyze(
            query=payload.query,
            thread_id=payload.thread_id,
            export_artifacts=payload.export_artifacts,
        )
        return _to_response(response)

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
        return _to_response(response)

    @app.post("/api/v1/analyze-data", response_model=AnalyzeResponse)
    def analyze_data(payload: AnalyzeDataRequest, _: None = Depends(auth_dependency)) -> AnalyzeResponse:
        response = service.analyze(
            query=payload.query,
            thread_id=payload.thread_id,
            export_artifacts=payload.export_artifacts,
            structured_metrics=payload.company_metrics,
        )
        return _to_response(response)

    @app.post("/api/v1/analyze-upload", response_model=AnalyzeResponse)
    async def analyze_upload(
        query: str = Form(...),
        thread_id: str | None = Form(default=None),
        export_artifacts: bool = Form(default=True),
        files: list[UploadFile] = File(...),
        _: None = Depends(auth_dependency),
    ) -> AnalyzeResponse:
        saved_paths = service.save_uploaded_files([(upload.filename or "document.pdf", await upload.read()) for upload in files])
        response = service.analyze(
            query=query,
            thread_id=thread_id,
            export_artifacts=export_artifacts,
            document_paths=saved_paths,
        )
        return _to_response(response)

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
