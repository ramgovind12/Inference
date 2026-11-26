"""FastAPI application for the inference worker."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry

from inference.backends.base import Backend
from inference.core.errors import (
    BackendError,
    BackendUnavailableError,
    WorkerNotReadyError,
)
from inference.core.schemas import ErrorResponse, InferenceRequest, InferenceResponse
from inference.observability.metrics import (
    METRICS_CONTENT_TYPE,
    WorkerMetrics,
    render,
)
from inference.observability.settings import ObservabilitySettings
from inference.worker import health
from inference.worker.factory import build_backend
from inference.worker.service import WorkerService, WorkerState
from inference.worker.settings import WorkerSettings


def create_app(
    backend: Backend | None = None,
    settings: WorkerSettings | None = None,
    observability: ObservabilitySettings | None = None,
) -> FastAPI:
    """Build the worker app. Defaults to a MockBackend."""
    settings = settings or WorkerSettings()
    observability = observability or ObservabilitySettings()
    backend = backend or build_backend(settings)
    registry = CollectorRegistry()
    metrics = WorkerMetrics(registry)
    service = WorkerService(
        backend, drain_timeout_s=settings.drain_timeout_s, metrics=metrics
    )
    metrics.bind_state(
        active=lambda: service.active_requests,
        ready=lambda: service.state is WorkerState.READY,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(title="InferScale Worker", lifespan=lifespan)
    app.state.service = service
    app.include_router(health.router)

    if observability.metrics_enabled:

        @app.get("/metrics", include_in_schema=False)
        async def metrics_endpoint() -> Response:
            """Prometheus metrics. Async, so gauges read state on the event loop."""
            return Response(render(registry), media_type=METRICS_CONTENT_TYPE)

    @app.post("/v1/inference", response_model=InferenceResponse)
    async def inference(body: InferenceRequest) -> InferenceResponse:
        return await service.infer(body)

    def _error(status_code: int, error: str, exc: Exception) -> JSONResponse:
        body = ErrorResponse(error=error, detail=str(exc))
        return JSONResponse(status_code=status_code, content=body.model_dump())

    @app.exception_handler(WorkerNotReadyError)
    async def _not_ready(_: Request, exc: WorkerNotReadyError) -> JSONResponse:
        return _error(503, "worker_not_ready", exc)

    @app.exception_handler(BackendUnavailableError)
    async def _unavailable(_: Request, exc: BackendUnavailableError) -> JSONResponse:
        return _error(503, "backend_unavailable", exc)

    @app.exception_handler(BackendError)
    async def _backend_error(_: Request, exc: BackendError) -> JSONResponse:
        return _error(502, "backend_error", exc)

    return app
