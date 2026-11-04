"""FastAPI application for the inference worker."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from inference.backends.base import Backend
from inference.backends.mock import MockBackend
from inference.core.errors import (
    BackendError,
    BackendUnavailableError,
    WorkerNotReadyError,
)
from inference.core.schemas import ErrorResponse, InferenceRequest, InferenceResponse
from inference.worker import health
from inference.worker.service import WorkerService
from inference.worker.settings import WorkerSettings


def create_app(
    backend: Backend | None = None, settings: WorkerSettings | None = None
) -> FastAPI:
    """Build the worker app. Defaults to a MockBackend."""
    settings = settings or WorkerSettings()
    backend = backend or MockBackend(latency_s=settings.mock_latency_s)
    service = WorkerService(backend, drain_timeout_s=settings.drain_timeout_s)

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
