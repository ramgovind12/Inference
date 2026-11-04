"""Health and readiness endpoints."""

from fastapi import APIRouter, Request, Response, status

from inference.worker.service import WorkerService

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request, response: Response) -> dict[str, str | int]:
    """Readiness: the worker can accept inference requests."""
    service: WorkerService = request.app.state.service
    is_ready = await service.is_ready()
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if is_ready else "not_ready",
        "state": service.state.value,
        "active_requests": service.active_requests,
    }
