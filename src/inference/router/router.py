"""Router: selects a worker per request and fails over when one is down."""

import asyncio
import contextlib
import logging

import httpx

from inference.core.errors import (
    NoAvailableWorkerError,
    WorkerNotReadyError,
    WorkerUnreachableError,
)
from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.router.client import HttpWorkerClient, WorkerClient
from inference.router.policies import RoutingPolicy, build_policy
from inference.router.registry import WorkerRegistry
from inference.router.settings import RouterSettings

logger = logging.getLogger(__name__)


class Router:
    """Routes requests to healthy workers that serve the requested model.

    A request is retried on another worker only when it provably never ran
    (worker unreachable or not ready). Backend errors are returned as-is.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        client: WorkerClient,
        policy: RoutingPolicy,
        max_attempts: int = 3,
        health_interval_s: float = 5.0,
    ) -> None:
        self._registry = registry
        self._client = client
        self._policy = policy
        self._max_attempts = max_attempts
        self._health_interval_s = health_interval_s
        self._health_task: asyncio.Task[None] | None = None

    async def route(self, request: InferenceRequest) -> InferenceResponse:
        """Send `request` to a worker. Raises NoAvailableWorkerError or BackendError."""
        tried: set[str] = set()
        last_error: Exception | None = None
        for _ in range(self._max_attempts):
            candidates = [
                w
                for w in self._registry.candidates(request.model)
                if w.endpoint.id not in tried
            ]
            if not candidates:
                break
            worker = self._policy.select(candidates)
            worker_id = worker.endpoint.id
            tried.add(worker_id)
            worker.in_flight += 1
            try:
                return await self._client.infer(worker.endpoint, request)
            except (WorkerUnreachableError, WorkerNotReadyError) as exc:
                last_error = exc
                self._registry.set_healthy(worker_id, False)
                logger.warning(
                    "worker failed, marked unhealthy",
                    extra={
                        "worker_id": worker_id,
                        "request_id": request.request_id,
                        "error": str(exc),
                    },
                )
            finally:
                worker.in_flight -= 1

        if not tried:
            message = f"no healthy worker serves model {request.model!r}"
        else:
            message = f"all {len(tried)} attempted worker(s) failed"
        raise NoAvailableWorkerError(message) from last_error

    async def refresh(self) -> None:
        """Poll every worker's readiness once and update its health flag."""
        workers = self._registry.all()
        results = await asyncio.gather(
            *(self._client.ready(w.endpoint) for w in workers)
        )
        for worker, is_ready in zip(workers, results, strict=True):
            if is_ready != worker.healthy:
                logger.info(
                    "worker health changed",
                    extra={"worker_id": worker.endpoint.id, "healthy": is_ready},
                )
            worker.healthy = is_ready

    async def start(self) -> None:
        """Start the periodic health-check loop."""
        if self._health_task is None:
            self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        """Stop the health-check loop."""
        if self._health_task is None:
            return
        self._health_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._health_task
        self._health_task = None

    async def _health_loop(self) -> None:
        while True:
            try:
                await self.refresh()
            except Exception:
                logger.exception("health check pass failed")
            await asyncio.sleep(self._health_interval_s)


def create_router(settings: RouterSettings, http: httpx.AsyncClient) -> Router:
    """Build a Router from settings and a shared httpx client."""
    return Router(
        registry=WorkerRegistry(settings.workers),
        client=HttpWorkerClient(http, timeout_s=settings.request_timeout_s),
        policy=build_policy(settings.policy),
        max_attempts=settings.max_attempts,
        health_interval_s=settings.health_interval_s,
    )
