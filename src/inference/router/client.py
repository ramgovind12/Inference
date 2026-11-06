"""Client used by the router to talk to workers."""

import logging
from typing import Protocol

import httpx

from inference.core.errors import (
    BackendError,
    WorkerNotReadyError,
    WorkerUnreachableError,
)
from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.router.registry import WorkerEndpoint

logger = logging.getLogger(__name__)


class WorkerClient(Protocol):
    """Interface for calling a worker."""

    async def ready(self, endpoint: WorkerEndpoint) -> bool:
        """Return True if the worker reports itself ready."""
        ...

    async def infer(
        self, endpoint: WorkerEndpoint, request: InferenceRequest
    ) -> InferenceResponse:
        """Send a request to the worker.

        Raises WorkerUnreachableError, WorkerNotReadyError or BackendError.
        """
        ...


class HttpWorkerClient:
    """WorkerClient implementation over HTTP using a shared httpx client."""

    def __init__(self, http: httpx.AsyncClient, timeout_s: float = 30.0) -> None:
        self._http = http
        self._timeout_s = timeout_s

    async def ready(self, endpoint: WorkerEndpoint) -> bool:
        try:
            response = await self._http.get(
                f"{endpoint.base_url}/ready", timeout=self._timeout_s
            )
        except httpx.TransportError as exc:
            logger.warning(
                "worker readiness check failed",
                extra={"worker_id": endpoint.id, "error": str(exc)},
            )
            return False
        return response.status_code == httpx.codes.OK

    async def infer(
        self, endpoint: WorkerEndpoint, request: InferenceRequest
    ) -> InferenceResponse:
        try:
            response = await self._http.post(
                f"{endpoint.base_url}/v1/inference",
                json=request.model_dump(),
                timeout=self._timeout_s,
            )
        except httpx.TransportError as exc:
            raise WorkerUnreachableError(
                f"worker {endpoint.id} unreachable: {exc!r}"
            ) from exc

        if response.status_code == httpx.codes.OK:
            return InferenceResponse.model_validate(response.json())
        detail = self._detail(response)
        if response.status_code == httpx.codes.SERVICE_UNAVAILABLE:
            raise WorkerNotReadyError(f"worker {endpoint.id}: {detail}")
        raise BackendError(
            f"worker {endpoint.id} returned {response.status_code}: {detail}"
        )

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        try:
            return str(response.json()["detail"])
        except (ValueError, KeyError, TypeError):
            return response.text
