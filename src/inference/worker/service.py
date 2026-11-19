"""Worker service: lifecycle management and request execution."""

import asyncio
import logging
import time
from enum import Enum

from prometheus_client import CollectorRegistry

from inference.backends.base import Backend
from inference.core.errors import BackendError, WorkerNotReadyError
from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.observability.metrics import WorkerMetrics

logger = logging.getLogger(__name__)


class WorkerState(str, Enum):
    """Lifecycle states of a worker."""

    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"


class WorkerService:
    """Runs inference requests against a backend and manages worker lifecycle."""

    def __init__(
        self,
        backend: Backend,
        drain_timeout_s: float = 30.0,
        metrics: WorkerMetrics | None = None,
    ) -> None:
        self._backend = backend
        self._drain_timeout_s = drain_timeout_s
        self._metrics = metrics or WorkerMetrics(CollectorRegistry())
        self._state = WorkerState.STARTING
        self._active_requests = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def state(self) -> WorkerState:
        return self._state

    @property
    def active_requests(self) -> int:
        return self._active_requests

    async def start(self) -> None:
        """Start the backend and mark the worker ready."""
        if self._state is not WorkerState.STARTING:
            raise RuntimeError(f"cannot start worker in state {self._state.value}")
        try:
            await self._backend.start()
        except Exception:
            self._state = WorkerState.STOPPED
            logger.exception("backend failed to start")
            raise
        self._state = WorkerState.READY
        logger.info("worker ready")

    async def is_ready(self) -> bool:
        """True if the worker accepts requests and its backend is ready."""
        return self._state is WorkerState.READY and await self._backend.is_ready()

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Execute a request. Raises WorkerNotReadyError or BackendError."""
        if self._state is not WorkerState.READY:
            self._metrics.requests.labels(outcome="not_ready").inc()
            raise WorkerNotReadyError(f"worker is {self._state.value}")
        self._active_requests += 1
        self._idle.clear()
        started = time.monotonic()
        outcome = "backend_error"
        try:
            response = await self._backend.generate(request)
        except BackendError:
            logger.exception("backend error", extra={"request_id": request.request_id})
            raise
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception as exc:
            logger.exception(
                "unexpected backend failure", extra={"request_id": request.request_id}
            )
            raise BackendError(f"backend failed: {exc}") from exc
        else:
            outcome = "ok"
            self._metrics.record_tokens(
                response.prompt_tokens, response.completion_tokens
            )
            return response
        finally:
            self._metrics.record_request(outcome, time.monotonic() - started)
            self._active_requests -= 1
            if self._active_requests == 0:
                self._idle.set()

    async def stop(self) -> None:
        """Stop accepting requests, drain in-flight ones, then stop the backend."""
        if self._state in (WorkerState.DRAINING, WorkerState.STOPPED):
            return
        self._state = WorkerState.DRAINING
        logger.info("worker draining", extra={"active": self._active_requests})
        try:
            await asyncio.wait_for(self._idle.wait(), self._drain_timeout_s)
        except TimeoutError:
            logger.warning(
                "drain timeout; stopping with requests in flight",
                extra={"active": self._active_requests},
            )
        await self._backend.stop()
        self._state = WorkerState.STOPPED
        logger.info("worker stopped")
