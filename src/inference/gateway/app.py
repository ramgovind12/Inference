"""FastAPI application for the gateway."""

import asyncio
import logging
import math
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse

from inference.core.errors import (
    BackendError,
    NoAvailableWorkerError,
    QueueClosedError,
    QueueFullError,
    SchedulerStoppedError,
)
from inference.core.schemas import ErrorResponse, InferenceResponse
from inference.gateway.auth import ApiKeyAuthenticator, AuthenticationError
from inference.gateway.rate_limit import RateLimitExceededError, TokenBucketLimiter
from inference.gateway.schemas import GatewayInferenceRequest
from inference.gateway.settings import GatewaySettings
from inference.queue.base import RequestQueue

logger = logging.getLogger(__name__)


class GatewayTimeoutError(Exception):
    """No result arrived within the request timeout."""


def create_app(
    queue: RequestQueue,
    settings: GatewaySettings | None = None,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
) -> FastAPI:
    """Build the gateway app. The queue is injected by the caller.

    `lifespan` lets the caller start and stop the components behind the queue.
    """
    settings = settings or GatewaySettings()
    if settings.auth_enabled and not settings.api_keys:
        raise ValueError("authentication is enabled but no API keys are configured")

    authenticator = ApiKeyAuthenticator(k.get_secret_value() for k in settings.api_keys)
    limiter = TokenBucketLimiter(settings.rate_limit_rps, settings.rate_limit_burst)

    app = FastAPI(title="InferScale Gateway", lifespan=lifespan)
    app.state.queue = queue

    async def authenticate(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> str:
        if not settings.auth_enabled:
            host = request.client.host if request.client else "unknown"
            return f"ip-{host}"
        return authenticator.authenticate(authorization, x_api_key)

    async def rate_limit(client_id: Annotated[str, Depends(authenticate)]) -> str:
        retry_after = limiter.check(client_id)
        if retry_after > 0:
            raise RateLimitExceededError(retry_after)
        return client_id

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.post("/v1/inference", response_model=InferenceResponse)
    async def inference(
        body: GatewayInferenceRequest,
        response: Response,
        client_id: Annotated[str, Depends(rate_limit)],
    ) -> InferenceResponse:
        request, priority = body.to_internal()
        item = await queue.put(request, priority)
        logger.info(
            "request queued",
            extra={
                "request_id": request.request_id,
                "client_id": client_id,
                "priority": priority.name,
            },
        )
        try:
            result = await asyncio.wait_for(item.future, settings.request_timeout_s)
        except TimeoutError:
            # wait_for has cancelled the future, so the scheduler skips or aborts it.
            raise GatewayTimeoutError(
                f"no result within {settings.request_timeout_s:g}s"
            ) from None
        response.headers["X-Request-ID"] = request.request_id
        return result

    def _error(
        status_code: int,
        error: str,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> JSONResponse:
        body = ErrorResponse(error=error, detail=detail)
        return JSONResponse(
            status_code=status_code, content=body.model_dump(), headers=headers
        )

    @app.exception_handler(AuthenticationError)
    async def _unauthorized(_: Request, exc: AuthenticationError) -> JSONResponse:
        return _error(401, "unauthorized", str(exc), {"WWW-Authenticate": "Bearer"})

    @app.exception_handler(RateLimitExceededError)
    async def _rate_limited(_: Request, exc: RateLimitExceededError) -> JSONResponse:
        retry_after = str(max(1, math.ceil(exc.retry_after_s)))
        return _error(
            429, "rate_limited", "rate limit exceeded", {"Retry-After": retry_after}
        )

    @app.exception_handler(QueueFullError)
    async def _queue_full(_: Request, exc: QueueFullError) -> JSONResponse:
        return _error(
            503, "queue_full", "the service is overloaded", {"Retry-After": "1"}
        )

    @app.exception_handler(QueueClosedError)
    @app.exception_handler(SchedulerStoppedError)
    async def _shutting_down(_: Request, exc: Exception) -> JSONResponse:
        return _error(503, "shutting_down", "the service is shutting down")

    @app.exception_handler(NoAvailableWorkerError)
    async def _no_worker(_: Request, exc: NoAvailableWorkerError) -> JSONResponse:
        logger.warning("no worker available", extra={"error": str(exc)})
        return _error(503, "no_worker_available", "no worker is available")

    @app.exception_handler(BackendError)
    async def _backend_error(_: Request, exc: BackendError) -> JSONResponse:
        logger.error("backend error", extra={"error": str(exc)})
        return _error(502, "backend_error", "the inference backend failed")

    @app.exception_handler(GatewayTimeoutError)
    async def _timeout(_: Request, exc: GatewayTimeoutError) -> JSONResponse:
        return _error(504, "timeout", str(exc))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unexpected gateway error", exc_info=exc)
        return _error(500, "internal_error", "internal server error")

    return app
