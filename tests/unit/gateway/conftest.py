from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import httpx
import pytest_asyncio

from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.gateway.app import create_app
from inference.gateway.settings import GatewaySettings
from inference.queue.base import Priority, QueuedRequest
from inference.queue.memory import InMemoryRequestQueue
from inference.scheduler.scheduler import Scheduler

API_KEY = "secret-key"


class SpyQueue(InMemoryRequestQueue):
    """Records every item accepted by `put`."""

    def __init__(self, max_size: int) -> None:
        super().__init__(max_size)
        self.items: list[QueuedRequest] = []

    async def put(
        self, request: InferenceRequest, priority: Priority = Priority.NORMAL
    ) -> QueuedRequest:
        item = await super().put(request, priority)
        self.items.append(item)
        return item


class FakeRouter:
    """RequestRouter double that raises the configured error for a prompt."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.errors: dict[str, Exception] = {}

    async def route(self, request: InferenceRequest) -> InferenceResponse:
        self.calls.append(request.prompt)
        if request.prompt in self.errors:
            raise self.errors[request.prompt]
        return InferenceResponse(
            request_id=request.request_id,
            model=request.model,
            output=f"done {request.prompt}",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=0.0,
        )


@dataclass
class Stack:
    client: httpx.AsyncClient
    queue: SpyQueue
    router: FakeRouter


@pytest_asyncio.fixture
async def make_stack() -> AsyncIterator[Callable[..., Awaitable[Stack]]]:
    cleanups: list[Callable[[], Awaitable[None]]] = []

    async def _make(
        *, with_scheduler: bool = True, queue_size: int = 10, **overrides: object
    ) -> Stack:
        settings_kwargs: dict[str, object] = {
            "api_keys": [API_KEY],
            "rate_limit_rps": 1000.0,
            "rate_limit_burst": 1000,
            "request_timeout_s": 2.0,
        }
        settings_kwargs.update(overrides)
        queue = SpyQueue(max_size=queue_size)
        router = FakeRouter()
        app = create_app(queue, GatewaySettings(**settings_kwargs))  # type: ignore[arg-type]
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://gateway",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        cleanups.append(client.aclose)
        if with_scheduler:
            scheduler = Scheduler(
                queue, router, max_concurrency=4, shutdown_timeout_s=1
            )
            await scheduler.start()
            cleanups.append(scheduler.stop)
        return Stack(client, queue, router)

    yield _make
    for cleanup in reversed(cleanups):
        await cleanup()
