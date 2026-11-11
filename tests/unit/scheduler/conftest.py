import asyncio
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio

from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.queue.memory import InMemoryRequestQueue
from inference.scheduler.scheduler import Scheduler


class FakeRouter:
    """RequestRouter double that records calls and can block, fail or hang."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.cancelled: list[str] = []
        self.errors: dict[str, Exception] = {}
        self.gate: asyncio.Event | None = None
        self.hang = False
        self.in_flight = 0
        self.peak = 0

    async def route(self, request: InferenceRequest) -> InferenceResponse:
        self.calls.append(request.prompt)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if request.prompt in self.errors:
                raise self.errors[request.prompt]
            if self.gate is not None:
                await self.gate.wait()
            if self.hang:
                await asyncio.Event().wait()
            return InferenceResponse(
                request_id=request.request_id,
                model=request.model,
                output=f"done {request.prompt}",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=0.0,
            )
        except asyncio.CancelledError:
            self.cancelled.append(request.prompt)
            raise
        finally:
            self.in_flight -= 1


@pytest.fixture
def queue() -> InMemoryRequestQueue:
    return InMemoryRequestQueue(max_size=100)


@pytest.fixture
def router() -> FakeRouter:
    return FakeRouter()


@pytest.fixture
def request_for() -> Callable[[str], InferenceRequest]:
    return lambda prompt: InferenceRequest(model="m", prompt=prompt)


@pytest_asyncio.fixture
async def make_scheduler(
    queue: InMemoryRequestQueue, router: FakeRouter
) -> AsyncIterator[Callable[..., Scheduler]]:
    created: list[Scheduler] = []

    def _make(**kwargs: object) -> Scheduler:
        scheduler = Scheduler(queue, router, **kwargs)  # type: ignore[arg-type]
        created.append(scheduler)
        return scheduler

    yield _make
    for scheduler in created:
        await scheduler.stop()
