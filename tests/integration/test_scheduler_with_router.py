"""Queue -> Scheduler -> Router -> worker, all real components in one process."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest
import pytest_asyncio

from inference.backends.mock import MockBackend
from inference.core.errors import NoAvailableWorkerError, QueueClosedError
from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.queue.base import Priority
from inference.queue.memory import InMemoryRequestQueue
from inference.router.client import HttpWorkerClient
from inference.router.policies import RoundRobinPolicy
from inference.router.registry import WorkerEndpoint, WorkerRegistry
from inference.router.router import Router
from inference.scheduler.scheduler import Scheduler
from inference.worker.app import create_app
from inference.worker.settings import WorkerSettings


class RecordingBackend(MockBackend):
    """Records the order in which prompts reach the backend."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.prompts.append(request.prompt)
        return await super().generate(request)


@dataclass
class Stack:
    backend: RecordingBackend
    router: Router
    queue: InMemoryRequestQueue


@pytest_asyncio.fixture
async def stack() -> AsyncIterator[Stack]:
    backend = RecordingBackend()
    app = create_app(backend, WorkerSettings())
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport) as http:
            registry = WorkerRegistry(
                [
                    WorkerEndpoint(
                        id="w", base_url="http://worker", models=["mock-model"]
                    )
                ]
            )
            router = Router(registry, HttpWorkerClient(http, 5.0), RoundRobinPolicy())
            yield Stack(backend, router, InMemoryRequestQueue(max_size=50))


def _request(prompt: str, model: str = "mock-model") -> InferenceRequest:
    return InferenceRequest(model=model, prompt=prompt)


@pytest.mark.asyncio
async def test_dispatch_follows_priority_order(stack: Stack):
    scheduler = Scheduler(stack.queue, stack.router, max_concurrency=1)
    items = [
        await stack.queue.put(_request("low"), Priority.LOW),
        await stack.queue.put(_request("normal"), Priority.NORMAL),
        await stack.queue.put(_request("high"), Priority.HIGH),
    ]

    await scheduler.start()
    await asyncio.wait_for(asyncio.gather(*(i.future for i in items)), timeout=5)
    await scheduler.stop()

    assert stack.backend.prompts == ["high", "normal", "low"]


@pytest.mark.asyncio
async def test_many_concurrent_requests_all_resolve(stack: Stack):
    scheduler = Scheduler(stack.queue, stack.router, max_concurrency=4)
    await scheduler.start()

    items = [await stack.queue.put(_request(f"r{i}")) for i in range(12)]
    responses = await asyncio.wait_for(
        asyncio.gather(*(i.future for i in items)), timeout=5
    )
    await scheduler.stop()

    assert [r.output for r in responses] == [
        f"mock response to: r{i}" for i in range(12)
    ]


@pytest.mark.asyncio
async def test_routing_failure_affects_only_its_own_request(stack: Stack):
    scheduler = Scheduler(stack.queue, stack.router)
    await scheduler.start()

    bad = await stack.queue.put(_request("x", model="unknown-model"))
    good = await stack.queue.put(_request("y"))

    with pytest.raises(NoAvailableWorkerError):
        await asyncio.wait_for(bad.future, timeout=5)
    assert (await asyncio.wait_for(good.future, timeout=5)).output.endswith("y")
    await scheduler.stop()


@pytest.mark.asyncio
async def test_stop_drains_buffered_requests_and_closes_queue(stack: Stack):
    scheduler = Scheduler(stack.queue, stack.router, max_concurrency=2)
    items = [await stack.queue.put(_request(f"r{i}")) for i in range(6)]
    await scheduler.start()

    await scheduler.stop()

    assert all(i.future.done() and i.future.exception() is None for i in items)
    assert len(stack.backend.prompts) == 6
    with pytest.raises(QueueClosedError):
        await stack.queue.put(_request("late"))
