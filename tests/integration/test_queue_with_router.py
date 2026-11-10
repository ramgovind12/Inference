"""Queue -> Router -> worker, with a test-only consumer standing in for the scheduler."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest
import pytest_asyncio

from inference.backends.mock import MockBackend
from inference.core.errors import (
    NoAvailableWorkerError,
    QueueClosedError,
    QueueFullError,
)
from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.queue.base import Priority, RequestQueue
from inference.queue.memory import InMemoryRequestQueue
from inference.router.client import HttpWorkerClient
from inference.router.policies import RoundRobinPolicy
from inference.router.registry import WorkerEndpoint, WorkerRegistry
from inference.router.router import Router
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
            yield Stack(backend, router)


def _request(prompt: str, model: str = "mock-model") -> InferenceRequest:
    return InferenceRequest(model=model, prompt=prompt)


async def _consume(queue: RequestQueue, router: Router) -> None:
    """Sequential consumer: the dispatch loop a scheduler will own later."""
    while True:
        try:
            item = await queue.get()
        except QueueClosedError:
            return
        try:
            item.future.set_result(await router.route(item.request))
        except Exception as exc:  # noqa: BLE001 - forwarded to the submitter's future
            item.future.set_exception(exc)


@pytest.mark.asyncio
async def test_dispatch_follows_priority_order(stack: Stack):
    queue = InMemoryRequestQueue(max_size=10)
    items = [
        await queue.put(_request("low"), Priority.LOW),
        await queue.put(_request("normal"), Priority.NORMAL),
        await queue.put(_request("high"), Priority.HIGH),
    ]
    consumer = asyncio.create_task(_consume(queue, stack.router))

    responses = await asyncio.wait_for(
        asyncio.gather(*(i.future for i in items)), timeout=5
    )
    await queue.close()
    await asyncio.wait_for(consumer, timeout=5)

    assert stack.backend.prompts == ["high", "normal", "low"]
    assert [r.output for r in responses] == [
        "mock response to: low",
        "mock response to: normal",
        "mock response to: high",
    ]


@pytest.mark.asyncio
async def test_full_queue_applies_backpressure_until_consumed(stack: Stack):
    queue = InMemoryRequestQueue(max_size=2)
    first = await queue.put(_request("a"))
    second = await queue.put(_request("b"))
    with pytest.raises(QueueFullError):
        await queue.put(_request("c"))

    consumer = asyncio.create_task(_consume(queue, stack.router))
    await asyncio.wait_for(asyncio.gather(first.future, second.future), timeout=5)
    third = await queue.put(_request("c"))  # space is available again
    await asyncio.wait_for(third.future, timeout=5)
    await queue.close()
    await asyncio.wait_for(consumer, timeout=5)

    assert stack.backend.prompts == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_close_drains_all_buffered_requests(stack: Stack):
    queue = InMemoryRequestQueue(max_size=10)
    items = [await queue.put(_request(f"r{i}")) for i in range(3)]
    await queue.close()

    await asyncio.wait_for(_consume(queue, stack.router), timeout=5)

    assert all(i.future.done() and i.future.exception() is None for i in items)
    assert stack.backend.prompts == ["r0", "r1", "r2"]


@pytest.mark.asyncio
async def test_routing_failure_propagates_to_the_submitters_future(stack: Stack):
    queue = InMemoryRequestQueue(max_size=10)
    bad = await queue.put(_request("x", model="unknown-model"))
    good = await queue.put(_request("y"))
    consumer = asyncio.create_task(_consume(queue, stack.router))

    with pytest.raises(NoAvailableWorkerError):
        await asyncio.wait_for(bad.future, timeout=5)
    assert (await asyncio.wait_for(good.future, timeout=5)).output.endswith("y")

    await queue.close()
    await asyncio.wait_for(consumer, timeout=5)
