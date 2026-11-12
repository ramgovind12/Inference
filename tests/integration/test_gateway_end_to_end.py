"""Client -> Gateway -> Queue -> Scheduler -> Router -> worker, all real components."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest
import pytest_asyncio
from pydantic import SecretStr

from inference.backends.mock import MockBackend
from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.gateway.app import create_app as create_gateway_app
from inference.gateway.settings import GatewaySettings
from inference.queue.memory import InMemoryRequestQueue
from inference.router.client import HttpWorkerClient
from inference.router.policies import RoundRobinPolicy
from inference.router.registry import WorkerEndpoint, WorkerRegistry
from inference.router.router import Router
from inference.scheduler.scheduler import Scheduler
from inference.worker.app import create_app as create_worker_app
from inference.worker.settings import WorkerSettings

KEY = "e2e-key"


class RecordingBackend(MockBackend):
    """Records the order in which prompts reach the backend."""

    def __init__(self, latency_s: float) -> None:
        super().__init__(latency_s=latency_s)
        self.prompts: list[str] = []

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.prompts.append(request.prompt)
        return await super().generate(request)


@dataclass
class Stack:
    client: httpx.AsyncClient
    backend: RecordingBackend
    scheduler: Scheduler


@pytest_asyncio.fixture
async def stack() -> AsyncIterator[Stack]:
    backend = RecordingBackend(latency_s=0.1)
    worker_app = create_worker_app(backend, WorkerSettings())
    async with (
        worker_app.router.lifespan_context(worker_app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=worker_app)) as worker_http,
    ):
        registry = WorkerRegistry(
            [WorkerEndpoint(id="w", base_url="http://worker", models=["mock-model"])]
        )
        router = Router(
            registry, HttpWorkerClient(worker_http, 5.0), RoundRobinPolicy()
        )
        queue = InMemoryRequestQueue(max_size=50)
        scheduler = Scheduler(queue, router, max_concurrency=1, shutdown_timeout_s=5)
        await scheduler.start()
        gateway = create_gateway_app(
            queue,
            GatewaySettings(
                api_keys=[SecretStr(KEY)],
                rate_limit_rps=1000.0,
                rate_limit_burst=1000,
                request_timeout_s=10.0,
            ),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=gateway, raise_app_exceptions=False),
            base_url="http://gateway",
            headers={"Authorization": f"Bearer {KEY}"},
        ) as client:
            yield Stack(client, backend, scheduler)
        await scheduler.stop()


def _body(prompt: str, **extra: object) -> dict[str, object]:
    return {"model": "mock-model", "prompt": prompt, **extra}


@pytest.mark.asyncio
async def test_request_travels_the_full_path(stack: Stack):
    response = await stack.client.post("/v1/inference", json=_body("hello world"))

    assert response.status_code == 200
    assert response.json()["output"] == "mock response to: hello world"
    assert stack.backend.prompts == ["hello world"]


@pytest.mark.asyncio
async def test_unknown_model_returns_503(stack: Stack):
    response = await stack.client.post(
        "/v1/inference", json={"model": "other-model", "prompt": "x"}
    )

    assert response.status_code == 503
    assert response.json()["error"] == "no_worker_available"


@pytest.mark.asyncio
async def test_high_priority_overtakes_low_priority_while_busy(stack: Stack):
    running = asyncio.create_task(
        stack.client.post("/v1/inference", json=_body("first"))
    )
    await asyncio.sleep(0.03)
    low = asyncio.create_task(
        stack.client.post("/v1/inference", json=_body("low", priority="low"))
    )
    await asyncio.sleep(0.01)
    high = asyncio.create_task(
        stack.client.post("/v1/inference", json=_body("high", priority="high"))
    )

    responses = await asyncio.wait_for(asyncio.gather(running, low, high), timeout=5)

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert stack.backend.prompts == ["first", "high", "low"]


@pytest.mark.asyncio
async def test_scheduler_stop_finishes_in_flight_then_gateway_returns_503(stack: Stack):
    in_flight = [
        asyncio.create_task(stack.client.post("/v1/inference", json=_body(f"r{i}")))
        for i in range(3)
    ]
    await asyncio.sleep(0.03)

    await stack.scheduler.stop()
    responses = await asyncio.gather(*in_flight)
    late = await stack.client.post("/v1/inference", json=_body("late"))

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert late.status_code == 503
    assert late.json()["error"] == "shutting_down"
