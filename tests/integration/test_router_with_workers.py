"""Router against real worker apps, wired in-process over ASGI transports."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from inference.backends.mock import MockBackend
from inference.core.errors import BackendError, NoAvailableWorkerError
from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.router.client import HttpWorkerClient
from inference.router.policies import LeastLoadedPolicy, RoundRobinPolicy, RoutingPolicy
from inference.router.registry import WorkerEndpoint, WorkerRegistry
from inference.router.router import Router
from inference.worker.app import create_app
from inference.worker.settings import WorkerSettings


class CountingBackend(MockBackend):
    def __init__(self, latency_s: float = 0.0) -> None:
        super().__init__(latency_s=latency_s)
        self.calls = 0

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        return await super().generate(request)


class HostTransport(httpx.AsyncBaseTransport):
    """Dispatches requests to a different ASGI app per host name."""

    def __init__(self, apps: dict[str, FastAPI]) -> None:
        self._transports = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transports[request.url.host].handle_async_request(request)


@dataclass
class Cluster:
    apps: dict[str, FastAPI]
    backends: dict[str, CountingBackend]
    http: httpx.AsyncClient

    def router(self, policy: RoutingPolicy) -> tuple[Router, WorkerRegistry]:
        registry = WorkerRegistry(
            WorkerEndpoint(id=name, base_url=f"http://{name}", models=["mock-model"])
            for name in self.apps
        )
        client = HttpWorkerClient(self.http, timeout_s=5.0)
        return Router(registry, client, policy), registry


@pytest_asyncio.fixture
async def cluster() -> AsyncIterator[Cluster]:
    backends = {
        name: CountingBackend(latency_s=0.05) for name in ("worker-a", "worker-b")
    }
    apps = {n: create_app(b, WorkerSettings()) for n, b in backends.items()}
    async with AsyncExitStack() as stack:
        for app in apps.values():
            await stack.enter_async_context(app.router.lifespan_context(app))
        http = await stack.enter_async_context(
            httpx.AsyncClient(transport=HostTransport(apps))
        )
        yield Cluster(apps, backends, http)


def _request() -> InferenceRequest:
    return InferenceRequest(model="mock-model", prompt="hello world")


@pytest.mark.asyncio
async def test_round_robin_spreads_load(cluster: Cluster):
    router, _ = cluster.router(RoundRobinPolicy())

    responses = [await router.route(_request()) for _ in range(4)]

    assert all(r.output == "mock response to: hello world" for r in responses)
    assert [b.calls for b in cluster.backends.values()] == [2, 2]


@pytest.mark.asyncio
async def test_least_loaded_balances_concurrent_requests(cluster: Cluster):
    router, registry = cluster.router(LeastLoadedPolicy())

    await asyncio.gather(*(router.route(_request()) for _ in range(4)))

    assert [b.calls for b in cluster.backends.values()] == [2, 2]
    assert [w.in_flight for w in registry.all()] == [0, 0]


@pytest.mark.asyncio
async def test_fails_over_when_worker_stopped(cluster: Cluster):
    router, registry = cluster.router(RoundRobinPolicy())
    await cluster.apps["worker-a"].state.service.stop()

    response = await router.route(_request())

    assert response.output == "mock response to: hello world"
    assert cluster.backends["worker-b"].calls == 1
    assert {w.endpoint.id: w.healthy for w in registry.all()} == {
        "worker-a": False,
        "worker-b": True,
    }


@pytest.mark.asyncio
async def test_refresh_detects_stopped_worker(cluster: Cluster):
    router, registry = cluster.router(RoundRobinPolicy())
    await cluster.apps["worker-b"].state.service.stop()

    await router.refresh()

    assert {w.endpoint.id: w.healthy for w in registry.all()} == {
        "worker-a": True,
        "worker-b": False,
    }


@pytest.mark.asyncio
async def test_all_workers_down_raises(cluster: Cluster):
    router, _ = cluster.router(RoundRobinPolicy())
    for app in cluster.apps.values():
        await app.state.service.stop()

    with pytest.raises(NoAvailableWorkerError):
        await router.route(_request())


@pytest.mark.asyncio
async def test_backend_error_propagates_without_failover(cluster: Cluster):
    router, registry = cluster.router(RoundRobinPolicy())
    cluster.backends["worker-a"].fail_with = BackendError("engine crashed")

    with pytest.raises(BackendError, match="engine crashed"):
        await router.route(_request())

    assert cluster.backends["worker-b"].calls == 0
    assert all(w.healthy for w in registry.all())
