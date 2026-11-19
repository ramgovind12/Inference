"""Tests for Prometheus metrics across gateway, scheduler, router and worker."""

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from inference.core.errors import (
    BackendError,
    NoAvailableWorkerError,
    WorkerUnreachableError,
)
from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.gateway.app import create_app
from inference.gateway.settings import GatewaySettings
from inference.observability.metrics import RouterMetrics, SchedulerMetrics
from inference.observability.settings import ObservabilitySettings
from inference.queue.memory import InMemoryRequestQueue
from inference.router.registry import WorkerEndpoint, WorkerRegistry, WorkerState
from inference.router.router import Router
from inference.scheduler.scheduler import Scheduler
from inference.worker.app import create_app as create_worker_app
from inference.worker.settings import WorkerSettings

BODY = {"model": "mock-model", "prompt": "hello"}


class FakeRouter:
    async def route(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            model=request.model,
            output="ok",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=2,
            latency_ms=1.0,
        )


class FirstPolicy:
    def select(self, candidates: Sequence[WorkerState]) -> WorkerState:
        return candidates[0]


@asynccontextmanager
async def running_gateway(
    registry: CollectorRegistry, **settings: Any
) -> AsyncIterator[httpx.AsyncClient]:
    queue = InMemoryRequestQueue(10)
    scheduler = Scheduler(queue, FakeRouter(), metrics=SchedulerMetrics(registry))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await scheduler.start()
        try:
            yield
        finally:
            await scheduler.stop()

    app = create_app(
        queue,
        GatewaySettings(auth_enabled=False, **settings),
        lifespan=lifespan,
        metrics_registry=registry,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            yield c


@pytest.mark.asyncio
async def test_success_is_counted() -> None:
    registry = CollectorRegistry()
    async with running_gateway(registry) as client:
        assert (await client.post("/v1/inference", json=BODY)).status_code == 200
    get = registry.get_sample_value
    assert get("inferscale_gateway_requests_total", {"status": "200"}) == 1
    assert get("inferscale_gateway_request_duration_seconds_count") == 1
    assert get("inferscale_scheduler_queue_wait_seconds_count") == 1


@pytest.mark.asyncio
async def test_validation_error_is_counted_as_422() -> None:
    registry = CollectorRegistry()
    async with running_gateway(registry) as client:
        await client.post("/v1/inference", json={"model": "m", "prompt": ""})
    value = registry.get_sample_value(
        "inferscale_gateway_requests_total", {"status": "422"}
    )
    assert value == 1


@pytest.mark.asyncio
async def test_rate_limit_counts_status_and_error_code() -> None:
    registry = CollectorRegistry()
    async with running_gateway(
        registry, rate_limit_rps=0.001, rate_limit_burst=1
    ) as client:
        await client.post("/v1/inference", json=BODY)
        assert (await client.post("/v1/inference", json=BODY)).status_code == 429
    get = registry.get_sample_value
    assert get("inferscale_gateway_requests_total", {"status": "429"}) == 1
    assert get("inferscale_gateway_errors_total", {"error": "rate_limited"}) == 1


@pytest.mark.asyncio
async def test_metrics_endpoint_serves_text_and_ignores_client_input() -> None:
    registry = CollectorRegistry()
    async with running_gateway(registry) as client:
        await client.post(
            "/v1/inference", json={"model": "weird-model-123", "prompt": "x"}
        )
        response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "inferscale_gateway_requests_total" in response.text
    assert "weird-model-123" not in response.text  # no client input in labels


@pytest.mark.asyncio
async def test_no_metrics_route_without_registry() -> None:
    app = create_app(InMemoryRequestQueue(1), GatewaySettings(auth_enabled=False))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        assert (await client.get("/metrics")).status_code == 404


def test_scrape_time_gauges_read_current_state() -> None:
    registry = CollectorRegistry()
    SchedulerMetrics(registry).bind_state(lambda: 3, lambda: 2)
    assert registry.get_sample_value("inferscale_queue_depth") == 3
    assert registry.get_sample_value("inferscale_scheduler_in_flight") == 2


def test_router_worker_gauges_read_current_state() -> None:
    registry = CollectorRegistry()
    state = {"healthy": True, "in_flight": 4}
    RouterMetrics(registry).bind_worker(
        "w1", lambda: state["healthy"], lambda: state["in_flight"]
    )
    labels = {"worker_id": "w1"}
    assert registry.get_sample_value("inferscale_router_worker_healthy", labels) == 1
    state["healthy"] = False
    assert registry.get_sample_value("inferscale_router_worker_healthy", labels) == 0
    assert registry.get_sample_value("inferscale_router_worker_in_flight", labels) == 4


class FakeClient:
    def __init__(self, behaviour: dict[str, Exception | None]) -> None:
        self._behaviour = behaviour

    async def infer(
        self, endpoint: WorkerEndpoint, request: InferenceRequest
    ) -> InferenceResponse:
        error = self._behaviour[endpoint.id]
        if error is not None:
            raise error
        return await FakeRouter().route(request)

    async def ready(self, endpoint: WorkerEndpoint) -> bool:
        return True


def make_router(
    registry: CollectorRegistry, behaviour: dict[str, Exception | None]
) -> Router:
    endpoints = [
        WorkerEndpoint(
            id=worker_id, base_url=f"http://{worker_id}", models=["mock-model"]
        )
        for worker_id in behaviour
    ]
    return Router(
        WorkerRegistry(endpoints),
        FakeClient(behaviour),
        FirstPolicy(),
        metrics=RouterMetrics(registry),
    )


@pytest.mark.asyncio
async def test_router_counts_failover_then_success() -> None:
    registry = CollectorRegistry()
    router = make_router(registry, {"a": WorkerUnreachableError("down"), "b": None})
    await router.route(InferenceRequest(**BODY))
    get = registry.get_sample_value
    failover = {"worker_id": "a", "outcome": "failover"}
    ok = {"worker_id": "b", "outcome": "ok"}
    assert get("inferscale_router_requests_total", failover) == 1
    assert get("inferscale_router_requests_total", ok) == 1
    assert (
        get("inferscale_router_request_duration_seconds_count", {"worker_id": "b"}) == 1
    )


@pytest.mark.asyncio
async def test_router_counts_backend_error_and_no_worker() -> None:
    registry = CollectorRegistry()
    router = make_router(registry, {"a": BackendError("boom")})
    with pytest.raises(BackendError):
        await router.route(InferenceRequest(**BODY))
    labels = {"worker_id": "a", "outcome": "backend_error"}
    assert registry.get_sample_value("inferscale_router_requests_total", labels) == 1

    with pytest.raises(NoAvailableWorkerError):
        await router.route(InferenceRequest(model="other", prompt="x"))
    assert registry.get_sample_value("inferscale_router_no_worker_total") == 1


@pytest.mark.asyncio
async def test_worker_metrics_after_a_request() -> None:
    app = create_worker_app(settings=WorkerSettings(mock_latency_s=0))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://w") as c:
            assert (await c.post("/v1/inference", json=BODY)).status_code == 200
            text = (await c.get("/metrics")).text
    assert 'inferscale_worker_requests_total{outcome="ok"} 1.0' in text
    assert 'inferscale_worker_tokens_total{type="completion"}' in text
    assert "inferscale_worker_ready 1.0" in text


@pytest.mark.asyncio
async def test_worker_metrics_can_be_disabled() -> None:
    app = create_worker_app(
        settings=WorkerSettings(mock_latency_s=0),
        observability=ObservabilitySettings(metrics_enabled=False),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://w") as c:
        assert (await c.get("/metrics")).status_code == 404
