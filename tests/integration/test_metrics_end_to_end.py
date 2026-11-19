"""A request travels gateway -> queue -> scheduler -> router -> worker; metrics follow."""

from collections.abc import Sequence

import httpx
import pytest
from prometheus_client import CollectorRegistry

from inference.gateway.app import create_app
from inference.gateway.settings import GatewaySettings
from inference.observability.metrics import RouterMetrics, SchedulerMetrics
from inference.queue.memory import InMemoryRequestQueue
from inference.router.client import HttpWorkerClient
from inference.router.registry import WorkerEndpoint, WorkerRegistry, WorkerState
from inference.router.router import Router
from inference.scheduler.scheduler import Scheduler
from inference.worker.app import create_app as create_worker_app
from inference.worker.settings import WorkerSettings


class FirstPolicy:
    def select(self, candidates: Sequence[WorkerState]) -> WorkerState:
        return candidates[0]


@pytest.mark.asyncio
async def test_metrics_follow_a_request_through_the_stack() -> None:
    worker_app = create_worker_app(settings=WorkerSettings(mock_latency_s=0))
    registry = CollectorRegistry()
    endpoint = WorkerEndpoint(id="w1", base_url="http://worker", models=["mock-model"])
    queue = InMemoryRequestQueue(10)

    async with worker_app.router.lifespan_context(worker_app):
        worker_transport = httpx.ASGITransport(app=worker_app)
        async with httpx.AsyncClient(transport=worker_transport) as worker_http:
            router = Router(
                WorkerRegistry([endpoint]),
                HttpWorkerClient(worker_http, timeout_s=5),
                FirstPolicy(),
                metrics=RouterMetrics(registry),
            )
            scheduler = Scheduler(queue, router, metrics=SchedulerMetrics(registry))
            await scheduler.start()
            app = create_app(
                queue, GatewaySettings(auth_enabled=False), metrics_registry=registry
            )
            gateway_transport = httpx.ASGITransport(app=app)
            try:
                async with httpx.AsyncClient(
                    transport=gateway_transport, base_url="http://gw"
                ) as gateway:
                    body = {"model": "mock-model", "prompt": "hi"}
                    response = await gateway.post("/v1/inference", json=body)
                    assert response.status_code == 200
                    gateway_text = (await gateway.get("/metrics")).text
                worker_text = (await worker_http.get("http://worker/metrics")).text
            finally:
                await scheduler.stop()

    get = registry.get_sample_value
    assert get("inferscale_gateway_requests_total", {"status": "200"}) == 1
    ok = {"worker_id": "w1", "outcome": "ok"}
    assert get("inferscale_router_requests_total", ok) == 1
    assert "inferscale_scheduler_queue_wait_seconds_count 1.0" in gateway_text
    assert 'inferscale_worker_requests_total{outcome="ok"} 1.0' in worker_text
