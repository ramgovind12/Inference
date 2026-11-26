"""Gateway -> queue -> scheduler -> router -> worker -> VllmBackend -> fake vLLM."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Response

from inference.backends.vllm import VllmBackend
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

BODY = {"model": "mock-model", "prompt": "hello world"}


def make_fake_vllm(
    calls: list[dict[str, Any]], state: dict[str, bool] | None = None
) -> FastAPI:
    state = state if state is not None else {"healthy": True}
    app = FastAPI()

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200 if state["healthy"] else 503)

    @app.post("/v1/completions")
    async def completions(body: dict[str, Any]) -> dict[str, Any]:
        calls.append(body)
        return {
            "choices": [{"text": f"fake:{body['prompt']}", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
        }

    return app


def refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def time_out(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("too slow", request=request)


class HostRouter(httpx.AsyncBaseTransport):
    """Sends each request to the ASGI app registered for its host name."""

    def __init__(self, apps: dict[str, FastAPI]) -> None:
        self._transports = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transports[request.url.host].handle_async_request(request)


@asynccontextmanager
async def running_stack(
    vllm_transports: dict[str, httpx.AsyncBaseTransport],
) -> AsyncIterator[tuple[httpx.AsyncClient, httpx.AsyncClient]]:
    """One worker per entry; each worker's vLLM is reached through its transport.

    Yields (gateway client, client that talks to the workers directly).
    """
    async with AsyncExitStack() as stack:
        worker_apps: dict[str, FastAPI] = {}
        for worker_id, transport in vllm_transports.items():
            vllm_http = await stack.enter_async_context(
                httpx.AsyncClient(transport=transport)
            )
            backend = VllmBackend("http://vllm", client=vllm_http)
            app = create_worker_app(backend, WorkerSettings())
            await stack.enter_async_context(app.router.lifespan_context(app))
            worker_apps[worker_id] = app

        worker_http = await stack.enter_async_context(
            httpx.AsyncClient(transport=HostRouter(worker_apps))
        )
        endpoints = [
            WorkerEndpoint(id=w, base_url=f"http://{w}", models=["mock-model"])
            for w in worker_apps
        ]
        router = Router(
            WorkerRegistry(endpoints),
            HttpWorkerClient(worker_http, 5.0),
            RoundRobinPolicy(),
        )
        queue = InMemoryRequestQueue(10)
        scheduler = Scheduler(queue, router)
        await scheduler.start()
        gateway = create_gateway_app(queue, GatewaySettings(auth_enabled=False))
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=gateway), base_url="http://gw"
            ) as client:
                yield client, worker_http
        finally:
            await scheduler.stop()


@pytest.mark.asyncio
async def test_request_reaches_vllm_and_back() -> None:
    calls: list[dict[str, Any]] = []
    fake = make_fake_vllm(calls)

    async with running_stack({"w1": httpx.ASGITransport(app=fake)}) as (client, _):
        response = await client.post("/v1/inference", json=BODY)

    assert response.status_code == 200
    assert response.json()["output"] == "fake:hello world"
    assert response.json()["completion_tokens"] == 2
    assert calls[0]["model"] == "mock-model"


@pytest.mark.asyncio
async def test_worker_is_not_ready_until_vllm_is_healthy() -> None:
    state = {"healthy": False}
    fake = make_fake_vllm([], state)

    async with running_stack({"w1": httpx.ASGITransport(app=fake)}) as (_, workers):
        assert (await workers.get("http://w1/ready")).status_code == 503
        state["healthy"] = True
        assert (await workers.get("http://w1/ready")).status_code == 200


@pytest.mark.asyncio
async def test_dead_vllm_on_one_worker_fails_over_to_the_other() -> None:
    calls: list[dict[str, Any]] = []
    transports: dict[str, httpx.AsyncBaseTransport] = {
        "w1": httpx.MockTransport(refuse),
        "w2": httpx.ASGITransport(app=make_fake_vllm(calls)),
    }

    async with running_stack(transports) as (client, _):
        response = await client.post("/v1/inference", json=BODY)

    assert response.status_code == 200
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_read_timeout_is_not_retried_on_another_worker() -> None:
    calls: list[dict[str, Any]] = []
    transports: dict[str, httpx.AsyncBaseTransport] = {
        "w1": httpx.MockTransport(time_out),
        "w2": httpx.ASGITransport(app=make_fake_vllm(calls)),
    }

    async with running_stack(transports) as (client, _):
        response = await client.post("/v1/inference", json=BODY)

    assert response.status_code == 502
    assert response.json()["error"] == "backend_error"
    assert calls == []  # w2 never saw the request
