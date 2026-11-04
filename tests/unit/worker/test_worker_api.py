from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from inference.backends.mock import MockBackend
from inference.core.errors import BackendError
from inference.worker.app import create_app
from inference.worker.service import WorkerService, WorkerState
from inference.worker.settings import WorkerSettings

PAYLOAD = {"model": "mock-model", "prompt": "hello world"}


async def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker"
    )


@pytest_asyncio.fixture
async def backend() -> MockBackend:
    return MockBackend()


@pytest_asyncio.fixture
async def app(backend: MockBackend) -> AsyncIterator[FastAPI]:
    application = create_app(backend, WorkerSettings())
    async with application.router.lifespan_context(application):
        yield application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with await _client(app) as c:
        yield c


@pytest.mark.asyncio
async def test_health_ok(client: httpx.AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ready_when_started(client: httpx.AsyncClient):
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_ready_503_when_backend_not_ready(
    client: httpx.AsyncClient, backend: MockBackend
):
    await backend.stop()
    response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


@pytest.mark.asyncio
async def test_inference_success(client: httpx.AsyncClient):
    response = await client.post("/v1/inference", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["output"] == "mock response to: hello world"
    assert body["model"] == "mock-model"
    assert body["request_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "hi"},
        {"model": "m"},
        {"model": "m", "prompt": ""},
        {"model": "m", "prompt": "hi", "max_tokens": 0},
    ],
)
async def test_inference_invalid_request_422(client: httpx.AsyncClient, payload: dict):
    response = await client.post("/v1/inference", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_inference_backend_error_502(
    client: httpx.AsyncClient, backend: MockBackend
):
    backend.fail_with = BackendError("engine crashed")
    response = await client.post("/v1/inference", json=PAYLOAD)
    assert response.status_code == 502
    assert response.json() == {"error": "backend_error", "detail": "engine crashed"}


@pytest.mark.asyncio
async def test_inference_unexpected_error_502(
    client: httpx.AsyncClient, backend: MockBackend
):
    backend.fail_with = RuntimeError("boom")
    response = await client.post("/v1/inference", json=PAYLOAD)
    assert response.status_code == 502
    assert response.json()["error"] == "backend_error"


@pytest.mark.asyncio
async def test_inference_backend_unavailable_503(
    client: httpx.AsyncClient, backend: MockBackend
):
    await backend.stop()
    response = await client.post("/v1/inference", json=PAYLOAD)
    assert response.status_code == 503
    assert response.json()["error"] == "backend_unavailable"


@pytest.mark.asyncio
async def test_inference_503_while_draining(app: FastAPI, client: httpx.AsyncClient):
    service: WorkerService = app.state.service
    await service.stop()
    assert service.state is WorkerState.STOPPED

    response = await client.post("/v1/inference", json=PAYLOAD)
    assert response.status_code == 503
    assert response.json()["error"] == "worker_not_ready"


@pytest.mark.asyncio
async def test_lifespan_shutdown_stops_worker(backend: MockBackend):
    application = create_app(backend, WorkerSettings())
    async with application.router.lifespan_context(application):
        assert application.state.service.state is WorkerState.READY
    assert application.state.service.state is WorkerState.STOPPED
    assert not await backend.is_ready()
