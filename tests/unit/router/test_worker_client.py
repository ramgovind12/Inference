import httpx
import pytest

from inference.core.errors import (
    BackendError,
    WorkerNotReadyError,
    WorkerUnreachableError,
)
from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.router.client import HttpWorkerClient
from inference.router.registry import WorkerEndpoint

ENDPOINT = WorkerEndpoint(id="w", base_url="http://worker", models=["m"])
REQUEST = InferenceRequest(model="m", prompt="hi")
RESPONSE = InferenceResponse(
    request_id=REQUEST.request_id,
    model="m",
    output="ok",
    finish_reason="stop",
    prompt_tokens=1,
    completion_tokens=1,
    latency_ms=1.0,
)


def _client(handler) -> HttpWorkerClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpWorkerClient(http, timeout_s=1.0)


@pytest.mark.asyncio
async def test_infer_success_posts_to_worker():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json=RESPONSE.model_dump())

    result = await _client(handler).infer(ENDPOINT, REQUEST)

    assert seen["url"] == "http://worker/v1/inference"
    assert result == RESPONSE


@pytest.mark.asyncio
async def test_503_maps_to_worker_not_ready():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, json={"error": "worker_not_ready", "detail": "worker is draining"}
        )

    with pytest.raises(WorkerNotReadyError, match="draining"):
        await _client(handler).infer(ENDPOINT, REQUEST)


@pytest.mark.asyncio
async def test_502_maps_to_backend_error_with_detail():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": "backend_error", "detail": "crash"})

    with pytest.raises(BackendError, match="crash"):
        await _client(handler).infer(ENDPOINT, REQUEST)


@pytest.mark.asyncio
async def test_unexpected_status_maps_to_backend_error():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="oops")

    with pytest.raises(BackendError, match="500"):
        await _client(handler).infer(ENDPOINT, REQUEST)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")]
)
async def test_transport_errors_map_to_unreachable(exc: Exception):
    def handler(_: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(WorkerUnreachableError):
        await _client(handler).infer(ENDPOINT, REQUEST)


@pytest.mark.asyncio
@pytest.mark.parametrize("status, expected", [(200, True), (503, False)])
async def test_ready_reflects_status(status: int, expected: bool):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={})

    assert await _client(handler).ready(ENDPOINT) is expected


@pytest.mark.asyncio
async def test_ready_false_when_unreachable():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert await _client(handler).ready(ENDPOINT) is False
