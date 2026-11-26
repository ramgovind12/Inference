"""Tests for VllmBackend using httpx.MockTransport (no vLLM needed)."""

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from inference.backends.vllm import VllmBackend
from inference.core.errors import BackendError, BackendUnavailableError
from inference.core.schemas import InferenceRequest

Handler = Callable[[httpx.Request], httpx.Response]


def ok_body() -> dict[str, Any]:
    return {
        "choices": [{"text": "hi there", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


def make_backend(handler: Handler, **kwargs: Any) -> VllmBackend:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return VllmBackend("http://vllm:8000", client=client, **kwargs)


def request(**kwargs: Any) -> InferenceRequest:
    return InferenceRequest(model="m", prompt="hello", **kwargs)


@pytest.mark.asyncio
async def test_generate_maps_request_and_response() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=ok_body())

    backend = make_backend(handler)
    await backend.start()
    result = await backend.generate(
        request(max_tokens=8, parameters={"temperature": 0.2})
    )

    assert seen[0].url.path == "/v1/completions"
    assert json.loads(seen[0].content) == {
        "model": "m",
        "prompt": "hello",
        "max_tokens": 8,
        "temperature": 0.2,
    }
    assert result.output == "hi there"
    assert result.finish_reason == "stop"
    assert (result.prompt_tokens, result.completion_tokens) == (3, 2)
    assert result.model == "m"
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_max_tokens_is_omitted_when_not_set() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json=ok_body())

    backend = make_backend(handler)
    await backend.start()
    await backend.generate(request())

    assert "max_tokens" not in bodies[0]


@pytest.mark.asyncio
async def test_unsupported_parameters_are_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json=ok_body())

    backend = make_backend(handler)
    await backend.start()
    with caplog.at_level(logging.WARNING):
        await backend.generate(
            request(parameters={"temperature": 0.5, "best_of": 99, "model": "evil"})
        )

    assert bodies[0]["temperature"] == 0.5
    assert "best_of" not in bodies[0]
    assert bodies[0]["model"] == "m"  # a client cannot override the model
    assert "dropping unsupported parameters" in caplog.text


@pytest.mark.asyncio
async def test_api_key_is_sent_but_never_leaks_into_errors() -> None:
    headers: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        headers.append(req.headers.get("authorization"))
        return httpx.Response(500, text="engine exploded")

    backend = make_backend(handler, api_key="secret-key")
    await backend.start()
    with pytest.raises(BackendError) as error:
        await backend.generate(request())

    assert headers == ["Bearer secret-key"]
    assert "secret-key" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("slow connect"),
        httpx.PoolTimeout("no connection free"),
    ],
)
async def test_request_that_never_reached_vllm_is_unavailable(exc: Exception) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc

    backend = make_backend(handler)
    await backend.start()
    with pytest.raises(BackendUnavailableError):
        await backend.generate(request())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc", [httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("dropped")]
)
async def test_request_that_may_have_run_is_a_plain_backend_error(
    exc: Exception,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise exc

    backend = make_backend(handler)
    await backend.start()
    with pytest.raises(BackendError) as error:
        await backend.generate(request())

    assert type(error.value) is BackendError  # not retried by the router


@pytest.mark.asyncio
async def test_status_503_is_unavailable() -> None:
    backend = make_backend(lambda req: httpx.Response(503, text="loading"))
    await backend.start()

    with pytest.raises(BackendUnavailableError):
        await backend.generate(request())


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404, 500])
async def test_other_error_statuses_are_backend_errors(status: int) -> None:
    backend = make_backend(lambda req: httpx.Response(status, text="nope"))
    await backend.start()

    with pytest.raises(BackendError) as error:
        await backend.generate(request())

    assert type(error.value) is BackendError
    assert str(status) in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"choices": [], "usage": {}}),
        httpx.Response(
            200,
            json={"choices": [{"text": "x"}], "usage": {"prompt_tokens": "many"}},
        ),
    ],
)
async def test_malformed_response_is_a_backend_error(response: httpx.Response) -> None:
    backend = make_backend(lambda req: response)
    await backend.start()

    with pytest.raises(BackendError):
        await backend.generate(request())


@pytest.mark.asyncio
async def test_generate_before_start_is_unavailable() -> None:
    backend = make_backend(lambda req: httpx.Response(200, json=ok_body()))

    with pytest.raises(BackendUnavailableError):
        await backend.generate(request())


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "expected"), [(200, True), (503, False)])
async def test_is_ready_follows_health_status(status: int, expected: bool) -> None:
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(status)

    backend = make_backend(handler)
    await backend.start()

    assert await backend.is_ready() is expected
    assert paths == ["/health"]


@pytest.mark.asyncio
async def test_is_ready_is_false_when_vllm_is_down_or_not_started() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    backend = make_backend(handler)
    assert await backend.is_ready() is False  # not started
    await backend.start()
    assert await backend.is_ready() is False  # started, vLLM down
    await backend.stop()
    assert await backend.is_ready() is False  # stopped
