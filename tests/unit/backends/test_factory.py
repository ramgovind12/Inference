"""Tests for the worker's backend factory."""

import httpx
import pytest
from pydantic import ValidationError

from inference.backends.mock import MockBackend
from inference.backends.vllm import VllmBackend
from inference.worker.app import create_app
from inference.worker.factory import build_backend
from inference.worker.settings import WorkerSettings


def test_default_backend_is_the_mock() -> None:
    assert isinstance(build_backend(WorkerSettings()), MockBackend)


def test_vllm_backend_is_built_from_settings() -> None:
    settings = WorkerSettings(backend="vllm", vllm_api_key="k")  # type: ignore[arg-type]

    assert isinstance(build_backend(settings), VllmBackend)


def test_empty_vllm_url_fails_at_startup() -> None:
    with pytest.raises(ValidationError):
        WorkerSettings(backend="vllm", vllm_base_url="")


def test_unknown_backend_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkerSettings(backend="gpt")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_an_injected_backend_wins_over_settings() -> None:
    app = create_app(MockBackend(), WorkerSettings(backend="vllm"))
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://w") as c,
    ):
        response = await c.post("/v1/inference", json={"model": "m", "prompt": "hello"})

    assert response.json()["output"] == "mock response to: hello"
