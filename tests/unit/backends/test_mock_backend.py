import pytest

from inference.backends.mock import MockBackend
from inference.core.errors import BackendUnavailableError
from inference.core.schemas import InferenceRequest


@pytest.mark.asyncio
async def test_generate_returns_structured_response(request_body: InferenceRequest):
    backend = MockBackend()
    await backend.start()

    response = await backend.generate(request_body)

    assert response.request_id == request_body.request_id
    assert response.model == "mock-model"
    assert response.output == "mock response to: hello world"
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 2
    assert response.completion_tokens == 5
    assert response.latency_ms >= 0


@pytest.mark.asyncio
async def test_ready_only_between_start_and_stop():
    backend = MockBackend()
    assert not await backend.is_ready()
    await backend.start()
    assert await backend.is_ready()
    await backend.stop()
    assert not await backend.is_ready()


@pytest.mark.asyncio
async def test_generate_before_start_raises(request_body: InferenceRequest):
    backend = MockBackend()
    with pytest.raises(BackendUnavailableError):
        await backend.generate(request_body)


@pytest.mark.asyncio
async def test_generate_raises_injected_failure(request_body: InferenceRequest):
    backend = MockBackend(fail_with=RuntimeError("boom"))
    await backend.start()
    with pytest.raises(RuntimeError, match="boom"):
        await backend.generate(request_body)
