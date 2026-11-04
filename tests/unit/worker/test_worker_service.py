import asyncio

import pytest

from inference.backends.mock import MockBackend
from inference.core.errors import (
    BackendError,
    BackendUnavailableError,
    WorkerNotReadyError,
)
from inference.core.schemas import InferenceRequest
from inference.worker.service import WorkerService, WorkerState


@pytest.mark.asyncio
async def test_lifecycle_start_and_stop():
    service = WorkerService(MockBackend())
    assert service.state is WorkerState.STARTING
    assert not await service.is_ready()

    await service.start()
    assert service.state is WorkerState.READY
    assert await service.is_ready()

    await service.stop()
    assert service.state is WorkerState.STOPPED
    assert not await service.is_ready()


@pytest.mark.asyncio
async def test_start_twice_raises():
    service = WorkerService(MockBackend())
    await service.start()
    with pytest.raises(RuntimeError):
        await service.start()


@pytest.mark.asyncio
async def test_stop_is_idempotent():
    service = WorkerService(MockBackend())
    await service.start()
    await service.stop()
    await service.stop()
    assert service.state is WorkerState.STOPPED


@pytest.mark.asyncio
async def test_backend_start_failure_stops_worker():
    class FailingStart(MockBackend):
        async def start(self) -> None:
            raise BackendUnavailableError("no engine")

    service = WorkerService(FailingStart())
    with pytest.raises(BackendUnavailableError):
        await service.start()
    assert service.state is WorkerState.STOPPED


@pytest.mark.asyncio
async def test_infer_before_start_raises(request_body: InferenceRequest):
    service = WorkerService(MockBackend())
    with pytest.raises(WorkerNotReadyError):
        await service.infer(request_body)


@pytest.mark.asyncio
async def test_infer_returns_response(request_body: InferenceRequest):
    service = WorkerService(MockBackend())
    await service.start()
    response = await service.infer(request_body)
    assert response.request_id == request_body.request_id
    assert service.active_requests == 0


@pytest.mark.asyncio
async def test_backend_error_propagates_and_clears_active(
    request_body: InferenceRequest,
):
    service = WorkerService(MockBackend(fail_with=BackendError("bad")))
    await service.start()
    with pytest.raises(BackendError, match="bad"):
        await service.infer(request_body)
    assert service.active_requests == 0


@pytest.mark.asyncio
async def test_unexpected_exception_wrapped_as_backend_error(
    request_body: InferenceRequest,
):
    service = WorkerService(MockBackend(fail_with=ValueError("oops")))
    await service.start()
    with pytest.raises(BackendError, match="oops") as exc_info:
        await service.infer(request_body)
    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_active_requests_tracked_while_in_flight(request_body: InferenceRequest):
    service = WorkerService(MockBackend(latency_s=0.1))
    await service.start()
    tasks = [asyncio.create_task(service.infer(request_body)) for _ in range(3)]
    await asyncio.sleep(0.02)
    assert service.active_requests == 3
    await asyncio.gather(*tasks)
    assert service.active_requests == 0


@pytest.mark.asyncio
async def test_stop_drains_in_flight_and_rejects_new(request_body: InferenceRequest):
    service = WorkerService(MockBackend(latency_s=0.2), drain_timeout_s=5)
    await service.start()
    in_flight = asyncio.create_task(service.infer(request_body))
    await asyncio.sleep(0.05)

    stopping = asyncio.create_task(service.stop())
    await asyncio.sleep(0.02)
    assert service.state is WorkerState.DRAINING
    assert not await service.is_ready()
    with pytest.raises(WorkerNotReadyError):
        await service.infer(request_body)

    await stopping
    response = await in_flight
    assert response.output.startswith("mock response")
    assert service.state is WorkerState.STOPPED


@pytest.mark.asyncio
async def test_stop_gives_up_after_drain_timeout(request_body: InferenceRequest):
    service = WorkerService(MockBackend(latency_s=0.5), drain_timeout_s=0.05)
    await service.start()
    in_flight = asyncio.create_task(service.infer(request_body))
    await asyncio.sleep(0.02)

    await service.stop()

    assert service.state is WorkerState.STOPPED
    assert service.active_requests == 1
    await in_flight
