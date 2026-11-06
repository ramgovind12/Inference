import asyncio

import pytest

from inference.core.errors import (
    BackendError,
    NoAvailableWorkerError,
    WorkerNotReadyError,
    WorkerUnreachableError,
)
from inference.core.schemas import InferenceRequest
from inference.router.policies import RoundRobinPolicy
from inference.router.registry import WorkerRegistry
from inference.router.router import Router


def _router(registry, client, **kwargs) -> Router:
    return Router(registry, client, RoundRobinPolicy(), **kwargs)


def _request(model: str = "m1") -> InferenceRequest:
    return InferenceRequest(model=model, prompt="hi")


def _healthy(registry: WorkerRegistry) -> dict[str, bool]:
    return {w.endpoint.id: w.healthy for w in registry.all()}


@pytest.mark.asyncio
async def test_routes_only_to_workers_serving_model(registry, fake_client):
    router = _router(registry, fake_client)
    for _ in range(4):
        await router.route(_request("m2"))
    assert set(fake_client.calls) == {"b", "c"}


@pytest.mark.asyncio
async def test_round_robin_spreads_requests(registry, fake_client):
    router = _router(registry, fake_client)
    for _ in range(4):
        await router.route(_request("m1"))
    assert fake_client.calls == ["a", "b", "a", "b"]


@pytest.mark.asyncio
async def test_unknown_model_raises_no_available_worker(registry, fake_client):
    router = _router(registry, fake_client)
    with pytest.raises(NoAvailableWorkerError, match="serves model"):
        await router.route(_request("nope"))
    assert fake_client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [WorkerUnreachableError("down"), WorkerNotReadyError("draining")]
)
async def test_fails_over_and_marks_worker_unhealthy(registry, fake_client, error):
    fake_client.errors["a"] = error
    router = _router(registry, fake_client)

    response = await router.route(_request("m1"))

    assert response.output == "from b"
    assert fake_client.calls == ["a", "b"]
    assert _healthy(registry) == {"a": False, "b": True, "c": True}


@pytest.mark.asyncio
async def test_unhealthy_worker_skipped_on_later_requests(registry, fake_client):
    fake_client.errors["a"] = WorkerUnreachableError("down")
    router = _router(registry, fake_client)
    await router.route(_request("m1"))
    fake_client.calls.clear()

    await router.route(_request("m1"))

    assert fake_client.calls == ["b"]


@pytest.mark.asyncio
async def test_backend_error_is_not_retried(registry, fake_client):
    fake_client.errors["a"] = BackendError("engine crashed")
    router = _router(registry, fake_client)

    with pytest.raises(BackendError, match="engine crashed"):
        await router.route(_request("m1"))

    assert fake_client.calls == ["a"]
    assert _healthy(registry)["a"] is True


@pytest.mark.asyncio
async def test_all_workers_failing_raises_with_cause(registry, fake_client):
    fake_client.errors["a"] = WorkerUnreachableError("a down")
    fake_client.errors["b"] = WorkerNotReadyError("b draining")
    router = _router(registry, fake_client)

    with pytest.raises(NoAvailableWorkerError, match="2 attempted") as exc_info:
        await router.route(_request("m1"))

    assert isinstance(exc_info.value.__cause__, WorkerNotReadyError)
    assert fake_client.calls == ["a", "b"]


@pytest.mark.asyncio
async def test_max_attempts_limits_failover(registry, fake_client):
    fake_client.errors["a"] = WorkerUnreachableError("down")
    router = _router(registry, fake_client, max_attempts=1)

    with pytest.raises(NoAvailableWorkerError):
        await router.route(_request("m1"))

    assert fake_client.calls == ["a"]
    assert _healthy(registry)["b"] is True


@pytest.mark.asyncio
async def test_in_flight_returns_to_zero_after_success_and_failure(
    registry, fake_client
):
    fake_client.errors["a"] = BackendError("boom")
    router = _router(registry, fake_client)

    with pytest.raises(BackendError):
        await router.route(_request("m1"))  # first pick is a, which raises
    await router.route(_request("m2"))  # served by c

    assert [w.in_flight for w in registry.all()] == [0, 0, 0]


@pytest.mark.asyncio
async def test_refresh_marks_workers_down_and_up(registry, fake_client):
    router = _router(registry, fake_client)

    fake_client.ready_map = {"a": False}
    await router.refresh()
    assert _healthy(registry) == {"a": False, "b": True, "c": True}

    fake_client.ready_map = {}
    await router.refresh()
    assert _healthy(registry) == {"a": True, "b": True, "c": True}


@pytest.mark.asyncio
async def test_health_loop_updates_health_and_stops(registry, fake_client):
    fake_client.ready_map = {"a": False}
    router = _router(registry, fake_client, health_interval_s=0.01)

    await router.start()
    await asyncio.sleep(0.1)
    assert _healthy(registry)["a"] is False

    fake_client.ready_map = {}
    await asyncio.sleep(0.1)
    assert _healthy(registry)["a"] is True

    await router.stop()
    fake_client.ready_map = {"a": False}
    await asyncio.sleep(0.05)
    assert _healthy(registry)["a"] is True


@pytest.mark.asyncio
async def test_health_loop_survives_check_failure(registry, fake_client):
    fake_client.ready_error = RuntimeError("check exploded")
    router = _router(registry, fake_client, health_interval_s=0.01)

    await router.start()
    await asyncio.sleep(0.05)
    fake_client.ready_error = None
    fake_client.ready_map = {"a": False}
    await asyncio.sleep(0.1)
    await router.stop()

    assert _healthy(registry)["a"] is False


@pytest.mark.asyncio
async def test_stop_without_start_is_noop(registry, fake_client):
    await _router(registry, fake_client).stop()
