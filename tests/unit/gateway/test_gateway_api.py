import asyncio
import logging

import pytest

from inference.core.errors import (
    BackendError,
    NoAvailableWorkerError,
    SchedulerStoppedError,
)
from inference.queue.base import Priority

BODY = {"model": "m", "prompt": "hello"}


@pytest.mark.asyncio
async def test_success_returns_response_and_request_id_header(make_stack):
    stack = await make_stack()

    response = await stack.client.post("/v1/inference", json=BODY)

    assert response.status_code == 200
    body = response.json()
    assert body["output"] == "done hello"
    assert response.headers["X-Request-ID"] == body["request_id"]


@pytest.mark.asyncio
async def test_client_supplied_request_id_is_kept(make_stack):
    stack = await make_stack()

    response = await stack.client.post(
        "/v1/inference", json={**BODY, "request_id": "abc"}
    )

    assert response.headers["X-Request-ID"] == "abc"
    assert response.json()["request_id"] == "abc"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value, expected",
    [("high", Priority.HIGH), ("normal", Priority.NORMAL), ("low", Priority.LOW)],
)
async def test_priority_reaches_the_queue(make_stack, value: str, expected: Priority):
    stack = await make_stack()

    await stack.client.post("/v1/inference", json={**BODY, "priority": value})

    assert stack.queue.items[0].priority is expected


@pytest.mark.asyncio
async def test_default_priority_is_normal(make_stack):
    stack = await make_stack()
    await stack.client.post("/v1/inference", json=BODY)
    assert stack.queue.items[0].priority is Priority.NORMAL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {**BODY, "priority": "urgent"},
        {"model": "m", "prompt": ""},
        {"prompt": "hi"},
        {**BODY, "max_tokens": 0},
    ],
)
async def test_invalid_body_returns_422_and_queues_nothing(make_stack, payload: dict):
    stack = await make_stack()

    response = await stack.client.post("/v1/inference", json=payload)

    assert response.status_code == 422
    assert stack.queue.items == []


@pytest.mark.asyncio
async def test_health_needs_no_key(make_stack):
    stack = await make_stack()
    response = await stack.client.get("/health", headers={"Authorization": ""})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_missing_key_returns_401(make_stack):
    stack = await make_stack()

    response = await stack.client.post(
        "/v1/inference", json=BODY, headers={"Authorization": ""}
    )

    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert stack.queue.items == []


@pytest.mark.asyncio
async def test_x_api_key_header_is_accepted(make_stack):
    stack = await make_stack()

    response = await stack.client.post(
        "/v1/inference",
        json=BODY,
        headers={"Authorization": "", "X-API-Key": "secret-key"},
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_authentication_is_checked_before_body_validation(make_stack):
    stack = await make_stack()

    response = await stack.client.post(
        "/v1/inference",
        json={"model": "m", "prompt": "", "priority": "bogus"},
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_wrong_key_is_never_logged_or_echoed(make_stack, caplog):
    stack = await make_stack()

    with caplog.at_level(logging.DEBUG):
        response = await stack.client.post(
            "/v1/inference", json=BODY, headers={"Authorization": "Bearer wrong-secret"}
        )

    assert response.status_code == 401
    assert "wrong-secret" not in response.text
    assert "wrong-secret" not in caplog.text


@pytest.mark.asyncio
async def test_auth_disabled_allows_requests_without_key(make_stack):
    stack = await make_stack(auth_enabled=False, api_keys=[])

    response = await stack.client.post(
        "/v1/inference", json=BODY, headers={"Authorization": ""}
    )

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_returns_429_with_retry_after(make_stack):
    stack = await make_stack(rate_limit_rps=0.1, rate_limit_burst=2)

    statuses = [
        (await stack.client.post("/v1/inference", json=BODY)).status_code
        for _ in range(3)
    ]
    limited = await stack.client.post("/v1/inference", json=BODY)

    assert statuses == [200, 200, 429]
    assert limited.json()["error"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1
    assert len(stack.queue.items) == 2
    assert stack.router.calls == ["hello", "hello"]


@pytest.mark.asyncio
async def test_queue_full_returns_503_with_retry_after(make_stack):
    stack = await make_stack(with_scheduler=False, queue_size=1, request_timeout_s=0.3)
    first = asyncio.create_task(stack.client.post("/v1/inference", json=BODY))
    await asyncio.sleep(0.05)

    second = await stack.client.post("/v1/inference", json=BODY)

    assert second.status_code == 503
    assert second.json()["error"] == "queue_full"
    assert second.headers["Retry-After"] == "1"
    assert (await first).status_code == 504


@pytest.mark.asyncio
async def test_closed_queue_returns_503_shutting_down(make_stack):
    stack = await make_stack()
    await stack.queue.close()

    response = await stack.client.post("/v1/inference", json=BODY)

    assert response.status_code == 503
    assert response.json()["error"] == "shutting_down"


@pytest.mark.asyncio
async def test_timeout_returns_504_and_cancels_the_future(make_stack):
    stack = await make_stack(with_scheduler=False, request_timeout_s=0.05)

    response = await stack.client.post("/v1/inference", json=BODY)

    assert response.status_code == 504
    assert response.json()["error"] == "timeout"
    assert stack.queue.items[0].future.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc, status, code",
    [
        (
            NoAvailableWorkerError("worker-1 and worker-2 down"),
            503,
            "no_worker_available",
        ),
        (BackendError("worker http://10.0.0.5:8000 crashed"), 502, "backend_error"),
        (SchedulerStoppedError("stopped"), 503, "shutting_down"),
        (RuntimeError("secret internal detail"), 500, "internal_error"),
    ],
)
async def test_routing_errors_map_to_http_errors_without_leaking_detail(
    make_stack, exc: Exception, status: int, code: str
):
    stack = await make_stack()
    stack.router.errors["hello"] = exc

    response = await stack.client.post("/v1/inference", json=BODY)

    assert response.status_code == status
    assert response.json()["error"] == code
    for leaked in ("worker-1", "10.0.0.5", "secret internal detail"):
        assert leaked not in response.text
