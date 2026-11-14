import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from inference import main as main_module
from inference.core.errors import QueueClosedError
from inference.core.schemas import InferenceRequest
from inference.main import create_control_plane_app, main
from inference.worker.app import create_app as create_worker_app

KEY = "test-key"


@pytest.fixture
def control_plane_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a control plane whose only worker refuses every connection.

    The control plane's own HTTP client is replaced, so the tests never touch
    the network (a refused loopback connection takes about 2 s on Windows).
    """
    monkeypatch.setenv("INFERENCE_GATEWAY_API_KEYS", json.dumps([KEY]))
    monkeypatch.setenv(
        "INFERENCE_ROUTER_WORKERS",
        json.dumps([{"id": "w", "base_url": "http://worker", "models": ["m"]}]),
    )

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def refusing_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(refuse))

    monkeypatch.setattr(
        main_module, "httpx", SimpleNamespace(AsyncClient=refusing_client)
    )


@pytest.fixture
def run_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run(app: object, **kwargs: Any) -> None:
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(main_module.uvicorn, "run", fake_run)
    monkeypatch.setattr(main_module, "configure_logging", lambda level: None)
    return calls


@pytest.mark.asyncio
async def test_control_plane_serves_health_and_stops_cleanly(control_plane_env):
    app = create_control_plane_app()
    queue = app.state.queue

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            response = await c.get("/health")
        assert response.status_code == 200

    with pytest.raises(QueueClosedError):
        await queue.put(InferenceRequest(model="m", prompt="late"))


@pytest.mark.asyncio
async def test_control_plane_requires_authentication(control_plane_env):
    app = create_control_plane_app()

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            response = await c.post("/v1/inference", json={"model": "m", "prompt": "x"})

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_unreachable_worker_gives_503_not_a_hang(
    control_plane_env, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("INFERENCE_ROUTER_HEALTH_INTERVAL_S", "60")
    app = create_control_plane_app()

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
            response = await c.post(
                "/v1/inference",
                json={"model": "m", "prompt": "x"},
                headers={"Authorization": f"Bearer {KEY}"},
            )

    # The only worker refuses connections, so the router fails over and gives up.
    assert response.status_code == 503
    assert response.json()["error"] == "no_worker_available"


def test_control_plane_fails_fast_without_api_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("INFERENCE_GATEWAY_API_KEYS", raising=False)

    with pytest.raises(ValueError, match="no API keys"):
        create_control_plane_app()


def test_serve_runs_the_control_plane_factory(run_calls):
    main(["serve", "--port", "9000", "--host", "0.0.0.0"])

    (call,) = run_calls
    assert call["app"] is create_control_plane_app
    assert call["factory"] is True
    assert (call["host"], call["port"]) == ("0.0.0.0", 9000)
    assert call["log_config"] is None


def test_worker_runs_the_worker_factory_with_defaults(run_calls):
    main(["worker"])

    (call,) = run_calls
    assert call["app"] is create_worker_app
    assert (call["host"], call["port"]) == ("127.0.0.1", 8001)


def test_serve_default_port_is_8000(run_calls):
    main(["serve"])

    assert run_calls[0]["port"] == 8000


def test_log_level_is_passed_to_logging(monkeypatch: pytest.MonkeyPatch):
    levels: list[str] = []
    monkeypatch.setattr(main_module, "configure_logging", levels.append)
    monkeypatch.setattr(main_module.uvicorn, "run", lambda *a, **k: None)

    main(["worker", "--log-level", "debug"])

    assert levels == ["debug"]


@pytest.mark.parametrize("argv", [[], ["unknown"], ["serve", "--port", "abc"]])
def test_invalid_arguments_exit_with_error(argv: list[str], run_calls):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert exit_info.value.code == 2
    assert run_calls == []
