import pytest

from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.router.registry import WorkerEndpoint, WorkerRegistry


class FakeWorkerClient:
    """WorkerClient double: records calls, raises configured errors per worker."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.errors: dict[str, Exception] = {}
        self.ready_map: dict[str, bool] = {}
        self.ready_error: Exception | None = None

    async def ready(self, endpoint: WorkerEndpoint) -> bool:
        if self.ready_error is not None:
            raise self.ready_error
        return self.ready_map.get(endpoint.id, True)

    async def infer(
        self, endpoint: WorkerEndpoint, request: InferenceRequest
    ) -> InferenceResponse:
        self.calls.append(endpoint.id)
        if endpoint.id in self.errors:
            raise self.errors[endpoint.id]
        return InferenceResponse(
            request_id=request.request_id,
            model=request.model,
            output=f"from {endpoint.id}",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=0.0,
        )


@pytest.fixture
def fake_client() -> FakeWorkerClient:
    return FakeWorkerClient()


@pytest.fixture
def endpoints() -> list[WorkerEndpoint]:
    return [
        WorkerEndpoint(id="a", base_url="http://a", models=["m1"]),
        WorkerEndpoint(id="b", base_url="http://b", models=["m1", "m2"]),
        WorkerEndpoint(id="c", base_url="http://c", models=["m2"]),
    ]


@pytest.fixture
def registry(endpoints: list[WorkerEndpoint]) -> WorkerRegistry:
    return WorkerRegistry(endpoints)
