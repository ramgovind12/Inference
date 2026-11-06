import pytest

from inference.router.registry import WorkerEndpoint, WorkerRegistry


def _ids(workers) -> list[str]:
    return [w.endpoint.id for w in workers]


def test_candidates_filter_by_model(registry: WorkerRegistry):
    assert _ids(registry.candidates("m1")) == ["a", "b"]
    assert _ids(registry.candidates("m2")) == ["b", "c"]


def test_candidates_unknown_model_is_empty(registry: WorkerRegistry):
    assert registry.candidates("nope") == []


def test_unhealthy_workers_excluded_and_restorable(registry: WorkerRegistry):
    registry.set_healthy("a", False)
    assert _ids(registry.candidates("m1")) == ["b"]
    registry.set_healthy("a", True)
    assert _ids(registry.candidates("m1")) == ["a", "b"]


def test_duplicate_worker_id_rejected():
    endpoint = WorkerEndpoint(id="a", base_url="http://a", models=["m"])
    with pytest.raises(ValueError, match="duplicate"):
        WorkerRegistry([endpoint, endpoint])


def test_endpoint_requires_models():
    with pytest.raises(ValueError):
        WorkerEndpoint(id="a", base_url="http://a", models=[])
