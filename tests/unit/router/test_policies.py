import pytest

from inference.core.errors import NoAvailableWorkerError
from inference.router.policies import (
    LeastLoadedPolicy,
    RoundRobinPolicy,
    build_policy,
)
from inference.router.registry import WorkerEndpoint, WorkerState


def _workers(*in_flight: int) -> list[WorkerState]:
    return [
        WorkerState(
            WorkerEndpoint(id=f"w{i}", base_url=f"http://w{i}", models=["m"]),
            in_flight=n,
        )
        for i, n in enumerate(in_flight)
    ]


def test_round_robin_cycles_in_order():
    policy = RoundRobinPolicy()
    workers = _workers(0, 0, 0)
    picked = [policy.select(workers).endpoint.id for _ in range(4)]
    assert picked == ["w0", "w1", "w2", "w0"]


def test_least_loaded_picks_lowest_in_flight():
    workers = _workers(3, 1, 2)
    assert LeastLoadedPolicy().select(workers).endpoint.id == "w1"


def test_least_loaded_rotates_among_ties():
    policy = LeastLoadedPolicy()
    workers = _workers(0, 0, 5)
    picked = [policy.select(workers).endpoint.id for _ in range(4)]
    assert picked == ["w0", "w1", "w0", "w1"]


@pytest.mark.parametrize("policy", [RoundRobinPolicy(), LeastLoadedPolicy()])
def test_empty_candidates_raise(policy):
    with pytest.raises(NoAvailableWorkerError):
        policy.select([])


def test_build_policy_by_name():
    assert isinstance(build_policy("round_robin"), RoundRobinPolicy)
    assert isinstance(build_policy("least_loaded"), LeastLoadedPolicy)


def test_build_policy_unknown_name():
    with pytest.raises(ValueError, match="unknown routing policy"):
        build_policy("random")
