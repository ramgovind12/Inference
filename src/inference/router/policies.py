"""Routing policies: choose one worker from a list of candidates."""

from collections.abc import Sequence
from typing import Protocol

from inference.core.errors import NoAvailableWorkerError
from inference.router.registry import WorkerState


class RoutingPolicy(Protocol):
    """Chooses a worker from non-empty candidates."""

    def select(self, candidates: Sequence[WorkerState]) -> WorkerState:
        """Return the chosen worker. Raises NoAvailableWorkerError if empty."""
        ...


class RoundRobinPolicy:
    """Cycles through candidates in order."""

    def __init__(self) -> None:
        self._next = 0

    def select(self, candidates: Sequence[WorkerState]) -> WorkerState:
        if not candidates:
            raise NoAvailableWorkerError("no candidate workers")
        worker = candidates[self._next % len(candidates)]
        self._next += 1
        return worker


class LeastLoadedPolicy:
    """Picks the worker with the fewest in-flight requests.

    Ties are broken round-robin so idle workers share load evenly.
    """

    def __init__(self) -> None:
        self._tie_breaker = RoundRobinPolicy()

    def select(self, candidates: Sequence[WorkerState]) -> WorkerState:
        if not candidates:
            raise NoAvailableWorkerError("no candidate workers")
        lowest = min(w.in_flight for w in candidates)
        return self._tie_breaker.select(
            [w for w in candidates if w.in_flight == lowest]
        )


def build_policy(name: str) -> RoutingPolicy:
    """Create a policy by its configured name."""
    policies: dict[str, type[RoundRobinPolicy | LeastLoadedPolicy]] = {
        "round_robin": RoundRobinPolicy,
        "least_loaded": LeastLoadedPolicy,
    }
    try:
        return policies[name]()
    except KeyError:
        raise ValueError(f"unknown routing policy: {name}") from None
