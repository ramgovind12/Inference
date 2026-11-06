"""Registry of known workers and their router-side state."""

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, Field


class WorkerEndpoint(BaseModel):
    """Static description of a worker: where it lives and which models it serves."""

    id: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    models: list[str] = Field(min_length=1)


@dataclass
class WorkerState:
    """Mutable router-side view of a worker."""

    endpoint: WorkerEndpoint
    healthy: bool = True
    in_flight: int = 0


class WorkerRegistry:
    """Holds worker endpoints with their health and in-flight request counts."""

    def __init__(self, endpoints: Iterable[WorkerEndpoint]) -> None:
        self._workers: dict[str, WorkerState] = {}
        for endpoint in endpoints:
            if endpoint.id in self._workers:
                raise ValueError(f"duplicate worker id: {endpoint.id}")
            self._workers[endpoint.id] = WorkerState(endpoint)

    def all(self) -> list[WorkerState]:
        """Return every registered worker."""
        return list(self._workers.values())

    def candidates(self, model: str) -> list[WorkerState]:
        """Return healthy workers that serve `model`."""
        return [
            w
            for w in self._workers.values()
            if w.healthy and model in w.endpoint.models
        ]

    def set_healthy(self, worker_id: str, healthy: bool) -> None:
        """Update a worker's health flag."""
        self._workers[worker_id].healthy = healthy
