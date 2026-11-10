"""Request queue interface and shared types."""

import asyncio
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from inference.core.schemas import InferenceRequest, InferenceResponse


class Priority(IntEnum):
    """Request priority. Lower values are dequeued first."""

    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass
class QueuedRequest:
    """A buffered request together with the future its submitter awaits.

    The queue only creates the future. Whoever dispatches the request
    (the scheduler) is responsible for resolving it.
    """

    request: InferenceRequest
    priority: Priority
    enqueued_at: float
    future: asyncio.Future[InferenceResponse]


class RequestQueue(Protocol):
    """Bounded, priority-ordered buffer of inference requests."""

    async def put(
        self, request: InferenceRequest, priority: Priority = Priority.NORMAL
    ) -> QueuedRequest:
        """Enqueue a request.

        Raises QueueFullError at capacity and QueueClosedError once closed.
        """
        ...

    async def get(self) -> QueuedRequest:
        """Wait for and return the next request.

        Raises QueueClosedError once the queue is closed and empty.
        """
        ...

    def qsize(self) -> int:
        """Return the number of buffered requests."""
        ...

    async def close(self) -> None:
        """Reject new requests, let buffered ones drain, wake waiting consumers."""
        ...
