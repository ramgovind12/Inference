"""In-memory priority queue with a hard size limit."""

import asyncio
import heapq
import itertools
import logging
import time

from inference.core.errors import QueueClosedError, QueueFullError
from inference.core.schemas import InferenceRequest
from inference.queue.base import Priority, QueuedRequest

logger = logging.getLogger(__name__)


class InMemoryRequestQueue:
    """Priority queue, FIFO within a priority level.

    Items are lost if the process exits.
    """

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self._max_size = max_size
        self._heap: list[tuple[int, int, QueuedRequest]] = []
        self._sequence = itertools.count()
        self._closed = False
        self._condition = asyncio.Condition()

    async def put(
        self, request: InferenceRequest, priority: Priority = Priority.NORMAL
    ) -> QueuedRequest:
        async with self._condition:
            if self._closed:
                raise QueueClosedError("queue is closed")
            if len(self._heap) >= self._max_size:
                logger.warning(
                    "queue full, rejecting request",
                    extra={
                        "request_id": request.request_id,
                        "max_size": self._max_size,
                    },
                )
                raise QueueFullError(f"queue is full (max_size={self._max_size})")
            item = QueuedRequest(
                request=request,
                priority=priority,
                enqueued_at=time.monotonic(),
                future=asyncio.get_running_loop().create_future(),
            )
            heapq.heappush(self._heap, (priority, next(self._sequence), item))
            self._condition.notify()
            return item

    async def get(self) -> QueuedRequest:
        async with self._condition:
            while not self._heap:
                if self._closed:
                    raise QueueClosedError("queue is closed and empty")
                try:
                    await self._condition.wait()
                except asyncio.CancelledError:
                    # A notification may have been consumed by this cancelled
                    # waiter; pass it on so the item is not stranded.
                    if self._heap:
                        self._condition.notify()
                    raise
            return heapq.heappop(self._heap)[2]

    def qsize(self) -> int:
        return len(self._heap)

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
