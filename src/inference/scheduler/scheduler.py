"""Scheduler: moves requests from the queue to the router."""

import asyncio
import logging
import time
from functools import partial
from typing import Protocol

from inference.core.errors import QueueClosedError, SchedulerStoppedError
from inference.core.schemas import InferenceRequest, InferenceResponse
from inference.queue.base import QueuedRequest, RequestQueue

logger = logging.getLogger(__name__)


class RequestRouter(Protocol):
    """Anything that can send a request to a worker (the Router)."""

    async def route(self, request: InferenceRequest) -> InferenceResponse:
        """Send the request and return the response."""
        ...


class Scheduler:
    """Dispatches queued requests to the router, with a limit on requests in flight.

    The scheduler resolves each request's future with the response or the error.
    It skips requests whose future was cancelled, and it cancels routing when the
    submitter cancels the future of a request that is already in flight.
    """

    def __init__(
        self,
        queue: RequestQueue,
        router: RequestRouter,
        max_concurrency: int = 8,
        shutdown_timeout_s: float = 30.0,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._queue = queue
        self._router = router
        self._shutdown_timeout_s = shutdown_timeout_s
        self._slots = asyncio.Semaphore(max_concurrency)
        self._inflight: dict[asyncio.Task[None], QueuedRequest] = {}
        self._loop_task: asyncio.Task[None] | None = None
        self._stopped = False

    async def start(self) -> None:
        """Start the dispatch loop."""
        if self._loop_task is not None:
            raise RuntimeError("scheduler already started")
        self._loop_task = asyncio.create_task(self._run())
        self._loop_task.add_done_callback(self._on_loop_done)

    async def stop(self) -> None:
        """Close the queue, drain it, and wait for requests in flight.

        When the shutdown timeout expires, the scheduler cancels the requests in
        flight and fails every request still queued with SchedulerStoppedError.
        """
        await self._queue.close()
        loop_task = self._loop_task
        if loop_task is None or self._stopped:
            return
        self._stopped = True

        deadline = time.monotonic() + self._shutdown_timeout_s
        await asyncio.wait({loop_task}, timeout=self._shutdown_timeout_s)
        while self._inflight:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.wait(set(self._inflight), timeout=remaining)

        if loop_task.done() and not self._inflight:
            logger.info("scheduler stopped")
            return
        logger.warning(
            "scheduler shutdown timeout; aborting",
            extra={"in_flight": len(self._inflight), "queued": self._queue.qsize()},
        )
        await self._abort(loop_task)

    async def _abort(self, loop_task: asyncio.Task[None]) -> None:
        loop_task.cancel()
        tasks = list(self._inflight)
        for task in tasks:
            future = self._inflight[task].future
            if not future.done():
                future.set_exception(SchedulerStoppedError("scheduler stopped"))
            task.cancel()
        await asyncio.gather(loop_task, *tasks, return_exceptions=True)
        while self._queue.qsize():
            item = await self._queue.get()
            if not item.future.done():
                item.future.set_exception(SchedulerStoppedError("scheduler stopped"))

    async def _run(self) -> None:
        while True:
            # Take a slot before taking a request. Requests then wait in the
            # priority-ordered queue, not inside the scheduler.
            await self._slots.acquire()
            try:
                item = await self._queue.get()
            except QueueClosedError:
                self._slots.release()
                return
            except BaseException:
                self._slots.release()
                raise
            if item.future.done():
                self._slots.release()
                logger.info(
                    "skipping cancelled request",
                    extra={"request_id": item.request.request_id},
                )
                continue
            task = asyncio.create_task(self._dispatch(item))
            self._inflight[task] = item
            task.add_done_callback(self._on_task_done)
            item.future.add_done_callback(partial(self._on_future_done, task))

    async def _dispatch(self, item: QueuedRequest) -> None:
        request_id = item.request.request_id
        logger.info(
            "dispatching request",
            extra={
                "request_id": request_id,
                "queue_wait_s": time.monotonic() - item.enqueued_at,
            },
        )
        try:
            response = await self._router.route(item.request)
        except asyncio.CancelledError:
            item.future.cancel()
            raise
        except Exception as exc:
            logger.exception("routing failed", extra={"request_id": request_id})
            if not item.future.done():
                item.future.set_exception(exc)
        else:
            if not item.future.done():
                item.future.set_result(response)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._inflight.pop(task, None)
        self._slots.release()

    @staticmethod
    def _on_future_done(
        task: asyncio.Task[None], future: asyncio.Future[object]
    ) -> None:
        if future.cancelled() and not task.done():
            task.cancel()

    @staticmethod
    def _on_loop_done(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("scheduler loop crashed", exc_info=error)
