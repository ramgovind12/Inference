import asyncio

import pytest

from inference.core.errors import (
    BackendError,
    QueueClosedError,
    SchedulerStoppedError,
)
from inference.queue.base import Priority
from inference.scheduler.scheduler import Scheduler


async def _settle(seconds: float = 0.05) -> None:
    await asyncio.sleep(seconds)


def test_invalid_max_concurrency_rejected(queue, router):
    with pytest.raises(ValueError, match="max_concurrency"):
        Scheduler(queue, router, max_concurrency=0)


@pytest.mark.asyncio
async def test_dispatches_request_and_resolves_future(
    queue, router, make_scheduler, request_for
):
    scheduler = make_scheduler()
    await scheduler.start()

    item = await queue.put(request_for("a"))
    response = await asyncio.wait_for(item.future, timeout=1)

    assert response.output == "done a"
    assert router.calls == ["a"]


@pytest.mark.asyncio
async def test_router_error_reaches_future_and_scheduler_continues(
    queue, router, make_scheduler, request_for
):
    router.errors["bad"] = BackendError("engine crashed")
    scheduler = make_scheduler()
    await scheduler.start()

    bad = await queue.put(request_for("bad"))
    good = await queue.put(request_for("good"))

    with pytest.raises(BackendError, match="engine crashed"):
        await asyncio.wait_for(bad.future, timeout=1)
    assert (await asyncio.wait_for(good.future, timeout=1)).output == "done good"


@pytest.mark.asyncio
async def test_concurrency_never_exceeds_limit(
    queue, router, make_scheduler, request_for
):
    router.gate = asyncio.Event()
    scheduler = make_scheduler(max_concurrency=2)
    items = [await queue.put(request_for(f"r{i}")) for i in range(5)]
    await scheduler.start()
    await _settle()

    assert router.in_flight == 2
    assert queue.qsize() == 3

    router.gate.set()
    await asyncio.wait_for(asyncio.gather(*(i.future for i in items)), timeout=2)
    assert router.peak == 2


@pytest.mark.asyncio
async def test_priority_order_holds_under_concurrency_limit(
    queue, router, make_scheduler, request_for
):
    router.gate = asyncio.Event()
    scheduler = make_scheduler(max_concurrency=1)
    first = await queue.put(request_for("first"))
    await scheduler.start()
    await _settle()

    low = await queue.put(request_for("low"), Priority.LOW)
    normal = await queue.put(request_for("normal"), Priority.NORMAL)
    high = await queue.put(request_for("high"), Priority.HIGH)
    router.gate.set()

    await asyncio.wait_for(
        asyncio.gather(first.future, low.future, normal.future, high.future),
        timeout=2,
    )
    assert router.calls == ["first", "high", "normal", "low"]


@pytest.mark.asyncio
async def test_future_cancelled_while_queued_is_skipped(
    queue, router, make_scheduler, request_for
):
    scheduler = make_scheduler()
    skipped = await queue.put(request_for("skip"))
    kept = await queue.put(request_for("keep"))
    skipped.future.cancel()

    await scheduler.start()
    await asyncio.wait_for(kept.future, timeout=1)

    assert router.calls == ["keep"]


@pytest.mark.asyncio
async def test_future_cancelled_in_flight_aborts_routing_and_frees_slot(
    queue, router, make_scheduler, request_for
):
    router.gate = asyncio.Event()
    scheduler = make_scheduler(max_concurrency=1)
    await scheduler.start()
    stuck = await queue.put(request_for("stuck"))
    await _settle()
    assert router.in_flight == 1

    stuck.future.cancel()
    await _settle()

    assert router.cancelled == ["stuck"]
    assert router.in_flight == 0

    router.gate.set()
    follow_up = await queue.put(request_for("next"))
    assert (await asyncio.wait_for(follow_up.future, timeout=1)).output == "done next"


@pytest.mark.asyncio
async def test_stop_drains_queued_requests(queue, router, make_scheduler, request_for):
    scheduler = make_scheduler(max_concurrency=2)
    await scheduler.start()
    items = [await queue.put(request_for(f"r{i}")) for i in range(5)]

    await scheduler.stop()

    assert all(i.future.done() and i.future.exception() is None for i in items)
    with pytest.raises(QueueClosedError):
        await queue.put(request_for("late"))


@pytest.mark.asyncio
async def test_stop_waits_for_requests_in_flight(
    queue, router, make_scheduler, request_for
):
    router.gate = asyncio.Event()
    scheduler = make_scheduler()
    await scheduler.start()
    item = await queue.put(request_for("slow"))
    await _settle()

    stopping = asyncio.create_task(scheduler.stop())
    await _settle()
    assert not stopping.done()

    router.gate.set()
    await asyncio.wait_for(stopping, timeout=1)
    assert (await item.future).output == "done slow"


@pytest.mark.asyncio
async def test_stop_timeout_aborts_in_flight_and_fails_queued(
    queue, router, make_scheduler, request_for
):
    router.hang = True
    scheduler = make_scheduler(max_concurrency=1, shutdown_timeout_s=0.05)
    await scheduler.start()
    running = await queue.put(request_for("running"))
    waiting = await queue.put(request_for("waiting"))
    await _settle()

    await asyncio.wait_for(scheduler.stop(), timeout=2)

    with pytest.raises(SchedulerStoppedError):
        await running.future
    with pytest.raises(SchedulerStoppedError):
        await waiting.future
    assert router.cancelled == ["running"]
    assert router.calls == ["running"]


@pytest.mark.asyncio
async def test_start_twice_raises(make_scheduler):
    scheduler = make_scheduler()
    await scheduler.start()
    with pytest.raises(RuntimeError, match="already started"):
        await scheduler.start()


@pytest.mark.asyncio
async def test_stop_is_idempotent(make_scheduler):
    scheduler = make_scheduler()
    await scheduler.start()
    await scheduler.stop()
    await scheduler.stop()


@pytest.mark.asyncio
async def test_stop_without_start_still_closes_queue(
    queue, make_scheduler, request_for
):
    scheduler = make_scheduler()
    await scheduler.stop()
    with pytest.raises(QueueClosedError):
        await queue.put(request_for("a"))
