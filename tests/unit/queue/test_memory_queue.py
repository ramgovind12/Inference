import asyncio

import pytest

from inference.core.errors import QueueClosedError, QueueFullError
from inference.core.schemas import InferenceRequest
from inference.queue.base import Priority
from inference.queue.memory import InMemoryRequestQueue


def _request(prompt: str = "hi") -> InferenceRequest:
    return InferenceRequest(model="m", prompt=prompt)


async def _drain(queue: InMemoryRequestQueue) -> list[str]:
    prompts = []
    while queue.qsize():
        prompts.append((await queue.get()).request.prompt)
    return prompts


def test_invalid_max_size_rejected():
    with pytest.raises(ValueError, match="max_size"):
        InMemoryRequestQueue(max_size=0)


@pytest.mark.asyncio
async def test_put_returns_item_with_pending_future():
    queue = InMemoryRequestQueue(max_size=5)
    request = _request()

    item = await queue.put(request, Priority.HIGH)

    assert item.request is request
    assert item.priority is Priority.HIGH
    assert item.enqueued_at > 0
    assert not item.future.done()
    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_default_priority_is_normal():
    queue = InMemoryRequestQueue(max_size=5)
    assert (await queue.put(_request())).priority is Priority.NORMAL


@pytest.mark.asyncio
async def test_dequeues_by_priority():
    queue = InMemoryRequestQueue(max_size=5)
    await queue.put(_request("low"), Priority.LOW)
    await queue.put(_request("normal"), Priority.NORMAL)
    await queue.put(_request("high"), Priority.HIGH)

    assert await _drain(queue) == ["high", "normal", "low"]


@pytest.mark.asyncio
async def test_fifo_within_a_priority():
    queue = InMemoryRequestQueue(max_size=10)
    for name in ("n1", "n2", "n3"):
        await queue.put(_request(name), Priority.NORMAL)

    assert await _drain(queue) == ["n1", "n2", "n3"]


@pytest.mark.asyncio
async def test_interleaved_priorities_keep_fifo_per_level():
    queue = InMemoryRequestQueue(max_size=10)
    await queue.put(_request("n1"), Priority.NORMAL)
    await queue.put(_request("h1"), Priority.HIGH)
    await queue.put(_request("n2"), Priority.NORMAL)
    await queue.put(_request("h2"), Priority.HIGH)

    assert await _drain(queue) == ["h1", "h2", "n1", "n2"]


@pytest.mark.asyncio
async def test_full_queue_rejects_and_leaves_contents_unchanged():
    queue = InMemoryRequestQueue(max_size=2)
    await queue.put(_request("a"))
    await queue.put(_request("b"))

    with pytest.raises(QueueFullError, match="max_size=2"):
        await queue.put(_request("c"), Priority.HIGH)

    assert queue.qsize() == 2
    assert await _drain(queue) == ["a", "b"]


@pytest.mark.asyncio
async def test_space_freed_by_get_allows_put_again():
    queue = InMemoryRequestQueue(max_size=1)
    await queue.put(_request("a"))
    await queue.get()

    await queue.put(_request("b"))

    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_get_waits_until_put():
    queue = InMemoryRequestQueue(max_size=5)
    getter = asyncio.create_task(queue.get())
    await asyncio.sleep(0.01)
    assert not getter.done()

    await queue.put(_request("late"))

    item = await asyncio.wait_for(getter, timeout=1)
    assert item.request.prompt == "late"


@pytest.mark.asyncio
async def test_multiple_getters_each_receive_distinct_item():
    queue = InMemoryRequestQueue(max_size=5)
    getters = [asyncio.create_task(queue.get()) for _ in range(3)]
    await asyncio.sleep(0.01)

    for name in ("a", "b", "c"):
        await queue.put(_request(name))

    items = await asyncio.wait_for(asyncio.gather(*getters), timeout=1)
    assert sorted(i.request.prompt for i in items) == ["a", "b", "c"]
    assert queue.qsize() == 0


@pytest.mark.asyncio
async def test_cancelled_get_loses_nothing():
    queue = InMemoryRequestQueue(max_size=5)
    getter = asyncio.create_task(queue.get())
    await asyncio.sleep(0.01)
    getter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await getter

    await queue.put(_request("kept"))

    item = await asyncio.wait_for(queue.get(), timeout=1)
    assert item.request.prompt == "kept"


@pytest.mark.asyncio
async def test_item_not_stranded_when_notified_getter_is_cancelled():
    queue = InMemoryRequestQueue(max_size=5)
    first = asyncio.create_task(queue.get())
    second = asyncio.create_task(queue.get())
    await asyncio.sleep(0.01)

    await queue.put(_request("x"))  # notifies the first waiter
    first.cancel()

    item = await asyncio.wait_for(second, timeout=1)
    assert item.request.prompt == "x"
    with pytest.raises(asyncio.CancelledError):
        await first


@pytest.mark.asyncio
async def test_put_after_close_raises():
    queue = InMemoryRequestQueue(max_size=5)
    await queue.close()
    with pytest.raises(QueueClosedError):
        await queue.put(_request())


@pytest.mark.asyncio
async def test_close_drains_remaining_items_then_raises():
    queue = InMemoryRequestQueue(max_size=5)
    await queue.put(_request("a"), Priority.LOW)
    await queue.put(_request("b"), Priority.HIGH)
    await queue.close()

    assert (await queue.get()).request.prompt == "b"
    assert (await queue.get()).request.prompt == "a"
    with pytest.raises(QueueClosedError):
        await queue.get()


@pytest.mark.asyncio
async def test_close_wakes_blocked_getters():
    queue = InMemoryRequestQueue(max_size=5)
    getters = [asyncio.create_task(queue.get()) for _ in range(2)]
    await asyncio.sleep(0.01)

    await queue.close()

    results = await asyncio.wait_for(
        asyncio.gather(*getters, return_exceptions=True), timeout=1
    )
    assert all(isinstance(r, QueueClosedError) for r in results)


@pytest.mark.asyncio
async def test_close_is_idempotent():
    queue = InMemoryRequestQueue(max_size=5)
    await queue.close()
    await queue.close()
