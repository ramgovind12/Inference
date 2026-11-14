# ADR 0003: The Scheduler closes the queue on stop

Status: Accepted

## Context
A graceful shutdown must stop new requests, finish the buffered ones, and wait
for the requests in flight. Draining requires a closed queue, so that a consumer
can tell "empty for now" from "empty for good".

## Decision
`Scheduler.stop()` closes the queue, drains it, and waits for work in flight up
to `shutdown_timeout_s`. After the timeout it cancels the work in flight and
fails every request still queued with `SchedulerStoppedError`.

## Consequences
- After `stop()`, `put` raises `QueueClosedError`. The Gateway returns 503
  `shutting_down`.
- No submitter waits forever: each future ends with a result, an error, or a
  cancellation.
- The application wiring does not close the queue itself.
