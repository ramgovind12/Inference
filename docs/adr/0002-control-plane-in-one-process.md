# ADR 0002: Gateway, Queue, Scheduler and Router run in one process

Status: Accepted

## Context
The request queue is in memory. A request enqueued by the Gateway must be
dequeued by the Scheduler, and the Scheduler resolves a future created by the
queue. Neither works across process boundaries.

## Decision
Run the Gateway, Queue, Scheduler and Router together as the control plane
(`inference serve`). Run each worker as its own process (`inference worker`).
`inference.main.create_control_plane_app` wires the components. It starts the
Router, then the Scheduler, and on shutdown stops the Scheduler (which drains
the queue), then the Router, then the HTTP client.

## Consequences
- Only workers scale out. There is one control plane instance.
- Rate limits and API keys are per process.
- Splitting the control plane needs a shared queue (Redis) and a different
  way to return results than an in-process future.
