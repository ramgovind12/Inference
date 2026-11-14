# Current Development Phase

## Project
InferScale

## Current Phase
Phase 6: Packaging and Wiring

## Status
IMPLEMENTED, AWAITING HUMAN ACCEPTANCE

## Objective
Run the complete system as real processes: one entry point for the control
plane, one for workers, container images, and a smoke test.

## Scope

IN SCOPE:
- `inference serve` and `inference worker` commands (`inference.main`)
- Wiring of Gateway, Queue, Scheduler and Router into one app, with ordered
  start and stop
- Structured JSON logging
- One Docker image, and a compose file with a gateway and two workers
- A smoke script
- Project documentation and ADRs

OUT OF SCOPE:
- Metrics and tracing
- Redis, vLLM, Kubernetes, autoscaling
- Worker self-registration and model discovery
- Any change to component interfaces other than the optional `lifespan`
  argument of the Gateway `create_app`

## Acceptance Criteria

1. `inference serve` starts the control plane and `inference worker` starts a worker.
2. A request sent to the control plane reaches a worker and returns a response.
3. On shutdown the control plane drains the queue, and then stops the Router and HTTP client.
4. Log lines are JSON and include the structured fields (for example `request_id`).
5. The docker compose stack starts, and the smoke script passes against it.
6. Unit tests cover the entry point, the wiring and the log formatter.
7. Existing project tests continue to pass.

## Known limitations
- The compose stack has not been run in this environment. Docker was not available when this phase was implemented.
- Keys and rate limits are per process and in memory.
- The control plane cannot be split into several processes until the queue has a shared backend (Redis).

## Approval Requirements

After implementation:
- Run tests.
- Report results.
- List known limitations.
- Wait for human acceptance.

Do not start the next phase automatically.
