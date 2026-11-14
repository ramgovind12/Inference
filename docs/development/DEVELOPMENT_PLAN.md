# Development Plan

Phases 1 to 5 are complete. Phase 6 is implemented and awaits acceptance.
Phases marked PROPOSED are suggestions only. None of them is approved until the
technical lead says so. See `CURRENT_PHASE.md` for the approved scope.

| Phase | Name | Status |
|---|---|---|
| 1 | Worker with a mock backend | Complete |
| 2 | Router: registry, policies, health checks, failover | Complete |
| 3 | Request queue: priority, limits, backpressure | Complete |
| 4 | Scheduler: dispatch loop, concurrency limit, graceful drain | Complete |
| 5 | Gateway: authentication, rate limiting, error mapping | Complete |
| 6 | Packaging and wiring: entry points, Docker, logging, smoke test | Implemented, awaiting acceptance |
| 7 | Observability: Prometheus metrics, tracing | PROPOSED |
| 8 | vLLM backend behind the `Backend` interface | PROPOSED |
| 9 | Redis: shared rate limits and keys, shared queue | PROPOSED |
| 10 | Autoscaling, worker discovery, Kubernetes | PROPOSED |

## Open issues
- Strict priority can starve LOW requests. Aging is not built.
- The Gateway does not cancel a request when its client disconnects. Only the timeout cancels it.
- Workers do not report which models they serve. The Router uses static configuration.
