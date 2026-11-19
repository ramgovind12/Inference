# ADR 0004: Prometheus metrics, one registry per process

Status: Accepted

## Context
The system had JSON logs but no numbers: queue depth, latency, worker health and
failover rates were not visible. `CLAUDE.md` names Prometheus as the metrics stack.

## Decision
- Use `prometheus-client`. Each process owns one `CollectorRegistry`. The global
  default registry is not used, so tests and several apps in one process do not collide.
- Components record event metrics through small holder classes in
  `inference.observability.metrics`. Every holder argument is optional. Without one,
  a component records into a throwaway registry.
- State metrics (queue depth, in-flight count, worker health) are read when
  Prometheus scrapes, through functions bound at wiring time.
- `GET /metrics` exists on the control plane and on every worker. It needs no key.
  `INFERENCE_METRICS_ENABLED=false` removes the route.
- Labels come only from fixed sets or static configuration. Never from client
  input (`model`, `client_id`, `request_id`, path).
- Tracing (OpenTelemetry) is a separate, later phase.

## Consequences
- Counters reset when a process restarts, as Prometheus expects.
- Histograms use the library's default buckets. They are an assumption and need
  tuning once a real backend is attached.
- `/metrics` is unauthenticated. Keep it off public ports.
- Metrics describe what each component did. A request that times out at the
  Gateway can still finish at a worker.
