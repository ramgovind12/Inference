"""Prometheus metrics. Each process owns one CollectorRegistry (no global state).

Event metrics are recorded by the components. State metrics (queue depth, worker
health, ...) are read when Prometheus scrapes, through functions bound at wiring
time. Labels come only from fixed sets or static configuration, never from
client input, so cardinality stays bounded.
"""

from collections.abc import Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST


def render(registry: CollectorRegistry) -> bytes:
    """Serialize the registry in the Prometheus text format."""
    return generate_latest(registry)


class GatewayMetrics:
    """Gateway counters. Labels: HTTP status and our fixed error codes only."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "inferscale_gateway_requests_total",
            "Inference requests by HTTP status.",
            ["status"],
            registry=registry,
        )
        self.duration = Histogram(
            "inferscale_gateway_request_duration_seconds",
            "Time from request received to response sent.",
            registry=registry,
        )
        self.errors = Counter(
            "inferscale_gateway_errors_total",
            "Error responses by error code.",
            ["error"],
            registry=registry,
        )


class SchedulerMetrics:
    """Queue and scheduler metrics."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.queue_wait = Histogram(
            "inferscale_scheduler_queue_wait_seconds",
            "Time a request waited in the queue before dispatch.",
            registry=registry,
        )
        self._queue_depth = Gauge(
            "inferscale_queue_depth",
            "Requests waiting in the queue.",
            registry=registry,
        )
        self._in_flight = Gauge(
            "inferscale_scheduler_in_flight",
            "Requests being routed right now.",
            registry=registry,
        )

    def bind_state(
        self, queue_depth: Callable[[], float], in_flight: Callable[[], float]
    ) -> None:
        """Read queue depth and in-flight count when Prometheus scrapes."""
        self._queue_depth.set_function(queue_depth)
        self._in_flight.set_function(in_flight)


class RouterMetrics:
    """Router metrics. `worker_id` comes from static configuration."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self._requests = Counter(
            "inferscale_router_requests_total",
            "Requests sent to a worker, by outcome (ok, backend_error, failover).",
            ["worker_id", "outcome"],
            registry=registry,
        )
        self._duration = Histogram(
            "inferscale_router_request_duration_seconds",
            "Time a worker took to answer.",
            ["worker_id"],
            registry=registry,
        )
        self.no_worker = Counter(
            "inferscale_router_no_worker_total",
            "Requests that found no available worker.",
            registry=registry,
        )
        self._healthy = Gauge(
            "inferscale_router_worker_healthy",
            "1 if the router considers the worker healthy.",
            ["worker_id"],
            registry=registry,
        )
        self._in_flight = Gauge(
            "inferscale_router_worker_in_flight",
            "Requests the router has in flight to the worker.",
            ["worker_id"],
            registry=registry,
        )

    def record(self, worker_id: str, outcome: str, duration_s: float) -> None:
        """Record a finished attempt that reached the worker."""
        self._requests.labels(worker_id=worker_id, outcome=outcome).inc()
        self._duration.labels(worker_id=worker_id).observe(duration_s)

    def record_failover(self, worker_id: str) -> None:
        """Record an attempt that failed before running, so the router retries."""
        self._requests.labels(worker_id=worker_id, outcome="failover").inc()

    def bind_worker(
        self,
        worker_id: str,
        healthy: Callable[[], bool],
        in_flight: Callable[[], float],
    ) -> None:
        """Read one worker's health and in-flight count at scrape time."""
        self._healthy.labels(worker_id=worker_id).set_function(
            lambda: 1.0 if healthy() else 0.0
        )
        self._in_flight.labels(worker_id=worker_id).set_function(in_flight)


class WorkerMetrics:
    """Worker metrics."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self.requests = Counter(
            "inferscale_worker_requests_total",
            "Requests by outcome (ok, backend_error, not_ready, cancelled).",
            ["outcome"],
            registry=registry,
        )
        self._duration = Histogram(
            "inferscale_worker_request_duration_seconds",
            "Time the backend took per request.",
            registry=registry,
        )
        self._tokens = Counter(
            "inferscale_worker_tokens_total",
            "Tokens processed.",
            ["type"],
            registry=registry,
        )
        self._active = Gauge(
            "inferscale_worker_active_requests",
            "Requests running now.",
            registry=registry,
        )
        self._ready = Gauge(
            "inferscale_worker_ready", "1 if the worker is ready.", registry=registry
        )

    def record_request(self, outcome: str, duration_s: float) -> None:
        """Record a finished request."""
        self.requests.labels(outcome=outcome).inc()
        self._duration.observe(duration_s)

    def record_tokens(self, prompt: int, completion: int) -> None:
        """Record token counts of a successful request."""
        self._tokens.labels(type="prompt").inc(prompt)
        self._tokens.labels(type="completion").inc(completion)

    def bind_state(
        self, active: Callable[[], float], ready: Callable[[], bool]
    ) -> None:
        """Read active requests and readiness when Prometheus scrapes."""
        self._active.set_function(active)
        self._ready.set_function(lambda: 1.0 if ready() else 0.0)
