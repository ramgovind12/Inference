# InferScale

A self-hosted, modular control plane for AI inference workloads. It routes,
queues, schedules and load-balances requests across inference workers. It is
not an inference engine: engines such as vLLM are external backends.

```
Client -> Gateway -> Queue -> Scheduler -> Router -> Worker -> Backend
```

| Component | Package | Job |
|---|---|---|
| Gateway | `inference.gateway` | API-key auth, rate limiting, validation, HTTP error mapping |
| Queue | `inference.queue` | Bounded priority buffer with backpressure |
| Scheduler | `inference.scheduler` | Dispatch loop with a concurrency limit |
| Router | `inference.router` | Model-aware worker selection, health checks, failover |
| Worker | `inference.worker` | Runs requests against a backend, health and readiness |
| Backend | `inference.backends` | Inference engine interface (a mock backend today) |

The Gateway, Queue, Scheduler and Router run in one process (the control plane,
`inference serve`) because the queue is in memory. Workers run as separate
processes (`inference worker`).

## Run locally

Install with [uv](https://docs.astral.sh/uv/): `uv sync`. Then start two
workers and the control plane in three terminals (PowerShell shown):

```powershell
uv run inference worker --port 8001
uv run inference worker --port 8002

$env:INFERENCE_GATEWAY_API_KEYS='["dev-key"]'
$env:INFERENCE_ROUTER_WORKERS='[{"id":"w1","base_url":"http://127.0.0.1:8001","models":["mock-model"]},{"id":"w2","base_url":"http://127.0.0.1:8002","models":["mock-model"]}]'
uv run inference serve --port 8000
```

Send a request:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/inference -H "Authorization: Bearer dev-key" -H "Content-Type: application/json" -d '{\"model\":\"mock-model\",\"prompt\":\"hello\"}'
```

The body also accepts `priority` (`high`, `normal` or `low`), `max_tokens`,
`parameters` and `request_id`. Run the smoke checks with
`uv run python scripts/smoke.py --key dev-key`.

## Run with Docker

```
docker compose -f docker/docker-compose.yml up --build
uv run python scripts/smoke.py --key dev-key
```

One image runs every role. The compose file starts the gateway and two workers.
Set `GATEWAY_API_KEY` to replace the development key.

## Configuration

All settings are environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `INFERENCE_GATEWAY_API_KEYS` | none | JSON list of accepted API keys |
| `INFERENCE_GATEWAY_AUTH_ENABLED` | `true` | Turn authentication off for development |
| `INFERENCE_GATEWAY_RATE_LIMIT_RPS` | `5` | Refill rate per client, requests per second |
| `INFERENCE_GATEWAY_RATE_LIMIT_BURST` | `10` | Bucket size per client |
| `INFERENCE_GATEWAY_REQUEST_TIMEOUT_S` | `60` | Wait for a result before returning 504 |
| `INFERENCE_QUEUE_MAX_SIZE` | `1000` | Queue capacity. A full queue returns 503 |
| `INFERENCE_SCHEDULER_MAX_CONCURRENCY` | `8` | Requests in flight to the Router |
| `INFERENCE_SCHEDULER_SHUTDOWN_TIMEOUT_S` | `30` | Time allowed to drain on shutdown |
| `INFERENCE_ROUTER_WORKERS` | none | JSON list of `{id, base_url, models}` |
| `INFERENCE_ROUTER_POLICY` | `round_robin` | `round_robin` or `least_loaded` |
| `INFERENCE_ROUTER_HEALTH_INTERVAL_S` | `5` | Seconds between worker readiness checks |
| `INFERENCE_ROUTER_REQUEST_TIMEOUT_S` | `30` | Timeout for one call to a worker |
| `INFERENCE_ROUTER_MAX_ATTEMPTS` | `3` | Workers tried before giving up |
| `INFERENCE_WORKER_DRAIN_TIMEOUT_S` | `30` | Time a worker waits for requests in flight on shutdown |
| `INFERENCE_WORKER_MOCK_LATENCY_S` | `0` | Delay added by the mock backend |

## Development

```
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

Project rules and the development workflow are in `CLAUDE.md`. The current
phase is in `docs/development/CURRENT_PHASE.md`, the roadmap in
`docs/development/DEVELOPMENT_PLAN.md`, and design decisions in `docs/adr/`.
