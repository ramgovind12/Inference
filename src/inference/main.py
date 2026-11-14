"""Entry points: `inference serve` (control plane) and `inference worker`."""

import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI

from inference.core.logging import configure_logging
from inference.gateway.app import create_app as create_gateway_app
from inference.gateway.settings import GatewaySettings
from inference.queue.memory import InMemoryRequestQueue
from inference.queue.settings import QueueSettings
from inference.router.router import create_router
from inference.router.settings import RouterSettings
from inference.scheduler.scheduler import Scheduler
from inference.scheduler.settings import SchedulerSettings
from inference.worker.app import create_app as create_worker_app


def create_control_plane_app() -> FastAPI:
    """Wire gateway, queue, scheduler and router into one app."""
    queue = InMemoryRequestQueue(QueueSettings().max_size)
    http = httpx.AsyncClient()
    router = create_router(RouterSettings(), http)
    scheduler_settings = SchedulerSettings()
    scheduler = Scheduler(
        queue,
        router,
        max_concurrency=scheduler_settings.max_concurrency,
        shutdown_timeout_s=scheduler_settings.shutdown_timeout_s,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await router.start()
        await scheduler.start()
        try:
            yield
        finally:
            await scheduler.stop()
            await router.stop()
            await http.aclose()

    return create_gateway_app(queue, GatewaySettings(), lifespan=lifespan)


def main(argv: list[str] | None = None) -> None:
    """Run the control plane or a worker."""
    parser = argparse.ArgumentParser(prog="inference")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, default_port in (("serve", 8000), ("worker", 8001)):
        command = commands.add_parser(name)
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=default_port)
        command.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    factory = create_control_plane_app if args.command == "serve" else create_worker_app
    uvicorn.run(factory, factory=True, host=args.host, port=args.port, log_config=None)
