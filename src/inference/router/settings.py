"""Router configuration, read from INFERENCE_ROUTER_* environment variables."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from inference.router.registry import WorkerEndpoint


class RouterSettings(BaseSettings):
    """Runtime settings for the router.

    `workers` is a JSON list, e.g.
    INFERENCE_ROUTER_WORKERS='[{"id":"a","base_url":"http://localhost:8001","models":["m"]}]'
    """

    model_config = SettingsConfigDict(env_prefix="INFERENCE_ROUTER_")

    workers: list[WorkerEndpoint] = Field(default_factory=list)
    policy: Literal["round_robin", "least_loaded"] = "round_robin"
    health_interval_s: float = Field(default=5.0, gt=0)
    request_timeout_s: float = Field(default=30.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)
