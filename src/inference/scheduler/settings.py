"""Scheduler configuration, read from INFERENCE_SCHEDULER_* environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SchedulerSettings(BaseSettings):
    """Runtime settings for the scheduler."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_SCHEDULER_")

    max_concurrency: int = Field(default=8, ge=1)
    shutdown_timeout_s: float = Field(default=30.0, gt=0)
