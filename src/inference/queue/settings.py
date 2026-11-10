"""Queue configuration, read from INFERENCE_QUEUE_* environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QueueSettings(BaseSettings):
    """Runtime settings for the request queue."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_QUEUE_")

    max_size: int = Field(default=1000, ge=1)
