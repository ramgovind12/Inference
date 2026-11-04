"""Worker configuration, read from INFERENCE_WORKER_* environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    """Runtime settings for the worker process."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_WORKER_")

    drain_timeout_s: float = 30.0
    mock_latency_s: float = 0.0
