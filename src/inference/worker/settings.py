"""Worker configuration, read from INFERENCE_WORKER_* environment variables."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    """Runtime settings for the worker process."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_WORKER_")

    drain_timeout_s: float = 30.0
    mock_latency_s: float = 0.0

    backend: Literal["mock", "vllm"] = "mock"
    vllm_base_url: str = Field(default="http://127.0.0.1:8000", min_length=1)
    vllm_api_key: SecretStr | None = None
    vllm_timeout_s: float = Field(default=120.0, gt=0)
    vllm_connect_timeout_s: float = Field(default=5.0, gt=0)
