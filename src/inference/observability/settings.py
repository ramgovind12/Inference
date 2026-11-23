from pydantic_settings import BaseSettings, SettingsConfigDict


class ObservabilitySettings(BaseSettings):
    """INFERENCE_METRICS_ENABLED=false removes the /metrics route."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_")

    metrics_enabled: bool = True
z