"""Gateway configuration, read from INFERENCE_GATEWAY_* environment variables."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    """Runtime settings for the gateway.

    `api_keys` is a JSON list, e.g. INFERENCE_GATEWAY_API_KEYS='["key-1","key-2"]'.
    """

    model_config = SettingsConfigDict(env_prefix="INFERENCE_GATEWAY_")

    api_keys: list[SecretStr] = Field(default_factory=list)
    auth_enabled: bool = False
    rate_limit_rps: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=10, ge=1)
    request_timeout_s: float = Field(default=60.0, gt=0)
