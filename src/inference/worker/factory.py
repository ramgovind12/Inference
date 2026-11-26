from inference.backends.base import Backend
from inference.backends.mock import MockBackend
from inference.backends.vllm import VllmBackend
from inference.worker.settings import WorkerSettings


def build_backend(settings: WorkerSettings) -> Backend:
    """Build the configured backend. The default is the mock."""
    if settings.backend == "vllm":
        key = settings.vllm_api_key
        return VllmBackend(
            base_url=settings.vllm_base_url,
            api_key=key.get_secret_value() if key else None,
            timeout_s=settings.vllm_timeout_s,
            connect_timeout_s=settings.vllm_connect_timeout_s,
        )
    return MockBackend(latency_s=settings.mock_latency_s)
