"""Backend interface abstracting the underlying inference engine."""

from typing import Protocol

from inference.core.schemas import InferenceRequest, InferenceResponse


class Backend(Protocol):
    """An inference engine the worker delegates execution to."""

    async def start(self) -> None:
        """Prepare the backend to accept requests."""
        ...

    async def stop(self) -> None:
        """Release backend resources."""
        ...

    async def is_ready(self) -> bool:
        """Return True if the backend can accept requests."""
        ...

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference. Raises BackendError on failure."""
        ...
