"""Mock inference backend for development and tests."""

import asyncio
import time

from inference.core.errors import BackendUnavailableError
from inference.core.schemas import InferenceRequest, InferenceResponse


class MockBackend:
    """Backend returning deterministic output without running a model.

    Set `fail_with` to make every `generate` call raise that exception.
    """

    def __init__(
        self,
        latency_s: float = 0.0,
        fail_with: Exception | None = None,
    ) -> None:
        self.latency_s = latency_s
        self.fail_with = fail_with
        self._ready = False

    async def start(self) -> None:
        self._ready = True

    async def stop(self) -> None:
        self._ready = False

    async def is_ready(self) -> bool:
        return self._ready

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        if not self._ready:
            raise BackendUnavailableError("mock backend is not started")
        started = time.perf_counter()
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)
        if self.fail_with is not None:
            raise self.fail_with
        output = f"mock response to: {request.prompt}"
        return InferenceResponse(
            request_id=request.request_id,
            model=request.model,
            output=output,
            finish_reason="stop",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(output.split()),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
