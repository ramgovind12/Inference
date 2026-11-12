"""Public request schema of the gateway."""

from typing import Literal

from inference.core.schemas import InferenceRequest
from inference.queue.base import Priority


class GatewayInferenceRequest(InferenceRequest):
    """The body clients send: an inference request plus an optional priority."""

    priority: Literal["high", "normal", "low"] = "normal"

    def to_internal(self) -> tuple[InferenceRequest, Priority]:
        """Split into the internal request and its queue priority."""
        request = InferenceRequest.model_validate(self.model_dump(exclude={"priority"}))
        return request, Priority[self.priority.upper()]
