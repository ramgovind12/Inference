"""Request and response schemas for inference."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """A single inference request."""

    request_id: str = Field(default_factory=lambda: uuid4().hex)
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    parameters: dict[str, Any] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    """The result of a successful inference request."""

    request_id: str
    model: str
    output: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class ErrorResponse(BaseModel):
    """Structured error body returned by the worker API."""

    error: str
    detail: str
