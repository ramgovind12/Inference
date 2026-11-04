import pytest

from inference.core.schemas import InferenceRequest


@pytest.fixture
def request_body() -> InferenceRequest:
    return InferenceRequest(model="mock-model", prompt="hello world")
