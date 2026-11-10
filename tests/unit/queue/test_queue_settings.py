import pytest
from pydantic import ValidationError

from inference.queue.settings import QueueSettings


def test_default_max_size():
    assert QueueSettings().max_size == 1000


def test_max_size_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_QUEUE_MAX_SIZE", "5")
    assert QueueSettings().max_size == 5


@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_max_size_rejected(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("INFERENCE_QUEUE_MAX_SIZE", value)
    with pytest.raises(ValidationError):
        QueueSettings()
