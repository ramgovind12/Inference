import pytest
from pydantic import ValidationError

from inference.scheduler.settings import SchedulerSettings


def test_defaults():
    settings = SchedulerSettings()
    assert settings.max_concurrency == 8
    assert settings.shutdown_timeout_s == 30.0


def test_environment_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_SCHEDULER_MAX_CONCURRENCY", "3")
    monkeypatch.setenv("INFERENCE_SCHEDULER_SHUTDOWN_TIMEOUT_S", "1.5")
    settings = SchedulerSettings()
    assert settings.max_concurrency == 3
    assert settings.shutdown_timeout_s == 1.5


@pytest.mark.parametrize(
    "name, value",
    [
        ("INFERENCE_SCHEDULER_MAX_CONCURRENCY", "0"),
        ("INFERENCE_SCHEDULER_SHUTDOWN_TIMEOUT_S", "0"),
        ("INFERENCE_SCHEDULER_SHUTDOWN_TIMEOUT_S", "-1"),
    ],
)
def test_invalid_values_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        SchedulerSettings()
