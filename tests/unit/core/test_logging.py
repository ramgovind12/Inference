import json
import logging

import pytest

from inference.core.logging import JsonFormatter, configure_logging


def _format(**record_kwargs: object) -> dict[str, object]:
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=record_kwargs.pop("msg", "hello %s"),  # type: ignore[arg-type]
        args=record_kwargs.pop("args", ("world",)),  # type: ignore[arg-type]
        exc_info=record_kwargs.pop("exc_info", None),  # type: ignore[arg-type]
    )
    for key, value in record_kwargs.items():
        setattr(record, key, value)
    return json.loads(JsonFormatter().format(record))


def test_formats_standard_fields():
    payload = _format()

    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert str(payload["time"]).endswith("+00:00")


def test_includes_extra_fields():
    payload = _format(request_id="r1", queue_wait_s=0.25)

    assert payload["request_id"] == "r1"
    assert payload["queue_wait_s"] == 0.25


def test_does_not_leak_internal_record_attributes():
    payload = _format()

    assert set(payload) == {"time", "level", "logger", "message"}


def test_non_serializable_extra_is_stringified():
    payload = _format(worker=object)

    assert "object" in str(payload["worker"])


def test_exception_info_is_included():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        payload = _format(exc_info=sys.exc_info())

    assert "ValueError: boom" in str(payload["exc_info"])


def test_output_is_a_single_line():
    record = logging.LogRecord("n", logging.INFO, __file__, 1, "a\nb", (), None)

    assert "\n" not in JsonFormatter().format(record)


def test_configure_logging_replaces_handlers_and_sets_level(
    capsys: pytest.CaptureFixture[str],
):
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("warning")
        assert root.level == logging.WARNING
        assert len(root.handlers) == 1

        logging.getLogger("t").warning("seen", extra={"request_id": "r9"})
        logging.getLogger("t").info("hidden")
        lines = capsys.readouterr().out.strip().splitlines()
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)

    assert [json.loads(line)["message"] for line in lines] == ["seen"]
    assert json.loads(lines[0])["request_id"] == "r9"
