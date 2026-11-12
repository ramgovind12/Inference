import pytest
from pydantic import SecretStr

from inference.gateway.app import create_app
from inference.gateway.auth import (
    ApiKeyAuthenticator,
    AuthenticationError,
    client_id_for,
)
from inference.gateway.settings import GatewaySettings
from inference.queue.memory import InMemoryRequestQueue


def _authenticator() -> ApiKeyAuthenticator:
    return ApiKeyAuthenticator(["key-one", "key-two"])


def test_bearer_key_accepted():
    assert _authenticator().authenticate("Bearer key-one", None) == client_id_for(
        "key-one"
    )


def test_bearer_scheme_is_case_insensitive():
    assert _authenticator().authenticate("bearer key-two", None) == client_id_for(
        "key-two"
    )


def test_x_api_key_header_accepted():
    assert _authenticator().authenticate(None, "key-one") == client_id_for("key-one")


def test_missing_key_rejected():
    with pytest.raises(AuthenticationError, match="missing"):
        _authenticator().authenticate(None, None)


def test_invalid_key_rejected():
    with pytest.raises(AuthenticationError, match="invalid"):
        _authenticator().authenticate("Bearer nope", None)


def test_non_bearer_scheme_without_other_key_counts_as_missing():
    with pytest.raises(AuthenticationError, match="missing"):
        _authenticator().authenticate("Basic key-one", None)


def test_non_bearer_authorization_is_ignored_when_x_api_key_is_present():
    # Either header may carry the key. A non-Bearer Authorization header
    # (for example Basic) is not a key, so the X-API-Key value is used.
    assert _authenticator().authenticate("Basic abc", "key-one") == client_id_for(
        "key-one"
    )


def test_client_id_is_stable_distinct_and_hides_the_key():
    assert client_id_for("key-one") == client_id_for("key-one")
    assert client_id_for("key-one") != client_id_for("key-two")
    assert "key-one" not in client_id_for("key-one")


def test_api_keys_read_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCE_GATEWAY_API_KEYS", '["k1","k2"]')
    settings = GatewaySettings()
    assert [k.get_secret_value() for k in settings.api_keys] == ["k1", "k2"]
    assert "k1" not in repr(settings)


def test_create_app_without_keys_fails_fast():
    with pytest.raises(ValueError, match="no API keys"):
        create_app(InMemoryRequestQueue(5), GatewaySettings(api_keys=[]))


def test_create_app_without_keys_allowed_when_auth_disabled():
    settings = GatewaySettings(api_keys=[], auth_enabled=False)
    assert create_app(InMemoryRequestQueue(5), settings) is not None


@pytest.mark.parametrize(
    "name, value",
    [
        ("INFERENCE_GATEWAY_RATE_LIMIT_RPS", "0"),
        ("INFERENCE_GATEWAY_RATE_LIMIT_BURST", "0"),
        ("INFERENCE_GATEWAY_REQUEST_TIMEOUT_S", "-1"),
    ],
)
def test_invalid_settings_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
):
    from pydantic import ValidationError

    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        GatewaySettings(api_keys=[SecretStr("k")])
