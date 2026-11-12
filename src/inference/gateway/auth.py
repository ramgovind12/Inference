"""API-key authentication."""

import hashlib
import hmac
from collections.abc import Iterable


class AuthenticationError(Exception):
    """The request carries no API key, or an invalid one."""


def client_id_for(api_key: str) -> str:
    """Return a stable client id for a key. The id does not reveal the key."""
    return "key-" + hashlib.sha256(api_key.encode()).hexdigest()[:12]


class ApiKeyAuthenticator:
    """Checks API keys sent as `Authorization: Bearer <key>` or `X-API-Key`."""

    def __init__(self, api_keys: Iterable[str]) -> None:
        self._keys = [key.encode() for key in api_keys]

    def authenticate(self, authorization: str | None, x_api_key: str | None) -> str:
        """Return the client id. Raises AuthenticationError if the key is bad."""
        key = self._extract(authorization, x_api_key)
        if key is None:
            raise AuthenticationError("missing API key")
        candidate = key.encode()
        # Compare against every key so the time taken does not show which one matched.
        matched = False
        for known in self._keys:
            if hmac.compare_digest(candidate, known):
                matched = True
        if not matched:
            raise AuthenticationError("invalid API key")
        return client_id_for(key)

    @staticmethod
    def _extract(authorization: str | None, x_api_key: str | None) -> str | None:
        if authorization is not None:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer" and value.strip():
                return value.strip()
        return x_api_key or None
