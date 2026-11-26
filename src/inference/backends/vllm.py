import logging
import time
from typing import Any

import httpx

from inference.core.errors import BackendError, BackendUnavailableError
from inference.core.schemas import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

# Sampling options a client may pass in `parameters`. Anything else is dropped.
ALLOWED_PARAMETERS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
    }
)

# The request never reached vLLM, so it is safe for the Router to retry it elsewhere.
_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class VllmBackend:
    """Backend that sends prompts to a vLLM server (`POST /v1/completions`)."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        connect_timeout_s: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._timeout = httpx.Timeout(timeout_s, connect=connect_timeout_s)
        self._probe_timeout = httpx.Timeout(connect_timeout_s)
        self._injected_client = client
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Create the HTTP client. Does not wait for the model to load."""
        self._client = self._injected_client or httpx.AsyncClient()

    async def stop(self) -> None:
        """Close the HTTP client (only if this backend created it)."""
        client, self._client = self._client, None
        if client is not None and self._injected_client is None:
            await client.aclose()

    async def is_ready(self) -> bool:
        """True when vLLM answers `GET /health` with 200."""
        if self._client is None:
            return False
        try:
            response = await self._client.get(
                f"{self._base_url}/health",
                headers=self._headers,
                timeout=self._probe_timeout,
            )
        except httpx.HTTPError as exc:
            logger.debug("vllm health check failed", extra={"error": repr(exc)})
            return False
        return response.status_code == httpx.codes.OK

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run one completion. Raises BackendUnavailableError or BackendError."""
        if self._client is None:
            raise BackendUnavailableError("vllm backend is not started")
        payload = self._payload(request)
        started = time.perf_counter()
        try:
            response = await self._client.post(
                f"{self._base_url}/v1/completions",
                json=payload,
                headers=self._headers,
                timeout=self._timeout,
            )
        except _NEVER_SENT as exc:
            raise BackendUnavailableError(f"cannot reach vLLM: {exc!r}") from exc
        except httpx.HTTPError as exc:
            # Read timeout or dropped connection: the request may have run.
            raise BackendError(f"vLLM request failed: {exc!r}") from exc

        if response.status_code == httpx.codes.SERVICE_UNAVAILABLE:
            raise BackendUnavailableError(f"vLLM unavailable: {self._detail(response)}")
        if response.status_code != httpx.codes.OK:
            raise BackendError(
                f"vLLM returned {response.status_code}: {self._detail(response)}"
            )
        return self._parse(request, response, started)

    @staticmethod
    def _payload(request: InferenceRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": request.model, "prompt": request.prompt}
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        dropped = sorted(set(request.parameters) - ALLOWED_PARAMETERS)
        if dropped:
            logger.warning(
                "dropping unsupported parameters",
                extra={"request_id": request.request_id, "parameters": dropped},
            )
        payload.update(
            {k: v for k, v in request.parameters.items() if k in ALLOWED_PARAMETERS}
        )
        return payload

    @staticmethod
    def _parse(
        request: InferenceRequest, response: httpx.Response, started: float
    ) -> InferenceResponse:
        try:
            data = response.json()
            choice = data["choices"][0]
            usage = data["usage"]
            return InferenceResponse(
                request_id=request.request_id,
                model=request.model,
                output=choice["text"],
                finish_reason=choice.get("finish_reason") or "unknown",
                prompt_tokens=int(usage["prompt_tokens"]),
                completion_tokens=int(usage["completion_tokens"]),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError("malformed response from vLLM") from exc

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        return response.text[:200]
