"""Per-client token-bucket rate limiter (in memory, per process)."""

import time
from collections.abc import Callable


class RateLimitExceededError(Exception):
    """A client sent requests faster than its limit."""

    def __init__(self, retry_after_s: float) -> None:
        super().__init__(f"rate limit exceeded; retry after {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s


class TokenBucketLimiter:
    """Gives each client `burst` tokens that refill at `rate_per_s`.

    Each request costs one token.
    """

    def __init__(
        self,
        rate_per_s: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_per_s <= 0 or burst < 1:
            raise ValueError("rate_per_s must be positive and burst at least 1")
        self._rate = rate_per_s
        self._burst = burst
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}

    def check(self, client_id: str) -> float:
        """Take one token for the client.

        Returns 0.0 when the request is allowed. Otherwise returns the seconds
        until a token is available, and takes nothing.
        """
        now = self._clock()
        tokens, last = self._buckets.get(client_id, (float(self._burst), now))
        tokens = min(float(self._burst), tokens + (now - last) * self._rate)
        if tokens >= 1.0:
            self._buckets[client_id] = (tokens - 1.0, now)
            return 0.0
        self._buckets[client_id] = (tokens, now)
        return (1.0 - tokens) / self._rate
