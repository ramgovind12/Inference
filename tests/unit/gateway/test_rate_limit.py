import pytest

from inference.gateway.rate_limit import RateLimitExceededError, TokenBucketLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _limiter(clock: FakeClock, rate: float = 1.0, burst: int = 3) -> TokenBucketLimiter:
    return TokenBucketLimiter(rate_per_s=rate, burst=burst, clock=clock)


def test_burst_is_allowed_then_refused():
    limiter = _limiter(FakeClock(), burst=3)
    assert [limiter.check("c") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.check("c") > 0


def test_retry_after_matches_refill_rate():
    clock = FakeClock()
    limiter = _limiter(clock, rate=2.0, burst=1)
    assert limiter.check("c") == 0.0
    assert limiter.check("c") == pytest.approx(0.5)
    clock.now += 0.25
    assert limiter.check("c") == pytest.approx(0.25)


def test_tokens_refill_over_time():
    clock = FakeClock()
    limiter = _limiter(clock, rate=1.0, burst=2)
    limiter.check("c")
    limiter.check("c")
    assert limiter.check("c") > 0

    clock.now += 1.0
    assert limiter.check("c") == 0.0
    assert limiter.check("c") > 0


def test_refill_is_capped_at_burst():
    clock = FakeClock()
    limiter = _limiter(clock, rate=1.0, burst=2)
    limiter.check("c")
    clock.now += 1000.0

    assert limiter.check("c") == 0.0
    assert limiter.check("c") == 0.0
    assert limiter.check("c") > 0


def test_refused_request_takes_no_token():
    clock = FakeClock()
    limiter = _limiter(clock, rate=1.0, burst=1)
    limiter.check("c")
    for _ in range(5):
        assert limiter.check("c") > 0
    clock.now += 1.0
    assert limiter.check("c") == 0.0


def test_clients_are_isolated():
    limiter = _limiter(FakeClock(), burst=1)
    assert limiter.check("a") == 0.0
    assert limiter.check("a") > 0
    assert limiter.check("b") == 0.0


@pytest.mark.parametrize("rate, burst", [(0, 1), (-1, 1), (1, 0)])
def test_invalid_configuration_rejected(rate: float, burst: int):
    with pytest.raises(ValueError):
        TokenBucketLimiter(rate_per_s=rate, burst=burst)


def test_exception_carries_retry_after():
    error = RateLimitExceededError(2.5)
    assert error.retry_after_s == 2.5
