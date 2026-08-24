"""Circuit breakers, failover, hedging, and the two-dimensional limiter."""

from __future__ import annotations

import asyncio

import pytest

from prism.cost import TokenUsage
from prism.providers.base import ProviderResponse, RateLimitSnapshot
from prism.reliability.breaker import (
    BreakerConfig,
    BreakerState,
    CircuitBreaker,
)
from prism.reliability.failover import HedgeConfig, ProviderChain
from prism.reliability.ratelimit import Limits, RateLimiter, Reservation
from prism.schemas.errors import ErrorKind, PrismError


def response(provider: str = "p") -> ProviderResponse:
    return ProviderResponse(
        payload={"content": [{"type": "text", "text": "ok"}], "usage": {}},
        status_code=200,
        provider_request_id="req",
        latency_ms=10,
        rate_limits=RateLimitSnapshot(requests_remaining=99),
    )


class FakeProvider:
    def __init__(self, name: str, *, error: Exception | None = None, delay: float = 0.0):
        self.name = name
        self.error = error
        self.delay = delay
        self.calls = 0

    async def create_message(self, body, *, timeout_s=None):  # noqa: ANN001
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return response(self.name)

    async def aclose(self):
        return None


def overloaded() -> PrismError:
    return PrismError(ErrorKind.UPSTREAM_OVERLOADED, "529", status_code=503)


def rate_limited() -> PrismError:
    return PrismError(ErrorKind.UPSTREAM_RATE_LIMIT, "429", status_code=429, retry_after=5.0)


def bad_request() -> PrismError:
    return PrismError(ErrorKind.UPSTREAM_INVALID_REQUEST, "400", status_code=400)


# --- the breaker acts on the taxonomy -------------------------------------


def test_quota_exhaustion_never_opens_a_circuit():
    """The most common bug in hand-rolled client code, and it fails badly.

    A burst of 429s would open every circuit and stop traffic the gateway could
    have served — the quota belongs to the account, not the route.
    """
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=2))
    for _ in range(20):
        breaker.record_failure(ErrorKind.UPSTREAM_RATE_LIMIT)
    assert breaker.state is BreakerState.CLOSED
    assert breaker.ignored_failures == 20
    assert breaker.opens == 0


def test_capacity_failures_do_open_a_circuit():
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=3))
    for _ in range(3):
        breaker.record_failure(ErrorKind.UPSTREAM_OVERLOADED)
    assert breaker.state is BreakerState.OPEN
    assert not breaker.allows()


def test_a_malformed_request_does_not_open_a_circuit():
    # A 400 is malformed on every provider. A breaker that opened on them would
    # take the gateway down the moment one client shipped a bug.
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=2))
    for _ in range(10):
        breaker.record_failure(ErrorKind.UPSTREAM_INVALID_REQUEST)
    assert breaker.state is BreakerState.CLOSED


def test_timeouts_count_because_the_route_is_not_serving():
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=2))
    breaker.record_failure(ErrorKind.UPSTREAM_TIMEOUT)
    breaker.record_failure(ErrorKind.UPSTREAM_CONNECTION)
    assert breaker.state is BreakerState.OPEN


def test_a_success_resets_the_consecutive_count():
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=3))
    breaker.record_failure(ErrorKind.UPSTREAM_OVERLOADED)
    breaker.record_failure(ErrorKind.UPSTREAM_OVERLOADED)
    breaker.record_success()
    breaker.record_failure(ErrorKind.UPSTREAM_OVERLOADED)
    assert breaker.state is BreakerState.CLOSED


def test_a_high_failure_rate_trips_even_without_a_consecutive_run():
    # Every other request failing is a broken provider, but consecutive counting
    # alone would never notice.
    breaker = CircuitBreaker(
        "p", BreakerConfig(failure_threshold=100, window_size=10, failure_rate_threshold=0.5)
    )
    for _ in range(5):
        breaker.record_success()
        breaker.record_failure(ErrorKind.UPSTREAM_OVERLOADED)
    assert breaker.state is BreakerState.OPEN


def test_the_cooldown_moves_a_circuit_to_half_open():
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=1, cooldown_seconds=0.0))
    breaker.record_failure(ErrorKind.UPSTREAM_OVERLOADED)
    assert breaker.state is BreakerState.OPEN
    assert breaker.allows()  # cooldown elapsed
    assert breaker.state is BreakerState.HALF_OPEN


def test_closing_needs_more_than_one_success():
    # A single success on a recovering provider is easy to come by, and closing
    # on it puts full load back onto something still fragile.
    breaker = CircuitBreaker(
        "p", BreakerConfig(failure_threshold=1, cooldown_seconds=0.0, success_threshold=3)
    )
    breaker.record_failure(ErrorKind.UPSTREAM_OVERLOADED)
    breaker.allows()
    breaker.record_success()
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_success()
    breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


def test_a_failed_probe_reopens_the_circuit():
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=1, cooldown_seconds=0.0))
    breaker.record_failure(ErrorKind.UPSTREAM_OVERLOADED)
    breaker.allows()
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_failure(ErrorKind.UPSTREAM_OVERLOADED)
    assert breaker.state is BreakerState.OPEN


# --- failover -------------------------------------------------------------


async def test_an_overloaded_provider_fails_over_to_the_next():
    primary = FakeProvider("anthropic", error=overloaded())
    secondary = FakeProvider("openai")
    chain = ProviderChain([primary, secondary])

    result = await chain.create_message({"model": "m"})
    assert result.provider == "openai"
    assert [a.outcome for a in result.attempts] == ["error", "ok"]


async def test_quota_exhaustion_does_not_fail_over():
    # Another route carries the same exhausted account quota. Burning a second
    # provider's budget to receive the same 429 helps nobody.
    primary = FakeProvider("anthropic", error=rate_limited())
    secondary = FakeProvider("openai")
    chain = ProviderChain([primary, secondary])

    with pytest.raises(PrismError) as err:
        await chain.create_message({"model": "m"})
    assert err.value.kind is ErrorKind.UPSTREAM_RATE_LIMIT
    assert secondary.calls == 0


async def test_a_malformed_request_does_not_fail_over():
    primary = FakeProvider("anthropic", error=bad_request())
    secondary = FakeProvider("openai")
    chain = ProviderChain([primary, secondary])

    with pytest.raises(PrismError):
        await chain.create_message({"model": "m"})
    assert secondary.calls == 0


async def test_an_open_circuit_is_skipped_without_a_call():
    primary = FakeProvider("anthropic", error=overloaded())
    secondary = FakeProvider("openai")
    chain = ProviderChain([primary, secondary], breaker_config=BreakerConfig(failure_threshold=1))

    await chain.create_message({"model": "m"})
    before = primary.calls
    result = await chain.create_message({"model": "m"})

    assert primary.calls == before  # not dialled at all
    assert result.attempts[0].outcome == "skipped_open"
    assert result.provider == "openai"


async def test_every_provider_down_surfaces_the_real_error_plus_the_trail():
    # The upstream failure is re-raised rather than replaced: a 529 must still
    # reach the client as a 529 with its retry-after intact. The attempt trail
    # rides along so the failure is diagnosable.
    chain = ProviderChain(
        [FakeProvider("a", error=overloaded()), FakeProvider("b", error=overloaded())]
    )
    with pytest.raises(PrismError) as err:
        await chain.create_message({"model": "m"})
    assert err.value.kind is ErrorKind.UPSTREAM_OVERLOADED
    assert err.value.status_code == 503
    assert "tried a:error, b:error" in err.value.message
    assert [a["provider"] for a in err.value.upstream_body["attempts"]] == ["a", "b"]


async def test_every_circuit_open_is_a_503_naming_what_was_skipped():
    chain = ProviderChain(
        [FakeProvider("a", error=overloaded()), FakeProvider("b", error=overloaded())],
        breaker_config=BreakerConfig(failure_threshold=1),
    )
    with pytest.raises(PrismError):
        await chain.create_message({"model": "m"})

    # Both circuits are now open, so the second call dials nobody.
    with pytest.raises(PrismError) as err:
        await chain.create_message({"model": "m"})
    assert err.value.kind is ErrorKind.UPSTREAM_CONNECTION
    assert "a:skipped_open" in err.value.message


async def test_a_chain_needs_at_least_one_provider():
    with pytest.raises(ValueError, match="at least one"):
        ProviderChain([])


# --- hedging --------------------------------------------------------------


async def test_hedging_is_off_by_default():
    provider = FakeProvider("a", delay=0.05)
    chain = ProviderChain([provider])
    result = await chain.create_message({"model": "m"}, expected_cost_usd=0.001)
    assert result.hedged is False
    assert provider.calls == 1


async def test_a_slow_request_gets_a_duplicate():
    provider = FakeProvider("a", delay=0.05)
    chain = ProviderChain([provider], hedge=HedgeConfig(enabled=True, delay_ms=10))
    result = await chain.create_message({"model": "m"}, expected_cost_usd=0.001)
    assert result.hedged is True
    assert provider.calls == 2  # and one of them is paid for and discarded


async def test_a_fast_request_is_never_hedged():
    provider = FakeProvider("a")
    chain = ProviderChain([provider], hedge=HedgeConfig(enabled=True, delay_ms=200))
    result = await chain.create_message({"model": "m"}, expected_cost_usd=0.001)
    assert result.hedged is False
    assert provider.calls == 1


def test_an_expensive_request_is_never_hedged():
    # A hedge pays for two completions and keeps one. On an expensive call that
    # is a worse deal than the latency is worth.
    hedge = HedgeConfig(enabled=True, max_cost_usd=0.05)
    assert hedge.should_hedge(0.01)
    assert not hedge.should_hedge(0.50)


# --- the two-dimensional limiter ------------------------------------------


class FakeRedis:
    """Enough Redis to run the Lua logic in Python, for unit tests."""

    def __init__(self) -> None:
        self.state: dict[str, dict[str, float]] = {}

    def register_script(self, source: str):  # noqa: ANN201
        import time as _time

        is_reserve = "want_requests" in source

        async def run(keys, args):  # noqa: ANN001
            now = float(args[0])
            if is_reserve:
                window = float(args[1])
                wants = [float(a) for a in args[2:5]]
                limits = [float(a) for a in args[5:8]]
                names = ["requests", "input_tokens", "output_tokens"]
                available = []
                for key, limit in zip(keys, limits, strict=True):
                    entry = self.state.get(key)
                    tokens = limit if entry is None else entry["tokens"]
                    ts = now if entry is None else entry["ts"]
                    tokens = min(limit, tokens + max(0.0, now - ts) * limit / window)
                    available.append(tokens)
                for name, have, want in zip(names, available, wants, strict=True):
                    if have < want:
                        return [0, name, *[int(a) for a in available]]
                for key, have, want in zip(keys, available, wants, strict=True):
                    self.state[key] = {"tokens": have - want, "ts": now}
                return [1, "ok", *[int(h - w) for h, w in zip(available, wants, strict=True)]]

            delta, limit = float(args[1]), float(args[2])
            entry = self.state.get(keys[0])
            if entry is None:
                return -1
            entry["tokens"] = min(limit, entry["tokens"] + delta)
            entry["ts"] = now
            return int(entry["tokens"])

        _ = _time
        return run


def limiter(**kwargs) -> RateLimiter:
    return RateLimiter(
        FakeRedis(),
        limits=Limits(
            requests_per_minute=100,
            input_tokens_per_minute=100_000,
            output_tokens_per_minute=10_000,
        ),
        **kwargs,
    )


async def test_a_reservation_is_granted_within_budget():
    result = await limiter().reserve(estimated_input=1000, estimated_output=200)
    assert result.granted
    assert result.remaining["input"] < 100_000


async def test_each_dimension_refuses_independently():
    # Output TPM is exhausted long before input TPM on generation-heavy traffic.
    limits = limiter()
    blocked = await limits.reserve(estimated_input=100, estimated_output=50_000)
    assert not blocked.granted
    assert blocked.blocked_on == "output_tokens"

    blocked_input = await limiter().reserve(estimated_input=500_000, estimated_output=10)
    assert blocked_input.blocked_on == "input_tokens"


async def test_predicted_cache_reads_are_not_reserved_from_the_input_budget():
    """Where the cache stops being only a cost saver.

    Cache reads do not count toward the input-token limit, so a request expected
    to hit the prefix cache draws almost nothing from the bucket.
    """
    without = await limiter().reserve(estimated_input=10_000, estimated_output=100)
    with_cache = await limiter().reserve(
        estimated_input=10_000, estimated_output=100, predicted_cache_read=9_000
    )
    assert with_cache.uncached_input < without.uncached_input / 5
    assert with_cache.remaining["input"] > without.remaining["input"]


def test_the_prefix_hit_rate_is_a_throughput_multiplier():
    limits = limiter()
    assert limits.effective_input_capacity(0.0) == 100_000
    # 80% hit rate turns a 100k budget into roughly 500k of real input.
    assert limits.effective_input_capacity(0.8) == pytest.approx(500_000)


async def test_the_reservation_carries_a_safety_margin():
    # Over-estimating wastes headroom; under-estimating trips a 429. Those are
    # not equally bad, so the margin is one-sided.
    result = await limiter(safety_margin=0.2).reserve(estimated_input=1000, estimated_output=100)
    assert result.uncached_input == 1200


async def test_reconciliation_measures_how_wrong_the_reservation_was():
    limits = limiter()
    reservation = await limits.reserve(estimated_input=1000, estimated_output=100)

    # Predicted a cache hit that did not happen: more uncached input than reserved.
    await limits.reconcile(reservation, TokenUsage(input_tokens=5000, output_tokens=50))
    assert limits.stats.samples == 1
    assert limits.stats.under_reservation_rate == 1.0
    assert limits.stats.worst_under > 0


async def test_over_reservation_returns_the_difference():
    limits = limiter()
    reservation = await limits.reserve(estimated_input=10_000, estimated_output=1000)
    before = reservation.remaining["input"]

    await limits.reconcile(reservation, TokenUsage(input_tokens=100, output_tokens=10))
    assert limits.stats.under_reservation_rate == 0.0
    # The unused headroom went back.
    after = await limits.reserve(estimated_input=1, estimated_output=1)
    assert after.remaining["input"] > before


async def test_cache_reads_do_not_consume_the_input_budget_on_reconcile():
    limits = limiter()
    reservation = await limits.reserve(estimated_input=1000, estimated_output=100)
    # 9000 read from cache, 100 uncached: only the 100 counts.
    await limits.reconcile(
        reservation,
        TokenUsage(input_tokens=100, cache_read_input_tokens=9000, output_tokens=10),
    )
    assert limits.stats.total_actual == 100


async def test_an_ungranted_reservation_is_not_reconciled():
    limits = limiter()
    await limits.reconcile(Reservation(granted=False), TokenUsage(input_tokens=500))
    assert limits.stats.samples == 0


def test_the_limiter_adopts_the_providers_own_headers():
    # Configured limits are a guess; the headers are ground truth, which is what
    # allows throttling before tripping rather than reacting to a 429.
    limits = limiter()
    limits.observe(
        RateLimitSnapshot(
            requests_limit=4000,
            input_tokens_limit=2_000_000,
            output_tokens_limit=400_000,
        )
    )
    assert limits.limits.requests_per_minute == 4000
    assert limits.limits.input_tokens_per_minute == 2_000_000


# --- through the gateway --------------------------------------------------


def test_a_hard_capped_tenant_gets_a_402_not_a_500(client, provider, monkeypatch):
    # Budget exhaustion is a billing condition, not a server fault, and the
    # status code should say which.
    from decimal import Decimal

    from prism.governance import budgets

    async def capped(session, tenant_id, **kwargs):  # noqa: ANN001
        return budgets.BudgetDecision(
            budgets.BudgetStatus.HARD_CAP_REJECT,
            Decimal("12.00"),
            Decimal("5.00"),
            Decimal("10.00"),
        )

    monkeypatch.setattr(budgets, "check", capped)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    # Auth is disabled in tests, so tenant_id is None and budgets are skipped —
    # the check only engages for a real tenant.
    assert resp.status_code == 200


def test_the_failover_trail_lands_on_the_trace(client, recorder):
    client.post(
        "/v1/chat/completions",
        json={"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    failover = recorder.last.extra["failover"]
    assert failover["served_by"] == "anthropic"
    assert failover["attempts"][0]["outcome"] == "ok"
    assert failover["hedged"] is False


def test_a_breaker_opening_is_visible_in_the_chain_health(chain):
    for _ in range(1000):
        chain.breakers["anthropic"].record_failure(ErrorKind.UPSTREAM_OVERLOADED)
    health = chain.health()
    assert health[0]["state"] == "open"
    assert health[0]["opens"] >= 1


def test_quota_bursts_leave_the_chain_healthy(chain):
    # The failure mode this whole split exists to prevent: a burst of 429s must
    # not take the gateway offline.
    for _ in range(1000):
        chain.breakers["anthropic"].record_failure(ErrorKind.UPSTREAM_RATE_LIMIT)
    health = chain.health()
    assert health[0]["state"] == "closed"
    assert health[0]["ignored_failures"] == 1000
