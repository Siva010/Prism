"""Rate-limit buckets against real Redis, skipped when there isn't one.

The unit tests in `test_reliability.py` reimplement the Lua logic in Python,
which checks the *reasoning* but not the script. The Lua has to run somewhere,
and "somewhere" cannot be production — that is the lesson `scripts/migrate.py`
taught this project the expensive way.

Atomicity is the specific thing only a real server can demonstrate: the whole
reason reservation is a script rather than a read-modify-write from Python is
that two concurrent requests must not both see room for the last thousand tokens.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid

import pytest

from prism.cost import TokenUsage
from prism.reliability.ratelimit import Limits, RateLimiter

REDIS_URL = os.environ.get("PRISM_TEST_REDIS_URL", "redis://localhost:6380/0")


def _reachable() -> bool:
    async def probe() -> bool:
        import redis.asyncio as aioredis

        client = aioredis.from_url(REDIS_URL)
        try:
            await client.ping()
            return True
        except Exception:  # noqa: BLE001
            return False
        finally:
            await client.aclose()

    with contextlib.suppress(Exception):
        return asyncio.run(probe())
    return False


pytestmark = pytest.mark.skipif(
    not _reachable(), reason=f"no Redis at {REDIS_URL}; run `docker compose up -d`"
)


@pytest.fixture
async def limiter():
    import redis.asyncio as aioredis

    client = aioredis.from_url(REDIS_URL)
    made = RateLimiter(
        client,
        provider=f"itest-{uuid.uuid4().hex[:8]}",
        limits=Limits(
            requests_per_minute=100,
            input_tokens_per_minute=100_000,
            output_tokens_per_minute=10_000,
        ),
    )
    yield made
    await client.delete(*made._keys())
    await client.aclose()


async def test_the_lua_script_actually_runs(limiter):
    result = await limiter.reserve(estimated_input=1000, estimated_output=100)
    assert result.granted
    assert result.remaining["input"] < 100_000
    assert result.remaining["requests"] == 99


async def test_each_dimension_refuses_independently_in_redis(limiter):
    blocked = await limiter.reserve(estimated_input=100, estimated_output=50_000)
    assert not blocked.granted
    assert blocked.blocked_on == "output_tokens"
    # A refusal must not have spent anything from the other dimensions.
    after = await limiter.reserve(estimated_input=100, estimated_output=10)
    assert after.remaining["requests"] == 99


async def test_reservation_is_atomic_under_concurrency(limiter):
    """The reason this is a script and not a read-modify-write.

    Twenty requests race for a budget that fits ten. Exactly ten must win — a
    check-then-write from Python would let far more through.
    """
    limiter.limits = Limits(
        requests_per_minute=10,
        input_tokens_per_minute=10_000_000,
        output_tokens_per_minute=10_000_000,
        window_seconds=600.0,  # long window, so refill cannot mask the race
    )
    results = await asyncio.gather(
        *(limiter.reserve(estimated_input=1, estimated_output=1) for _ in range(20))
    )
    granted = sum(r.granted for r in results)
    assert granted == 10, f"expected exactly 10 grants, got {granted}"


async def test_reconciliation_returns_unused_budget_to_redis(limiter):
    reservation = await limiter.reserve(estimated_input=50_000, estimated_output=5_000)
    assert reservation.granted

    # Nowhere near the reservation: the headroom should come back.
    await limiter.reconcile(reservation, TokenUsage(input_tokens=100, output_tokens=10))
    after = await limiter.reserve(estimated_input=40_000, estimated_output=10)
    assert after.granted, "returned budget was not actually returned"


async def test_cache_reads_do_not_consume_the_input_bucket(limiter):
    """The coupling between week 7 and week 10, end to end.

    A request that reads 90k tokens from the prefix cache and 1k uncached should
    leave the bucket almost untouched.
    """
    reservation = await limiter.reserve(
        estimated_input=91_000, estimated_output=100, predicted_cache_read=90_000
    )
    assert reservation.granted
    assert reservation.uncached_input < 1_500
    # Enough left for a second large request, which would be impossible if the
    # cached tokens had been charged to the bucket.
    second = await limiter.reserve(
        estimated_input=91_000, estimated_output=100, predicted_cache_read=90_000
    )
    assert second.granted
