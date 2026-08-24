"""Token-aware rate limiting, in two dimensions.

Providers limit three things independently: requests per minute, *input* tokens
per minute, and *output* tokens per minute. Tripping any one of them is a 429, so
a limiter that tracks only request count is a limiter that does not work.

**The wrinkle that makes this interesting.** Cache reads do not count toward the
input-token limit on most models. So the input budget is really two budgets —
cached and uncached — and the prefix cache stops being only a cost saver and
becomes a **throughput multiplier**. An 80% prefix hit rate against a 2M ITPM
limit is roughly 10M effective input tokens per minute. That coupling between the
caching layer and the rate-limiting layer only shows up once both exist, and it
means the reservation has to *predict cache behaviour* to reserve correctly.

Which creates the real difficulty: a request expected to hit the prefix cache
reserves almost nothing from the uncached bucket, and if that prediction is wrong
the bucket has been over-committed. So every reservation is reconciled against
the actual `usage` the provider reports, and the reconciliation error is measured
rather than assumed.

**Reserve pessimistically, reconcile honestly.** The estimator's error
distribution (see `tokens.py`) is asymmetric in its consequences here: over-
estimating wastes headroom, under-estimating trips a 429. So the reservation adds
a margin taken from the estimator's *upper* error percentile, not its mean.

The buckets live in Redis and are updated by a Lua script, because reserving
against three dimensions has to be atomic. Checking then writing from Python
would let two concurrent requests both see room for the last thousand tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..cost import TokenUsage

# One script, one round trip, atomic across all three dimensions. A
# check-then-write from Python would let two concurrent requests both see room
# for the last thousand tokens and both proceed.
_RESERVE_LUA = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local want_requests = tonumber(ARGV[3])
local want_input = tonumber(ARGV[4])
local want_output = tonumber(ARGV[5])
local limit_requests = tonumber(ARGV[6])
local limit_input = tonumber(ARGV[7])
local limit_output = tonumber(ARGV[8])

local function refill(key, limit)
  local state = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(state[1])
  local ts = tonumber(state[2])
  if tokens == nil then
    tokens = limit
    ts = now
  end
  -- Continuous refill rather than a fixed window: a fixed window lets a caller
  -- spend the whole budget at 59.9s and the whole budget again at 60.1s.
  local elapsed = math.max(0, now - ts)
  tokens = math.min(limit, tokens + (elapsed * limit / window))
  return tokens
end

local r_key = KEYS[1]
local i_key = KEYS[2]
local o_key = KEYS[3]

local r = refill(r_key, limit_requests)
local i = refill(i_key, limit_input)
local o = refill(o_key, limit_output)

local function refused(dimension)
  return {0, dimension, math.floor(r), math.floor(i), math.floor(o)}
end

if r < want_requests then return refused('requests') end
if i < want_input    then return refused('input_tokens') end
if o < want_output   then return refused('output_tokens') end

redis.call('HMSET', r_key, 'tokens', r - want_requests, 'ts', now)
redis.call('HMSET', i_key, 'tokens', i - want_input, 'ts', now)
redis.call('HMSET', o_key, 'tokens', o - want_output, 'ts', now)
redis.call('EXPIRE', r_key, math.ceil(window * 2))
redis.call('EXPIRE', i_key, math.ceil(window * 2))
redis.call('EXPIRE', o_key, math.ceil(window * 2))

return {1, 'ok',
        math.floor(r - want_requests),
        math.floor(i - want_input),
        math.floor(o - want_output)}
"""

# Returning tokens after reconciliation. Never pushes a bucket above its limit.
_SETTLE_LUA = """
local now = tonumber(ARGV[1])
local delta = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
if tokens == nil then return -1 end
tokens = math.min(limit, tokens + delta)
redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
return math.floor(tokens)
"""


@dataclass(frozen=True)
class Limits:
    """Per-provider, per-minute ceilings.

    Defaults are placeholders; the real values arrive on every response in the
    `anthropic-ratelimit-*` headers, so `RateLimiter.observe` overwrites them
    from ground truth rather than trusting configuration.
    """

    requests_per_minute: int = 1000
    input_tokens_per_minute: int = 2_000_000
    output_tokens_per_minute: int = 400_000
    window_seconds: float = 60.0


@dataclass(frozen=True)
class Reservation:
    granted: bool
    #: Which dimension refused, when it did. All three are tracked separately
    #: because they are exhausted at different times by different traffic.
    blocked_on: str = "ok"
    requests: int = 0
    uncached_input: int = 0
    output: int = 0
    remaining: dict[str, int] = field(default_factory=dict)
    #: Tokens the request is expected to read from the prefix cache. Reserved
    #: from nothing, because they do not count against the input limit — this is
    #: where the cache becomes a throughput multiplier.
    predicted_cache_read: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "granted": self.granted,
            "blocked_on": self.blocked_on,
            "reserved_input": self.uncached_input,
            "reserved_output": self.output,
            "predicted_cache_read": self.predicted_cache_read,
            "remaining": self.remaining,
        }


@dataclass
class ReconciliationStats:
    """How wrong the reservations were, which is the metric worth reporting."""

    samples: int = 0
    over_reserved: int = 0
    under_reserved: int = 0
    total_reserved: int = 0
    total_actual: int = 0
    worst_under: float = 0.0

    def record(self, reserved: int, actual: int) -> None:
        self.samples += 1
        self.total_reserved += reserved
        self.total_actual += actual
        if reserved >= actual:
            self.over_reserved += 1
        else:
            self.under_reserved += 1
            if actual:
                self.worst_under = max(self.worst_under, (actual - reserved) / actual)

    @property
    def accuracy(self) -> float:
        """Reserved over actual. Above 1.0 means headroom is being wasted."""
        return self.total_reserved / self.total_actual if self.total_actual else 1.0

    @property
    def under_reservation_rate(self) -> float:
        """The dangerous one: these are the requests that can trip a 429."""
        return self.under_reserved / self.samples if self.samples else 0.0

    def as_json(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "accuracy": self.accuracy,
            "under_reservation_rate": self.under_reservation_rate,
            "worst_under_reservation": self.worst_under,
        }


class RateLimiter:
    def __init__(
        self,
        redis: Any,
        *,
        provider: str = "anthropic",
        limits: Limits | None = None,
        safety_margin: float = 0.15,
    ) -> None:
        self.redis = redis
        self.provider = provider
        self.limits = limits or Limits()
        # Taken from the estimator's upper error percentile, not its mean:
        # over-estimating wastes headroom, under-estimating trips a 429, and
        # those are not equally bad.
        self.safety_margin = safety_margin
        self.stats = ReconciliationStats()
        self._reserve = None
        self._settle = None

    def _keys(self) -> list[str]:
        base = f"prism:rl:{self.provider}"
        return [f"{base}:requests", f"{base}:input", f"{base}:output"]

    async def _scripts(self) -> tuple[Any, Any]:
        if self._reserve is None:
            self._reserve = self.redis.register_script(_RESERVE_LUA)
            self._settle = self.redis.register_script(_SETTLE_LUA)
        return self._reserve, self._settle

    async def reserve(
        self,
        *,
        estimated_input: int,
        estimated_output: int,
        predicted_cache_read: int = 0,
    ) -> Reservation:
        """Take budget before dispatch.

        `predicted_cache_read` is subtracted from the input reservation because
        cache reads do not draw on the input-token limit. Predicting it too
        generously over-commits the bucket, which is exactly what reconciliation
        is there to measure.
        """
        uncached = max(0, estimated_input - predicted_cache_read)
        want_input = int(uncached * (1 + self.safety_margin))
        want_output = int(estimated_output * (1 + self.safety_margin))

        reserve, _ = await self._scripts()
        granted, blocked_on, r, i, o = await reserve(
            keys=self._keys(),
            args=[
                time.time(),
                self.limits.window_seconds,
                1,
                want_input,
                want_output,
                self.limits.requests_per_minute,
                self.limits.input_tokens_per_minute,
                self.limits.output_tokens_per_minute,
            ],
        )
        if isinstance(blocked_on, bytes):
            blocked_on = blocked_on.decode()

        return Reservation(
            granted=bool(granted),
            blocked_on=str(blocked_on),
            requests=1,
            uncached_input=want_input,
            output=want_output,
            remaining={"requests": int(r), "input": int(i), "output": int(o)},
            predicted_cache_read=predicted_cache_read,
        )

    async def reconcile(self, reservation: Reservation, usage: TokenUsage) -> None:
        """Return the difference between what was reserved and what was used.

        The prediction is scored here rather than trusted. A request that was
        expected to hit the cache and did not consumed uncached input the bucket
        never accounted for, and `stats.under_reservation_rate` is what makes
        that visible before it becomes a 429.
        """
        if not reservation.granted:
            return

        # Cache reads are free against the input limit, so only uncached input
        # and cache writes count.
        actual_input = usage.input_tokens + usage.cache_creation_input_tokens
        self.stats.record(reservation.uncached_input, actual_input)

        _, settle = await self._scripts()
        keys = self._keys()
        for key, reserved, actual, limit in (
            (
                keys[1],
                reservation.uncached_input,
                actual_input,
                self.limits.input_tokens_per_minute,
            ),
            (
                keys[2],
                reservation.output,
                usage.output_tokens,
                self.limits.output_tokens_per_minute,
            ),
        ):
            delta = reserved - actual
            if delta:
                await settle(keys=[key], args=[time.time(), delta, limit])

    def observe(self, snapshot: Any) -> None:
        """Adopt the provider's own numbers.

        Every response carries the real limits in `anthropic-ratelimit-*`
        headers. Configured values are a starting guess; these are ground truth,
        so throttling happens *before* tripping rather than in reaction to a 429.
        """
        if snapshot is None:
            return
        self.limits = Limits(
            requests_per_minute=snapshot.requests_limit or self.limits.requests_per_minute,
            input_tokens_per_minute=(
                snapshot.input_tokens_limit or self.limits.input_tokens_per_minute
            ),
            output_tokens_per_minute=(
                snapshot.output_tokens_limit or self.limits.output_tokens_per_minute
            ),
            window_seconds=self.limits.window_seconds,
        )

    def effective_input_capacity(self, prefix_hit_rate: float) -> float:
        """Input tokens per minute achievable at a given prefix-cache hit rate.

        The number that makes the coupling concrete: at an 80% hit rate a 2M
        ITPM limit carries roughly 10M input tokens per minute, because four in
        five never touch the budget.
        """
        hit_rate = min(0.999, max(0.0, prefix_hit_rate))
        return self.limits.input_tokens_per_minute / (1.0 - hit_rate)
