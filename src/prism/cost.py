"""Cost accounting over the five token classes.

The naive model — ``total_tokens * unit_price`` — is wrong in both directions once
caching is on: it overcharges cache reads by 10x and undercharges cache writes by
25%. Prism therefore never stores a total; it stores the classes and derives cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal

from .registry import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_1H_MULTIPLIER,
    CACHE_WRITE_5M_MULTIPLIER,
    ModelSpec,
)

_PER_MTOK = Decimal(1_000_000)


@dataclass(frozen=True)
class TokenUsage:
    """The response ``usage`` object, normalized.

    ``output_tokens`` already includes thinking tokens — they bill as output.
    ``thinking_tokens`` is carried separately for attribution, not for billing,
    and adding it to the cost would double-count.
    """

    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def billable_input_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    @property
    def prefix_cache_hit_rate(self) -> float:
        """Share of input tokens served from the provider prefix cache."""
        total = self.billable_input_tokens
        return self.cache_read_input_tokens / total if total else 0.0

    @property
    def write_read_ratio(self) -> float | None:
        """Cache writes per cache read.

        The single number that says whether breakpoints are placed correctly. A
        ratio persistently above ~1 means the breakpoints sit on volatile content:
        every request pays the write premium and nothing is ever read back, which
        is strictly worse than placing no breakpoint at all.
        """
        if self.cache_read_input_tokens == 0:
            return None if self.cache_creation_input_tokens == 0 else float("inf")
        return self.cache_creation_input_tokens / self.cache_read_input_tokens


@dataclass(frozen=True)
class CostBreakdown:
    uncached_input_usd: Decimal
    cached_input_usd: Decimal
    cache_write_usd: Decimal
    output_usd: Decimal

    @property
    def total_usd(self) -> Decimal:
        return (
            self.uncached_input_usd
            + self.cached_input_usd
            + self.cache_write_usd
            + self.output_usd
        )

    def as_json(self) -> dict[str, str]:
        # Serialized as strings: JSONB would otherwise round-trip Decimal through
        # float and lose cents at aggregate scale.
        return {k: str(v) for k, v in asdict(self).items()} | {
            "total_usd": str(self.total_usd)
        }


def compute_cost(
    usage: TokenUsage,
    spec: ModelSpec,
    *,
    cache_ttl: str = "5m",
    on: date | None = None,
) -> CostBreakdown:
    input_rate, output_rate = spec.rates(on)
    write_multiplier = (
        CACHE_WRITE_1H_MULTIPLIER if cache_ttl == "1h" else CACHE_WRITE_5M_MULTIPLIER
    )

    def price(tokens: int, rate: Decimal) -> Decimal:
        return (Decimal(tokens) * rate) / _PER_MTOK

    return CostBreakdown(
        uncached_input_usd=price(usage.input_tokens, input_rate),
        cached_input_usd=price(
            usage.cache_read_input_tokens, input_rate * CACHE_READ_MULTIPLIER
        ),
        cache_write_usd=price(
            usage.cache_creation_input_tokens, input_rate * write_multiplier
        ),
        output_usd=price(usage.output_tokens, output_rate),
    )


def naive_cost(usage: TokenUsage, spec: ModelSpec, *, on: date | None = None) -> Decimal:
    """What a `total_tokens * unit_price` model would have reported.

    Kept so the dashboard can show the gap between the naive model and the real
    one — the size of that gap is the argument for having built this.
    """
    input_rate, output_rate = spec.rates(on)
    return (
        Decimal(usage.billable_input_tokens) * input_rate
        + Decimal(usage.output_tokens) * output_rate
    ) / _PER_MTOK
