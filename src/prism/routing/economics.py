"""When routing down is worth it — derived, not tuned.

Most routers pick a threshold on predicted difficulty by hand and defend it with
a plot. There is no need: the threshold falls out of the cost structure.

Let `p` be the probability the cheap configuration produces an acceptable answer,
`c` the cost of the cheap call, `e` the cost of the expensive one, and assume a
failure is retried on the expensive tier. Then:

    E[cost | route cheap]     = c + (1 - p) * e
    E[cost | route expensive] = e

Routing cheap is worth it exactly when `c + (1 - p)e < e`, which simplifies to

    p > c / e

**The threshold is the cost ratio.** Nothing to tune. For Haiku against Opus
that is roughly 0.2 — route down whenever the cheap tier has better than a 1-in-5
chance of being right, because four wasted Haiku calls still cost less than one
avoided Opus call. For Sonnet against Opus it is ~0.6, which is far less
permissive, and that difference is a fact about the price list rather than a
judgement call.

Two refinements matter in practice:

* **A verifier is not free.** Verify-then-escalate pays `v` on every request, so
  the rule becomes `p > (c + v) / e`. A verifier that costs a third of the
  expensive call destroys most of the advantage, which is why the cheap tier is
  usually its own verifier or there is no verifier at all.
* **Without escalation, the currency is quality, not money.** If a bad cheap
  answer is simply delivered, routing down always saves money and the constraint
  has to be a quality floor instead. Both modes are implemented; the difference
  is explicit rather than a flag nobody reads.

This is the same shape of argument as the semantic cache's threshold: name the
two errors, work out what each costs, and let the arithmetic pick the operating
point.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from ..cost import TokenUsage, compute_cost
from ..registry import MODELS, ModelSpec

# A representative request, used only to turn two price lists into one ratio.
# The ratio is insensitive to the exact shape as long as both tiers see the same
# one, which is the point of holding it fixed.
_REFERENCE = TokenUsage(input_tokens=2000, output_tokens=500)


class EscalationMode(StrEnum):
    #: Cheap answer is delivered as-is. Routing down trades quality, not money.
    NONE = "none"
    #: Cheap answer is checked; a failure is retried on the expensive tier.
    VERIFY_THEN_ESCALATE = "verify_then_escalate"


@dataclass(frozen=True)
class RoutingEconomics:
    cheap: ModelSpec
    expensive: ModelSpec
    mode: EscalationMode = EscalationMode.VERIFY_THEN_ESCALATE
    #: Verifier cost as a fraction of the cheap call. 0 means the cheap tier
    #: verifies itself (a second short call), which is the usual arrangement.
    verifier_fraction: float = 0.0
    #: Only used in NONE mode: the minimum acceptable success probability.
    quality_floor: float = 0.85

    @property
    def cheap_cost(self) -> Decimal:
        return compute_cost(_REFERENCE, self.cheap).total_usd

    @property
    def expensive_cost(self) -> Decimal:
        return compute_cost(_REFERENCE, self.expensive).total_usd

    @property
    def cost_ratio(self) -> float:
        """c / e on a fixed reference request."""
        expensive = self.expensive_cost
        if expensive == 0:
            return 1.0
        return float(self.cheap_cost / expensive)

    @property
    def threshold(self) -> float:
        """The predicted-success probability above which routing down pays.

        In escalation mode this is (c + v) / e, derived above. In NONE mode there
        is no retry to pay for, so money always favours the cheap tier and the
        binding constraint is the quality floor instead.
        """
        if self.mode is EscalationMode.NONE:
            return self.quality_floor
        verifier = self.cost_ratio * self.verifier_fraction
        return min(1.0, self.cost_ratio + verifier)

    def expected_cost(self, p_success: float) -> Decimal:
        """Expected spend when routing down at this success probability."""
        c, e = self.cheap_cost, self.expensive_cost
        if self.mode is EscalationMode.NONE:
            return c
        verifier = c * Decimal(str(self.verifier_fraction))
        return c + verifier + Decimal(str(1.0 - p_success)) * e

    def saving_vs_expensive(self, p_success: float) -> Decimal:
        return self.expensive_cost - self.expected_cost(p_success)

    def worth_routing_down(self, p_success: float) -> bool:
        return p_success > self.threshold

    def as_json(self) -> dict[str, object]:
        return {
            "cheap": self.cheap.id,
            "expensive": self.expensive.id,
            "mode": str(self.mode),
            "cost_ratio": self.cost_ratio,
            "threshold": self.threshold,
            "cheap_cost_usd": str(self.cheap_cost),
            "expensive_cost_usd": str(self.expensive_cost),
        }


def for_tiers(
    cheap: str,
    expensive: str,
    *,
    mode: EscalationMode = EscalationMode.VERIFY_THEN_ESCALATE,
    verifier_fraction: float = 0.0,
    quality_floor: float = 0.85,
) -> RoutingEconomics:
    return RoutingEconomics(
        cheap=MODELS[cheap],
        expensive=MODELS[expensive],
        mode=mode,
        verifier_fraction=verifier_fraction,
        quality_floor=quality_floor,
    )


def break_even_table(
    mode: EscalationMode = EscalationMode.VERIFY_THEN_ESCALATE,
) -> list[dict[str, object]]:
    """Every downgrade on the ladder and the probability that justifies it.

    Worth printing, because the numbers are unintuitive: the Haiku threshold is
    low enough that routing down is right far more often than instinct suggests,
    while the Sonnet-to-Opus one is strict enough that it rarely is.
    """
    ladder = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]
    rows = []
    for i, cheap in enumerate(ladder):
        for expensive in ladder[i + 1 :]:
            economics = for_tiers(cheap, expensive, mode=mode)
            rows.append(economics.as_json())
    return rows
