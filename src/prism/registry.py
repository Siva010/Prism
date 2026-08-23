"""Model registry: capabilities and pricing.

Two jobs, deliberately in one place:

1. **Capabilities.** The Haiku/Sonnet/Opus ladder does not have a uniform request
   surface. Opus 5 and Sonnet 5 reject ``temperature``/``top_p``/``top_k`` with a
   400; Haiku 4.5 accepts them. Opus 5 and Sonnet 5 take adaptive thinking; Haiku
   4.5 takes a ``budget_tokens`` budget. A gateway that forwards an OpenAI request
   verbatim to whichever tier the router picked will 400 on some tiers and not
   others, so the translation layer consults this table.

2. **Pricing, per token class.** Uncached input, cached input, cache writes, and
   output price differently. ``total_tokens * unit_price`` is wrong by a large
   margin once caching is active, which is the entire point of the cost work.

Prices are USD per million tokens and drift faster than any other constant here.
Pin them against https://platform.claude.com/docs/en/about-claude/pricing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Literal

ThinkingStyle = Literal["adaptive", "budget_tokens", "none"]

# Multipliers applied to the base input price.
CACHE_WRITE_5M_MULTIPLIER = Decimal("1.25")
CACHE_WRITE_1H_MULTIPLIER = Decimal("2.0")
CACHE_READ_MULTIPLIER = Decimal("0.1")


@dataclass(frozen=True)
class ModelSpec:
    id: str
    tier: Literal["haiku", "sonnet", "opus"]
    context_window: int
    max_output_tokens: int
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    thinking: ThinkingStyle
    supports_sampling_params: bool
    supports_effort: bool
    # Introductory pricing, when active, replaces the rates above until the date.
    intro_input_per_mtok: Decimal | None = None
    intro_output_per_mtok: Decimal | None = None
    intro_until: date | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def rates(self, on: date | None = None) -> tuple[Decimal, Decimal]:
        """(input, output) $/MTok, honouring introductory pricing windows."""
        if self.intro_until is None:
            return self.input_per_mtok, self.output_per_mtok
        today = on or date.today()
        if today <= self.intro_until:
            return (
                self.intro_input_per_mtok or self.input_per_mtok,
                self.intro_output_per_mtok or self.output_per_mtok,
            )
        return self.input_per_mtok, self.output_per_mtok


MODELS: dict[str, ModelSpec] = {
    "claude-haiku-4-5": ModelSpec(
        id="claude-haiku-4-5",
        tier="haiku",
        context_window=200_000,
        max_output_tokens=64_000,
        input_per_mtok=Decimal("1.00"),
        output_per_mtok=Decimal("5.00"),
        thinking="budget_tokens",
        supports_sampling_params=True,
        supports_effort=False,
    ),
    "claude-sonnet-5": ModelSpec(
        id="claude-sonnet-5",
        tier="sonnet",
        context_window=1_000_000,
        max_output_tokens=128_000,
        input_per_mtok=Decimal("3.00"),
        output_per_mtok=Decimal("15.00"),
        intro_input_per_mtok=Decimal("2.00"),
        intro_output_per_mtok=Decimal("10.00"),
        intro_until=date(2026, 8, 31),
        thinking="adaptive",
        supports_sampling_params=False,
        supports_effort=True,
    ),
    "claude-opus-5": ModelSpec(
        id="claude-opus-5",
        tier="opus",
        context_window=1_000_000,
        max_output_tokens=128_000,
        input_per_mtok=Decimal("5.00"),
        output_per_mtok=Decimal("25.00"),
        thinking="adaptive",
        supports_sampling_params=False,
        supports_effort=True,
    ),
}

# The drop-in pitch means clients arrive with OpenAI model names. Map them onto
# the ladder rather than 404-ing, and record both names on the trace so the
# dashboard can show what was asked for and what was served.
MODEL_ALIASES: dict[str, str] = {
    "gpt-4o-mini": "claude-haiku-4-5",
    "gpt-4o": "claude-sonnet-5",
    "gpt-4-turbo": "claude-sonnet-5",
    "gpt-4": "claude-opus-5",
    "gpt-5": "claude-opus-5",
    # Convenience aliases for the tiers themselves.
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
    "prism-auto": "claude-opus-5",  # replaced by the router in week 9
}


class UnknownModelError(ValueError):
    pass


def resolve_model(requested: str) -> ModelSpec:
    """Map a client-supplied model name onto a concrete Anthropic model."""
    name = requested.strip()
    if name in MODELS:
        return MODELS[name]
    aliased = MODEL_ALIASES.get(name) or MODEL_ALIASES.get(name.lower())
    if aliased and aliased in MODELS:
        return MODELS[aliased]
    raise UnknownModelError(
        f"Unknown model {requested!r}. Known models: "
        + ", ".join(sorted(MODELS) + sorted(MODEL_ALIASES))
    )
