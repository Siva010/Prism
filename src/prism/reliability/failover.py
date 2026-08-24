"""Provider failover, and when hedging is worth the money.

A chain of providers, each behind its own circuit breaker. A request walks the
chain until one accepts it. Three rules make this less obvious than it looks:

**Not every failure is worth failing over on.** A 400 is malformed on every
provider; retrying it elsewhere burns a second provider's quota to receive the
same error. Only capacity-class failures and transport errors move to the next
link. A 429 is the awkward middle case — it is retryable, but *not* on another
route belonging to the same account, so it is surfaced with `retry-after`
rather than failed over.

**Hedging is a cost decision, not a latency trick.** Firing a duplicate at the
p95 mark buys tail latency and pays for it in tokens: every hedge that loses is
a completion billed and discarded. It is worth it only when the tail is long
relative to the median and the request is cheap relative to the delay's cost.
And a hedged request must never double-reserve rate-limit budget — the second
call has to draw on the same reservation, or the limiter will be wrong by
exactly the hedge rate.

**Degrading beats failing.** When every provider is unavailable, dropping to a
cheaper tier on a healthy one is better than a 503, and it is a decision the
caller should be told about rather than one made silently.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ..logging_setup import get_logger
from ..providers.base import ProviderResponse
from ..schemas.errors import ErrorKind, PrismError
from .breaker import BreakerConfig, CircuitBreaker

log = get_logger(__name__)


@dataclass
class Attempt:
    provider: str
    outcome: str  # ok | error | skipped_open
    error_kind: str | None = None
    latency_ms: int = 0


@dataclass
class FailoverResult:
    response: ProviderResponse
    provider: str
    attempts: list[Attempt] = field(default_factory=list)
    hedged: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "served_by": self.provider,
            "hedged": self.hedged,
            "attempts": [
                {
                    "provider": a.provider,
                    "outcome": a.outcome,
                    "error_kind": a.error_kind,
                    "latency_ms": a.latency_ms,
                }
                for a in self.attempts
            ],
        }


@dataclass
class HedgeConfig:
    enabled: bool = False
    #: Fire the duplicate at this delay, normally the observed p95.
    delay_ms: int = 2000
    #: Never hedge a request whose expected cost exceeds this. A hedge on an
    #: expensive completion pays for two and keeps one.
    max_cost_usd: float = 0.05

    def should_hedge(self, expected_cost_usd: float) -> bool:
        return self.enabled and expected_cost_usd <= self.max_cost_usd


class ProviderChain:
    """Ordered providers, each with a breaker."""

    def __init__(
        self,
        providers: list[Any],
        *,
        breaker_config: BreakerConfig | None = None,
        hedge: HedgeConfig | None = None,
    ) -> None:
        if not providers:
            raise ValueError("a provider chain needs at least one provider")
        self.providers = providers
        self.hedge = hedge or HedgeConfig()
        self.breakers = {
            p.name: CircuitBreaker(p.name, breaker_config or BreakerConfig()) for p in providers
        }

    def health(self) -> list[dict[str, Any]]:
        return [b.as_json() for b in self.breakers.values()]

    async def create_message(
        self,
        body: dict[str, Any],
        *,
        timeout_s: float | None = None,
        expected_cost_usd: float = 0.0,
    ) -> FailoverResult:
        attempts: list[Attempt] = []
        last_error: PrismError | None = None

        for provider in self.providers:
            breaker = self.breakers[provider.name]
            if not breaker.allows():
                attempts.append(Attempt(provider.name, "skipped_open"))
                continue

            started = time.perf_counter()
            try:
                if self.hedge.should_hedge(expected_cost_usd):
                    response, hedged = await self._hedged_call(provider, body, timeout_s)
                else:
                    response, hedged = (
                        await provider.create_message(body, timeout_s=timeout_s),
                        False,
                    )
            except PrismError as exc:
                latency = int((time.perf_counter() - started) * 1000)
                breaker.record_failure(exc.kind)
                attempts.append(Attempt(provider.name, "error", str(exc.kind), latency))
                last_error = exc

                if exc.kind is ErrorKind.UPSTREAM_RATE_LIMIT:
                    # Quota belongs to the account, not the route. Failing over
                    # carries the same exhausted quota somewhere else.
                    exc.upstream_body = {"attempts": [a.provider for a in attempts]}
                    raise
                if not exc.kind.is_retryable:
                    # A 400 is malformed everywhere. Trying the next provider
                    # spends its quota to receive the same error.
                    raise
                continue

            breaker.record_success()
            attempts.append(
                Attempt(
                    provider.name,
                    "ok",
                    None,
                    int((time.perf_counter() - started) * 1000),
                )
            )
            return FailoverResult(response, provider.name, attempts, hedged)

        trail = ", ".join(f"{a.provider}:{a.outcome}" for a in attempts)
        if last_error is not None:
            # Re-raise the real upstream failure rather than a generic one: a 529
            # should still reach the client as a 529 with its retry-after intact.
            # The attempt trail is attached so the failure is still diagnosable.
            last_error.message = f"{last_error.message} (tried {trail})"
            last_error.args = (last_error.message,)
            last_error.upstream_body = {
                "attempts": [
                    {"provider": a.provider, "outcome": a.outcome, "error": a.error_kind}
                    for a in attempts
                ]
            }
            raise last_error

        # Nothing was even dialled — every circuit was open.
        raise PrismError(
            ErrorKind.UPSTREAM_CONNECTION,
            f"every provider in the chain is unavailable ({trail})",
            status_code=503,
        )

    async def _hedged_call(
        self, provider: Any, body: dict[str, Any], timeout_s: float | None
    ) -> tuple[ProviderResponse, bool]:
        """Fire a duplicate after the hedge delay; keep whichever lands first.

        The loser is cancelled, but tokens it already generated are still
        billed — which is the whole cost of the technique and the reason it is
        off by default.
        """
        first = asyncio.create_task(provider.create_message(body, timeout_s=timeout_s))
        done, _ = await asyncio.wait({first}, timeout=self.hedge.delay_ms / 1000)
        if done:
            return first.result(), False

        second = asyncio.create_task(provider.create_message(body, timeout_s=timeout_s))
        done, pending = await asyncio.wait({first, second}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        winner = done.pop()
        result: ProviderResponse = winner.result()
        log.info(
            "hedge_fired",
            provider=provider.name,
            winner="hedge" if winner is second else "original",
        )
        return result, True
