"""Circuit breakers, and the distinction they exist to act on.

The error taxonomy from week 1 finally pays off here. `ErrorKind` already knows
which failures are the *provider's* problem, and only those open a circuit:

* **429 — you are out of quota.** Your account, not the provider. Opening a
  circuit would stop traffic that a moment's backoff would serve, and failing
  over would carry the same exhausted quota to another route. It counts for
  nothing here; `retry-after` is the correct response.
* **529 — the provider is over capacity.** Quota is irrelevant, retrying
  immediately makes it worse, and another provider probably *is* healthy. This
  is what a breaker is for.
* **Timeouts and connection failures** count too: whatever is wrong, this route
  is not currently serving.
* **400s do not.** A malformed request will be malformed on every provider, and
  a breaker that opens on them would take the whole gateway down the moment one
  client shipped a bug.

Conflating quota exhaustion with capacity failure is the single most common bug
in hand-rolled client code, and it fails in the worst direction: a burst of 429s
opens every circuit, and the gateway stops serving traffic it could have served.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..schemas.errors import ErrorKind


class BreakerState(StrEnum):
    CLOSED = "closed"  # serving
    OPEN = "open"  # refusing, waiting out the cooldown
    HALF_OPEN = "half_open"  # letting one request through to test the water


class CircuitOpenError(RuntimeError):
    def __init__(self, provider: str, retry_after_s: float) -> None:
        super().__init__(f"circuit for {provider!r} is open; retry in {retry_after_s:.1f}s")
        self.provider = provider
        self.retry_after_s = retry_after_s


@dataclass
class BreakerConfig:
    #: Consecutive capacity-class failures before opening.
    failure_threshold: int = 5
    #: ...or this share of a rolling window, whichever trips first. Consecutive
    #: counting alone misses the case where every other request fails.
    failure_rate_threshold: float = 0.5
    window_size: int = 20
    #: How long to stay open before testing again.
    cooldown_seconds: float = 30.0
    #: Successes needed in half-open before closing. More than one, because a
    #: single success on a recovering provider is easy to come by and closing on
    #: it puts the full load straight back onto something still fragile.
    success_threshold: int = 3


@dataclass
class CircuitBreaker:
    provider: str
    config: BreakerConfig = field(default_factory=BreakerConfig)

    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    half_open_successes: int = 0
    opened_at: float = 0.0
    #: Rolling outcomes, True = capacity-class failure.
    _window: deque[bool] = field(default_factory=lambda: deque(maxlen=20))
    #: Counted but never acted on, so the split is visible in the metrics.
    ignored_failures: int = 0
    opens: int = 0

    def __post_init__(self) -> None:
        self._window = deque(maxlen=self.config.window_size)

    @property
    def failure_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def _now(self) -> float:
        return time.monotonic()

    def allows(self) -> bool:
        """May a request be dispatched to this provider?"""
        if self.state is BreakerState.CLOSED:
            return True
        if self.state is BreakerState.OPEN:
            if self._now() - self.opened_at >= self.config.cooldown_seconds:
                self.state = BreakerState.HALF_OPEN
                self.half_open_successes = 0
                return True
            return False
        # Half-open: allow probes through.
        return True

    def retry_after(self) -> float:
        if self.state is not BreakerState.OPEN:
            return 0.0
        return max(0.0, self.config.cooldown_seconds - (self._now() - self.opened_at))

    def record_success(self) -> None:
        self._window.append(False)
        self.consecutive_failures = 0
        if self.state is BreakerState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.config.success_threshold:
                self.state = BreakerState.CLOSED
                self._window.clear()

    def record_failure(self, kind: ErrorKind) -> None:
        """Only capacity-class failures move the breaker."""
        if not kind.counts_toward_circuit_breaker:
            # A 429 is quota, not capacity. Counting it would open every circuit
            # during a quota burst and stop traffic the gateway could serve.
            self.ignored_failures += 1
            return

        self._window.append(True)
        self.consecutive_failures += 1

        if self.state is BreakerState.HALF_OPEN:
            # The probe failed. Straight back to open, with a fresh cooldown.
            self._open()
            return

        tripped_consecutive = self.consecutive_failures >= self.config.failure_threshold
        tripped_rate = (
            len(self._window) >= self.config.window_size
            and self.failure_rate >= self.config.failure_rate_threshold
        )
        if tripped_consecutive or tripped_rate:
            self._open()

    def _open(self) -> None:
        self.state = BreakerState.OPEN
        self.opened_at = self._now()
        self.half_open_successes = 0
        self.opens += 1

    def as_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "state": str(self.state),
            "consecutive_failures": self.consecutive_failures,
            "failure_rate": self.failure_rate,
            "retry_after_s": self.retry_after(),
            "opens": self.opens,
            "ignored_failures": self.ignored_failures,
        }
