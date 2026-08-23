"""Statistics for small evaluation sets.

Eval sets are small — 200 examples is normal — and small samples are where
honest measurement goes wrong. Three things this module exists to prevent:

1. **Point estimates.** On 200 examples a 3% improvement is usually noise.
   Every headline number here comes with a bootstrap confidence interval, and
   the interval is what decides whether a change shipped or regressed.
2. **Unpaired comparisons.** Candidate and baseline are run on the *same*
   examples, so the comparison is paired. Bootstrapping the two arms
   independently throws away that pairing and inflates the interval, hiding
   real differences behind example-level variance.
3. **Trusting the judge.** An LLM judge is a measuring instrument, and an
   uncalibrated instrument reports whatever it likes. Cohen's kappa against
   human labels is how you find out whether it agrees with people any better
   than chance would.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def excludes_zero(self) -> bool:
        """True when the interval supports a real effect in one direction."""
        return self.low > 0.0 or self.high < 0.0

    def __str__(self) -> str:
        pct = int(self.confidence * 100)
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}] ({pct}% CI)"

    def as_json(self) -> dict[str, float]:
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "confidence": self.confidence,
        }


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for a statistic over one sample.

    The seed is fixed and explicit. A confidence interval that moves between
    runs of the same data invites re-rolling until the answer is the one you
    wanted, which is exactly the failure mode a CI gate exists to prevent.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), confidence)
    if array.size == 1:
        v = float(array[0])
        return Interval(v, v, v, confidence)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, array.size, size=(resamples, array.size))
    draws = np.apply_along_axis(statistic, 1, array[idx])

    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return Interval(float(statistic(array)), float(low), float(high), confidence)


def paired_bootstrap_ci(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """CI for the mean paired difference (candidate - baseline).

    Resamples *examples*, not arms. Both arms saw the same inputs, so an example
    that is hard for one is hard for the other; keeping the pairing removes that
    shared difficulty from the variance and is what makes a small eval set able
    to detect a small real change at all.
    """
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired comparison needs equal-length arms, got {a.shape} and {b.shape}")
    return bootstrap_ci((a - b).tolist(), resamples=resamples, confidence=confidence, seed=seed)


def regression_detected(diff: Interval, *, tolerance: float = 0.0) -> bool:
    """Gate decision: did quality drop by more than noise?

    Fails only when the *entire* interval sits below the tolerance — the whole
    plausible range is a regression, not merely the point estimate. A gate that
    fires on the point estimate alone would fail roughly half of all no-op
    changes and get switched off within a week.
    """
    return diff.high < -abs(tolerance)


def cohens_kappa(
    rater_a: Sequence[str], rater_b: Sequence[str], *, labels: Sequence[str] | None = None
) -> float:
    """Chance-corrected agreement between two raters.

    Raw agreement is misleading whenever the label distribution is skewed: two
    raters who both say "tie" 80% of the time agree 64% of the time by accident.
    Kappa subtracts that expected agreement, so it answers "better than chance?"
    rather than "how often did they match?".

    Returns 1.0 for perfect agreement, 0.0 for chance-level, negative for worse
    than chance.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same examples")
    n = len(rater_a)
    if n == 0:
        return float("nan")

    categories = sorted(set(labels) if labels is not None else set(rater_a) | set(rater_b))
    index = {label: i for i, label in enumerate(categories)}
    k = len(categories)
    if k < 2:
        # Everything got the same label. Agreement is total but uninformative,
        # and kappa is undefined rather than 1.0.
        return float("nan")

    matrix = np.zeros((k, k), dtype=float)
    for x, y in zip(rater_a, rater_b, strict=True):
        matrix[index[x], index[y]] += 1

    observed = float(np.trace(matrix)) / n
    expected = float(np.sum(matrix.sum(axis=0) * matrix.sum(axis=1)) / (n * n))
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch bands. Reported alongside the number, never instead of it."""
    if kappa != kappa:  # NaN
        return "undefined"
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


def wilson_interval(successes: int, trials: int, *, confidence: float = 0.95) -> Interval:
    """Score interval for a proportion.

    Preferred over the normal approximation for the small-n, near-0, and near-1
    cases this harness lives in — a 0/20 false-hit rate is not "0% ± 0%", and
    the semantic cache's operating point depends on knowing how wide that
    uncertainty really is.
    """
    if trials == 0:
        return Interval(float("nan"), float("nan"), float("nan"), confidence)

    from math import sqrt

    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(confidence, 1.9600)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin), confidence)
