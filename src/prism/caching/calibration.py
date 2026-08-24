"""Choosing the similarity threshold — the intellectual core of the cache.

The semantic cache is a binary classifier over "are these two requests
equivalent?". So it has an ROC curve, and picking a threshold is picking an
operating point on it. The only question is which errors you are willing to make.

**The two errors are not comparable.** A false negative is a cache miss: you pay
for a completion you already had. A false positive is a *false hit*: a real user
receives an answer to a question they did not ask. One costs cents, the other
costs correctness — so precision is weighted far above recall, and the operating
point is chosen by fixing a maximum tolerable false-hit rate and taking the best
recall available under it, rather than by maximising F1 or accuracy.

**The reported false-hit rate needs an interval.** A labelled set of 200 pairs
that produces 0 false hits does not mean the false-hit rate is zero; it means it
is below roughly 2% with 95% confidence. `wilson_interval` supplies that, and the
upper bound is what a threshold decision should be defended with.

**Report the threshold's exposure, not its hit rate.** A high hit rate at an
uncalibrated threshold is not a result.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..eval.stats import Interval, wilson_interval
from .embeddings import Embedder, cosine


@dataclass(frozen=True)
class LabelledPair:
    """Two requests a human judged equivalent (or not) for caching purposes.

    "Equivalent" is a stronger claim than "similar": it means an answer to one is
    a correct and complete answer to the other. "What is the capital of France?"
    and "What is the capital of Germany?" are highly similar and not equivalent,
    which is exactly the pair a naive threshold gets wrong.
    """

    query_a: str
    query_b: str
    equivalent: bool
    notes: str = ""


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def hit_rate(self) -> float:
        """Share of all pairs served from cache — what a naive report shows."""
        total = (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )
        return (self.true_positives + self.false_positives) / total if total else 0.0

    @property
    def precision(self) -> float:
        served = self.true_positives + self.false_positives
        return self.true_positives / served if served else 1.0

    @property
    def recall(self) -> float:
        available = self.true_positives + self.false_negatives
        return self.true_positives / available if available else 0.0

    @property
    def false_hit_rate(self) -> float:
        """Share of *served* responses that were wrong. The number that matters."""
        served = self.true_positives + self.false_positives
        return self.false_positives / served if served else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Share of non-equivalent pairs wrongly served — the ROC x-axis."""
        negatives = self.false_positives + self.true_negatives
        return self.false_positives / negatives if negatives else 0.0

    def false_hit_interval(self, confidence: float = 0.95) -> Interval:
        served = self.true_positives + self.false_positives
        return wilson_interval(self.false_positives, served, confidence=confidence)

    def as_json(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "hit_rate": self.hit_rate,
            "precision": self.precision,
            "recall": self.recall,
            "false_hit_rate": self.false_hit_rate,
            "false_hit_upper_bound": self.false_hit_interval().high,
            "false_positive_rate": self.false_positive_rate,
            "counts": {
                "tp": self.true_positives,
                "fp": self.false_positives,
                "tn": self.true_negatives,
                "fn": self.false_negatives,
            },
        }


@dataclass
class CalibrationResult:
    curve: list[ThresholdPoint]
    chosen: ThresholdPoint | None
    max_false_hit_rate: float
    n_pairs: int
    embedder: str

    @property
    def auc(self) -> float:
        """Area under the ROC curve — how separable the classes are at all.

        Reported because it is threshold-independent: a low AUC means no operating
        point is good, and no amount of threshold tuning will fix it.
        """
        points = sorted(
            {(p.false_positive_rate, p.recall) for p in self.curve} | {(0.0, 0.0), (1.0, 1.0)}
        )
        area = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
            area += (x1 - x0) * (y0 + y1) / 2
        return area

    def as_json(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "embedder": self.embedder,
            "auc": self.auc,
            "max_false_hit_rate": self.max_false_hit_rate,
            "chosen": self.chosen.as_json() if self.chosen else None,
            "curve": [p.as_json() for p in self.curve],
        }

    def render(self) -> str:
        lines = [
            f"threshold calibration on n={self.n_pairs} labelled pairs ({self.embedder})",
            f"  AUC {self.auc:.4f}",
            "",
            "  threshold   hit_rate  precision  recall  false_hit  fh_upper",
        ]
        for point in self.curve:
            marker = " <-" if self.chosen and point.threshold == self.chosen.threshold else ""
            upper = point.false_hit_interval().high
            # Undefined rather than zero when nothing was served: an interval
            # over no observations is not evidence of safety.
            upper_text = "       -" if upper != upper else f"{upper:8.3f}"
            lines.append(
                f"  {point.threshold:9.3f}  {point.hit_rate:8.3f}  "
                f"{point.precision:9.3f}  {point.recall:6.3f}  "
                f"{point.false_hit_rate:9.3f}  {upper_text}{marker}"
            )
        lines.append("")
        if self.chosen is None:
            lines.append(
                "  NO OPERATING POINT: no threshold keeps the false-hit rate under "
                f"{self.max_false_hit_rate:.1%} while serving anything. The honest "
                "conclusion is that this cache should stay off for this corpus."
            )
        else:
            c = self.chosen
            interval = c.false_hit_interval()
            lines.append(
                f"  chosen threshold {c.threshold:.3f}: serves {c.hit_rate:.1%} of "
                f"requests, recall {c.recall:.1%}, false-hit rate "
                f"{c.false_hit_rate:.1%} (95% CI upper bound {interval.high:.1%})"
            )
        return "\n".join(lines)


def sweep(
    pairs: Sequence[LabelledPair],
    embedder: Embedder,
    *,
    thresholds: Sequence[float] | None = None,
) -> list[ThresholdPoint]:
    """Score every labelled pair once, then evaluate each candidate threshold."""
    if not pairs:
        return []

    texts_a = embedder.encode([p.query_a for p in pairs], is_query=True)
    texts_b = embedder.encode([p.query_b for p in pairs], is_query=False)
    scored = [(cosine(a, b), p.equivalent) for a, b, p in zip(texts_a, texts_b, pairs, strict=True)]

    if thresholds is None:
        thresholds = [round(0.70 + 0.01 * i, 2) for i in range(31)]

    curve: list[ThresholdPoint] = []
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for similarity, equivalent in scored:
            served = similarity >= threshold
            if served and equivalent:
                tp += 1
            elif served and not equivalent:
                fp += 1
            elif not served and equivalent:
                fn += 1
            else:
                tn += 1
        curve.append(ThresholdPoint(threshold, tp, fp, tn, fn))
    return curve


def choose_operating_point(
    curve: Sequence[ThresholdPoint],
    *,
    max_false_hit_rate: float = 0.01,
    use_upper_bound: bool = True,
    min_served: int = 5,
) -> ThresholdPoint | None:
    """Best recall subject to a false-hit ceiling.

    `use_upper_bound` defaults to True: the constraint is applied to the *upper
    bound* of the false-hit interval, not the point estimate. On a small labelled
    set the point estimate is often 0.0, and treating that as proof of safety is
    how a cache ships with an unmeasured error rate.

    `min_served` rejects thresholds so high that almost nothing is served — with
    two hits and no misses the observed rate is meaningless.

    Returns None when no threshold satisfies the constraint. That is a real
    answer: it means this cache should stay off for this corpus.
    """
    viable = []
    for point in curve:
        served = point.true_positives + point.false_positives
        if served < min_served:
            continue
        observed = point.false_hit_interval().high if use_upper_bound else point.false_hit_rate
        if observed <= max_false_hit_rate:
            viable.append(point)

    if not viable:
        return None
    # Among safe thresholds, take the most recall; break ties toward the *higher*
    # threshold, which is the more conservative of two equally-performing points.
    return max(viable, key=lambda p: (p.recall, p.threshold))


def calibrate(
    pairs: Sequence[LabelledPair],
    embedder: Embedder,
    *,
    max_false_hit_rate: float = 0.01,
    thresholds: Sequence[float] | None = None,
) -> CalibrationResult:
    curve = sweep(pairs, embedder, thresholds=thresholds)
    return CalibrationResult(
        curve=curve,
        chosen=choose_operating_point(curve, max_false_hit_rate=max_false_hit_rate),
        max_false_hit_rate=max_false_hit_rate,
        n_pairs=len(pairs),
        embedder=embedder.model_name,
    )


def load_pairs(path: str | Path) -> list[LabelledPair]:
    """Read JSONL: {"query_a": ..., "query_b": ..., "equivalent": true|false}."""
    path = Path(path)
    pairs: list[LabelledPair] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
                pairs.append(
                    LabelledPair(
                        query_a=raw["query_a"],
                        query_b=raw["query_b"],
                        equivalent=bool(raw["equivalent"]),
                        notes=raw.get("notes", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return pairs
