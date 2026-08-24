"""The difficulty classifier.

**Why logistic regression and not a fine-tuned model.** This is a small tabular
binary problem — a few hundred rows, a few dozen features, predicting "will the
cheap configuration produce an acceptable answer?". Logistic regression trains in
under a second, needs no GPU, has readable coefficients, and produces calibrated
probabilities. That last property is not a nicety: the routing rule in
`economics.py` compares a predicted probability against a cost ratio, so a model
that outputs confident-but-uncalibrated scores would route by a number that does
not mean what the arithmetic assumes it means.

Fine-tuning a language model to make this decision would cost orders of magnitude
more, take longer, be unexplainable, and be worse — because the signal lives in
request shape and a handful of embedding directions, not in language modelling.
Reaching for the simplest sufficient tool is the whole point.

The model reports its own operating curve. Accuracy is not the metric: the two
errors cost different amounts, and `economics.py` already says how much.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .features import RequestFeatures


class RouterUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingRow:
    """One labelled outcome: did the cheap configuration succeed here?"""

    example_id: str
    features: RequestFeatures
    cheap_succeeded: bool
    cheap_model: str = ""
    expensive_model: str = ""


@dataclass
class OperatingPoint:
    threshold: float
    routed_down: int
    correct_downgrades: int
    bad_downgrades: int
    routed_up: int
    missed_savings: int

    @property
    def downgrade_rate(self) -> float:
        total = self.routed_down + self.routed_up
        return self.routed_down / total if total else 0.0

    @property
    def precision(self) -> float:
        """Of the requests routed down, how many the cheap tier actually handled."""
        return self.correct_downgrades / self.routed_down if self.routed_down else 1.0

    @property
    def recall(self) -> float:
        """Of the requests the cheap tier could have handled, how many were sent there."""
        available = self.correct_downgrades + self.missed_savings
        return self.correct_downgrades / available if available else 0.0

    def as_json(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "downgrade_rate": self.downgrade_rate,
            "precision": self.precision,
            "recall": self.recall,
            "routed_down": self.routed_down,
            "bad_downgrades": self.bad_downgrades,
            "missed_savings": self.missed_savings,
        }


@dataclass
class RouterReport:
    n_train: int
    n_test: int
    auc: float
    brier: float
    curve: list[OperatingPoint] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    base_rate: float = 0.0

    @property
    def beats_base_rate(self) -> bool:
        """Is the model better than always routing the same way?

        A router with AUC ~0.5 is a coin flip wearing a coefficient vector, and
        the correct action is to delete it and pick one tier.
        """
        return self.auc > 0.60

    def as_json(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "auc": self.auc,
            "brier": self.brier,
            "base_rate": self.base_rate,
            "beats_base_rate": self.beats_base_rate,
            "curve": [p.as_json() for p in self.curve],
            "top_coefficients": dict(
                sorted(self.coefficients.items(), key=lambda kv: -abs(kv[1]))[:12]
            ),
        }

    def render(self) -> str:
        lines = [
            f"router: trained on {self.n_train}, tested on {self.n_test} held-out",
            f"  AUC {self.auc:.4f}   Brier {self.brier:.4f}   base rate {self.base_rate:.1%}",
            "",
            "  threshold  down_rate  precision  recall  bad_downgrades",
        ]
        for point in self.curve:
            lines.append(
                f"  {point.threshold:9.2f}  {point.downgrade_rate:9.3f}  "
                f"{point.precision:9.3f}  {point.recall:6.3f}  "
                f"{point.bad_downgrades:14d}"
            )
        if self.coefficients:
            lines += ["", "  strongest signals (positive = cheap tier likely to cope):"]
            top = sorted(self.coefficients.items(), key=lambda kv: -abs(kv[1]))[:8]
            for name, weight in top:
                lines.append(f"    {name:<26} {weight:+.4f}")
        if not self.beats_base_rate:
            lines += [
                "",
                "  WARNING: AUC below 0.60. This router is close to a coin flip; "
                "picking one tier outright would be simpler and no worse.",
            ]
        return "\n".join(lines)


class DifficultyRouter:
    """Predicts P(cheap configuration succeeds)."""

    def __init__(self, *, project_dimensions: int = 8, seed: int = 0) -> None:
        self.project_dimensions = project_dimensions
        self.seed = seed
        self._pipeline: Any = None
        self.feature_names = RequestFeatures.names(project_dimensions=project_dimensions)

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    def _build(self) -> Any:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RouterUnavailableError(
                "scikit-learn is not installed. Install the extra:\n"
                '    uv pip install -e ".[routing]"'
            ) from exc

        return Pipeline(
            [
                # Features span log-token-counts and 0/1 flags; without scaling
                # the regularizer would penalise them wildly unevenly.
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        # Strong L2. The training set is small and the embedding
                        # projection is correlated, so the prior is deliberately
                        # toward underfitting.
                        C=0.5,
                        max_iter=2000,
                        random_state=self.seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        )

    def _matrix(self, rows: Sequence[TrainingRow]) -> list[list[float]]:
        return [r.features.vector(project_dimensions=self.project_dimensions) for r in rows]

    def fit(self, rows: Sequence[TrainingRow]) -> None:
        if len({r.cheap_succeeded for r in rows}) < 2:
            raise ValueError(
                "training data has only one outcome class; the router cannot "
                "learn a boundary from examples that all succeeded or all failed"
            )
        self._pipeline = self._build()
        self._pipeline.fit(self._matrix(rows), [r.cheap_succeeded for r in rows])

    def predict_proba(self, features: RequestFeatures) -> float:
        if self._pipeline is None:
            raise RouterUnavailableError("router has not been trained")
        vector = features.vector(project_dimensions=self.project_dimensions)
        return float(self._pipeline.predict_proba([vector])[0][1])

    def evaluate(
        self, rows: Sequence[TrainingRow], *, thresholds: Sequence[float] | None = None
    ) -> tuple[float, float, list[OperatingPoint]]:
        from sklearn.metrics import brier_score_loss, roc_auc_score

        probabilities = [self.predict_proba(r.features) for r in rows]
        truth = [r.cheap_succeeded for r in rows]

        auc = float(roc_auc_score(truth, probabilities)) if len(set(truth)) > 1 else 0.5
        # Brier score, because the routing rule compares a probability against a
        # cost ratio. A model that ranks well but is badly calibrated would route
        # by a number that does not mean what the arithmetic assumes.
        brier = float(brier_score_loss(truth, probabilities))

        curve = []
        for threshold in thresholds or [round(0.1 * i, 1) for i in range(1, 10)]:
            routed_down = correct = bad = up = missed = 0
            for probability, succeeded in zip(probabilities, truth, strict=True):
                if probability > threshold:
                    routed_down += 1
                    correct += succeeded
                    bad += not succeeded
                else:
                    up += 1
                    missed += succeeded
            curve.append(OperatingPoint(threshold, routed_down, correct, bad, up, missed))
        return auc, brier, curve

    def coefficients(self) -> dict[str, float]:
        """Readable weights — the payoff for choosing a linear model."""
        if self._pipeline is None:
            return {}
        weights = self._pipeline.named_steps["model"].coef_[0]
        return dict(zip(self.feature_names, (float(w) for w in weights), strict=False))

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self._pipeline,
                "project_dimensions": self.project_dimensions,
                "feature_names": self.feature_names,
            },
            path,
        )
        # A sidecar the dashboard and a human can read without unpickling.
        path.with_suffix(".json").write_text(
            json.dumps(self.coefficients(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> DifficultyRouter:
        import joblib

        payload = joblib.load(Path(path))
        router = cls(project_dimensions=payload["project_dimensions"])
        router._pipeline = payload["pipeline"]
        router.feature_names = payload["feature_names"]
        return router
