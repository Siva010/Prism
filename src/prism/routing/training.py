"""Turning eval outcomes into router training data.

The router's labels come from the evaluation harness: run the same examples on
the cheap tier and the expensive one, score both, and the label is "did the cheap
one produce an acceptable answer?".

**This is where contamination gets in.** The router trains on eval outcomes, so
if it is then *scored* on those same examples, its reported quality retention is
memorisation. `dataset.assert_disjoint` guards the split, and `build` refuses to
mix them — the check runs before any fitting rather than as a review step
somebody might skip.

The second trap is subtler: the label depends on a scoring threshold. "The cheap
tier succeeded" for a programmatic example means it scored 1.0 on its strictest
metric; for a judged example it means the judge did not prefer the expensive
answer. Both are choices, and moving either one changes every label. So the
threshold used is recorded on the dataset rather than left implicit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..eval.dataset import Dataset, DatasetError, Example, assert_disjoint
from ..eval.judge import PairwiseResult, Verdict
from ..eval.runner import ArmScores
from .features import RequestFeatures, extract


@dataclass(frozen=True)
class LabelledOutcome:
    example_id: str
    cheap_succeeded: bool
    cheap_score: float
    expensive_score: float
    source: str  # programmatic | judge

    def as_json(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "cheap_succeeded": self.cheap_succeeded,
            "cheap_score": self.cheap_score,
            "expensive_score": self.expensive_score,
            "source": self.source,
        }


def label_from_scores(
    examples: Sequence[Example],
    cheap: ArmScores,
    expensive: ArmScores,
    *,
    judge_results: Sequence[PairwiseResult] = (),
    success_threshold: float = 1.0,
) -> list[LabelledOutcome]:
    """Label each example by whether the cheap arm was good enough.

    `success_threshold` is explicit and recorded. At 1.0 a cheap answer must be
    exactly right; at 0.8 partial credit counts. Lowering it makes the router
    look better and the product worse, which is precisely why it is a parameter
    rather than a constant buried in a comparison.
    """
    judged = {r.example_id: r for r in judge_results if r.is_usable}
    out: list[LabelledOutcome] = []

    for index, example in enumerate(examples):
        verdict = judged.get(example.id)
        if example.is_judged and verdict is not None:
            # Pairwise: the cheap arm succeeded unless the judge preferred the
            # expensive one outright. A tie counts as success, because paying
            # more for an answer nobody can distinguish is the waste the router
            # exists to remove.
            succeeded = verdict.verdict is not Verdict.B
            out.append(
                LabelledOutcome(
                    example.id,
                    succeeded,
                    1.0 if succeeded else 0.0,
                    1.0,
                    "judge",
                )
            )
            continue

        cheap_score = _score_at(cheap, index)
        expensive_score = _score_at(expensive, index)
        if cheap_score is None:
            continue
        out.append(
            LabelledOutcome(
                example.id,
                cheap_score >= success_threshold,
                cheap_score,
                expensive_score if expensive_score is not None else 0.0,
                "programmatic",
            )
        )
    return out


def _score_at(scores: ArmScores, index: int) -> float | None:
    """The strictest available metric for one example."""
    for metric in ("exact_match", "schema_conformance", "contains", "json_valid"):
        values = scores.per_metric.get(metric)
        if values and index < len(values):
            return values[index]
    values = scores.per_metric.get("token_f1")
    if values and index < len(values):
        return values[index]
    return None


def build(
    dataset: Dataset,
    outcomes: Sequence[LabelledOutcome],
    *,
    embedder: Any = None,
    holdout: Dataset | None = None,
) -> list[Any]:
    """Assemble training rows, refusing anything that would contaminate the split."""
    if holdout is not None:
        # Runs before fitting, not as a review step somebody might skip.
        assert_disjoint(dataset, holdout)

    by_id = {e.id: e for e in dataset}
    labelled = [o for o in outcomes if o.example_id in by_id]
    if not labelled:
        raise DatasetError(
            "no outcomes match this split. The router must train on the "
            "router_train split, not on the split it will be scored against."
        )

    from .model import TrainingRow

    rows: list[TrainingRow] = []
    for outcome in labelled:
        example = by_id[outcome.example_id]
        body = {
            "model": "claude-opus-5",
            "max_tokens": 1024,
            "messages": example.messages,
        }
        if example.schema is not None:
            body["output_config"] = {"format": {"type": "json_schema"}}

        embedding = None
        if embedder is not None:
            from ..caching.keys import query_text

            embedding = embedder.encode([query_text(body)], is_query=True)[0]

        rows.append(
            TrainingRow(
                example_id=example.id,
                features=extract(body, embedding=embedding),
                cheap_succeeded=outcome.cheap_succeeded,
            )
        )
    return rows


def features_for_request(body: dict[str, Any], *, embedder: Any = None) -> RequestFeatures:
    """Feature extraction on the serving path, mirroring training exactly.

    Shared with `build` so a feature added in one place cannot silently go
    missing in the other — train/serve skew is the classic way a router that
    looked good offline routes badly in production.
    """
    embedding = None
    if embedder is not None:
        from ..caching.keys import query_text

        query = query_text(body)
        if query:
            embedding = embedder.encode([query], is_query=True)[0]
    return extract(body, embedding=embedding)


def save_outcomes(outcomes: Sequence[LabelledOutcome], path: str | Path) -> int:
    Path(path).write_text(
        "\n".join(json.dumps(o.as_json()) for o in outcomes) + "\n", encoding="utf-8"
    )
    return len(outcomes)


def load_outcomes(path: str | Path) -> list[LabelledOutcome]:
    out: list[LabelledOutcome] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        raw = json.loads(line)
        out.append(
            LabelledOutcome(
                example_id=raw["example_id"],
                cheap_succeeded=bool(raw["cheap_succeeded"]),
                cheap_score=float(raw.get("cheap_score", 0.0)),
                expensive_score=float(raw.get("expensive_score", 0.0)),
                source=raw.get("source", "programmatic"),
            )
        )
    return out
