"""The golden dataset.

One JSONL file, one example per line, versioned in git next to the code. Two
properties matter more than the format:

**Splits are a contamination boundary, not a convenience.** The week-9 router
trains on labelled outcomes produced by this harness. If it trains and is scored
on the same examples, its reported quality retention is memorisation, and the
number is worthless. ``split`` is therefore required on every example, and
``load()`` refuses a file whose ids repeat across splits.

**Scoring method is a property of the example, not of the run.** An example with
references is scored programmatically; one without goes to the judge. That
decision lives in the dataset so a run cannot quietly promote hard examples to
the judge (which is generous) or demote open-ended ones to exact match (which is
brutal) and change the headline number without changing the data.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Split = Literal["dev", "test", "router_train", "calibration"]
TaskType = Literal["qa", "extraction", "classification", "summarization", "open_ended"]


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    split: Split
    task_type: TaskType

    # Stored in OpenAI shape because that is what the gateway ingests: an eval
    # example is literally a request Prism can serve, so the harness exercises
    # the real path rather than a parallel one.
    messages: list[dict[str, Any]]

    references: list[str] = Field(default_factory=list)
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    metrics: list[str] = Field(default_factory=list)

    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def schema(self) -> dict[str, Any] | None:  # type: ignore[override]
        return self.schema_

    @property
    def is_judged(self) -> bool:
        """No references means no programmatic answer key — the judge decides."""
        return not self.references and not self.schema_

    @model_validator(mode="after")
    def _check_scoring_is_possible(self) -> Example:
        if self.metrics:
            unknown = set(self.metrics) - set(_KNOWN_METRICS)
            if unknown:
                raise ValueError(f"{self.id}: unknown metrics {sorted(unknown)}")
            if "schema_conformance" in self.metrics and self.schema_ is None:
                raise ValueError(f"{self.id}: schema_conformance requires a schema")
            needs_refs = {"exact_match", "token_f1", "contains"} & set(self.metrics)
            if needs_refs and not self.references:
                raise ValueError(f"{self.id}: {sorted(needs_refs)} require at least one reference")
        elif self.task_type != "open_ended" and self.is_judged:
            raise ValueError(
                f"{self.id}: {self.task_type} examples need references or a schema; "
                "only open_ended examples may fall through to the judge"
            )
        if not self.messages:
            raise ValueError(f"{self.id}: messages must not be empty")
        return self


_KNOWN_METRICS = (
    "exact_match",
    "token_f1",
    "contains",
    "json_valid",
    "schema_conformance",
)


class DatasetError(ValueError):
    pass


class Dataset:
    def __init__(self, examples: list[Example], *, source: Path | None = None) -> None:
        self.examples = examples
        self.source = source

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def split(self, name: Split) -> Dataset:
        return Dataset([e for e in self.examples if e.split == name], source=self.source)

    def judged(self) -> Dataset:
        return Dataset([e for e in self.examples if e.is_judged], source=self.source)

    def programmatic(self) -> Dataset:
        return Dataset([e for e in self.examples if not e.is_judged], source=self.source)

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.examples),
            "by_split": dict(Counter(e.split for e in self.examples)),
            "by_task_type": dict(Counter(e.task_type for e in self.examples)),
            "judged": sum(1 for e in self.examples if e.is_judged),
            "programmatic": sum(1 for e in self.examples if not e.is_judged),
        }


def load(path: str | Path) -> Dataset:
    """Parse and validate a JSONL golden set."""
    path = Path(path)
    examples: list[Example] = []
    seen: dict[str, Split] = {}

    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: invalid JSON — {exc}") from exc
            try:
                example = Example.model_validate(raw)
            except ValueError as exc:
                raise DatasetError(f"{path}:{lineno}: {exc}") from exc

            if example.id in seen:
                # The same prompt appearing in two splits is exactly the
                # contamination this guard exists to catch.
                raise DatasetError(
                    f"{path}:{lineno}: duplicate id {example.id!r} "
                    f"(already in split {seen[example.id]!r})"
                )
            seen[example.id] = example.split
            examples.append(example)

    if not examples:
        raise DatasetError(f"{path}: no examples")
    return Dataset(examples, source=path)


def assert_disjoint(train: Dataset, test: Dataset) -> None:
    """Guard the router's train/test boundary.

    Called before any router training run. Failing loudly here is much cheaper
    than discovering afterwards that a reported quality-retention figure was
    measured on examples the model had already been fitted to.
    """
    overlap = {e.id for e in train} & {e.id for e in test}
    if overlap:
        raise DatasetError(
            f"train and test splits share {len(overlap)} example(s): {sorted(overlap)[:5]}"
        )
