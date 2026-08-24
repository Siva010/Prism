"""What the router looks at.

Two groups, kept separable on purpose.

**Metadata features** are hand-built, interpretable, and cheap: length, tool
complexity, conversation depth, whether structured output was demanded. With a
linear model their coefficients are readable, so "why did this get routed up?"
has an answer that fits in a sentence. That is most of the argument for choosing
logistic regression here.

**Embedding features** carry what the metadata cannot — that "prove this identity
holds for all n" is harder than "what is the capital of France", despite similar
length and identical structure. They come from the same local model the semantic
cache already loads, so they cost nothing extra on a request that was going to be
embedded anyway.

The embedding is 384-dimensional and a router's training set is a few hundred
rows, so using it raw invites memorisation. `project_dimensions` reduces it to a
handful of components before it reaches the model. That is a deliberate bias
toward underfitting: a router that memorises its training set produces a quality
retention figure that is a lie, and the eval harness would have to catch it after
the fact rather than the design preventing it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from ..tokens import estimate_request

# Cheap lexical signals for task shape. Not a classifier — just features whose
# weights the model is free to learn as zero.
_REASONING_MARKERS = (
    "prove",
    "derive",
    "explain why",
    "step by step",
    "analyse",
    "analyze",
    "compare",
    "trade-off",
    "tradeoff",
    "design",
    "architect",
    "debug",
    "optimise",
    "optimize",
    "refactor",
    "implement",
)
_LOOKUP_MARKERS = (
    "what is",
    "who is",
    "when did",
    "where is",
    "how many",
    "list the",
    "name the",
    "define",
    "translate",
    "convert",
)

FEATURE_NAMES = [
    "log_input_tokens",
    "log_output_budget",
    "n_messages",
    "n_tools",
    "log_tool_schema_chars",
    "tool_required_fields",
    "has_output_schema",
    "query_chars",
    "query_words",
    "avg_word_length",
    "n_question_marks",
    "n_code_fences",
    "n_digits_ratio",
    "reasoning_markers",
    "lookup_markers",
    "has_system_prompt",
]


@dataclass
class RequestFeatures:
    metadata: list[float]
    embedding: list[float] = field(default_factory=list)

    def vector(self, *, project_dimensions: int = 0) -> list[float]:
        if not self.embedding or project_dimensions <= 0:
            return list(self.metadata)
        return [*self.metadata, *_project(self.embedding, project_dimensions)]

    @staticmethod
    def names(*, project_dimensions: int = 0) -> list[str]:
        if project_dimensions <= 0:
            return list(FEATURE_NAMES)
        return [*FEATURE_NAMES, *(f"emb_{i}" for i in range(project_dimensions))]


def _project(embedding: list[float], dimensions: int) -> list[float]:
    """Average-pool the embedding into `dimensions` buckets.

    A fixed, deterministic reduction rather than a learned one (PCA would need
    fitting, and a fitted reduction is one more thing that can leak the test set
    into training). Crude, and enough: the model needs a coarse sense of *where*
    in embedding space a request sits, not a faithful reconstruction.
    """
    if dimensions >= len(embedding):
        return list(embedding)
    size = math.ceil(len(embedding) / dimensions)
    out = []
    for i in range(dimensions):
        chunk = embedding[i * size : (i + 1) * size]
        out.append(sum(chunk) / len(chunk) if chunk else 0.0)
    return out


def _query_text(body: dict[str, Any]) -> str:
    for message in reversed(body.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
    return ""


def extract(body: dict[str, Any], *, embedding: list[float] | None = None) -> RequestFeatures:
    """Build features from an Anthropic-shaped request body."""
    query = _query_text(body)
    lowered = query.lower()
    words = query.split()
    tools = body.get("tools") or []
    tool_json = json.dumps([t.get("input_schema") or {} for t in tools])
    required = sum(len((t.get("input_schema") or {}).get("required") or []) for t in tools)
    digits = sum(c.isdigit() for c in query)

    metadata = [
        math.log1p(estimate_request(body)),
        math.log1p(int(body.get("max_tokens") or 0)),
        float(len(body.get("messages") or [])),
        float(len(tools)),
        math.log1p(len(tool_json)),
        float(required),
        1.0 if (body.get("output_config") or {}).get("format") else 0.0,
        math.log1p(len(query)),
        math.log1p(len(words)),
        (sum(len(w) for w in words) / len(words)) if words else 0.0,
        float(query.count("?")),
        float(query.count("```")),
        (digits / len(query)) if query else 0.0,
        float(sum(marker in lowered for marker in _REASONING_MARKERS)),
        float(
            sum(lowered.startswith(marker) or f" {marker}" in lowered for marker in _LOOKUP_MARKERS)
        ),
        1.0 if body.get("system") else 0.0,
    ]
    return RequestFeatures(metadata=metadata, embedding=list(embedding or []))
