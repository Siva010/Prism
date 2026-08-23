"""Programmatic metrics.

Every example that can be scored without a model *should* be. A judge costs
money, adds latency, and introduces a second source of error that then has to be
calibrated; exact match costs nothing and is not wrong. The judge exists for the
open-ended residual, not as the default.

Each scorer returns a float in [0, 1] so metrics compose into one per-example
score and feed the same bootstrap.
"""

from __future__ import annotations

import json
import re
import string
from collections.abc import Sequence
from typing import Any

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip articles and punctuation, collapse whitespace.

    The SQuAD convention. Without it, exact match measures formatting habits
    rather than correctness — "The answer is Paris." and "paris" would score 0.
    """
    lowered = text.lower()
    lowered = _ARTICLES.sub(" ", lowered)
    lowered = lowered.translate(_PUNCT)
    return _WHITESPACE.sub(" ", lowered).strip()


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize(prediction) == normalize(reference) else 0.0


def any_exact_match(prediction: str, references: Sequence[str]) -> float:
    """Several phrasings can be right. Credit the best one, not the first."""
    return max((exact_match(prediction, r) for r in references), default=0.0)


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 — partial credit where exact match is too brittle."""
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        # Both empty is a match; one empty is not.
        return float(pred_tokens == ref_tokens)

    common: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > common.get(token, 0):
            common[token] = common.get(token, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def any_token_f1(prediction: str, references: Sequence[str]) -> float:
    return max((token_f1(prediction, r) for r in references), default=0.0)


def contains(prediction: str, reference: str) -> float:
    return 1.0 if normalize(reference) in normalize(prediction) else 0.0


def json_valid(prediction: str) -> float:
    try:
        json.loads(prediction)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    return 1.0


def schema_conformance(prediction: str, schema: dict[str, Any]) -> float:
    """Does the output satisfy the JSON Schema it was asked for?

    Deliberately limited to the constraints this harness checks itself —
    ``type``, ``required``, ``properties``, ``enum``, ``additionalProperties``.
    That is roughly the surface the provider's constrained decoding also covers,
    which is the point: week 8 reports *schema-surface coverage*, the fraction of
    a schema's constraints that are grammar-enforced versus post-validated, and
    that split only means something if both sides are enumerated rather than
    delegated to a library.
    """
    try:
        value = json.loads(prediction)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    return 1.0 if _conforms(value, schema) else 0.0


def _conforms(value: Any, schema: dict[str, Any]) -> bool:
    expected = schema.get("type")
    if expected and not _type_matches(value, expected):
        return False

    if "enum" in schema and value not in schema["enum"]:
        return False

    if expected == "object" or isinstance(value, dict):
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                return False
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return False
        for name, sub_schema in properties.items():
            if name in value and not _conforms(value[name], sub_schema):
                return False

    if expected == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return all(_conforms(item, item_schema) for item in value)

    return True


def _type_matches(value: Any, expected: str | list[str]) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(value, e) for e in expected)
    checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        # bool is a subclass of int in Python; JSON Schema does not agree.
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    check = checks.get(expected)
    return check(value) if check else True


SCORERS = {
    "exact_match": lambda pred, ex: any_exact_match(pred, ex.references),
    "token_f1": lambda pred, ex: any_token_f1(pred, ex.references),
    "contains": lambda pred, ex: max((contains(pred, r) for r in ex.references), default=0.0),
    "json_valid": lambda pred, ex: json_valid(pred),
    "schema_conformance": lambda pred, ex: schema_conformance(pred, ex.schema or {}),
}
