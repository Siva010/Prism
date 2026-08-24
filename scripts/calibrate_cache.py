"""Calibrate the semantic cache threshold against labelled query pairs.

    python scripts/calibrate_cache.py --pairs datasets/cache_pairs.jsonl

Prints the full ROC table and the chosen operating point. Exits non-zero when no
threshold keeps the false-hit rate under the ceiling — which is a real answer,
not a failure: it means the cache should stay off for this corpus.

    --max-false-hit-rate  the ceiling, applied to the interval's UPPER bound
    --embedder            local (real) | hashing (stub, for plumbing checks only)
    --point-estimate      apply the ceiling to the observed rate instead of the
                          upper bound. Reports a more flattering threshold and a
                          less defensible one.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from prism.caching.calibration import calibrate, load_pairs  # noqa: E402
from prism.caching.embeddings import (  # noqa: E402
    EmbeddingUnavailableError,
    build_embedder,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="datasets/cache_pairs.jsonl")
    parser.add_argument("--max-false-hit-rate", type=float, default=0.01)
    parser.add_argument("--embedder", default="local", choices=["local", "hashing"])
    parser.add_argument("--point-estimate", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    pairs = load_pairs(args.pairs)
    if len(pairs) < 40:
        print(
            f"WARNING: only {len(pairs)} labelled pairs. Below ~200 the false-hit "
            "interval is wider than the ceiling being enforced, so the chosen "
            "threshold is not yet defensible.",
            file=sys.stderr,
        )

    try:
        embedder = build_embedder(args.embedder)
    except EmbeddingUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not embedder.is_production_grade:
        print(
            "WARNING: the hashing stub is not semantically meaningful. Its curve "
            "exercises the plumbing and says nothing about retrieval quality.",
            file=sys.stderr,
        )

    result = calibrate(pairs, embedder, max_false_hit_rate=args.max_false_hit_rate)
    if args.point_estimate and result.chosen is None:
        from prism.caching.calibration import choose_operating_point

        result.chosen = choose_operating_point(
            result.curve,
            max_false_hit_rate=args.max_false_hit_rate,
            use_upper_bound=False,
        )

    print(json.dumps(result.as_json(), indent=2) if args.json else result.render())

    if result.chosen is None:
        return 1
    print(
        f"\nTo use it:\n    PRISM_SEMANTIC_CACHE_ENABLED=true\n"
        f"    PRISM_SEMANTIC_CACHE_THRESHOLD={result.chosen.threshold}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
