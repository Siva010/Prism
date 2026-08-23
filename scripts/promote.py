"""Promote a prompt version to active — and roll one back.

    python scripts/promote.py assistant v2      # ship it
    python scripts/promote.py assistant v1      # roll it back

Both directions are the same operation, which is the point: rollback is not a
separate emergency procedure, it is the ordinary one run backwards.

Promoting changes a cache-key input, so the next request against every affected
prefix is a cache write rather than a read. That cost is predictable and worth
stating out loud rather than discovering in a bill.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from prism.prompts import PromptError, Registry  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        print(__doc__)
        return 2

    registry = Registry("prompts")
    name = argv[0]

    if len(argv) == 1:
        try:
            active = registry.active(name)
        except PromptError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        for version in registry.versions(name):
            marker = "  <- active" if version == active else ""
            prompt = registry.get(name, version)
            print(f"  {version}  {prompt.content_hash}{marker}")
        return 0

    try:
        before = registry.active(name)
    except PromptError:
        before = None

    try:
        registry.set_active(name, argv[1])
    except PromptError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    after = registry.active(name)
    direction = "rollback" if before is not None and after < before else "promotion"
    print(f"{name}: {before} -> {after}  ({direction})")
    print(
        "Prompt version is a cache-key input: expect the next request against "
        "each affected prefix to be a cache write rather than a read."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
