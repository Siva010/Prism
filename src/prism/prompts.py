"""Versioned prompt registry.

Prompts are files in `prompts/<name>/<version>.md`, with `prompts/manifest.json`
naming the version currently serving traffic. Three things depend on that being
explicit rather than implicit:

* **The CI gate** (week 6) compares the newest version in a branch against the
  active one. Both live in the tree at once, which is what makes the comparison a
  self-contained job rather than a git-checkout dance.
* **Cache keys** (weeks 7-8) include the prompt version, so a version bump has a
  predictable, measurable cache cost instead of a mysterious hit-rate cliff.
* **Prefix-cache breakpoints** sit on version boundaries. A prompt whose text
  changes without its version changing silently invalidates every cached prefix,
  which is why `content_hash` is checked against the version rather than trusted.

The hash is over the exact bytes served. Nothing here normalises whitespace: a
trailing newline changes the prefix the provider caches, so it must change the
hash too.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import total_ordering
from pathlib import Path

DEFAULT_ROOT = Path("prompts")
_VERSION_RE = re.compile(r"^v(\d+)$")


class PromptError(ValueError):
    pass


@total_ordering
@dataclass(frozen=True)
class Version:
    """A `vN` version, ordered numerically rather than lexically.

    Sorting these as strings would put v10 before v2, and the gate would compare
    the wrong pair — silently, and only once a prompt reached its tenth revision.
    """

    number: int

    @classmethod
    def parse(cls, raw: str) -> Version:
        match = _VERSION_RE.match(raw.strip())
        if not match:
            raise PromptError(f"invalid version {raw!r}; expected the form 'v1'")
        return cls(int(match.group(1)))

    def __str__(self) -> str:
        return f"v{self.number}"

    def __lt__(self, other: Version) -> bool:
        return self.number < other.number


@dataclass(frozen=True)
class Prompt:
    name: str
    version: Version
    text: str

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def content_hash(self) -> str:
        """SHA-256 of the exact bytes served, truncated for readability.

        A cache-key input, so it hashes the text verbatim — normalising
        whitespace here would let a change that invalidates the provider's prefix
        cache pass through looking identical.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


class Registry:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self._manifest_path = self.root / "manifest.json"

    # --- reading ---------------------------------------------------------

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )

    def versions(self, name: str) -> list[Version]:
        directory = self.root / name
        if not directory.is_dir():
            raise PromptError(f"no prompt named {name!r} under {self.root}")
        found = []
        for path in directory.glob("*.md"):
            try:
                found.append(Version.parse(path.stem))
            except PromptError:
                continue  # not a version file; ignore rather than fail the run
        if not found:
            raise PromptError(f"prompt {name!r} has no vN.md files")
        return sorted(found)

    def latest(self, name: str) -> Version:
        return self.versions(name)[-1]

    def get(self, name: str, version: str | Version | None = None) -> Prompt:
        resolved = (
            self.active(name)
            if version is None
            else (version if isinstance(version, Version) else Version.parse(version))
        )
        path = self.root / name / f"{resolved}.md"
        if not path.is_file():
            raise PromptError(f"prompt {name}@{resolved} not found at {path}")
        return Prompt(name=name, version=resolved, text=path.read_text(encoding="utf-8"))

    def resolve(self, ref: str) -> Prompt:
        """Parse `name@v2`, or bare `name` for the active version."""
        if "@" in ref:
            name, version = ref.split("@", 1)
            return self.get(name.strip(), version.strip())
        return self.get(ref.strip())

    # --- the manifest ----------------------------------------------------

    def manifest(self) -> dict[str, dict[str, str]]:
        if not self._manifest_path.is_file():
            return {}
        try:
            loaded = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PromptError(f"{self._manifest_path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise PromptError(f"{self._manifest_path}: expected a JSON object")
        return loaded

    def active(self, name: str) -> Version:
        """The version currently serving traffic — the gate's baseline."""
        entry = self.manifest().get(name)
        if entry is None or "active" not in entry:
            raise PromptError(
                f"prompt {name!r} has no active version in {self._manifest_path}; "
                "the CI gate needs a declared baseline to compare against"
            )
        return Version.parse(entry["active"])

    def set_active(self, name: str, version: str | Version) -> None:
        """Promote a version. This is the one-click rollback, both directions."""
        resolved = version if isinstance(version, Version) else Version.parse(version)
        self.get(name, resolved)  # fail before writing if it does not exist
        manifest = self.manifest()
        manifest.setdefault(name, {})["active"] = str(resolved)
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # --- what the gate needs ---------------------------------------------

    def pending(self, name: str) -> tuple[Prompt, Prompt] | None:
        """(candidate, baseline) when a newer version exists than the active one.

        Returns None when the newest version *is* the active one — there is
        nothing to gate, and a gate that invented a comparison in that case would
        burn a full eval run on every unrelated commit.
        """
        active = self.active(name)
        latest = self.latest(name)
        if latest <= active:
            return None
        return self.get(name, latest), self.get(name, active)

    def all_pending(self) -> dict[str, tuple[Prompt, Prompt]]:
        out: dict[str, tuple[Prompt, Prompt]] = {}
        for name in self.names():
            try:
                pair = self.pending(name)
            except PromptError:
                continue
            if pair is not None:
                out[name] = pair
        return out
