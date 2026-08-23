"""The prompt registry and what the CI gate reads from it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism.prompts import Prompt, PromptError, Registry, Version

REPO_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    (tmp_path / "assistant").mkdir()
    (tmp_path / "assistant" / "v1.md").write_text("first", encoding="utf-8")
    (tmp_path / "assistant" / "v2.md").write_text("second", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"assistant": {"active": "v1"}}), encoding="utf-8"
    )
    return Registry(tmp_path)


def test_versions_sort_numerically_not_lexically():
    # Sorted as strings, v10 comes before v2 and the gate silently compares the
    # wrong pair — but only once a prompt reaches its tenth revision.
    versions = sorted(Version.parse(v) for v in ("v2", "v10", "v1"))
    assert [str(v) for v in versions] == ["v1", "v2", "v10"]


def test_latest_picks_v10_over_v9(tmp_path: Path):
    directory = tmp_path / "p"
    directory.mkdir()
    for n in (1, 2, 9, 10):
        (directory / f"v{n}.md").write_text(f"body {n}", encoding="utf-8")
    assert str(Registry(tmp_path).latest("p")) == "v10"


def test_an_invalid_version_string_is_rejected():
    with pytest.raises(PromptError, match="invalid version"):
        Version.parse("2.0")


def test_the_content_hash_covers_the_exact_bytes():
    # It is a cache-key input. Normalising whitespace here would let a change
    # that invalidates the provider's prefix cache pass through looking
    # identical, which is the exact failure it exists to catch.
    a = Prompt("p", Version(1), "text")
    b = Prompt("p", Version(1), "text\n")
    assert a.content_hash != b.content_hash
    assert Prompt("p", Version(2), "text").content_hash == a.content_hash


def test_resolving_a_ref(registry: Registry):
    assert registry.resolve("assistant@v2").text == "second"
    # A bare name means the active version, not the newest one.
    assert registry.resolve("assistant").text == "first"


def test_a_missing_active_entry_is_an_error_not_a_guess(tmp_path: Path):
    # Defaulting to "newest" would make the gate compare a version against
    # itself and pass every time.
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "v1.md").write_text("x", encoding="utf-8")
    with pytest.raises(PromptError, match="no active version"):
        Registry(tmp_path).active("p")


def test_pending_is_what_the_gate_compares(registry: Registry):
    candidate, baseline = registry.pending("assistant")
    assert candidate.ref == "assistant@v2"
    assert baseline.ref == "assistant@v1"


def test_nothing_is_pending_once_the_newest_version_is_active(registry: Registry):
    # A gate that invented a comparison here would burn a full eval run on every
    # unrelated commit.
    registry.set_active("assistant", "v2")
    assert registry.pending("assistant") is None
    assert registry.all_pending() == {}


def test_promotion_and_rollback_are_the_same_operation(registry: Registry):
    registry.set_active("assistant", "v2")
    assert str(registry.active("assistant")) == "v2"
    registry.set_active("assistant", "v1")
    assert str(registry.active("assistant")) == "v1"


def test_promoting_a_version_that_does_not_exist_changes_nothing(registry: Registry):
    with pytest.raises(PromptError, match="not found"):
        registry.set_active("assistant", "v9")
    # The manifest must not have been half-written.
    assert str(registry.active("assistant")) == "v1"


def test_non_version_files_are_ignored_rather_than_fatal(tmp_path: Path):
    directory = tmp_path / "p"
    directory.mkdir()
    (directory / "v1.md").write_text("x", encoding="utf-8")
    (directory / "NOTES.md").write_text("scratch", encoding="utf-8")
    assert [str(v) for v in Registry(tmp_path).versions("p")] == ["v1"]


# --- the registry actually shipped in this repo ---------------------------


def test_the_repo_registry_is_well_formed():
    registry = Registry(REPO_PROMPTS)
    assert registry.names()
    for name in registry.names():
        active = registry.active(name)
        assert active in registry.versions(name)
        assert registry.get(name, active).text.strip()


def test_every_shipped_version_has_distinct_content():
    # Two versions with identical bytes mean a version bump that pays the
    # prefix-cache write cost and buys nothing.
    registry = Registry(REPO_PROMPTS)
    for name in registry.names():
        hashes = {registry.get(name, v).content_hash: str(v) for v in registry.versions(name)}
        assert len(hashes) == len(registry.versions(name)), (
            f"{name} has versions with identical content: {hashes}"
        )
