"""Prefix-cache breakpoint placement — cost policy and trust boundary."""

from __future__ import annotations

from prism.caching.breakpoints import MAX_BREAKPOINTS, CachePolicy, Scope, apply, audit

LONG = "word " * 2000  # comfortably over any minimum


def body(**kwargs):
    base = {
        "model": "claude-opus-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    base.update(kwargs)
    return base


def markers(payload) -> list[str]:
    """Where cache_control ended up, as level names."""
    found = []
    for tool in payload.get("tools") or []:
        if "cache_control" in tool:
            found.append("tools")
    system = payload.get("system")
    if isinstance(system, list):
        found += ["system" for b in system if "cache_control" in b]
    for i, message in enumerate(payload.get("messages") or []):
        content = message.get("content")
        if isinstance(content, list):
            found += [f"messages[{i}]" for b in content if "cache_control" in b]
    return found


# --- the security boundary ------------------------------------------------


def test_a_tenant_scoped_system_prompt_never_gets_a_breakpoint():
    # Prefix entries are isolated per organization, not per tenant. Under a
    # shared API key, caching a prefix built from tenant A's data makes it
    # reachable by tenant B.
    payload, report = apply(
        body(system=[{"type": "text", "text": LONG}]),
        CachePolicy(),
        system_scope=Scope.TENANT,
    )
    assert markers(payload) == []
    assert any("cross-tenant leak" in s for s in report.skipped)


def test_a_shared_system_prompt_does_get_one():
    payload, report = apply(
        body(system=[{"type": "text", "text": LONG}]),
        CachePolicy(),
        system_scope=Scope.SHARED,
    )
    assert markers(payload) == ["system"]
    assert report.placements[0].level == "system"


def test_the_default_scope_is_the_safe_one():
    # A system prompt assembled at runtime routinely carries tenant context, so
    # the caller must opt in to SHARED rather than out of it.
    payload, _ = apply(body(system=[{"type": "text", "text": LONG}]), CachePolicy())
    assert markers(payload) == []


def test_tenant_scoped_tools_are_refused_too():
    tools = [{"name": "f", "input_schema": {"type": "object"}, "description": LONG}]
    payload, report = apply(body(tools=tools), CachePolicy(), tools_scope=Scope.TENANT)
    assert markers(payload) == []
    assert any("tenant-scoped" in s for s in report.skipped)


def test_conversation_history_is_tenant_scoped_by_default():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": LONG}]},
        {"role": "assistant", "content": [{"type": "text", "text": LONG}]},
        {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
    ]
    payload, report = apply(body(messages=messages), CachePolicy(cache_conversation_prefix=True))
    assert markers(payload) == []
    assert any("tenant-scoped" in s for s in report.skipped)


# --- a bad breakpoint is worse than none ----------------------------------


def test_a_prefix_below_the_minimum_is_not_marked():
    # Below the floor the marker is ignored entirely, so placing one spends a
    # scarce slot to buy nothing.
    payload, report = apply(
        body(system=[{"type": "text", "text": "short"}]),
        CachePolicy(),
        system_scope=Scope.SHARED,
    )
    assert markers(payload) == []
    assert any("below the" in s for s in report.skipped)


def test_haiku_has_a_higher_floor_than_opus():
    # ~1500 tokens: cacheable on Opus, not on Haiku.
    text = "word " * 1200
    opus = CachePolicy.for_tier("opus")
    haiku = CachePolicy.for_tier("haiku")
    assert haiku.min_cacheable_tokens > opus.min_cacheable_tokens

    on_opus, _ = apply(
        body(system=[{"type": "text", "text": text}]), opus, system_scope=Scope.SHARED
    )
    on_haiku, _ = apply(
        body(system=[{"type": "text", "text": text}]), haiku, system_scope=Scope.SHARED
    )
    assert markers(on_opus) == ["system"]
    assert markers(on_haiku) == []


def test_the_final_message_is_never_marked():
    # It differs on every request: a guaranteed write and a guaranteed miss.
    messages = [
        {"role": "user", "content": [{"type": "text", "text": LONG}]},
        {"role": "assistant", "content": [{"type": "text", "text": LONG}]},
        {"role": "user", "content": [{"type": "text", "text": "and now?"}]},
    ]
    payload, _ = apply(
        body(messages=messages),
        CachePolicy(cache_conversation_prefix=True),
        conversation_prefix_scope=Scope.SHARED,
    )
    placed = markers(payload)
    assert placed == ["messages[1]"]
    assert "messages[2]" not in placed


def test_conversation_caching_is_off_by_default():
    # History only pays back on multi-turn traffic; on single-turn it is pure
    # write cost for reads that never arrive.
    messages = [
        {"role": "user", "content": [{"type": "text", "text": LONG}]},
        {"role": "assistant", "content": [{"type": "text", "text": LONG}]},
        {"role": "user", "content": [{"type": "text", "text": "next"}]},
    ]
    payload, _ = apply(
        body(messages=messages), CachePolicy(), conversation_prefix_scope=Scope.SHARED
    )
    assert markers(payload) == []


# --- spending the four slots ----------------------------------------------


def test_tools_do_not_get_their_own_slot_when_system_subsumes_them():
    # A marker on system already caches the tools before it. Two markers for one
    # prefix wastes a slot the API only gives you four of.
    tools = [{"name": "f", "description": LONG, "input_schema": {"type": "object"}}]
    payload, report = apply(
        body(tools=tools, system=[{"type": "text", "text": LONG}]),
        CachePolicy(),
        tools_scope=Scope.SHARED,
        system_scope=Scope.SHARED,
    )
    assert markers(payload) == ["system"]
    assert any("covered by the system breakpoint" in s for s in report.skipped)


def test_tools_get_their_own_slot_when_nothing_downstream_is_cacheable():
    tools = [{"name": "f", "description": LONG, "input_schema": {"type": "object"}}]
    payload, _ = apply(body(tools=tools), CachePolicy(), tools_scope=Scope.SHARED)
    assert markers(payload) == ["tools"]


def test_placement_never_exceeds_the_api_limit():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": LONG}]},
        {"role": "assistant", "content": [{"type": "text", "text": LONG}]},
        {"role": "user", "content": [{"type": "text", "text": "x"}]},
    ]
    payload, _ = apply(
        body(
            tools=[{"name": "f", "description": LONG, "input_schema": {}}],
            system=[{"type": "text", "text": LONG}],
            messages=messages,
        ),
        CachePolicy(cache_conversation_prefix=True),
        tools_scope=Scope.SHARED,
        system_scope=Scope.SHARED,
        conversation_prefix_scope=Scope.SHARED,
    )
    assert len(markers(payload)) <= MAX_BREAKPOINTS


# --- policy knobs and the audit -------------------------------------------


def test_disabling_the_policy_leaves_the_body_untouched():
    original = body(system=[{"type": "text", "text": LONG}])
    payload, report = apply(original, CachePolicy(enabled=False), system_scope=Scope.SHARED)
    assert payload is original
    assert markers(payload) == []
    assert report.skipped == ["caching disabled"]


def test_a_one_hour_ttl_is_declared_on_the_marker():
    # 1h writes cost 2x base instead of 1.25x, so this is never implicit.
    payload, _ = apply(
        body(system=[{"type": "text", "text": LONG}]),
        CachePolicy(ttl="1h"),
        system_scope=Scope.SHARED,
    )
    assert payload["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_the_default_ttl_marker_omits_the_field():
    payload, _ = apply(
        body(system=[{"type": "text", "text": LONG}]),
        CachePolicy(),
        system_scope=Scope.SHARED,
    )
    assert payload["system"][-1]["cache_control"] == {"type": "ephemeral"}


def test_a_string_system_prompt_is_promoted_so_it_can_carry_a_marker():
    payload, _ = apply(body(system=LONG), CachePolicy(), system_scope=Scope.SHARED)
    assert isinstance(payload["system"], list)
    assert "cache_control" in payload["system"][-1]


def test_the_original_body_is_not_mutated():
    original = body(system=[{"type": "text", "text": LONG}])
    apply(original, CachePolicy(), system_scope=Scope.SHARED)
    assert "cache_control" not in original["system"][0]


def test_the_audit_catches_a_marker_on_the_final_message():
    bad = body(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "x",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
    )
    problems = audit(bad)
    assert any("final message" in p for p in problems)


def test_the_audit_catches_too_many_markers():
    marker = {"type": "ephemeral"}
    bad = body(
        system=[{"type": "text", "text": f"b{i}", "cache_control": marker} for i in range(5)]
    )
    assert any("accepts 4" in p for p in audit(bad))


def test_a_clean_body_audits_clean():
    payload, _ = apply(
        body(system=[{"type": "text", "text": LONG}]),
        CachePolicy(),
        system_scope=Scope.SHARED,
    )
    assert audit(payload) == []


def test_the_report_serializes_for_the_trace():
    _, report = apply(
        body(system=[{"type": "text", "text": LONG}]),
        CachePolicy(),
        system_scope=Scope.SHARED,
    )
    payload = report.as_json()
    assert payload["breakpoints"][0]["level"] == "system"
    assert payload["estimated_cacheable_tokens"] > 0
