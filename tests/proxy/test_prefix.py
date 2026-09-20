"""I-2 `prefix`: classification is a pure function; the lever never mutates."""

from __future__ import annotations

from pathlib import Path

from calm_proxy import lint
from calm_proxy.middleware import prefix

from .harness import post, run, stack, trace_records

SYSTEM = {"role": "system", "content": "you are a coding agent with a long prompt " * 20}
USER = {"role": "user", "content": "make the tests pass"}
TOOL_A = {"role": "tool", "tool_call_id": "a", "content": "A" * 400}
TOOL_B = {"role": "tool", "tool_call_id": "b", "content": "B" * 400}


def analyze(previous: list[dict], current: list[dict]) -> dict:
    return prefix.analyze(prefix.views(previous), prefix.views(current))


def test_append_only_is_the_good_case() -> None:
    result = analyze([SYSTEM, USER], [SYSTEM, USER, TOOL_A])
    assert result["divergence"] == prefix.APPEND_ONLY
    assert result["common_messages"] == 2
    assert result["common_bytes"] == sum(
        len(m["content"].encode()) for m in (SYSTEM, USER)
    )
    assert result["achievable_cached_tokens_est"] == result["common_bytes"] // 4


def test_first_request_has_no_previous() -> None:
    result = prefix.analyze(None, prefix.views([SYSTEM, USER]))
    assert result == {
        "common_messages": 0,
        "common_bytes": 0,
        "divergence": prefix.NO_PREVIOUS,
        "achievable_cached_tokens_est": 0,
    }


def test_system_prompt_churn() -> None:
    churned = {"role": "system", "content": SYSTEM["content"] + " (v2)"}
    result = analyze([SYSTEM, USER], [churned, USER])
    assert result["divergence"] == prefix.SYSTEM_CHANGED
    assert result["common_messages"] == 0
    # ... but most of the bytes of the divergent message still match.
    assert result["common_bytes"] == len(SYSTEM["content"].encode())


def test_history_truncation_is_compaction() -> None:
    long = [SYSTEM, USER, TOOL_A, TOOL_B]
    compacted = [SYSTEM, TOOL_B]
    assert analyze(long, compacted)["divergence"] == prefix.HISTORY_TRUNCATED


def test_tool_result_reordering() -> None:
    result = analyze([SYSTEM, USER, TOOL_A, TOOL_B], [SYSTEM, USER, TOOL_B, TOOL_A])
    assert result["divergence"] == prefix.TOOL_RESULT_REORDERED
    assert result["common_messages"] == 2


def test_timestamp_like_divergence() -> None:
    old = {"role": "user", "content": "now is 2026-09-20T02:45 and the tree is clean"}
    new = {"role": "user", "content": "now is 2026-09-20T03:11 and the tree is clean"}
    assert analyze([SYSTEM, old], [SYSTEM, new])["divergence"] == prefix.TIMESTAMP_LIKE


def test_real_content_change_is_not_blamed_on_timestamps() -> None:
    old = {"role": "user", "content": "implement parse_arguments"}
    new = {"role": "user", "content": "implement render_template"}
    assert analyze([SYSTEM, old], [SYSTEM, new])["divergence"] == prefix.CONTENT_CHANGED


def test_prefix_is_recorded_and_request_is_untouched(tmp_path: Path) -> None:
    turn1 = {"model": "fake-model", "messages": [SYSTEM, USER], "temperature": 0}
    turn2 = {"model": "fake-model", "messages": [SYSTEM, USER, TOOL_A], "temperature": 0}

    async def scenario() -> None:
        async with stack(tmp_path, enable="trace,prefix") as s:
            await post(s.proxy_url, turn1, {"X-Calm-Session": "s1"})
            await post(s.proxy_url, turn2, {"X-Calm-Session": "s1"})
            assert s.upstream.requests[0]["messages"] == turn1["messages"]
            assert s.upstream.requests[1]["messages"] == turn2["messages"]

    run(scenario())
    records = trace_records(tmp_path)
    assert records[0]["prefix"]["divergence"] == prefix.NO_PREVIOUS
    assert records[1]["prefix"]["divergence"] == prefix.APPEND_ONLY
    assert records[1]["prefix"]["common_messages"] == 2
    assert records[1]["prefix"]["achievable_cached_tokens_est"] > 0


def test_prefix_state_is_per_session(tmp_path: Path) -> None:
    payload = {"model": "fake-model", "messages": [SYSTEM, USER]}

    async def scenario() -> None:
        async with stack(tmp_path, enable="trace,prefix") as s:
            await post(s.proxy_url, payload, {"X-Calm-Session": "s1"})
            await post(s.proxy_url, payload, {"X-Calm-Session": "s2"})

    run(scenario())
    assert [r["prefix"]["divergence"] for r in trace_records(tmp_path)] == [
        prefix.NO_PREVIOUS,
        prefix.NO_PREVIOUS,
    ]


def test_prefix_absent_when_lever_is_off(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with stack(tmp_path, enable="trace") as s:
            await post(s.proxy_url, {"model": "fake-model", "messages": [SYSTEM, USER]})

    run(scenario())
    assert "prefix" not in trace_records(tmp_path)[0]


def test_lint_table(tmp_path: Path) -> None:
    turn1 = {"model": "fake-model", "messages": [SYSTEM, USER]}
    turn2 = {"model": "fake-model", "messages": [SYSTEM, USER, TOOL_A]}

    async def scenario() -> None:
        async with stack(tmp_path, enable="trace,prefix") as s:
            await post(s.proxy_url, turn1, {"X-Calm-Session": "s1"})
            await post(s.proxy_url, turn2, {"X-Calm-Session": "s1"})

    run(scenario())
    text = lint.report(lint.load_records(tmp_path))
    assert "requests: 2   sessions: 1" in text
    assert "append_only" in text
    assert "prompt_tok" in text
