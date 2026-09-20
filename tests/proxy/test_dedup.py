"""I-4 `dedup`: deterministic, prefix-stable, and never silent."""

from __future__ import annotations

from pathlib import Path

from calm_proxy.hashing import sha256_hex
from calm_proxy.middleware import dedup

from .harness import post, run, stack, trace_records

BIG = "E" * 500
SYSTEM = {"role": "system", "content": "you are a coding agent"}
RESULT = {"role": "tool", "tool_call_id": "a", "content": BIG}
RESULT_AGAIN = {"role": "tool", "tool_call_id": "b", "content": BIG}
OTHER = {"role": "tool", "tool_call_id": "c", "content": "F" * 500}


def test_repeat_is_replaced_by_a_stable_reference() -> None:
    out, info = dedup.apply([SYSTEM, RESULT, OTHER, RESULT_AGAIN])
    assert info["replaced"] == 1
    assert out[0] is SYSTEM and out[1] is RESULT
    assert out[3]["content"] == f"[identical to result #1 (sha256:{sha256_hex(BIG)[:12]})]"
    assert out[3]["tool_call_id"] == "b"
    assert info["bytes_saved"] == len(BIG) - len(out[3]["content"])


def test_replacement_is_a_function_of_the_history_only() -> None:
    history = [SYSTEM, RESULT, OTHER, RESULT_AGAIN]
    first, _ = dedup.apply(history)
    second, _ = dedup.apply(list(history))
    assert [m.get("content") for m in first] == [m.get("content") for m in second]
    # A later turn keeps the same replacement for the same earlier messages,
    # so the KV prefix stays aligned.
    longer, _ = dedup.apply(history + [{"role": "user", "content": "next"}])
    assert [m.get("content") for m in longer[: len(first)]] == [
        m.get("content") for m in first
    ]


def test_small_results_are_left_alone() -> None:
    small = {"role": "tool", "content": "ok"}
    out, info = dedup.apply([small, dict(small)], min_bytes=200)
    assert info == {"replaced": 0, "bytes_saved": 0}
    assert [m["content"] for m in out] == ["ok", "ok"]


def test_non_tool_messages_are_never_touched() -> None:
    user = {"role": "user", "content": BIG}
    out, info = dedup.apply([user, dict(user)])
    assert info["replaced"] == 0
    assert [m["content"] for m in out] == [BIG, BIG]


def test_user_shaped_tool_results_are_detected() -> None:
    observation = {"role": "user", "content": "OBSERVATION:\n" + BIG}
    out, info = dedup.apply([observation, dict(observation)])
    assert info["replaced"] == 1
    assert out[1]["content"].startswith("[identical to result #1")


def test_dedup_is_recorded_in_the_trace_and_sent_upstream(tmp_path: Path) -> None:
    payload = {
        "model": "fake-model",
        "messages": [SYSTEM, RESULT, OTHER, RESULT_AGAIN],
        "temperature": 0,
    }

    async def scenario() -> None:
        async with stack(tmp_path, enable="trace,dedup") as s:
            status, _, _ = await post(s.proxy_url, payload)
            assert status == 200
            sent = s.upstream.requests[-1]["messages"]
            assert sent[3]["content"].startswith("[identical to result #1")
            assert sent[1]["content"] == BIG

    run(scenario())
    record = trace_records(tmp_path)[0]
    assert record["dedup"]["replaced"] == 1
    assert record["dedup"]["bytes_saved"] > 400
    # The trace's message records describe what the *server* saw.
    assert record["messages"][3]["bytes"] < 100


def test_dedup_off_sends_the_original(tmp_path: Path) -> None:
    payload = {"model": "fake-model", "messages": [SYSTEM, RESULT, RESULT_AGAIN]}

    async def scenario() -> None:
        async with stack(tmp_path, enable="trace") as s:
            await post(s.proxy_url, payload)
            assert s.upstream.requests[-1]["messages"] == payload["messages"]

    run(scenario())
    assert trace_records(tmp_path)[0]["dedup"] == {"replaced": 0, "bytes_saved": 0}


def test_min_bytes_flag_is_honoured(tmp_path: Path) -> None:
    payload = {"model": "fake-model", "messages": [SYSTEM, RESULT, RESULT_AGAIN]}

    async def scenario() -> None:
        async with stack(tmp_path, enable="trace,dedup", dedup_min_bytes=10_000) as s:
            await post(s.proxy_url, payload)
            assert s.upstream.requests[-1]["messages"] == payload["messages"]

    run(scenario())
    assert trace_records(tmp_path)[0]["dedup"]["replaced"] == 0
