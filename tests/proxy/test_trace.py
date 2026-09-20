"""The trace record contract (shared with LC's fallback proxy)."""

from __future__ import annotations

from pathlib import Path

from calm_proxy.hashing import sha256_hex

from .harness import post, run, stack, trace_records

CHAT = {
    "model": "fake-model",
    "messages": [
        {"role": "system", "content": "you are a coding agent"},
        {"role": "user", "content": "make the tests pass"},
    ],
    "temperature": 0,
    "seed": 7,
    "max_tokens": 1024,
}

REQUIRED_KEYS = {
    "ts",
    "session",
    "seq",
    "model",
    "stream",
    "params",
    "messages",
    "prompt_sha",
    "usage",
    "timing_ms",
    "memo",
    "dedup",
    "upstream_status",
    "error",
}


def test_record_schema_and_bodies(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with stack(tmp_path) as s:
            await post(s.proxy_url, CHAT)

    run(scenario())
    records = trace_records(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert REQUIRED_KEYS <= set(record)
    assert record["model"] == "fake-model"
    assert record["stream"] is False
    assert record["upstream_status"] == 200
    assert record["error"] is None
    assert record["params"] == {"temperature": 0, "seed": 7, "max_tokens": 1024}
    assert [m["role"] for m in record["messages"]] == ["system", "user"]
    assert record["messages"][1]["bytes"] == len("make the tests pass")
    assert record["usage"]["prompt_tokens"] > 0
    assert record["usage"]["completion_tokens"] > 0
    assert record["timing_ms"]["done"] is not None
    assert record["memo"] == {"hit": False, "key": None}
    assert record["dedup"] == {"replaced": 0, "bytes_saved": 0}
    for message in record["messages"]:
        assert (tmp_path / "bodies" / f"{message['sha']}.txt").exists()


def test_session_from_header_and_seq(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with stack(tmp_path) as s:
            for _ in range(3):
                await post(s.proxy_url, CHAT, headers={"X-Calm-Session": "abc"})

    run(scenario())
    records = trace_records(tmp_path)
    assert [r["seq"] for r in records] == [0, 1, 2]
    assert {r["session"] for r in records} == {"abc"}


def test_session_defaults_to_first_system_message(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with stack(tmp_path) as s:
            await post(s.proxy_url, CHAT)
            await post(
                s.proxy_url,
                {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]},
            )

    run(scenario())
    records = trace_records(tmp_path)
    assert records[0]["session"] == sha256_hex("you are a coding agent")
    assert records[1]["session"] == sha256_hex("hi")
    assert records[0]["seq"] == records[1]["seq"] == 0


def test_streamed_usage_is_captured(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with stack(tmp_path, tok_ms=1) as s:
            await post(s.proxy_url, dict(CHAT, stream=True))

    run(scenario())
    record = trace_records(tmp_path)[0]
    assert record["stream"] is True
    assert record["usage"]["prompt_tokens"] > 0
    assert record["usage"]["cached_tokens"] is not None
    assert record["timing_ms"]["first_token"] is not None
    assert record["timing_ms"]["done"] >= record["timing_ms"]["first_token"]


def test_cached_tokens_null_when_upstream_is_silent(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with stack(tmp_path) as s:
            # The fake upstream always reports cached_tokens; a 4xx body does not.
            await post(s.proxy_url, CHAT, headers={"X-Fake-Status": "500"})

    run(scenario())
    record = trace_records(tmp_path)[0]
    assert record["usage"]["cached_tokens"] is None
    assert record["upstream_status"] == 500
    assert record["error"] == "fake upstream failure"


def test_prompt_sha_is_stable_and_content_addressed(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with stack(tmp_path) as s:
            await post(s.proxy_url, CHAT)
            await post(s.proxy_url, CHAT)
            await post(
                s.proxy_url,
                dict(CHAT, messages=CHAT["messages"] + [{"role": "user", "content": "more"}]),
            )

    run(scenario())
    records = trace_records(tmp_path)
    assert records[0]["prompt_sha"] == records[1]["prompt_sha"]
    assert records[2]["prompt_sha"] != records[0]["prompt_sha"]


def test_tracing_off_writes_nothing(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with stack(tmp_path, enable="") as s:
            status, _, _ = await post(s.proxy_url, CHAT)
            assert status == 200

    run(scenario())
    assert trace_records(tmp_path) == []
