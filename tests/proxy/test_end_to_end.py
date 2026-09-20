"""A scripted 5-turn agent, with every lever on, must not be able to tell.

Two exclusions, both by design rather than by convenience. memo is excluded
because a replay is the one case where the proxy answers without asking. And
the byte-identity turns carry *distinct* observations: dedup rewrites repeated
tool results on purpose, so identity there would only be testable against a
model that ignores the rewrite. The repeated-observation conversation is used
for the lever-reporting assertions instead.
"""

from __future__ import annotations

from pathlib import Path

import aiohttp

from .harness import post, run, sse_chunks, stack, stable, trace_records

ALL_LEVERS = "trace,prefix,dedup"

BIG_RESULT = "OBSERVATION:\n" + ("tests/test_store.py::test_add_def PASSED\n" * 40)


def conversation(repeat: bool = True) -> list[list[dict[str, str]]]:
    """Five turns: system, user, then tool results (the agent loop)."""
    history = [
        {"role": "system", "content": "You are a coding agent. " + "Follow the rules. " * 20},
        {"role": "user", "content": "Make the failing test pass."},
    ]
    turns = [list(history)]
    for step in range(4):
        result = BIG_RESULT if repeat else BIG_RESULT + f"ran at step {step}\n"
        history = history + [
            {"role": "assistant", "content": f"Running the tests (attempt {step})."},
            {"role": "user", "content": result},
        ]
        turns.append(list(history))
    return turns


def payload(messages: list[dict[str, str]], **extra) -> dict:
    return {
        "model": "qwen2.5-coder:7b",
        "temperature": 0,
        "messages": messages,
        **extra,
    }


def test_five_turns_identical_non_streaming(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable=ALL_LEVERS) as st:
            direct, through = [], []
            for messages in conversation(repeat=False):
                _, a, _ = await post(
                    st.upstream_url, payload(messages), headers={"X-Calm-Session": "direct"}
                )
                _, b, _ = await post(
                    st.proxy_url, payload(messages), headers={"X-Calm-Session": "proxied"}
                )
                direct.append(stable(a))
                through.append(stable(b))
            return direct, through

    direct, through = run(go())
    for a, b in zip(direct, through):
        a.pop("id", None)
        b.pop("id", None)
        assert a["choices"] == b["choices"]


def test_five_turns_identical_streaming(tmp_path: Path):
    async def stream(base: str, messages: list[dict[str, str]], session: str) -> list:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                base + "/v1/chat/completions",
                json=payload(messages, stream=True),
                headers={"X-Calm-Session": session},
            ) as resp:
                assert resp.headers["Content-Type"].startswith("text/event-stream")
                return sse_chunks(await resp.read())

    async def go():
        async with stack(trace_dir=tmp_path, enable=ALL_LEVERS) as st:
            pairs = []
            for messages in conversation(repeat=False):
                a = await stream(st.upstream_url, messages, "direct")
                b = await stream(st.proxy_url, messages, "proxied")
                pairs.append((a, b))
            return pairs

    for a, b in run(go()):
        for chunk in (*a, *b):
            if isinstance(chunk, dict):
                chunk.pop("id", None)
        assert a == b, "the client must not be able to tell the proxy is there"


def test_levers_all_report_in_the_trace(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable=ALL_LEVERS + ",memo") as st:
            for messages in conversation():
                await post(st.proxy_url, payload(messages), headers={"X-Calm-Session": "s"})

    run(go())
    records = trace_records(tmp_path)
    assert [r["seq"] for r in records] == [0, 1, 2, 3, 4]
    assert records[0]["prefix"]["divergence"] == "no_previous_request"
    assert all(r["prefix"]["divergence"] == "append_only" for r in records[1:])
    # turn n has n-1 repeats of the same observation before the newest one
    assert [r["dedup"]["replaced"] for r in records] == [0, 0, 1, 2, 3]
    assert records[-1]["dedup"]["bytes_saved"] > 3 * 1000
    assert all(r["memo"]["key"] for r in records)
    assert [r["memo"]["hit"] for r in records] == [False] * 5
