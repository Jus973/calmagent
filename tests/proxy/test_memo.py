"""I-3 memo: replay stored responses for byte-identical deterministic requests."""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp

from calm_proxy.middleware import memo as memomw

from .harness import post, run, stack, stable, sse_chunks, trace_records


def body(text: str = "hello", **extra):
    return {
        "model": "qwen2.5-coder:7b",
        "temperature": 0,
        "messages": [{"role": "user", "content": text}],
        **extra,
    }


def test_eligibility():
    assert memomw.eligible({"temperature": 0})
    assert memomw.eligible({"seed": 7, "temperature": 0.8})
    assert not memomw.eligible({"temperature": 0.7})
    assert not memomw.eligible({})


def test_key_is_content_addressed():
    a = memomw.memo_key(body("x"))
    assert a == memomw.memo_key(body("x"))
    assert a != memomw.memo_key(body("y"))
    assert a != memomw.memo_key(body("x", seed=1))


def test_hit_replays_identical_response(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,memo") as st:
            _, first, _ = await post(st.proxy_url, body())
            _, second, headers = await post(st.proxy_url, body())
            return stable(first), stable(second), headers

    first, second, headers = run(go())
    assert first == second
    assert headers.get(memomw.MEMO_HEADER) == "hit"
    records = trace_records(tmp_path)
    assert records[0]["memo"]["hit"] is False
    assert records[1]["memo"]["hit"] is True
    assert records[1]["memo"]["key"] == records[0]["memo"]["key"]
    assert records[1]["usage"]["completion_tokens"] == records[0]["usage"]["completion_tokens"]


def test_miss_records_key_without_hit(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,memo") as st:
            await post(st.proxy_url, body("one"))
            await post(st.proxy_url, body("two"))

    run(go())
    records = trace_records(tmp_path)
    assert [r["memo"]["hit"] for r in records] == [False, False]
    assert records[0]["memo"]["key"] != records[1]["memo"]["key"]


def test_nondeterministic_requests_are_not_memoized(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,memo") as st:
            payload = body()
            payload["temperature"] = 0.9
            await post(st.proxy_url, payload)
            await post(st.proxy_url, payload)

    run(go())
    records = trace_records(tmp_path)
    assert [r["memo"] for r in records] == [
        {"hit": False, "key": None},
        {"hit": False, "key": None},
    ]
    assert not (tmp_path / "memo").exists() or not list((tmp_path / "memo").glob("*.json"))


def test_disabled_by_default(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace") as st:
            await post(st.proxy_url, body())
            await post(st.proxy_url, body())

    run(go())
    records = trace_records(tmp_path)
    assert all(r["memo"] == {"hit": False, "key": None} for r in records)
    assert not (tmp_path / "memo").exists()


def test_errors_are_not_stored(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,memo") as st:
            status, _, _ = await post(st.proxy_url, body(), headers={"X-Fake-Status": "500"})
            status2, _, headers = await post(st.proxy_url, body())
            return status, status2, headers

    status, status2, headers = run(go())
    assert status == 500
    assert status2 == 200
    assert memomw.MEMO_HEADER not in headers


async def _stream(base: str, payload: dict) -> tuple[bytes, dict[str, str]]:
    async with aiohttp.ClientSession() as session:
        async with session.post(base + "/v1/chat/completions", json=payload) as resp:
            return await resp.read(), dict(resp.headers)


def test_stream_hit_replays_the_same_text(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,memo") as st:
            first, _ = await _stream(st.proxy_url, body(stream=True))
            second, headers = await _stream(st.proxy_url, body(stream=True))
            return first, second, headers

    first, second, headers = run(go())
    assert headers.get(memomw.MEMO_HEADER) == "hit"

    def text(raw: bytes) -> str:
        out = []
        for chunk in sse_chunks(raw):
            if chunk == "[DONE]":
                continue
            for choice in chunk.get("choices", []):
                out.append(choice.get("delta", {}).get("content") or "")
        return "".join(out)

    assert text(first) == text(second)
    assert sse_chunks(second)[-1] == "[DONE]"
    assert not any(
        isinstance(c, dict) and c.get("usage") for c in sse_chunks(second)
    ), "client did not ask for usage, so the replay must not include a usage chunk"

    records = trace_records(tmp_path)
    assert records[1]["memo"]["hit"] is True
    assert records[1]["usage"]["completion_tokens"] == records[0]["usage"]["completion_tokens"]


def test_stream_hit_includes_usage_when_client_asked(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,memo") as st:
            payload = body(stream=True, stream_options={"include_usage": True})
            await _stream(st.proxy_url, payload)
            second, _ = await _stream(st.proxy_url, payload)
            return second

    chunks = sse_chunks(run(go()))
    assert any(isinstance(c, dict) and c.get("usage") for c in chunks)


def test_store_is_grow_only(tmp_path: Path):
    store = memomw.MemoStore(tmp_path)
    store.put("k" * 8, {"id": "a"}, "m")
    store.put("k" * 8, {"id": "b"}, "m")
    assert store.get("k" * 8)["response"] == {"id": "a"}
    index = (tmp_path / "memo" / "index.jsonl").read_text().splitlines()
    assert len(index) == 1
    assert json.loads(index[0])["key"] == "k" * 8
