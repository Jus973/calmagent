"""The agent must not be able to tell the proxy is there."""

from __future__ import annotations

from pathlib import Path

from .harness import post, run, sse_chunks, stable, stack

CHAT = {
    "model": "fake-model",
    "messages": [
        {"role": "system", "content": "you are a coding agent"},
        {"role": "user", "content": "make the tests pass"},
    ],
    "temperature": 0,
}

# Distinct upstream sessions, so the fake's cached_tokens (longest common prefix
# with the *previous* request in the same session) is comparable.
A = {"X-Calm-Session": "direct"}
B = {"X-Calm-Session": "via-proxy"}


def test_non_stream_passthrough_is_identical(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with stack(tmp_path) as s:
            direct_status, direct, _ = await post(s.upstream_url, CHAT, A)
            proxy_status, through, headers = await post(s.proxy_url, CHAT, B)
            assert direct_status == proxy_status == 200
            assert stable(direct) == stable(through)
            assert headers["Content-Type"].startswith("application/json")

    run(scenario())


def test_non_stream_tool_calls_pass_through(tmp_path: Path) -> None:
    payload = dict(CHAT)
    payload["tools"] = [{"type": "function", "function": {"name": "bash"}}]
    payload["messages"] = CHAT["messages"] + [{"role": "user", "content": "USE_TOOL now"}]

    async def scenario() -> None:
        async with stack(tmp_path) as s:
            _, direct, _ = await post(s.upstream_url, payload, A)
            _, through, _ = await post(s.proxy_url, payload, B)
            assert stable(direct) == stable(through)
            message = stable(through)["choices"][0]["message"]
            assert message["tool_calls"][0]["function"]["name"] == "bash"

    run(scenario())


def test_stream_passthrough_is_identical(tmp_path: Path) -> None:
    payload = dict(CHAT, stream=True)

    async def scenario() -> None:
        async with stack(tmp_path) as s:
            _, direct, direct_headers = await post(s.upstream_url, payload, A)
            _, through, headers = await post(s.proxy_url, payload, B)
            assert headers["Content-Type"] == direct_headers["Content-Type"]
            assert sse_chunks(direct) == sse_chunks(through)
            assert through.endswith(b"data: [DONE]\n\n")

    run(scenario())


def test_stream_usage_chunk_kept_when_client_asks(tmp_path: Path) -> None:
    payload = dict(CHAT, stream=True, stream_options={"include_usage": True})

    async def scenario() -> None:
        async with stack(tmp_path) as s:
            _, through, _ = await post(s.proxy_url, payload)
            chunks = [c for c in sse_chunks(through) if c != "[DONE]"]
            assert any(c.get("usage") for c in chunks)

    run(scenario())


def test_stream_usage_chunk_stripped_when_client_did_not_ask(tmp_path: Path) -> None:
    payload = dict(CHAT, stream=True)

    async def scenario() -> None:
        async with stack(tmp_path) as s:
            _, through, _ = await post(s.proxy_url, payload)
            chunks = [c for c in sse_chunks(through) if c != "[DONE]"]
            assert not any(c.get("usage") for c in chunks)
            # ... but the proxy asked upstream for it, so the trace has usage.
            assert s.upstream.requests[-1]["stream_options"]["include_usage"] is True

    run(scenario())


def test_legacy_completions_endpoint(tmp_path: Path) -> None:
    payload = {"model": "fake-model", "prompt": "def add(a, b):", "temperature": 0}

    async def scenario() -> None:
        async with stack(tmp_path) as s:
            _, direct, _ = await post(s.upstream_url, payload, A, path="/v1/completions")
            _, through, _ = await post(s.proxy_url, payload, B, path="/v1/completions")
            assert stable(direct) == stable(through)

    run(scenario())


def test_unknown_path_is_forwarded(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with stack(tmp_path) as s:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(s.proxy_url + "/api/tags") as resp:
                    assert resp.status == 200
                    assert (await resp.json())["models"][0]["name"] == "fake-model"

    run(scenario())
