"""DV-2: the request shapes real agents send, the errors real servers return,
and the one upstream surface that reports a cache signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import aiohttp
import pytest

from calm_proxy import stats as stats_cmd
from calm_proxy.middleware import memo as memomw

from .harness import post, run, sse_chunks, stack, stable, trace_records

MODEL = "qwen2.5-coder:7b"


def chat(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}


# -- request shapes ------------------------------------------------------


SHAPES = {
    "plain": chat(),
    "stop": chat(stop=["\n\n", "OBSERVATION:"]),
    "n": chat(n=1),
    "response_format": chat(response_format={"type": "json_object"}),
    "tool_choice": chat(
        tools=[
            {
                "type": "function",
                "function": {"name": "bash", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
    ),
    "seed_and_top_p": chat(seed=7, top_p=0.95, frequency_penalty=0.1),
    "content_parts": {
        "model": MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    },
    "tool_result_history": {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "run the tests"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command": "pytest"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "1 failed"},
        ],
    },
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_agent_shape_is_handled(name: str, tmp_path: Path):
    payload = SHAPES[name]

    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,prefix,dedup,memo") as st:
            direct = await post(st.upstream_url, payload, headers={"X-Calm-Session": "d"})
            through = await post(st.proxy_url, payload, headers={"X-Calm-Session": "p"})
            return direct, through

    (status_a, a, _), (status_b, b, _) = run(go())
    assert status_a == status_b == 200
    assert stable(a)["choices"] == stable(b)["choices"]
    records = trace_records(tmp_path)
    assert len(records) == 1
    assert records[0]["messages"], "every shape must produce message records"


def test_tool_call_response_survives_streaming(tmp_path: Path):
    payload = chat(
        stream=True,
        tools=[{"type": "function", "function": {"name": "bash"}}],
        messages=[{"role": "user", "content": "USE_TOOL please"}],
    )

    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,prefix,dedup,memo") as st:
            async with aiohttp.ClientSession() as http:
                async with http.post(
                    st.proxy_url + "/v1/chat/completions", json=payload
                ) as resp:
                    return sse_chunks(await resp.read())

    chunks = run(go())
    tool_deltas = [
        c
        for c in chunks
        if isinstance(c, dict)
        and any((choice.get("delta") or {}).get("tool_calls") for choice in c.get("choices", []))
    ]
    assert tool_deltas, "tool_calls must reach the client"


# -- errors --------------------------------------------------------------


def test_upstream_5xx_reaches_the_client_and_is_traced(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,memo") as st:
            return await post(
                st.proxy_url, chat(temperature=0), headers={"X-Fake-Status": "503"}
            )

    status, body, headers = run(go())
    assert status == 503
    assert json.loads(body)["error"]["message"] == "fake upstream failure"
    record = trace_records(tmp_path)[0]
    assert record["upstream_status"] == 503
    assert record["error"] == "fake upstream failure"
    assert record["memo"]["hit"] is False
    assert not list((tmp_path / "memo").glob("*.json")), "errors are never memoized"


def test_upstream_timeout_becomes_504_and_is_traced(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,memo", timeout_s=0.2) as st:
            return await post(st.proxy_url, chat(temperature=0), headers={"X-Fake-Delay": "2"})

    status, body, _ = run(go())
    assert status == 504
    assert "upstream" in json.loads(body)["error"]["message"]
    record = trace_records(tmp_path)[0]
    assert record["error"] and "Timeout" in record["error"]
    assert not list((tmp_path / "memo").glob("*.json"))


def test_a_traced_error_does_not_poison_the_next_request(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,memo") as st:
            await post(st.proxy_url, chat(temperature=0), headers={"X-Fake-Status": "500"})
            return await post(st.proxy_url, chat(temperature=0))

    status, _, headers = run(go())
    assert status == 200
    assert memomw.MEMO_HEADER not in headers


# -- Ollama's native surface (REQ-LC-1) ----------------------------------


def native(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}


def test_native_api_chat_is_traced_with_upstream_timings(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,prefix") as st:
            status, body, _ = await post(
                st.proxy_url, native(stream=False), path="/api/chat"
            )
            return status, json.loads(body)

    status, body = run(go())
    assert status == 200
    assert body["done"] is True
    record = trace_records(tmp_path)[0]
    assert record["api"] == "ollama"
    assert record["upstream"]["prompt_eval_ms"] is not None
    assert record["upstream"]["eval_ms"] == 2.0
    assert record["upstream"]["load_ms"] == 1.0
    assert record["usage"]["prompt_tokens"] and record["usage"]["cached_tokens"] is None


def test_native_stream_is_forwarded_verbatim(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,prefix") as st:
            async with aiohttp.ClientSession() as http:
                async with http.post(
                    st.proxy_url + "/api/chat",
                    json=native(),
                    headers={"X-Calm-Session": "p"},
                ) as via:
                    proxied = await via.read()
                async with http.post(
                    st.upstream_url + "/api/chat",
                    json=native(),
                    headers={"X-Calm-Session": "d"},
                ) as direct:
                    plain = await direct.read()
            return proxied, plain

    proxied, plain = run(go())
    assert proxied == plain
    lines = [json.loads(line) for line in proxied.splitlines() if line.strip()]
    assert lines[-1]["done"] is True
    record = trace_records(tmp_path)[0]
    assert record["stream"] is True
    assert record["upstream"]["prompt_eval_ms"] is not None


def test_prompt_eval_falls_when_the_prefix_is_reused(tmp_path: Path):
    """The fake server bills cold prefill only for the non-shared suffix, the
    way LC measured the real one: an append-only turn is nearly free."""

    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,prefix") as st:
            history = [{"role": "system", "content": "context " * 500}]
            await post(
                st.proxy_url,
                {"model": MODEL, "messages": history, "stream": False},
                headers={"X-Calm-Session": "s"},
                path="/api/chat",
            )
            await post(
                st.proxy_url,
                {
                    "model": MODEL,
                    "messages": history + [{"role": "user", "content": "go"}],
                    "stream": False,
                },
                headers={"X-Calm-Session": "s"},
                path="/api/chat",
            )

    run(go())
    cold, warm = trace_records(tmp_path)
    assert warm["upstream"]["prompt_eval_ms"] < cold["upstream"]["prompt_eval_ms"] / 10
    assert warm["prefix"]["divergence"] == "append_only"


# -- stats ---------------------------------------------------------------


def test_stats_reports_every_lever(tmp_path: Path):
    big = "OBSERVATION:\n" + ("x" * 500)

    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,prefix,dedup,memo") as st:
            history = [{"role": "system", "content": "sys"}, {"role": "user", "content": big}]
            await post(st.proxy_url, {"model": MODEL, "temperature": 0, "messages": history})
            history = history + [
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": big},
            ]
            await post(st.proxy_url, {"model": MODEL, "temperature": 0, "messages": history})
            await post(st.proxy_url, {"model": MODEL, "temperature": 0, "messages": history})
            await post(st.proxy_url, native(stream=False), path="/api/chat")

    run(go())
    text = stats_cmd.report(trace_records(tmp_path))
    assert "requests 4" in text
    assert "1/3 eligible requests replayed" in text
    assert "2 tool results referenced" in text
    assert "prompt_eval" in text
    assert "prefix gap" in text


def test_stats_is_honest_about_a_v1_only_trace(tmp_path: Path):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,prefix") as st:
            await post(st.proxy_url, chat())

    run(go())
    text = stats_cmd.report(trace_records(tmp_path))
    assert "n/a on the /v1 surface" in text


def test_stats_cli(tmp_path: Path, capsys):
    async def go():
        async with stack(trace_dir=tmp_path, enable="trace,prefix") as st:
            await post(st.proxy_url, chat())

    run(go())
    from calm_proxy import cli

    assert cli.main(["stats", str(tmp_path)]) == 0
    assert "requests 1" in capsys.readouterr().out
