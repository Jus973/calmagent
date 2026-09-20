"""In-process proxy + fake upstream, for tests."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import aiohttp
from aiohttp import web

from calm_proxy import fake_upstream, server
from calm_proxy.config import Config, parse_enabled


@dataclass
class Stack:
    proxy_url: str
    upstream_url: str
    trace_dir: Path
    upstream: fake_upstream.FakeUpstream
    proxy: server.Proxy


async def _serve(app: web.Application) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, port


@asynccontextmanager
async def stack(
    trace_dir: Path | None = None,
    enable: str = "trace",
    tok_ms: int = 0,
    reply: str | None = None,
    **config_kwargs: Any,
) -> AsyncIterator[Stack]:
    fake_app = fake_upstream.create_app(tok_ms=tok_ms, reply=reply)
    fake_runner, fake_port = await _serve(fake_app)
    upstream_url = f"http://127.0.0.1:{fake_port}"
    config = Config(
        upstream=upstream_url,
        trace_dir=trace_dir,
        enabled=parse_enabled(enable),
        **config_kwargs,
    )
    proxy_app = server.create_app(config)
    proxy_runner, proxy_port = await _serve(proxy_app)
    try:
        yield Stack(
            proxy_url=f"http://127.0.0.1:{proxy_port}",
            upstream_url=upstream_url,
            trace_dir=trace_dir or Path("."),
            upstream=fake_app["upstream"],
            proxy=proxy_app["proxy"],
        )
    finally:
        await proxy_runner.cleanup()
        await fake_runner.cleanup()


async def post(
    base: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    path: str = "/v1/chat/completions",
) -> tuple[int, bytes, dict[str, str]]:
    async with aiohttp.ClientSession() as session:
        async with session.post(base + path, json=payload, headers=headers or {}) as resp:
            body = await resp.read()
            return resp.status, body, dict(resp.headers)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def trace_records(trace_dir: Path) -> list[dict[str, Any]]:
    path = trace_dir / "trace.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sse_chunks(raw: bytes) -> list[Any]:
    """Decoded SSE payloads with the volatile ``created`` field removed."""
    out: list[Any] = []
    for block in raw.split(b"\n\n"):
        line = block.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[len(b"data:") :].strip()
        if data == b"[DONE]":
            out.append("[DONE]")
            continue
        chunk = json.loads(data)
        chunk.pop("created", None)
        out.append(chunk)
    return out


def stable(body: bytes) -> Any:
    parsed = json.loads(body)
    parsed.pop("created", None)
    return parsed
