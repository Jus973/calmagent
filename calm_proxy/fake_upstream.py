"""An OpenAI-compatible fake model server, so the proxy can be developed and
tested without a GPU or a real model.

It returns canned completions, streams them token by token (``--tok-ms`` of
sleep per token), and reports ``prompt_tokens_details.cached_tokens`` as the
length of the longest common prefix with the previous request in the same
session — which is what makes the prefix lever testable.

Test hooks (all opt-in, so ordinary requests stay boring):

* header ``X-Fake-Status: 500`` → that status with an OpenAI error body
* header ``X-Fake-Delay: 2.5`` → sleep that many seconds before responding
* header ``X-Fake-Tool-Calls: 1`` or a request with ``tools`` and a last
  message containing ``USE_TOOL`` → respond with ``tool_calls``
* header ``X-Fake-Reply: …`` → use that text as the completion
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from aiohttp import web

from .hashing import message_text, sha256_hex
from .middleware.trace import session_id

CHARS_PER_TOKEN = 4


def _tokens(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def _prompt_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{m.get('role')}:{message_text(m)}" for m in messages)


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


TOOL_CALLS = [
    {
        "id": "call_fake_0",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "pytest -q"}'},
    }
]


class FakeUpstream:
    def __init__(self, tok_ms: int = 0, reply: str | None = None):
        self.tok_ms = tok_ms
        self.reply = reply
        self.last_prompt: dict[str, str] = {}
        self.requests: list[dict[str, Any]] = []

    # -- helpers -----------------------------------------------------------
    def _reply_for(self, request: web.Request, body: dict[str, Any], prompt: str) -> str:
        forced = request.headers.get("X-Fake-Reply")
        if forced:
            return forced
        if self.reply:
            return self.reply
        return f"canned reply for {sha256_hex(prompt)[:12]}"

    def _wants_tool_calls(self, request: web.Request, body: dict[str, Any]) -> bool:
        if request.headers.get("X-Fake-Tool-Calls"):
            return True
        if not body.get("tools"):
            return False
        messages = body.get("messages") or []
        return bool(messages) and "USE_TOOL" in message_text(messages[-1])

    def _usage(self, prompt: str, completion: str, session: str) -> dict[str, Any]:
        previous = self.last_prompt.get(session, "")
        cached = _tokens_of_prefix(_common_prefix_len(previous, prompt))
        prompt_tokens = _tokens(prompt)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": _tokens(completion),
            "total_tokens": prompt_tokens + _tokens(completion),
            "prompt_tokens_details": {"cached_tokens": min(cached, prompt_tokens)},
        }

    # -- routes ------------------------------------------------------------
    async def chat_completions(self, request: web.Request) -> web.StreamResponse:
        return await self._completion(request, chat=True)

    async def completions(self, request: web.Request) -> web.StreamResponse:
        return await self._completion(request, chat=False)

    async def _completion(self, request: web.Request, *, chat: bool) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        delay = request.headers.get("X-Fake-Delay")
        if delay:
            await asyncio.sleep(float(delay))
        status = request.headers.get("X-Fake-Status")
        if status:
            return web.json_response(
                {"error": {"message": "fake upstream failure", "type": "server_error"}},
                status=int(status),
            )

        messages = body.get("messages")
        if not isinstance(messages, list):
            messages = [{"role": "user", "content": body.get("prompt", "")}]
        prompt = _prompt_text(messages)
        session = session_id(request.headers.get("X-Calm-Session"), messages)
        text = self._reply_for(request, body, prompt)
        tool_calls = TOOL_CALLS if (chat and self._wants_tool_calls(request, body)) else None
        if tool_calls is not None:
            text = ""
        usage = self._usage(prompt, text, session)
        self.last_prompt[session] = prompt

        options = body.get("stream_options") or {}
        include_usage = bool(options.get("include_usage"))
        if body.get("stream"):
            return await self._stream(request, body, text, tool_calls, usage, include_usage, chat)
        return web.json_response(
            _completion_body(body, text, tool_calls, usage, chat), status=200
        )

    async def _stream(
        self,
        request: web.Request,
        body: dict[str, Any],
        text: str,
        tool_calls: list[dict[str, Any]] | None,
        usage: dict[str, Any],
        include_usage: bool,
        chat: bool,
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        )
        await response.prepare(request)
        model = body.get("model", "fake-model")
        created = int(time.time())
        stream_id = "chatcmpl-fake"

        async def send(chunk: dict[str, Any]) -> None:
            await response.write(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n")

        def frame(delta: dict[str, Any], finish: Any = None) -> dict[str, Any]:
            if chat:
                choice = {"index": 0, "delta": delta, "finish_reason": finish}
            else:
                choice = {
                    "index": 0,
                    "text": delta.get("content", ""),
                    "finish_reason": finish,
                }
            return {
                "id": stream_id,
                "object": "chat.completion.chunk" if chat else "text_completion",
                "created": created,
                "model": model,
                "choices": [choice],
            }

        if chat:
            await send(frame({"role": "assistant", "content": ""}))
        if tool_calls is not None:
            await send(frame({"tool_calls": tool_calls}))
            await send(frame({}, finish="tool_calls"))
        else:
            for token in _tokenize(text):
                if self.tok_ms:
                    await asyncio.sleep(self.tok_ms / 1000)
                await send(frame({"content": token}))
            await send(frame({}, finish="stop"))
        if include_usage:
            await send(
                {
                    "id": stream_id,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
            )
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def tags(self, _request: web.Request) -> web.Response:
        return web.json_response({"models": [{"name": "fake-model"}]})


def _tokens_of_prefix(chars: int) -> int:
    return chars // CHARS_PER_TOKEN


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    parts = text.split(" ")
    return [parts[0]] + [" " + part for part in parts[1:]]


def _completion_body(
    body: dict[str, Any],
    text: str,
    tool_calls: list[dict[str, Any]] | None,
    usage: dict[str, Any],
    chat: bool,
) -> dict[str, Any]:
    model = body.get("model", "fake-model")
    created = int(time.time())
    if chat:
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        finish = "stop"
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
            finish = "tool_calls"
        choices = [{"index": 0, "message": message, "finish_reason": finish}]
        obj = "chat.completion"
    else:
        choices = [{"index": 0, "text": text, "finish_reason": "stop"}]
        obj = "text_completion"
    return {
        "id": "chatcmpl-fake",
        "object": obj,
        "created": created,
        "model": model,
        "choices": choices,
        "usage": usage,
    }


def create_app(tok_ms: int = 0, reply: str | None = None) -> web.Application:
    upstream = FakeUpstream(tok_ms=tok_ms, reply=reply)
    app = web.Application(client_max_size=1024 ** 3)
    app["upstream"] = upstream
    app.router.add_post("/v1/chat/completions", upstream.chat_completions)
    app.router.add_post("/v1/completions", upstream.completions)
    app.router.add_get("/api/tags", upstream.tags)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m calm_proxy.fake_upstream")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tok-ms", type=int, default=0, help="sleep per streamed token")
    parser.add_argument("--reply", default=None, help="canned completion text")
    args = parser.parse_args(argv)
    web.run_app(
        create_app(tok_ms=args.tok_ms, reply=args.reply),
        host=args.host,
        port=args.port,
        print=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
