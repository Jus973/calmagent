"""A model server with the timing behaviour of a local one and none of its variance (§latency bench).

A real 7B on a CPU box answers a slot request in tens of seconds, so a wall-clock comparison of two
schedules is mostly sampling noise: each arm sees different text. This transport returns *fixed* text
per (slot, sample index) and spends time the way a saturated local server does. There is one queue and
one aggregate throughput: producing a token costs `1/tok_s` of server time and prefilling a prompt
costs `ttft_ms`, whoever asked for it. So the wall clock of a schedule is the server time it demanded,
and two schedules against the same fixtures differ only in what they asked for and when.

It is an httpx transport, so `Client` talks to it over its normal code path, including SSE streaming
and an early cut: tokens after the cut are never produced, cost no server time, and the gap between
`offered_tokens` and `decoded_tokens` shows exactly how many they were.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field

import httpx

_IMPLEMENT = re.compile(r"Implement `(\w+)` now\.")


@dataclass
class FakeModelConfig:
    ttft_ms: float = 400.0        # server time a prompt's prefill occupies
    tok_s: float = 24.0           # aggregate decode throughput, shared by every open request
    chars_per_token: int = 4


@dataclass
class FakeModelStats:
    requests: int = 0
    decoded_tokens: int = 0       # tokens the server actually produced
    offered_tokens: int = 0       # tokens it would have produced had no request been cut short
    ttft_ms: list[int] = field(default_factory=list)

    @property
    def skipped_tokens(self) -> int:
        return self.offered_tokens - self.decoded_tokens

    def to_json(self) -> dict:
        return {"requests": self.requests, "decoded_tokens": self.decoded_tokens,
                "offered_tokens": self.offered_tokens, "skipped_tokens": self.skipped_tokens}


def _tokens(text: str, chars: int) -> list[str]:
    return [text[i:i + chars] for i in range(0, len(text), chars)] or [""]


class FakeModel:
    """`samples[slot][i]` is the completion for sample i of that slot; `tail` is appended to each."""

    def __init__(self, samples: dict[str, list[str]], *, tail: str = "",
                 cfg: FakeModelConfig | None = None):
        self.samples = samples
        self.tail = tail
        self.cfg = cfg or FakeModelConfig()
        self.stats = FakeModelStats()
        self._busy_until = 0.0        # the server's queue, as the time it is booked until

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    # ------------------------------------------------------------------ the one queue
    async def _server_time(self, seconds: float) -> None:
        """Occupy the server for `seconds`, queueing behind whatever it is already doing."""
        now = time.monotonic()
        self._busy_until = max(self._busy_until, now) + seconds
        await asyncio.sleep(max(0.0, self._busy_until - now))

    async def _prefill(self) -> None:
        await self._server_time(self.cfg.ttft_ms / 1000)
        self.stats.ttft_ms.append(int(self.cfg.ttft_ms))

    async def _token(self) -> None:
        await self._server_time(1 / self.cfg.tok_s)
        self.stats.decoded_tokens += 1

    # ------------------------------------------------------------------ routing
    def _completion(self, body: dict) -> str:
        """Which canned completion this request gets: the slot it names, the sample index in its seed."""
        prompt = "\n".join(m.get("content", "") for m in body.get("messages", []))
        m = _IMPLEMENT.search(prompt)
        if m is None or m.group(1) not in self.samples:
            raise KeyError(f"fake model has no samples for this request: {prompt[-120:]!r}")
        texts = self.samples[m.group(1)]
        return texts[(body.get("seed") or 0) % 100 % len(texts)] + self.tail

    # ------------------------------------------------------------------ transport
    async def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        toks = _tokens(self._completion(body), self.cfg.chars_per_token)[:body.get("max_tokens") or None]
        self.stats.requests += 1
        self.stats.offered_tokens += len(toks)
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=self._sse(toks))
        await self._prefill()
        for _ in toks:
            await self._token()
        return httpx.Response(200, json={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(toks)},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": sum(len(m["content"]) for m in body["messages"]) // 4,
                      "completion_tokens": len(toks)},
        })

    async def _sse(self, toks: list[str]):
        """SSE frames, one token each. A caller that stops reading stops the server: the generator is
        suspended at its `yield`, so the remaining tokens never cost anything."""
        await self._prefill()
        for t in toks:
            await self._token()
            yield _frame({"choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})
        yield _frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 0, "completion_tokens": len(toks)}})
        yield b"data: [DONE]\n\n"


def _frame(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"
