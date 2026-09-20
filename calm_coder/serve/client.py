"""OpenAI-compatible chat client (§3.6). Configured by env only:

CALM_BASE_URL (default http://localhost:11434/v1), CALM_MODEL, CALM_API_KEY,
CALM_MAX_INFLIGHT (default 32), CALM_NO_N=1 (server ignores `n`: send n requests instead),
CALM_STREAM=1 (stream completions: measures TTFT and lets a caller cut a completion short),
CALM_TIMEOUT_S (default 600) for the HTTP read timeout.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class Sample:
    text: str
    completion_tokens: int
    prompt_tokens: int            # with server-side n>1 the prompt is charged once: on sample 0 only
    latency_ms: int
    seed: int | None
    cached_tokens: int | None     # None = unknown. Never estimated.
    tokens_estimated: bool        # usage absent: len(text)//4
    finish_reason: str | None
    system_fingerprint: str | None = None
    ttft_ms: int | None = None    # None = not streamed, so prefill and decode aren't separable
    stopped_early: bool = False   # the caller had its fill and ended the request
    model: str | None = None      # which member of a Fleet produced this sample

    def to_json(self) -> dict:
        return asdict(self)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def key_for_url(base_url: str, explicit: str | None = None) -> str | None:
    """Host-specific key first, then the generic env. The key itself is never logged."""
    if explicit:
        return explicit
    if "api.x.ai" in (base_url or ""):
        return (os.environ.get("XAI_API_KEY") or os.environ.get("CALM_API_KEY")
                or os.environ.get("OPENAI_API_KEY"))
    return os.environ.get("CALM_API_KEY") or os.environ.get("OPENAI_API_KEY")


def auth_label(base_url: str, key: str | None) -> str | None:
    """Which env supplied the key, for client.json. Never the secret."""
    if not key:
        return None
    return "xai" if "api.x.ai" in (base_url or "") else "env"


def client_from_spec(spec: str | None = None, **kw) -> "Client | Fleet":
    return Fleet.from_spec(spec, **kw) if spec else Client(**kw)


class Client:
    def __init__(self, base_url: str | None = None, model: str | None = None, *,
                 api_key: str | None = None, max_inflight: int | None = None,
                 no_n: bool | None = None, stream: bool | None = None,
                 timeout_s: float | httpx.Timeout | None = None, transport=None):
        timeout_s = timeout_s if timeout_s is not None else httpx.Timeout(
            600, read=float(os.environ.get("CALM_TIMEOUT_S", "600")))
        self.base_url = (base_url or os.environ.get("CALM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.environ.get("CALM_MODEL", "qwen2.5-coder:7b")
        self.no_n = _env_flag("CALM_NO_N") if no_n is None else no_n
        self.stream = _env_flag("CALM_STREAM") if stream is None else stream
        key = key_for_url(self.base_url, api_key)
        self.auth = auth_label(self.base_url, key)
        self._sem = asyncio.Semaphore(max_inflight or int(os.environ.get("CALM_MAX_INFLIGHT", "32")))
        self._http = httpx.AsyncClient(
            timeout=timeout_s, transport=transport,
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )

    def config(self) -> dict:
        return {"base_url": self.base_url, "model": self.model, "no_n": self.no_n,
                "stream": self.stream, "auth": self.auth}

    @property
    def members(self) -> list["Client"]:
        return [self]

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _post(self, body: dict) -> tuple[dict, int]:
        delay = 1.0
        for attempt in range(4):
            async with self._sem:
                t0 = time.monotonic()
                try:
                    r = await self._http.post(f"{self.base_url}/chat/completions", json=body)
                except (httpx.TransportError, httpx.TimeoutException):
                    if attempt == 3:
                        raise
                    r = None
                ms = int((time.monotonic() - t0) * 1000)
            if r is not None and r.status_code not in RETRY_STATUS:
                r.raise_for_status()
                return r.json(), ms
            if attempt == 3:
                r.raise_for_status()
            await asyncio.sleep(delay)
            delay *= 2
        raise RuntimeError("unreachable")

    async def _request(self, messages, n, temperature, max_tokens, seed, top_p) -> list[Sample]:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "n": n}
        if seed is not None:
            body["seed"] = seed
        if top_p is not None:
            body["top_p"] = top_p
        data, ms = await self._post(body)
        return _samples(data, messages, ms, seed, self.model)

    async def _stream_request(self, messages, temperature, max_tokens, seed, top_p,
                              stop_when: Callable[[str], int | None] | None) -> list[Sample]:
        """One streamed sample. `stop_when` sees the text so far at every newline and returns the index to
        cut at; cutting ends the request, so the tokens after that index are never decoded.

        A server that ignores `stream` and answers with a whole completion is read as one: no TTFT, and
        no cut, but the sample is the same. Retries match `_post`: an attempt that fails before it is
        cut or finished is restarted from empty, so no partial text survives it."""
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "n": 1, "stream": True,
                "stream_options": {"include_usage": True}}
        if seed is not None:
            body["seed"] = seed
        if top_p is not None:
            body["top_p"] = top_p
        delay = 1.0
        for attempt in range(4):
            parts: list[str] = []
            ttft, finish, usage, stopped, check, retry = None, None, {}, False, False, False
            async with self._sem:
                t0 = time.monotonic()
                try:
                    async with self._http.stream("POST", f"{self.base_url}/chat/completions",
                                                 json=body) as r:
                        if r.status_code >= 400:
                            await r.aread()
                            if r.status_code not in RETRY_STATUS or attempt == 3:
                                r.raise_for_status()
                            retry = True
                        elif "event-stream" not in r.headers.get("content-type", ""):
                            await r.aread()
                            return _samples(r.json(), messages,
                                            int((time.monotonic() - t0) * 1000), seed, self.model)
                        else:
                            async for line in r.aiter_lines():
                                if not line.startswith("data:"):
                                    continue
                                payload = line[5:].strip()
                                if payload == "[DONE]":
                                    break
                                chunk = json.loads(payload)
                                usage = chunk.get("usage") or usage
                                for c in chunk.get("choices", []):
                                    finish = c.get("finish_reason") or finish
                                    text = (c.get("delta") or {}).get("content") or ""
                                    if not text:
                                        continue
                                    if ttft is None:
                                        ttft = int((time.monotonic() - t0) * 1000)
                                    parts.append(text)
                                    check = check or "\n" in text
                                if stop_when is not None and check:
                                    check = False
                                    cut = stop_when("".join(parts))
                                    if cut is not None:
                                        parts = ["".join(parts)[:cut]]
                                        stopped = True
                                        break
                except (httpx.TransportError, httpx.TimeoutException):
                    if attempt == 3:
                        raise
                    retry = True
                ms = int((time.monotonic() - t0) * 1000)
            if retry:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            out = "".join(parts)
            # A cut request never gets the usage frame, and a streamed one only if the server sends it.
            estimated = stopped or "completion_tokens" not in usage
            return [Sample(
                text=out, completion_tokens=len(out) // 4 if estimated else usage["completion_tokens"],
                prompt_tokens=usage.get("prompt_tokens", sum(len(m["content"]) for m in messages) // 4),
                latency_ms=ms, seed=seed,
                cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                tokens_estimated=estimated, finish_reason="stopped_early" if stopped else finish,
                ttft_ms=ttft, stopped_early=stopped, model=self.model,
            )]
        raise RuntimeError("unreachable")

    async def sample(self, messages: list[dict], n: int = 1, temperature: float = 0.8,
                     max_tokens: int = 512, seed: int | None = 0, top_p: float | None = None,
                     stop_when: Callable[[str], int | None] | None = None) -> list[Sample]:
        """n samples. Without server-side n, sample i uses seed + i so samples differ but are reproducible.

        Streaming (`CALM_STREAM=1`, or any `stop_when` rule) is one request per sample: it measures TTFT
        and can end a completion as soon as the caller has what it asked for.
        """
        if self.stream or stop_when is not None:
            reqs = [self._stream_request(messages, temperature, max_tokens,
                                         seed if seed is None or n == 1 else seed + i, top_p, stop_when)
                    for i in range(n)]
            return [s for batch in await asyncio.gather(*reqs) for s in batch]
        if n == 1 or not self.no_n:
            return await self._request(messages, n, temperature, max_tokens, seed, top_p)
        reqs = [self._request(messages, 1, temperature, max_tokens,
                              None if seed is None else seed + i, top_p) for i in range(n)]
        return [s for batch in await asyncio.gather(*reqs) for s in batch]

    async def metrics_snapshot(self) -> dict[str, float] | None:
        """vLLM /metrics prefix-cache counters (names vary by version). None if not exposed."""
        root = re.sub(r"/v1$", "", self.base_url)
        try:
            r = await self._http.get(f"{root}/metrics", timeout=5)
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            return None
        out = {}
        for line in r.text.splitlines():
            if line.startswith("#") or "prefix_cache" not in line:
                continue
            name, _, val = line.rpartition(" ")
            try:
                out[name] = float(val)
            except ValueError:
                pass
        return out


class Fleet:
    """Several `Client`s behind one `sample`, so the agents writing into a store are different
    models rather than clones of one.

    A round of n samples is split across members in a fixed order, so which member produces which
    sample index is a function of (n, member order) alone and not of timing — two runs with the
    same seed draw the same sample from the same model. Member j's samples start at
    `seed + offset_j`, which keeps sample identities distinct across members.

    Nothing about the store changes: a definition is still keyed by the hash of its canonical form,
    so two models that write the same method write one definition, and which model got there first
    is provenance (`Sample.model`, carried onto the emission), never identity.
    """

    def __init__(self, clients: Sequence["Client"]):
        if not clients:
            raise ValueError("Fleet needs at least one client")
        self._clients = list(clients)
        self.model = "+".join(c.model for c in self._clients)

    @classmethod
    def from_spec(cls, spec: str, **kw) -> "Fleet":
        """`model[@base_url]` entries, comma separated; a missing base_url falls back to the env."""
        out = []
        for part in spec.split(","):
            if not part.strip():
                continue
            model, _, url = part.strip().partition("@")
            out.append(Client(base_url=url or None, model=model, **kw))
        return cls(out)

    @property
    def members(self) -> list["Client"]:
        return list(self._clients)

    def shares(self, n: int) -> list[int]:
        """n split as evenly as possible; earlier members take the remainder."""
        q, r = divmod(n, len(self._clients))
        return [q + (1 if i < r else 0) for i in range(len(self._clients))]

    def config(self) -> dict:
        return {"fleet": [c.config() for c in self._clients], "model": self.model}

    async def aclose(self) -> None:
        await asyncio.gather(*(c.aclose() for c in self._clients))

    async def __aenter__(self) -> "Fleet":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def metrics_snapshot(self) -> dict[str, float] | None:
        snaps = await asyncio.gather(*(c.metrics_snapshot() for c in self._clients))
        out = {f"{c.model}|{k}": v for c, s in zip(self._clients, snaps) if s for k, v in s.items()}
        return out or None

    async def sample(self, messages: list[dict], n: int = 1, temperature: float = 0.8,
                     max_tokens: int = 512, seed: int | None = 0, top_p: float | None = None,
                     stop_when: Callable[[str], int | None] | None = None) -> list[Sample]:
        shares = self.shares(n)
        reqs, off = [], 0
        for c, k in zip(self._clients, shares):
            if k:
                reqs.append(c.sample(messages, n=k, temperature=temperature, max_tokens=max_tokens,
                                     seed=None if seed is None else seed + off, top_p=top_p,
                                     stop_when=stop_when))
                off += k
        return [s for batch in await asyncio.gather(*reqs) for s in batch]


def _samples(data: dict, messages: list[dict], ms: int, seed: int | None,
             model: str | None = None) -> list[Sample]:
    """An unstreamed response body -> one Sample per choice."""
    choices = sorted(data.get("choices", []), key=lambda c: c.get("index", 0))
    texts = [(c.get("message") or {}).get("content") or "" for c in choices]
    usage = data.get("usage") or {}
    estimated = "completion_tokens" not in usage
    total_c = usage.get("completion_tokens", sum(len(t) // 4 for t in texts))
    prompt = usage.get("prompt_tokens", sum(len(m["content"]) for m in messages) // 4)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    # Usage is per request; with several choices, split completion tokens by text length.
    total_len = sum(len(t) for t in texts) or 1
    out = []
    for i, (c, t) in enumerate(zip(choices, texts)):
        ct = total_c if len(texts) == 1 else round(total_c * len(t) / total_len)
        out.append(Sample(
            text=t, completion_tokens=ct, prompt_tokens=prompt if i == 0 else 0, latency_ms=ms,
            seed=seed, cached_tokens=cached if i == 0 else None,
            tokens_estimated=estimated or len(texts) > 1, finish_reason=c.get("finish_reason"),
            system_fingerprint=data.get("system_fingerprint"), model=model,
        ))
    return out
