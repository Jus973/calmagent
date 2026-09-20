"""OpenAI-compatible chat client (§3.6). Configured by env only:

CALM_BASE_URL (default http://localhost:11434/v1), CALM_MODEL, CALM_API_KEY,
CALM_MAX_INFLIGHT (default 32), CALM_NO_N=1 (server ignores `n`: send n requests instead),
CALM_TIMEOUT_S (default 600) for the HTTP read timeout.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import asdict, dataclass

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

    def to_json(self) -> dict:
        return asdict(self)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


class Client:
    def __init__(self, base_url: str | None = None, model: str | None = None, *,
                 api_key: str | None = None, max_inflight: int | None = None,
                 no_n: bool | None = None, timeout_s: float | None = None, transport=None):
        timeout_s = timeout_s if timeout_s is not None else float(os.environ.get("CALM_TIMEOUT_S", "600"))
        self.base_url = (base_url or os.environ.get("CALM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.environ.get("CALM_MODEL", "qwen2.5-coder:7b")
        self.no_n = _env_flag("CALM_NO_N") if no_n is None else no_n
        key = api_key or os.environ.get("CALM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._sem = asyncio.Semaphore(max_inflight or int(os.environ.get("CALM_MAX_INFLIGHT", "32")))
        self._http = httpx.AsyncClient(
            timeout=timeout_s, transport=transport,
            headers={"Authorization": f"Bearer {key}"} if key else {},
        )

    def config(self) -> dict:
        return {"base_url": self.base_url, "model": self.model, "no_n": self.no_n}

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
                system_fingerprint=data.get("system_fingerprint"),
            ))
        return out

    async def sample(self, messages: list[dict], n: int = 1, temperature: float = 0.8,
                     max_tokens: int = 512, seed: int | None = 0, top_p: float | None = None) -> list[Sample]:
        """n samples. Without server-side n, sample i uses seed + i so samples differ but are reproducible."""
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
