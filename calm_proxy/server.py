"""The proxy itself: an OpenAI-compatible passthrough with middlewares.

Everything the proxy does is recorded in the trace record; nothing it does is
invisible. Requests it does not understand are forwarded untouched.
"""

from __future__ import annotations

import json
import time
from typing import Any

import aiohttp
from aiohttp import web

from .config import Config
from .middleware import trace as tracemw
from .middleware.trace import TraceWriter
from . import sse

DROP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "upgrade",
    "accept-encoding",
    "expect",
}

DROP_RESPONSE_HEADERS = {
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "content-encoding",
}


def _request_headers(request: web.Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in DROP_REQUEST_HEADERS
    }


def _response_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in DROP_RESPONSE_HEADERS
    }


class Proxy:
    def __init__(self, config: Config):
        self.config = config
        trace_dir = config.trace_dir if config.has("trace") else None
        self.tracer = TraceWriter(trace_dir)
        self._client: aiohttp.ClientSession | None = None

    # -- lifecycle ---------------------------------------------------------
    async def startup(self, _app: web.Application | None = None) -> None:
        if self._client is None:
            timeout = aiohttp.ClientTimeout(total=None, sock_read=self.config.timeout_s)
            self._client = aiohttp.ClientSession(timeout=timeout, auto_decompress=False)

    async def cleanup(self, _app: web.Application | None = None) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> aiohttp.ClientSession:
        assert self._client is not None, "proxy not started"
        return self._client

    def upstream_url(self, request: web.Request) -> str:
        base = self.config.upstream.rstrip("/")
        target = request.rel_url.path
        if request.rel_url.query_string:
            target = f"{target}?{request.rel_url.query_string}"
        return base + target

    # -- routing -----------------------------------------------------------
    async def handle(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        path = request.rel_url.path
        body: Any = None
        if request.method == "POST" and path.endswith(("/chat/completions", "/completions")):
            try:
                body = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                body = None
        if isinstance(body, dict):
            return await self._handle_completion(request, raw, body)
        return await self._passthrough(request, raw)

    async def _passthrough(self, request: web.Request, raw: bytes) -> web.StreamResponse:
        try:
            async with self.client.request(
                request.method,
                self.upstream_url(request),
                data=raw or None,
                headers=_request_headers(request),
            ) as upstream:
                payload = await upstream.read()
                return web.Response(
                    body=payload,
                    status=upstream.status,
                    headers=_response_headers(upstream.headers),
                )
        except aiohttp.ClientError as exc:
            return _gateway_error(exc)

    # -- the interesting path ---------------------------------------------
    async def _handle_completion(
        self, request: web.Request, raw: bytes, body: dict[str, Any]
    ) -> web.StreamResponse:
        messages = _messages_of(body)
        session = tracemw.session_id(request.headers.get("X-Calm-Session"), messages)
        seq = self.tracer.next_seq(session)
        client_stream = bool(body.get("stream"))

        upstream_body = dict(body)
        strip_usage_chunk = False
        if client_stream:
            options = body.get("stream_options")
            if not (isinstance(options, dict) and options.get("include_usage")):
                upstream_body["stream_options"] = {
                    **(options if isinstance(options, dict) else {}),
                    "include_usage": True,
                }
                strip_usage_chunk = True

        payload = raw if upstream_body == body else json.dumps(upstream_body).encode("utf-8")
        writer = self.tracer if self.tracer.enabled else None
        message_recs = tracemw.message_records(messages, writer)

        record: dict[str, Any] = {
            "ts": time.time(),
            "session": session,
            "seq": seq,
            "model": body.get("model"),
            "stream": client_stream,
            "params": tracemw.params_of(body),
            "messages": message_recs,
            "prompt_sha": tracemw.prompt_sha(message_recs),
            "usage": tracemw.usage_record(None),
            "timing_ms": {"submit": 0, "first_token": None, "done": None},
            "memo": {"hit": False, "key": None},
            "dedup": {"replaced": 0, "bytes_saved": 0},
            "upstream_status": None,
            "error": None,
        }

        started = time.monotonic()
        try:
            headers = _request_headers(request)
            headers["Content-Length"] = str(len(payload))
            upstream = await self.client.post(
                self.upstream_url(request), data=payload, headers=headers
            )
        except aiohttp.ClientError as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["timing_ms"]["done"] = _ms(started)
            self.tracer.append(record)
            return _gateway_error(exc)
        except TimeoutError as exc:
            record["error"] = f"TimeoutError: {exc}"
            record["timing_ms"]["done"] = _ms(started)
            self.tracer.append(record)
            return _gateway_error(exc, status=504)

        record["upstream_status"] = upstream.status
        try:
            if client_stream or _is_event_stream(upstream):
                return await self._stream_response(
                    request, upstream, record, started, strip_usage_chunk, client_stream
                )
            return await self._buffered_response(upstream, record, started)
        finally:
            upstream.release()

    async def _buffered_response(
        self, upstream: aiohttp.ClientResponse, record: dict[str, Any], started: float
    ) -> web.Response:
        data = await upstream.read()
        record["timing_ms"]["first_token"] = _ms(started)
        record["timing_ms"]["done"] = _ms(started)
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            record["usage"] = tracemw.usage_record(parsed.get("usage"))
            if upstream.status >= 400:
                record["error"] = _error_text(parsed)
        elif upstream.status >= 400:
            record["error"] = f"upstream status {upstream.status}"
        self.tracer.append(record)
        return web.Response(
            body=data,
            status=upstream.status,
            headers=_response_headers(upstream.headers),
        )

    async def _stream_response(
        self,
        request: web.Request,
        upstream: aiohttp.ClientResponse,
        record: dict[str, Any],
        started: float,
        strip_usage_chunk: bool,
        client_stream: bool,
    ) -> web.StreamResponse:
        out = web.StreamResponse(
            status=upstream.status, headers=_response_headers(upstream.headers)
        )
        await out.prepare(request)
        buffer = b""
        usage: dict[str, Any] | None = None
        completion = []
        async for chunk in upstream.content.iter_any():
            if record["timing_ms"]["first_token"] is None:
                record["timing_ms"]["first_token"] = _ms(started)
            buffer += chunk
            events, buffer = sse.split_events(buffer)
            for event in events:
                forward = True
                for payload in sse.data_payloads(event):
                    if isinstance(payload, dict) and payload.get("usage") is not None:
                        usage = payload["usage"]
                    completion.append(sse.chunk_text(payload))
                    if strip_usage_chunk and sse.is_usage_only(payload):
                        forward = False
                if forward:
                    await out.write(event)
        if buffer:
            await out.write(buffer)
        record["usage"] = tracemw.usage_record(usage)
        record["timing_ms"]["done"] = _ms(started)
        if upstream.status >= 400:
            record["error"] = f"upstream status {upstream.status}"
        if not client_stream:
            record["stream"] = False
        self.tracer.append(record)
        await out.write_eof()
        return out


def _messages_of(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if isinstance(messages, list):
        return [m for m in messages if isinstance(m, dict)]
    prompt = body.get("prompt")
    if prompt is not None:
        return [{"role": "user", "content": prompt}]
    return []


def _is_event_stream(upstream: aiohttp.ClientResponse) -> bool:
    return "text/event-stream" in (upstream.headers.get("Content-Type") or "")


def _error_text(parsed: dict[str, Any]) -> str | None:
    error = parsed.get("error")
    if error is None:
        return None
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error)


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _gateway_error(exc: BaseException, status: int = 502) -> web.Response:
    if isinstance(exc, (TimeoutError, aiohttp.ServerTimeoutError)):
        status = 504
    payload = {
        "error": {
            "message": f"calm-proxy could not reach upstream: {type(exc).__name__}: {exc}",
            "type": "upstream_error",
        }
    }
    return web.json_response(payload, status=status)


def create_app(config: Config) -> web.Application:
    proxy = Proxy(config)
    app = web.Application(client_max_size=1024 ** 3)
    app["proxy"] = proxy
    app.on_startup.append(proxy.startup)
    app.on_cleanup.append(proxy.cleanup)
    app.router.add_route("*", "/{tail:.*}", proxy.handle)
    return app


def run(config: Config) -> None:
    app = create_app(config)
    web.run_app(app, host=config.host, port=config.port, print=None)
