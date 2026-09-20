"""I-3 `memo` — exact-request memoization for deterministic requests.

Key = sha256 over (model, canonical messages, temperature, seed, max_tokens,
tools). Only requests that are deterministic by construction are eligible:
`temperature == 0` or an explicit `seed`. The store is a grow-only,
content-addressed JSONL directory, so a hit is a replay of a response the
upstream already produced — never a guess, and never an error (non-200
responses are not stored).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from ..hashing import canonical_json, sha256_hex

MEMO_HEADER = "X-Calm-Memo"


def eligible(body: dict[str, Any]) -> bool:
    if body.get("seed") is not None:
        return True
    temperature = body.get("temperature")
    return temperature is not None and float(temperature) == 0.0


def memo_key(body: dict[str, Any]) -> str:
    material = {
        "model": body.get("model"),
        "messages": body.get("messages"),
        "prompt": body.get("prompt"),
        "temperature": body.get("temperature"),
        "seed": body.get("seed"),
        "max_tokens": body.get("max_tokens", body.get("max_completion_tokens")),
        "tools": body.get("tools"),
        "tool_choice": body.get("tool_choice"),
        "response_format": body.get("response_format"),
        "stop": body.get("stop"),
        "n": body.get("n"),
    }
    return sha256_hex("memo|" + canonical_json(material))


class MemoStore:
    """Grow-only: one file per key, written once, plus an append-only index."""

    def __init__(self, trace_dir: str | os.PathLike[str]):
        self.dir = Path(trace_dir) / "memo"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path_for(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def put(self, key: str, response: dict[str, Any], model: str | None = None) -> None:
        path = self.path_for(key)
        if path.exists():
            return
        entry = {"key": key, "ts": time.time(), "model": model, "response": response}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        with self._lock:
            with (self.dir / "index.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "ts": entry["ts"], "model": model}) + "\n")


def _choice_payload(response: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None, str]:
    choices = response.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message") or {}
    text = message.get("content") or choice.get("text") or ""
    tool_calls = message.get("tool_calls")
    finish = choice.get("finish_reason") or "stop"
    return text, tool_calls, finish


def sse_frames(response: dict[str, Any], include_usage: bool, chat: bool = True) -> list[bytes]:
    """Re-stream a stored response as SSE, the way the upstream would have."""
    text, tool_calls, finish = _choice_payload(response)
    stream_id = response.get("id", "chatcmpl-memo")
    model = response.get("model")
    created = response.get("created", int(time.time()))
    obj = "chat.completion.chunk" if chat else "text_completion"

    def frame(delta: dict[str, Any], finish_reason: Any = None) -> bytes:
        if chat:
            choice: dict[str, Any] = {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        else:
            choice = {
                "index": 0,
                "text": delta.get("content", ""),
                "finish_reason": finish_reason,
            }
        chunk = {
            "id": stream_id,
            "object": obj,
            "created": created,
            "model": model,
            "choices": [choice],
        }
        return b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"

    frames: list[bytes] = []
    if chat:
        frames.append(frame({"role": "assistant", "content": ""}))
    if tool_calls:
        frames.append(frame({"tool_calls": tool_calls}))
    elif text:
        frames.append(frame({"content": text}))
    frames.append(frame({}, finish_reason=finish))
    if include_usage and response.get("usage") is not None:
        frames.append(
            b"data: "
            + json.dumps(
                {
                    "id": stream_id,
                    "object": obj,
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": response["usage"],
                }
            ).encode("utf-8")
            + b"\n\n"
        )
    frames.append(b"data: [DONE]\n\n")
    return frames


def response_from_stream(
    text: str,
    tool_calls: list[dict[str, Any]] | None,
    finish_reason: str | None,
    usage: dict[str, Any] | None,
    model: str | None,
    stream_id: str | None,
    chat: bool = True,
) -> dict[str, Any]:
    """Rebuild the non-streamed response object from accumulated chunks."""
    if chat:
        message: dict[str, Any] = {"role": "assistant", "content": text or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        choices = [
            {"index": 0, "message": message, "finish_reason": finish_reason or "stop"}
        ]
    else:
        choices = [{"index": 0, "text": text, "finish_reason": finish_reason or "stop"}]
    return {
        "id": stream_id or "chatcmpl-memo",
        "object": "chat.completion" if chat else "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": usage,
    }
