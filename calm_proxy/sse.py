"""Minimal SSE framing helpers.

The proxy forwards server-sent events byte-for-byte: it splits the upstream
stream on event boundaries, inspects the JSON payloads for the trace, and
re-emits the original bytes of every event it keeps.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

SEPARATOR = b"\n\n"


def split_events(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Split ``buffer`` into complete events (separator included) and a rest."""
    events: list[bytes] = []
    while True:
        index = buffer.find(SEPARATOR)
        if index == -1:
            return events, buffer
        events.append(buffer[: index + len(SEPARATOR)])
        buffer = buffer[index + len(SEPARATOR) :]


def data_payloads(event: bytes) -> Iterator[Any]:
    """Yield the decoded JSON of every ``data:`` line that is not ``[DONE]``."""
    for line in event.split(b"\n"):
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:") :].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def is_usage_only(chunk: Any) -> bool:
    """True for the extra final chunk Ollama/OpenAI send for include_usage."""
    return (
        isinstance(chunk, dict)
        and chunk.get("usage") is not None
        and not chunk.get("choices")
    )


def chunk_text(chunk: Any) -> str:
    """The text a streamed chunk contributes to the assistant message."""
    if not isinstance(chunk, dict):
        return ""
    out = []
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            out.append(content)
        text = choice.get("text")
        if isinstance(text, str):
            out.append(text)
    return "".join(out)


def encode_event(chunk: Any) -> bytes:
    return b"data: " + json.dumps(chunk).encode("utf-8") + SEPARATOR
