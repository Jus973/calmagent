"""Content addressing helpers shared by every middleware.

Every store the proxy keeps is grow-only and keyed by a content hash, so it is
idempotent and replay-safe.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_text(content: Any) -> str:
    """Flatten a message ``content`` (str, list of parts, or None) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return canonical_json(content)


def message_text(message: dict[str, Any]) -> str:
    """The hashed text of a whole message: content plus any tool_calls."""
    text = content_text(message.get("content"))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        text = text + "\n" + canonical_json(tool_calls)
    return text
