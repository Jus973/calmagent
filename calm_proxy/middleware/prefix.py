"""I-2 `prefix` — why the server's prompt cache is missing.

For every request we compare its message list with the previous request in the
same session: how many leading messages are byte-identical, how many bytes of
the first divergent message still match, and *what kind* of divergence it is.
The lever never mutates a request; it only explains the loss.

`achievable_cached_tokens_est` is the token estimate of that common prefix. The
gap between it and the server-reported `cached_tokens` is the headroom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..hashing import message_text, sha256_hex

CHARS_PER_TOKEN = 4

APPEND_ONLY = "append_only"
SYSTEM_CHANGED = "system_changed"
HISTORY_TRUNCATED = "history_truncated"
TOOL_RESULT_REORDERED = "tool_result_reordered"
TIMESTAMP_LIKE = "timestamp_like"
CONTENT_CHANGED = "content_changed"
NO_PREVIOUS = "no_previous_request"

TIMESTAMP_PATTERNS = (
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}",          # 2026-09-20T02:45
    r"\d{4}-\d{2}-\d{2}",                          # 2026-09-20
    r"\d{2}:\d{2}:\d{2}",                          # 02:45:07
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
    r"\b[0-9a-f]{16,}\b",                          # long hex ids
    r"\b1[0-9]{9}(?:\.[0-9]+)?\b",                 # unix seconds
    r"\b\d+\.\d+s\b",                              # 12.3s durations
    r"0x[0-9a-fA-F]{6,}",                          # object addresses
)
TIMESTAMP_RE = re.compile("|".join(TIMESTAMP_PATTERNS))


@dataclass(frozen=True)
class MessageView:
    role: str
    sha: str
    text: str

    @property
    def nbytes(self) -> int:
        return len(self.text.encode("utf-8"))


def views(messages: Iterable[dict[str, Any]]) -> list[MessageView]:
    out = []
    for message in messages:
        text = message_text(message)
        out.append(
            MessageView(role=str(message.get("role")), sha=sha256_hex(text), text=text)
        )
    return out


def _common_message_count(previous: list[MessageView], current: list[MessageView]) -> int:
    count = 0
    for old, new in zip(previous, current):
        if old.sha != new.sha:
            break
        count += 1
    return count


def _common_byte_count(old: str, new: str) -> int:
    old_bytes = old.encode("utf-8")
    new_bytes = new.encode("utf-8")
    limit = min(len(old_bytes), len(new_bytes))
    index = 0
    while index < limit and old_bytes[index] == new_bytes[index]:
        index += 1
    return index


def _diff_region(old: str, new: str, common_bytes: int) -> str:
    """The text around the divergence, on both sides, for classification."""
    start = max(0, common_bytes - 40)
    return old[start : common_bytes + 200] + "\n" + new[start : common_bytes + 200]


def classify(
    previous: list[MessageView], current: list[MessageView], common_messages: int
) -> str:
    if common_messages == len(previous):
        return APPEND_ONLY
    old = previous[common_messages]
    new = current[common_messages] if common_messages < len(current) else None
    if old.role == "system" or (new is not None and new.role == "system"):
        return SYSTEM_CHANGED
    remaining_old = previous[common_messages:]
    remaining_new = current[common_messages:] if new is not None else []
    old_shas = [m.sha for m in remaining_old]
    new_shas = [m.sha for m in remaining_new]
    if sorted(old_shas) == sorted(new_shas) and old_shas != new_shas:
        return TOOL_RESULT_REORDERED
    if _is_truncation(old_shas, new_shas):
        return HISTORY_TRUNCATED
    if new is not None and TIMESTAMP_RE.search(
        _diff_region(old.text, new.text, _common_byte_count(old.text, new.text))
    ):
        return TIMESTAMP_LIKE
    return CONTENT_CHANGED


def _is_truncation(old_shas: list[str], new_shas: list[str]) -> bool:
    """The new history is the old one with leading messages dropped."""
    if not new_shas or len(new_shas) >= len(old_shas):
        return False
    for start in range(1, len(old_shas)):
        if old_shas[start : start + len(new_shas)] == new_shas:
            return True
    return False


def analyze(previous: list[MessageView] | None, current: list[MessageView]) -> dict[str, Any]:
    if previous is None:
        common_messages = 0
        common_bytes = 0
        divergence = NO_PREVIOUS
    else:
        common_messages = _common_message_count(previous, current)
        common_bytes = sum(m.nbytes for m in current[:common_messages])
        if common_messages < len(previous) and common_messages < len(current):
            common_bytes += _common_byte_count(
                previous[common_messages].text, current[common_messages].text
            )
        divergence = classify(previous, current, common_messages)
    return {
        "common_messages": common_messages,
        "common_bytes": common_bytes,
        "divergence": divergence,
        "achievable_cached_tokens_est": common_bytes // CHARS_PER_TOKEN,
    }


class PrefixTracker:
    """Keeps the previous request's message list per session. Never mutates."""

    def __init__(self) -> None:
        self._last: dict[str, list[MessageView]] = {}

    def observe(self, session: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        current = views(messages)
        result = analyze(self._last.get(session), current)
        self._last[session] = current
        return result
