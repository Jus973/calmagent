"""I-4 `dedup` — content-addressed tool-result dedup inside one request.

When a tool result the agent is re-sending is byte-identical to one that
already appears *earlier in the same request's message list*, its content is
replaced by a short stable reference. The replacement is a deterministic
function of the history, so the same history always yields the same bytes and
the KV prefix stays aligned across turns — a dedup that is not prefix-stable
costs more computed tokens than it saves.

Nothing is deleted: the original body is still in the trace's `bodies/`.
"""

from __future__ import annotations

from typing import Any

from ..hashing import content_text, sha256_hex

# Client conventions for "this user message is really a tool result".
# mini-swe-agent and aider both wrap command output in a user turn.
TOOL_RESULT_MARKERS = (
    "OBSERVATION:",
    "<output>",
    "The command output",
    "Tool output",
    "Command output",
    "```output",
)


def looks_like_tool_result(message: dict[str, Any]) -> bool:
    role = message.get("role")
    if role in {"tool", "function"}:
        return True
    if role != "user":
        return False
    text = content_text(message.get("content")).lstrip()
    return text.startswith(TOOL_RESULT_MARKERS)


def reference(ordinal: int, sha: str) -> str:
    return f"[identical to result #{ordinal} (sha256:{sha[:12]})]"


def apply(
    messages: list[dict[str, Any]], min_bytes: int = 200
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (messages, {"replaced": n, "bytes_saved": b}).

    The input is never mutated; unchanged messages are returned as-is.
    """
    out: list[dict[str, Any]] = []
    first_seen: dict[str, int] = {}
    ordinal = 0
    replaced = 0
    saved = 0
    for message in messages:
        if not looks_like_tool_result(message):
            out.append(message)
            continue
        ordinal += 1
        text = content_text(message.get("content"))
        sha = sha256_hex(text)
        size = len(text.encode("utf-8"))
        earlier = first_seen.get(sha)
        if earlier is not None and size >= min_bytes:
            ref = reference(earlier, sha)
            out.append({**message, "content": ref})
            replaced += 1
            saved += size - len(ref.encode("utf-8"))
            continue
        first_seen.setdefault(sha, ordinal)
        out.append(message)
    return out, {"replaced": replaced, "bytes_saved": saved}
