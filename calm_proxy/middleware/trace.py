"""I-1 `trace` — the instrument.

Writes one JSON object per upstream request to ``<trace-dir>/trace.jsonl`` and
every message body once to ``<trace-dir>/bodies/<sha>.txt`` (content-addressed,
grow-only).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from ..hashing import canonical_json, message_text, sha256_hex


class TraceWriter:
    """Append-only trace store. Safe for concurrent use within one process."""

    def __init__(self, trace_dir: str | os.PathLike[str] | None):
        self.dir = Path(trace_dir) if trace_dir is not None else None
        self._lock = threading.Lock()
        self._seq: dict[str, int] = {}
        if self.dir is not None:
            (self.dir / "bodies").mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    @property
    def path(self) -> Path | None:
        return None if self.dir is None else self.dir / "trace.jsonl"

    def next_seq(self, session: str) -> int:
        with self._lock:
            seq = self._seq.get(session, 0)
            self._seq[session] = seq + 1
        return seq

    def write_body(self, text: str) -> str:
        """Store ``text`` under its sha256 and return the sha."""
        sha = sha256_hex(text)
        if self.dir is None:
            return sha
        target = self.dir / "bodies" / f"{sha}.txt"
        if not target.exists():
            tmp = target.with_suffix(".txt.tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, target)
        return sha

    def append(self, record: dict[str, Any]) -> None:
        if self.dir is None:
            return
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with (self.dir / "trace.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def session_id(header_value: str | None, messages: list[dict[str, Any]] | None) -> str:
    """``X-Calm-Session`` if present, else sha of the first system message, else
    sha of the first user message, else ``"unknown"``."""
    if header_value:
        return header_value
    for role in ("system", "user"):
        for message in messages or []:
            if message.get("role") == role:
                return sha256_hex(message_text(message))
    return "unknown"


def message_records(
    messages: list[dict[str, Any]], writer: TraceWriter | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for message in messages:
        text = message_text(message)
        sha = writer.write_body(text) if writer is not None else sha256_hex(text)
        record: dict[str, Any] = {
            "role": message.get("role"),
            "sha": sha,
            "bytes": len(text.encode("utf-8")),
        }
        tool_call_id = message.get("tool_call_id")
        if tool_call_id:
            record["tool_call_id"] = tool_call_id
        records.append(record)
    return records


def prompt_sha(message_recs: list[dict[str, Any]]) -> str:
    return sha256_hex(canonical_json(message_recs))


def params_of(body: dict[str, Any]) -> dict[str, Any]:
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    return {
        "temperature": body.get("temperature"),
        "seed": body.get("seed"),
        "max_tokens": max_tokens,
    }


def usage_record(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached if cached is not None else None,
        "completion_tokens": usage.get("completion_tokens"),
    }
