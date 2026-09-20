"""`calm-proxy lint <trace-dir>` — per-session prompt-cache report.

Reads a trace directory written by the `trace` + `prefix` middlewares and
prints, per session, how many prompt tokens the server actually computed, how
many a perfectly prefix-aligned agent could have avoided, and which divergence
is costing the most.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .middleware.prefix import NO_PREVIOUS

COLUMNS = (
    ("session", 14),
    ("reqs", 5),
    ("prompt_tok", 11),
    ("cached_tok", 11),
    ("computed_tok", 13),
    ("cached%", 8),
    ("achievable%", 12),
    ("gap%", 7),
    ("top divergence", 28),
)


def load_records(trace_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(trace_dir)
    if path.is_dir():
        path = path / "trace.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class SessionSummary:
    def __init__(self, session: str):
        self.session = session
        self.requests = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.achievable = 0
        self.divergences: Counter[str] = Counter()
        self.cached_reported = False

    def add(self, record: dict[str, Any]) -> None:
        self.requests += 1
        usage = record.get("usage") or {}
        prompt = usage.get("prompt_tokens") or 0
        cached = usage.get("cached_tokens")
        self.prompt_tokens += prompt
        if cached is not None:
            self.cached_reported = True
            self.cached_tokens += cached
        prefix = record.get("prefix") or {}
        estimate = prefix.get("achievable_cached_tokens_est") or 0
        self.achievable += min(estimate, prompt) if prompt else estimate
        divergence = prefix.get("divergence")
        if divergence and divergence != NO_PREVIOUS:
            self.divergences[divergence] += 1

    @property
    def computed_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens

    def _share(self, value: int) -> str:
        if not self.prompt_tokens:
            return "—"
        return f"{100 * value / self.prompt_tokens:.1f}%"

    def row(self) -> list[str]:
        top = self.divergences.most_common(1)
        gap = max(0, self.achievable - self.cached_tokens)
        return [
            self.session[:12],
            str(self.requests),
            str(self.prompt_tokens),
            str(self.cached_tokens) if self.cached_reported else "n/a",
            str(self.computed_tokens),
            self._share(self.cached_tokens) if self.cached_reported else "n/a",
            self._share(self.achievable),
            self._share(gap),
            f"{top[0][0]} ({top[0][1]})" if top else "—",
        ]


def summarize(records: Iterable[dict[str, Any]]) -> tuple[list[SessionSummary], SessionSummary]:
    sessions: dict[str, SessionSummary] = {}
    total = SessionSummary("TOTAL")
    for record in records:
        session = str(record.get("session", "unknown"))
        sessions.setdefault(session, SessionSummary(session)).add(record)
        total.add(record)
    ordered = sorted(sessions.values(), key=lambda s: -s.prompt_tokens)
    return ordered, total


def _format_table(rows: list[list[str]]) -> str:
    header = "  ".join(name.ljust(width) for name, width in COLUMNS)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append("  ".join(cell.ljust(width) for cell, (_, width) in zip(row, COLUMNS)))
    return "\n".join(lines)


def report(records: list[dict[str, Any]]) -> str:
    sessions, total = summarize(records)
    rows = [s.row() for s in sessions]
    if len(sessions) > 1:
        rows.append(total.row())
    lines = [
        f"requests: {total.requests}   sessions: {len(sessions)}",
        "",
        _format_table(rows),
    ]
    if total.divergences:
        lines += ["", "divergence reasons (pooled):"]
        for name, count in total.divergences.most_common():
            lines.append(f"  {name:<24} {count}")
    if not total.cached_reported:
        lines += ["", "upstream never reported cached_tokens; cached columns are n/a"]
    if not any(r.get("prefix") for r in records):
        lines += ["", "no prefix data in this trace: run the proxy with --enable trace,prefix"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="calm-proxy lint")
    parser.add_argument("trace_dir", help="run directory, or a trace.jsonl path")
    args = parser.parse_args(argv)
    print(report(load_records(args.trace_dir)))
    return 0
