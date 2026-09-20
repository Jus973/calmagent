"""JSONL read/append helpers. Every log this repo writes is append-only JSONL (§0)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    """Rows of a JSONL file; an absent file reads as no rows, never as an error."""
    p = Path(path)
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    Path(path).write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def append_jsonl(path: str | Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
