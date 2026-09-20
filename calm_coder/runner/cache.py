"""Cross-run outcome cache: (test class, the bindings it can reach) -> result.

Definition identity is a content hash and the test module is fixed, so a test class's result on a
composition depends only on the fills it can reach. The cache is append-only JSONL, keyed by that
content hash, so reusing an entry can only add facts the store would have derived anyway. Reused
outcomes carry `reused:` in their detail — the log never claims we executed a test we didn't.
"""
from __future__ import annotations

from pathlib import Path

from calm_coder.jsonl import append_jsonl, read_jsonl


class OutcomeCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._d: dict[str, dict] = {}
        for row in read_jsonl(self.path):
            self._d.setdefault(row["key"], row)

    def get(self, key: str) -> dict | None:
        return self._d.get(key)

    def put(self, key: str, test_id: str, result: str, detail: str = "") -> None:
        if key in self._d:
            return
        row = {"key": key, "test_id": test_id, "result": result, "detail": detail[:512]}
        self._d[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(self.path, row)
