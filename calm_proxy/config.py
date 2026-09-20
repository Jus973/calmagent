from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

KNOWN_MIDDLEWARES = ("trace", "prefix", "dedup", "memo", "affinity")


@dataclass
class Config:
    upstream: str = "http://127.0.0.1:11434"
    port: int = 8787
    host: str = "127.0.0.1"
    trace_dir: Path | None = None
    enabled: frozenset[str] = frozenset({"trace"})
    dedup_min_bytes: int = 200
    timeout_s: float = 600.0

    def has(self, name: str) -> bool:
        return name in self.enabled


def parse_enabled(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    names = {part.strip() for part in value.split(",") if part.strip()}
    unknown = names - set(KNOWN_MIDDLEWARES)
    if unknown:
        raise ValueError(
            f"unknown middleware(s): {','.join(sorted(unknown))}; "
            f"known: {','.join(KNOWN_MIDDLEWARES)}"
        )
    return frozenset(names)
