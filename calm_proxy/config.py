from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

KNOWN_MIDDLEWARES = ("trace", "prefix", "dedup", "memo", "affinity")

# Names that are recognised but have no middleware behind them yet. Enabling one used to start the
# proxy happily and do nothing, which during a demo is indistinguishable from the lever not
# working -- `--enable dedup` would report `replaced: 0` on every request and look like a null
# result. Refuse instead, and name the implementation that does have it.
UNIMPLEMENTED: dict[str, str] = {
    "dedup": "bench_agent.trace_proxy --dedup",
    "prefix": "bench_agent/probe_headroom.py and bench_agent/cross_run.py (offline, on a trace)",
    "memo": "",
    "affinity": "",
}


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
    missing = sorted(names & set(UNIMPLEMENTED))
    if missing:
        hints = "; ".join(
            f"{n} -> {UNIMPLEMENTED[n]}" if UNIMPLEMENTED[n] else f"{n} -> not built"
            for n in missing
        )
        raise ValueError(
            f"middleware not implemented in calm_proxy yet: {','.join(missing)}. "
            f"Enabling it would silently do nothing. Use: {hints}"
        )
    return frozenset(names)
