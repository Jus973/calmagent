"""Grow-only store (I1): exactly two writes, both idempotent set inserts, plus an append-only log."""
from __future__ import annotations

import time
from typing import FrozenSet, Iterable, Mapping

from calm_coder.store.defs import Definition, Outcome


class Store:
    def __init__(self) -> None:
        self._defs: dict[str, Definition] = {}
        self._by_slot: dict[str, dict[str, Definition]] = {}
        self._outcomes: dict[Outcome, Outcome] = {}
        self._log: list[dict] = []

    # ---- the only two writes
    def add_def(self, d: Definition) -> bool:
        if d.hash in self._defs:
            return False
        self._defs[d.hash] = d
        if d.kind == "fill":
            self._by_slot.setdefault(d.slot, {})[d.hash] = d
        self._log.append({"t": time.time(), "kind": "def", "data": d.to_json()})
        return True

    def add_outcome(self, o: Outcome) -> bool:
        if o in self._outcomes:
            return False
        self._outcomes[o] = o
        self._log.append({"t": time.time(), "kind": "outcome", "data": o.to_json()})
        return True

    # ---- reads
    def get(self, h: str) -> Definition | None:
        return self._defs.get(h)

    def defs(self) -> FrozenSet[Definition]:
        return frozenset(self._defs.values())

    def defs_for_slot(self, slot_id: str) -> FrozenSet[Definition]:
        return frozenset(self._by_slot.get(slot_id, {}).values())

    def helpers(self) -> Mapping[str, Definition]:
        return {h: d for h, d in self._defs.items() if d.kind == "helper"}

    def outcomes(self) -> FrozenSet[Outcome]:
        return frozenset(self._outcomes)

    def event_log(self) -> list[dict]:
        return list(self._log)

    @classmethod
    def from_events(cls, events: Iterable[dict]) -> "Store":
        s = cls()
        for e in events:
            if e["kind"] == "def":
                s.add_def(Definition.from_json(e["data"]))
            elif e["kind"] == "outcome":
                s.add_outcome(Outcome.from_json(e["data"]))
        return s
