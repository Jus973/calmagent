"""Phases 1–2 (§3.12). The only place ordering policy lives.

Everything read here is a monotone set (store defs/outcomes); everything written goes through
add_outcome. The policy decides *when* facts appear, never *which* facts are derivable.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

from calm_coder.runner import tests as rt
from calm_coder.runner.sandbox import run_tests_async
from calm_coder.store.defs import Composition, Outcome
from calm_coder.store.derive import slot_evidence, verified
from calm_coder.store.materialize import materialize
from calm_coder.store.store import Store

if TYPE_CHECKING:
    from calm_coder.task import Task

BAD = ("fail", "error", "timeout")


@dataclass
class TraceRow:
    comp_id: str
    bindings: dict[str, str]
    results: dict[str, str]          # test class -> result
    wall_ms: int
    cached: bool
    all_stub_pass: bool              # every bound fill passed its slot test in stub context
    slot_level_pass: bool            # all method-level test classes pass
    class_level_pass: bool | None    # all class-level classes pass (None if the task has none)


@dataclass
class SearchResult:
    verified_comp: str | None
    trace: list[TraceRow] = field(default_factory=list)
    test_ms: int = 0                 # sum of full-module run times until the exit (cached runs count their recorded time)
    unsolvable_reason: str | None = None


class Scheduler:
    def __init__(self, task: "Task", store: Store, *, max_comps: int = 64, per_class_timeout_s: float = 5,
                 wall_timeout_s: float = 20, on_event: Callable[[dict], None] | None = None):
        self.task, self.store = task, store
        self.max_comps = max_comps
        self.per_class, self.wall = per_class_timeout_s, wall_timeout_s
        self.on_event = on_event
        # scheduling caches (not facts; facts live in the store)
        self._full: dict[str, tuple[dict[str, str], int]] = {}
        self._stub: dict[str, str] = {}
        self._details: dict[str, dict[str, str]] = {}

    def _emit(self, ev: dict) -> None:
        if self.on_event:
            self.on_event(ev)

    def _record(self, comp: Composition, res: Mapping[str, dict]) -> None:
        for t, r in res.items():
            new = self.store.add_outcome(Outcome(t, comp.id, r["result"], detail=r.get("detail", "")[:2048],
                                                 wall_ms=r.get("wall_ms", 0), bindings=comp.bindings))
            self._emit({"kind": "outcome", "new": new, "test": t, "result": r["result"], "comp": comp})

    # ---------------------------------------------------------------- phase 1
    async def stub_test(self, fill_hash: str) -> str:
        """Run only the fill's own slot test with every other slot stubbed."""
        if fill_hash in self._stub:
            return self._stub[fill_hash]
        d = self.store.get(fill_hash)
        t = rt.slot_test_class(self.task, d.slot)
        comp = Composition.make({d.slot: d.hash})
        if t is None:
            self._stub[fill_hash] = "inconclusive"
            return "inconclusive"
        missing = rt.predicted_inconclusive(self.task, t, {d.slot})
        if missing:
            res = {t: {"result": "inconclusive", "detail": f"predicted: needs {sorted(missing)}", "wall_ms": 0}}
        else:
            src = materialize(self.task, self.store, comp)
            res = await run_tests_async(src, self.task.test_src, [t], per_class_timeout_s=self.per_class,
                                        wall_timeout_s=self.wall)
        self._record(comp, res)
        self._stub[fill_hash] = res[t]["result"]
        return res[t]["result"]

    async def phase1(self, fill_hashes: Sequence[str]) -> dict[str, str]:
        hs = sorted(set(fill_hashes))
        results = await asyncio.gather(*(self.stub_test(h) for h in hs))
        return dict(zip(hs, results))

    # ---------------------------------------------------------------- phase 2
    async def run_full(self, comp: Composition) -> tuple[dict[str, str], int, bool]:
        if comp.id in self._full:
            res, ms = self._full[comp.id]
            return res, ms, True
        t0 = time.monotonic()
        src = materialize(self.task, self.store, comp)
        raw = await run_tests_async(src, self.task.test_src, list(self.task.test_classes),
                                    per_class_timeout_s=self.per_class, wall_timeout_s=self.wall)
        ms = int((time.monotonic() - t0) * 1000)
        self._record(comp, raw)
        res = {t: r["result"] for t, r in raw.items()}
        self._full[comp.id] = (res, ms)
        self._details[comp.id] = {t: r.get("detail", "") for t, r in raw.items()}
        return res, ms, False

    def _callee_closure(self, slots: set[str], bindings: Mapping[str, str]) -> set[str]:
        """A failing slot also implicates every slot its bound fill calls (through helpers too)."""
        out, todo = set(), list(slots)
        while todo:
            s, todo = todo[-1], todo[:-1]
            if s in out:
                continue
            out.add(s)
            if s not in bindings:
                continue
            hs, seen = [bindings[s]], set()
            while hs:
                h, hs = hs[-1], hs[:-1]
                d = self.store.get(h)
                if d is None or h in seen:
                    continue
                seen.add(h)
                todo += sorted(d.slot_refs)
                hs += sorted(d.helper_refs)
        return out

    def rank(self, fill_hash: str, fan_in: Mapping[str, int], arrival: Mapping[str, int]) -> tuple:
        ev = slot_evidence(self.store, self.task, self.store.get(fill_hash))
        bad = ev["pass"] == 0 and sum(ev[k] for k in BAD) > 0
        return (bad, -ev["pass"], -ev["inconclusive"], -fan_in.get(fill_hash, 0), arrival.get(fill_hash, 1 << 30))

    async def search(self, cands: Mapping[str, Sequence[str]], fan_in: Mapping[str, int],
                     arrival: Mapping[str, int]) -> SearchResult:
        """cands: slot -> distinct fill hashes available at this budget."""
        task = self.task
        out = SearchResult(verified_comp=None)
        empty = [s.id for s in task.slots if not cands.get(s.id)]
        if empty:
            out.unsolvable_reason = f"no_fill_for: {empty}"
            return out
        ranked = {s: sorted(set(hs), key=lambda h: self.rank(h, fan_in, arrival)) for s, hs in cands.items()}
        slot_ids = [s.id for s in task.slots]
        class_level = rt.class_level_tests(task)
        own = {t for t in task.slot_tests.values()}
        start = Composition.make({s: ranked[s][0] for s in slot_ids})
        frontier: deque[Composition] = deque([start])
        queued = {start.id}
        tried: set[str] = set()
        while frontier and len(tried) < self.max_comps:
            comp = frontier.popleft()
            if comp.id in tried:
                continue
            tried.add(comp.id)
            res, ms, cached = await self.run_full(comp)
            out.test_ms += ms
            b = comp.binding
            out.trace.append(TraceRow(
                comp_id=comp.id, bindings=b, results=res, wall_ms=ms, cached=cached,
                all_stub_pass=all(self._stub.get(h) == "pass" for h in b.values()),
                slot_level_pass=all(res.get(t) == "pass" for t in own),
                class_level_pass=all(res.get(t) == "pass" for t in class_level) if class_level else None,
            ))
            self._emit({"kind": "comp", "comp": comp, "results": res})
            if comp.id in verified(self.store, task):      # ∃ verified — the exit
                out.verified_comp = comp.id
                break
            failing = [t for t, r in res.items() if r != "pass"]
            details = self._details.get(comp.id, {})
            implicated: set[str] = set()
            for t in failing:
                implicated |= rt.test_slot_deps(task, t) | rt.slots_in_traceback(task, details.get(t, ""))
            implicated = self._callee_closure(implicated, b) or set(slot_ids)

            def remaining(s: str) -> int:
                return sum(1 for h in ranked[s] if comp.with_(s, h).id not in tried)

            for s in sorted(implicated, key=lambda s: (remaining(s), s)):
                alt = next((comp.with_(s, h) for h in ranked[s]
                            if comp.with_(s, h).id not in tried and comp.with_(s, h).id not in queued), None)
                if alt is not None:
                    frontier.append(alt)
                    queued.add(alt.id)
        if out.verified_comp is None and not out.unsolvable_reason:
            out.unsolvable_reason = "budget_exhausted" if frontier else "frontier_exhausted"
        return out
