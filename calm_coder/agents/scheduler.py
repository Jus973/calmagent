"""Phases 1–2 (§3.12). The only place ordering policy lives.

Everything read here is a monotone set (store defs/outcomes); everything written goes through
add_outcome. The policy decides *when* facts appear, never *which* facts are derivable.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator, Callable, Mapping, Sequence

from calm_coder.runner import tests as rt
from calm_coder.runner.cache import OutcomeCache
from calm_coder.runner.sandbox import run_tests_async
from calm_coder.store.defs import Composition, Outcome, sha256
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
    test_wall_ms: int = 0            # wall clock of the search itself (< test_ms once width > 1)
    unsolvable_reason: str | None = None


class Scheduler:
    def __init__(self, task: "Task", store: Store, *, max_comps: int = 64, per_class_timeout_s: float = 5,
                 wall_timeout_s: float = 20, on_event: Callable[[dict], None] | None = None,
                 width: int = 1, reuse_class_outcomes: bool = False, cache: OutcomeCache | None = None,
                 fail_fast: bool = False):
        self.task, self.store = task, store
        self.max_comps = max_comps
        self.per_class, self.wall = per_class_timeout_s, wall_timeout_s
        self.on_event = on_event
        self.width = max(1, width)
        self.fail_fast = fail_fast
        self.reuse_class_outcomes = reuse_class_outcomes or cache is not None
        self.cache = cache
        # scheduling caches (not facts; facts live in the store)
        self._full: dict[str, tuple[dict[str, str], int]] = {}
        self._stub: dict[str, str] = {}
        self._details: dict[str, dict[str, str]] = {}
        self._class_results: dict[str, tuple[str, str]] = {}   # class key -> (result, detail)
        self._stub_inflight: dict[str, asyncio.Future] = {}
        self._test_src_hash = sha256(task.test_src)

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
        fut = self._stub_inflight.get(fill_hash)
        if fut is None:
            fut = self._stub_inflight[fill_hash] = asyncio.ensure_future(self._stub_test(fill_hash))
        return await asyncio.shield(fut)

    async def _stub_test(self, fill_hash: str) -> str:
        d = self.store.get(fill_hash)
        t = rt.slot_test_class(self.task, d.slot)
        comp = Composition.make({d.slot: d.hash})
        if t is None:
            self._stub[fill_hash] = "inconclusive"
            return "inconclusive"
        missing = rt.predicted_inconclusive(self.task, t, {d.slot})
        hit = self._reused(t, comp)
        if missing:
            res = {t: {"result": "inconclusive", "detail": f"predicted: needs {sorted(missing)}", "wall_ms": 0}}
        elif hit is not None:
            res = {t: {"result": hit[0], "detail": hit[1], "wall_ms": 0}}
        else:
            src = materialize(self.task, self.store, comp)
            res = await run_tests_async(src, self.task.test_src, [t], per_class_timeout_s=self.per_class,
                                        wall_timeout_s=self.wall)
            key = self.class_key(t, comp)
            self._class_results[key] = (res[t]["result"], res[t].get("detail", "")[:512])
            if self.cache is not None:
                self.cache.put(key, t, res[t]["result"], res[t].get("detail", ""))
        self._record(comp, res)
        self._stub[fill_hash] = res[t]["result"]
        return res[t]["result"]

    async def phase1(self, fill_hashes: Sequence[str]) -> dict[str, str]:
        hs = sorted(set(fill_hashes))
        results = await asyncio.gather(*(self.stub_test(h) for h in hs))
        return dict(zip(hs, results))

    async def drain(self) -> None:
        """Await stub tests still in flight — a latency-first caller stops waiting on them at the ∃ exit."""
        if self._stub_inflight:
            await asyncio.gather(*self._stub_inflight.values(), return_exceptions=True)

    def has_stub_pass(self, slot_id: str) -> bool:
        """Scheduling input: does some fill of this slot already pass its slot test in stub context?"""
        return any(self._stub.get(d.hash) == "pass" for d in self.store.defs_for_slot(slot_id))

    # ---------------------------------------------------------------- phase 2
    def class_key(self, test_id: str, comp: Composition) -> str:
        """Content hash of everything `test_id` can reach in `comp`: the test module, the slots it
        calls closed under the bound fills' own calls, and which of those are stubbed."""
        b = comp.binding
        slots = self._callee_closure(rt.test_slot_deps(self.task, test_id), b)
        bound = sorted((s, b[s]) for s in slots if s in b)
        stubbed = sorted(s for s in slots if s not in b)
        return sha256(f"{self._test_src_hash}|{test_id}|{bound}|{stubbed}")

    def _reused(self, test_id: str, comp: Composition) -> tuple[str, str] | None:
        if not self.reuse_class_outcomes:
            return None
        key = self.class_key(test_id, comp)
        hit = self._class_results.get(key)
        if hit is None and self.cache is not None:
            row = self.cache.get(key)
            if row is not None:
                hit = (row["result"], row.get("detail", ""))
        if hit is None:
            return None
        return hit[0], f"reused: same reachable fills ({hit[1]})"[:2048]

    async def run_full(self, comp: Composition) -> tuple[dict[str, str], int, bool]:
        if comp.id in self._full:
            res, ms = self._full[comp.id]
            return res, ms, True
        t0 = time.monotonic()
        reused = {t: r for t in self.task.test_classes if (r := self._reused(t, comp)) is not None}
        pending = [t for t in self.task.test_classes if t not in reused]
        raw: dict[str, dict] = {t: {"result": r, "detail": d, "wall_ms": 0} for t, (r, d) in reused.items()}
        if pending:
            src = materialize(self.task, self.store, comp)
            raw |= await self._run_classes(src, pending)
            for t in [t for t in pending if t in raw]:
                key = self.class_key(t, comp)
                self._class_results[key] = (raw[t]["result"], raw[t].get("detail", "")[:512])
                if self.cache is not None:
                    self.cache.put(key, t, raw[t]["result"], raw[t].get("detail", ""))
        ms = int((time.monotonic() - t0) * 1000)
        self._record(comp, raw)
        res = {t: r["result"] for t, r in raw.items()}
        self._full[comp.id] = (res, ms)
        self._details[comp.id] = {t: r.get("detail", "") for t, r in raw.items()}
        return res, ms, not pending

    async def _race(self, batch: list[Composition]) -> AsyncIterator[tuple[Composition, tuple[dict[str, str], int, bool]]]:
        """Run the batch concurrently, yielding each composition as it finishes so the ∃ exit doesn't wait
        on the slowest one. Abandoning a run in flight only means its outcomes arrive later or not at all."""
        if len(batch) == 1:
            yield batch[0], await self.run_full(batch[0])
            return

        async def one(c: Composition) -> tuple[Composition, tuple[dict[str, str], int, bool]]:
            return c, await self.run_full(c)

        futs = [asyncio.ensure_future(one(c)) for c in batch]
        try:
            for fin in asyncio.as_completed(futs):
                yield await fin
        finally:
            for f in futs:
                f.cancel()
            await asyncio.gather(*futs, return_exceptions=True)

    async def _run_classes(self, src: str, pending: list[str]) -> dict[str, dict]:
        """One subprocess per class, racing, stopping at the first non-pass: a composition needs *every*
        class to pass, so once one doesn't, the rest only add facts nobody is waiting for. Off by default
        (the experiment's trace rows need the full result map); `calm solve` turns it on."""
        if not self.fail_fast or len(pending) == 1:
            return await run_tests_async(src, self.task.test_src, pending,
                                         per_class_timeout_s=self.per_class, wall_timeout_s=self.wall)

        async def one(t: str) -> dict[str, dict]:
            return await run_tests_async(src, self.task.test_src, [t],
                                         per_class_timeout_s=self.per_class, wall_timeout_s=self.wall)

        futs = [asyncio.ensure_future(one(t)) for t in pending]
        out: dict[str, dict] = {}
        try:
            for fin in asyncio.as_completed(futs):
                r = await fin
                out |= r
                if any(v["result"] != "pass" for v in r.values()):
                    break
        finally:
            for f in futs:
                f.cancel()
            for r in await asyncio.gather(*futs, return_exceptions=True):   # already-finished races still count
                if isinstance(r, dict):
                    out |= r
        return out

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
        t_search = time.monotonic()
        while frontier and len(tried) < self.max_comps:
            # Frontier compositions are independent facts, so `width` of them race; the first verified
            # one to *finish* is the exit. Racing cannot change which facts are derivable.
            batch: list[Composition] = []
            while frontier and len(batch) < self.width and len(tried) + len(batch) < self.max_comps:
                comp = frontier.popleft()
                if comp.id not in tried:
                    batch.append(comp)
            if not batch:
                break
            tried |= {c.id for c in batch}
            expand: list[tuple[Composition, dict[str, str]]] = []
            async with contextlib.aclosing(self._race(batch)) as runs:
                async for comp, (res, ms, cached) in runs:
                    out.test_ms += ms
                    b = comp.binding
                    out.trace.append(TraceRow(
                        comp_id=comp.id, bindings=b, results=res, wall_ms=ms, cached=cached,
                        all_stub_pass=all(self._stub.get(h) == "pass" for h in b.values()),
                        slot_level_pass=all(res.get(t) == "pass" for t in own),
                        class_level_pass=all(res.get(t) == "pass" for t in class_level) if class_level else None,
                    ))
                    self._emit({"kind": "comp", "comp": comp, "results": res})
                    if comp.id in verified(self.store, task):       # ∃ verified — the exit
                        out.verified_comp = comp.id
                        break
                    expand.append((comp, res))
            if out.verified_comp is not None:
                break
            for comp, res in expand:
                b = comp.binding
                failing = [t for t, r in res.items() if r != "pass"]
                details = self._details.get(comp.id, {})
                implicated: set[str] = set()
                for t in failing:
                    implicated |= rt.test_slot_deps(task, t) | rt.slots_in_traceback(task, details.get(t, ""))
                implicated = self._callee_closure(implicated, b) or set(slot_ids)

                def remaining(s: str, comp: Composition = comp) -> int:
                    return sum(1 for h in ranked[s] if comp.with_(s, h).id not in tried)

                for s in sorted(implicated, key=lambda s: (remaining(s), s)):
                    alt = next((comp.with_(s, h) for h in ranked[s]
                                if comp.with_(s, h).id not in tried and comp.with_(s, h).id not in queued), None)
                    if alt is not None:
                        frontier.append(alt)
                        queued.add(alt.id)
        out.test_wall_ms = int((time.monotonic() - t_search) * 1000)
        if out.verified_comp is None and not out.unsolvable_reason:
            out.unsolvable_reason = "budget_exhausted" if frontier else "frontier_exhausted"
        return out
