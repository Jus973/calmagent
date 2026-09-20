"""Post-hoc, zero-token analyses over a finished run (NOT pre-registered; reported separately from H1–H6).

1. ceiling:   exhaustive search over every combination of the fills CALM already sampled (N=8).
              Separates "the search policy missed a working combination" from "none existed".
2. recombine: split each of C's whole-class samples into per-method fills, store them in a fresh
              grow-only store, and search. Same tokens as C@8; tests recombination at method granularity.

python -m calm_coder.bench.posthoc runs/<dir> [--cap 3000] [--par 6]
Writes runs/<dir>/posthoc/{results.jsonl, events/*.jsonl}; never touches the run's own logs.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import itertools
import time
from pathlib import Path

from calm_coder.agents.scheduler import Scheduler
from calm_coder.bench.classeval import load_rows, task_from_row
from calm_coder.jsonl import append_jsonl, read_jsonl, write_jsonl
from calm_coder.runner.sandbox import run_tests_async
from calm_coder.serve.extract import ExtractError, extract_class
from calm_coder.store.defs import Composition, Outcome, emission_to_defs
from calm_coder.store.derive import verified
from calm_coder.store.materialize import materialize
from calm_coder.store.store import Store


async def exhaustive(task, store: Store, cands: dict[str, list[str]], cap: int, par: int) -> dict:
    """Try combinations (rank order: fills with stub-context passes first) until one verifies or cap."""
    slots = [s.id for s in task.slots]
    combos = itertools.product(*(cands[s] for s in slots))
    total = 1
    for s in slots:
        total *= len(cands[s])
    tried, found, t0 = 0, None, time.monotonic()
    sem = asyncio.Semaphore(par)

    async def one(bind: dict[str, str]):
        comp = Composition.make(bind)
        async with sem:
            res = await run_tests_async(materialize(task, store, comp), task.test_src, list(task.test_classes))
        for t, r in res.items():
            store.add_outcome(Outcome(t, comp.id, r["result"], detail=r.get("detail", "")[:512],
                                      wall_ms=r.get("wall_ms", 0), bindings=comp.bindings))
        return comp.id

    while found is None and tried < cap:
        batch = [dict(zip(slots, c)) for c in itertools.islice(combos, min(par * 4, cap - tried))]
        if not batch:
            break
        await asyncio.gather(*(one(b) for b in batch))
        tried += len(batch)
        v = verified(store, task)
        found = next(iter(sorted(v)), None)
    return {"space": total, "tried": tried, "verified": found is not None,
            "exhausted": tried >= total, "wall_s": round(time.monotonic() - t0, 1)}


def _ranked_cands(task, store: Store, stub_pass: set[str]) -> dict[str, list[str]]:
    return {s.id: sorted((d.hash for d in store.defs_for_slot(s.id)), key=lambda h: (h not in stub_pass, h))
            for s in task.slots}


async def ceiling(task, events: list[dict], cap: int, par: int) -> dict:
    store = Store.from_events(events)
    stub_pass = {o.bindings[0][1] for o in store.outcomes() if o.result == "pass" and len(o.bindings) == 1}
    fresh = Store.from_events([e for e in events if e["kind"] == "def"])
    cands = _ranked_cands(task, fresh, stub_pass)
    if any(not v for v in cands.values()):
        return {"space": 0, "tried": 0, "verified": False, "exhausted": True, "wall_s": 0.0}
    return await exhaustive(task, fresh, cands, cap, par) | {"events": fresh.event_log()}


async def recombine(task, samples: list[dict], cap: int, par: int, max_comps: int) -> dict:
    store = Store()
    for s in samples:
        src = extract_class(s["text"], task)
        if isinstance(src, ExtractError):
            continue
        try:
            mod = ast.parse(src)
        except SyntaxError:
            continue
        cdef = next((n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == task.class_name), None)
        if cdef is None:
            continue
        fns = {n.name: n for n in cdef.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        extra = [f for n, f in fns.items() if n not in task.slot_ids and n != "__init__"]
        for slot in task.slots:
            if slot.id in fns:
                for d in emission_to_defs(task, slot, [fns[slot.id], *extra],
                                          {"from": "c_sample", "sample_idx": s["sample_idx"]}):
                    store.add_def(d)
    cands = {s.id: sorted(d.hash for d in store.defs_for_slot(s.id)) for s in task.slots}
    if any(not v for v in cands.values()):
        return {"policy": {"verified": False, "comps": 0}, "exhaustive": {"verified": False, "space": 0, "tried": 0},
                "distinct_fills": {k: len(v) for k, v in cands.items()}, "events": store.event_log()}
    # same policy as the CALM arm (stub tests, then blame-driven search), then the exhaustive ceiling
    sched = Scheduler(task, store, max_comps=max_comps)
    await sched.phase1([h for hs in cands.values() for h in hs])
    res = await sched.search(cands, {}, {})
    stub_pass = {h for h, r in sched._stub.items() if r == "pass"}
    ex = await exhaustive(task, store, _ranked_cands(task, store, stub_pass), cap, par) \
        if res.verified_comp is None else {"verified": True, "space": None, "tried": 0}
    return {"policy": {"verified": res.verified_comp is not None, "comps": len(res.trace)}, "exhaustive": ex,
            "distinct_fills": {k: len(v) for k, v in cands.items()}, "events": store.event_log()}


async def main_async(d: Path, cap: int, par: int) -> None:
    out = d / "posthoc"
    (out / "events").mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r["analysis"]) for r in read_jsonl(out / "results.jsonl")}
    rows = {r["task_id"]: r for r in load_rows()}
    results = read_jsonl(d / "results.jsonl")
    calm8 = {r["task_id"]: r for r in results if r["arm"] == "calm" and r["N"] == 8 and r["seed"] == 0}
    c8 = {r["task_id"]: r for r in results if r["arm"] == "c" and r["N"] == 8 and r["seed"] == 0}
    samples = {}
    for s in read_jsonl(d / "class_samples.jsonl"):
        if s["arm"] == "c" and s["seed"] == 0:
            samples.setdefault(s["task_id"], {})[s["sample_idx"]] = s
    for tid in sorted(set(calm8) & set(c8), key=lambda t: int(t.split("_")[1])):
        task = task_from_row(rows[tid])
        if (tid, "ceiling") not in done:
            ev = read_jsonl(d / "events" / f"{tid}__calm__s0.jsonl")
            r = await ceiling(task, ev, cap, par)
            write_jsonl(out / "events" / f"{tid}__ceiling.jsonl", r.pop("events", []))
            row = {"task_id": tid, "analysis": "ceiling", "calm_solved": calm8[tid]["solved"],
                   "calm_comps": calm8[tid]["comps_tested"], **r}
            append_jsonl(out / "results.jsonl", row)
            print(row)
        if (tid, "recombine") not in done:
            r = await recombine(task, list(samples.get(tid, {}).values()), cap, par, max_comps=64)
            write_jsonl(out / "events" / f"{tid}__recombine.jsonl", r.pop("events", []))
            row = {"task_id": tid, "analysis": "recombine", "c8_solved": c8[tid]["solved"],
                   "c8_n_pass": c8[tid]["n_pass"], **r}
            append_jsonl(out / "results.jsonl", row)
            print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--par", type=int, default=6)
    a = ap.parse_args()
    asyncio.run(main_async(Path(a.run_dir), a.cap, a.par))


if __name__ == "__main__":
    main()
