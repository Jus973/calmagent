"""calm solve — skeleton + tests in, verified class out (§3.18). A thin wrapper; no new logic.

python -m calm_coder.cli solve path/to/skeleton.py --tests path/to/test_x.py --N 4 [--live] [--out solved.py]
Exit 0 on verified; 1 when the budget is exhausted (the failure exit waits for every phase: it's the one barrier).

The default path is latency-first: stub tests start while later samples are still decoding, the search
tests several compositions at once, and the ∃ exit stops generation as soon as anything verifies.
`--sequential` is the experiment's phase-by-phase path.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

from rich.console import Console

from calm_coder.agents.fill import Emission, generate_fills
from calm_coder.agents.scheduler import Scheduler, SearchResult
from calm_coder.bench.experiment import budget_view
from calm_coder.jsonl import write_jsonl
from calm_coder.runner.cache import OutcomeCache
from calm_coder.serve.client import Client
from calm_coder.store.defs import Composition
from calm_coder.store.materialize import STUB_EXC, materialize
from calm_coder.store.store import Store
from calm_coder.task import task_from_files, task_from_implementation
from calm_coder.v2 import state
from calm_coder.v2.harness import run_from_seed

console = Console(stderr=True)


async def _pipelined(client: Client, task, store: Store, sched: Scheduler, *, n: int, seed: int,
                     cb, stop_at_fill: bool = True, adaptive: bool = True) -> SearchResult:
    """Phases 0-2 overlapped: fills are stub-tested as they land and composed as soon as every slot has
    one, so a verified composition can arrive before the budget is spent. Late emissions are inert, so
    cancelling generation after the ∃ exit only costs facts nobody needed."""
    emissions: list[Emission] = []
    stubs: dict[str, asyncio.Task] = {}
    landed = asyncio.Event()                       # a fill arrived: compose again without polling for it

    def on_emission(e: Emission) -> None:
        emissions.append(e)
        if e.fill_hash and e.fill_hash not in stubs:
            stubs[e.fill_hash] = asyncio.ensure_future(sched.stub_test(e.fill_hash))
        landed.set()

    gen = asyncio.ensure_future(generate_fills(client, task, store, n=n, seed=seed, on_event=cb,
                                               on_emission=on_emission,
                                               skip_slot=sched.has_stub_pass if adaptive else None,
                                               stop_at_fill=stop_at_fill))
    res: SearchResult | None = None
    try:
        while not gen.done():
            landed.clear()                         # nothing new to compose until another fill lands
            if all(store.defs_for_slot(s.id) for s in task.slots):
                _, cands, fan_in, arrival = budget_view(emissions, n)
                res = await sched.search(cands, fan_in, arrival)
                if res.verified_comp:
                    gen.cancel()
                    break
            waiter = asyncio.ensure_future(landed.wait())
            await asyncio.wait({waiter, gen}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await gen
    if res is not None and res.verified_comp:
        for t in stubs.values():
            t.cancel()
        await asyncio.gather(*stubs.values(), return_exceptions=True)
        await sched.drain()                        # the tests themselves finish; their outcomes are facts
        return res
    await asyncio.gather(*stubs.values())
    _, cands, fan_in, arrival = budget_view(emissions, n)
    return await sched.search(cands, fan_in, arrival)


async def solve(task, *, n: int, seed: int, live: bool, max_comps: int, events_out: Path | None,
                sequential: bool = False, width: int = 4, cache: Path | None = None,
                stop_at_fill: bool = True):
    store = Store()
    view = None
    if live:
        from calm_coder.viz.live import LiveView
        view = LiveView(task, store, title=f"calm solve · {task.task_id} · N={n}")
        view.__enter__()
    try:
        cb = view.on_event if view else None
        sched = Scheduler(task, store, max_comps=max_comps, on_event=cb,
                          width=1 if sequential else width, reuse_class_outcomes=not sequential,
                          cache=OutcomeCache(cache) if cache else None, fail_fast=not sequential)
        async with Client() as client:
            if sequential:
                emissions = await generate_fills(client, task, store, n=n, seed=seed, on_event=cb)
                _, cands, fan_in, arrival = budget_view(emissions, n)
                await sched.phase1([h for hs in cands.values() for h in hs])
                res = await sched.search(cands, fan_in, arrival)
            else:
                res = await _pipelined(client, task, store, sched, n=n, seed=seed, cb=cb,
                                       stop_at_fill=stop_at_fill)
    finally:
        if view:
            view.__exit__(None, None, None)
    if events_out:
        write_jsonl(events_out, store.event_log())
    if res.verified_comp is None:
        return None, res
    tr = next(t for t in res.trace if t.comp_id == res.verified_comp)
    src = materialize(task, store, Composition.make(tr.bindings))
    return src.replace(STUB_EXC, "", 1).lstrip("\n"), res


def _materialize_verified(task, res):
    vid = res.row.get("verified_comp")
    if not vid:
        return None
    binds = state.comp_bindings(res.store).get(vid)
    if not binds:
        return None
    src = materialize(task, res.store, Composition.make(binds))
    return src.replace(STUB_EXC, "", 1).lstrip("\n")


async def repair(task, seed_src: str, *, n: int, seed: int, live: bool, budget_tokens: int,
                 rewrite: bool, events_out: Path | None, rounds: int = 3):
    """Keep methods that already pass. Sample only dead slots, unless `--rewrite` (the baseline)."""
    store = Store()
    view = None
    if live:
        from calm_coder.viz.live import LiveView
        view = LiveView(task, store, title=f"calm repair · {task.task_id}")
        view.__enter__()
    try:
        cb = view.on_event if view else None
        async with Client() as client:
            res = await run_from_seed(
                client, task, seed_src, mode="rewrite" if rewrite else "repair",
                budget_tokens=budget_tokens, seed=seed, repair_n=n, rounds=rounds,
                warm=False, store=store, on_event=cb)
    finally:
        if view:
            view.__exit__(None, None, None)
    if events_out:
        write_jsonl(events_out, res.store.event_log())
    return _materialize_verified(task, res), res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="calm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="fill every method of a class skeleton until the tests pass")
    s.add_argument("skeleton")
    s.add_argument("--tests", required=True)
    s.add_argument("--N", type=int, default=4, help="samples per method")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-comps", type=int, default=64)
    s.add_argument("--width", type=int, default=4, help="compositions tested at once in the search")
    s.add_argument("--sequential", action="store_true",
                   help="the experiment's path: generate everything, then phase 1, then search")
    s.add_argument("--no-stop-at-fill", action="store_true",
                   help="decode every completion to its end instead of stopping at the method")
    s.add_argument("--cache", help="JSONL outcome cache reused across runs (keyed by content hashes)")
    s.add_argument("--live", action="store_true")
    s.add_argument("--out")
    s.add_argument("--events", help="write the store's event log here (for --replay / --replay-shuffled)")
    r = sub.add_parser("repair", help="keep passing methods of an existing class; resample only dead slots")
    r.add_argument("implementation")
    r.add_argument("--tests", required=True)
    r.add_argument("--N", type=int, default=4, help="samples per repair round")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--budget-tokens", type=int, default=4000)
    r.add_argument("--rewrite", action="store_true",
                   help="baseline: regenerate the whole class (the usual agent loop)")
    r.add_argument("--live", action="store_true")
    r.add_argument("--out")
    r.add_argument("--events", help="write the store's event log here")
    a = ap.parse_args(argv)
    if a.cmd == "solve":
        task = task_from_files(a.skeleton, a.tests)
        src, res = asyncio.run(solve(task, n=a.N, seed=a.seed, live=a.live, max_comps=a.max_comps,
                                     events_out=Path(a.events) if a.events else None,
                                     sequential=a.sequential, width=a.width,
                                     cache=Path(a.cache) if a.cache else None,
                                     stop_at_fill=not a.no_stop_at_fill))
        if src is None:
            console.print(f"[red]no verified composition within budget ({res.unsolvable_reason}, "
                          f"{len(res.trace)} compositions tested)[/red]")
            return 1
        console.print(f"[green]verified after {len(res.trace)} composition(s), "
                      f"{res.test_wall_ms} ms wall in the search[/green]")
    else:
        task, seed_src = task_from_implementation(a.implementation, a.tests)
        src, res = asyncio.run(repair(task, seed_src, n=a.N, seed=a.seed, live=a.live,
                                      budget_tokens=a.budget_tokens, rewrite=a.rewrite,
                                      events_out=Path(a.events) if a.events else None,
                                      rounds=a.rounds))
        row = res.row
        if src is None:
            console.print(f"[red]no verified composition within budget "
                          f"(dead slots {row.get('dead_slots')}, "
                          f"{row.get('decode_tokens')} decode tokens)[/red]")
            return 1
        kept = row.get("seed_slots_kept") or []
        mode = "rewrite" if a.rewrite else "repair"
        console.print(f"[green]{mode} verified; kept {len(kept)}/{len(task.slots)} seed methods "
                      f"{kept}; {row.get('decode_tokens')} decode tokens[/green]")
    if a.out:
        Path(a.out).write_text(src)
        console.print(f"wrote {a.out}")
    else:
        sys.stdout.write(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
