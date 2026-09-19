"""calm solve — skeleton + tests in, verified class out (§3.18). A thin wrapper; no new logic.

python -m calm_coder.cli solve path/to/skeleton.py --tests path/to/test_x.py --N 4 [--live] [--out solved.py]
Exit 0 on verified; 1 when the budget is exhausted (the failure exit waits for every phase: it's the one barrier).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console

from calm_coder.agents.fill import generate_fills
from calm_coder.agents.scheduler import Scheduler
from calm_coder.bench.experiment import _budget_view
from calm_coder.serve.client import Client
from calm_coder.store.defs import Composition
from calm_coder.store.derive import verified
from calm_coder.store.materialize import STUB_EXC, materialize
from calm_coder.store.store import Store
from calm_coder.task import task_from_files

console = Console(stderr=True)


async def solve(task, *, n: int, seed: int, live: bool, max_comps: int, events_out: Path | None):
    store = Store()
    view = None
    if live:
        from calm_coder.viz.live import LiveView
        view = LiveView(task, store, title=f"calm solve · {task.task_id} · N={n}")
        view.__enter__()
    try:
        cb = view.on_event if view else None
        async with Client() as client:
            emissions = await generate_fills(client, task, store, n=n, seed=seed, on_event=cb)
        _, cands, fan_in, arrival = _budget_view(emissions, n)
        sched = Scheduler(task, store, max_comps=max_comps, on_event=cb)
        await sched.phase1([h for hs in cands.values() for h in hs])
        res = await sched.search(cands, fan_in, arrival)
    finally:
        if view:
            view.__exit__(None, None, None)
    if events_out:
        events_out.write_text("\n".join(json.dumps(e) for e in store.event_log()) + "\n")
    if res.verified_comp is None:
        return None, res
    tr = next(t for t in res.trace if t.comp_id == res.verified_comp)
    src = materialize(task, store, Composition.make(tr.bindings))
    return src.replace(STUB_EXC, "", 1).lstrip("\n"), res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="calm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="fill every method of a class skeleton until the tests pass")
    s.add_argument("skeleton")
    s.add_argument("--tests", required=True)
    s.add_argument("--N", type=int, default=4, help="samples per method")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-comps", type=int, default=64)
    s.add_argument("--live", action="store_true")
    s.add_argument("--out")
    s.add_argument("--events", help="write the store's event log here (for --replay / --replay-shuffled)")
    a = ap.parse_args(argv)
    task = task_from_files(a.skeleton, a.tests)
    src, res = asyncio.run(solve(task, n=a.N, seed=a.seed, live=a.live, max_comps=a.max_comps,
                                 events_out=Path(a.events) if a.events else None))
    if src is None:
        console.print(f"[red]no verified composition within budget ({res.unsolvable_reason}, "
                      f"{len(res.trace)} compositions tested)[/red]")
        return 1
    console.print(f"[green]verified after {len(res.trace)} composition(s)[/green]")
    if a.out:
        Path(a.out).write_text(src)
        console.print(f"wrote {a.out}")
    else:
        sys.stdout.write(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
