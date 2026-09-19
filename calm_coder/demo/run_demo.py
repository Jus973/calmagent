"""Right terminal of the demo (§7): the store. N=4 agents fill every slot concurrently, conflicts stay 0,
duplicates collapse, the root turns green; then the same event log replayed in 20 shuffled orders.

python -m calm_coder.demo.run_demo            # live (needs the model server); records the event log
python -m calm_coder.demo.run_demo --offline  # replays the recording (no GPU/server needed)
python -m calm_coder.demo.run_demo --git      # left terminal: git worktrees + order-flip diff
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from rich.console import Console

from calm_coder.bench.confluence import derived
from calm_coder.cli import solve
from calm_coder.store.store import Store
from calm_coder.task import task_from_files
from calm_coder.viz.live import replay

HERE = Path(__file__).parent
EVENTS = HERE / "recorded" / "events.jsonl"
console = Console()


def shuffled(task, events: list[dict], orders: int = 20) -> int:
    console.rule("[bold]replay the same facts in shuffled orders")
    base = derived(Store.from_events(events), task)
    rng = random.Random(0)
    same = 0
    for i in range(orders):
        ev = list(events)
        rng.shuffle(ev)
        ok = derived(Store.from_events(ev), task) == base
        same += ok
        console.print(f"order {i + 1:>2}: {len(ev)} events → " + ("[green]identical[/green]" if ok else "[red]DIFFERENT[/red]"))
        time.sleep(0.05)
    console.print(f"[bold]{same}/{orders}: derived facts identical in every order. Nothing to order.[/bold]")
    return same


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--git", action="store_true")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--delay", type=float, default=0.06)
    a = ap.parse_args()
    if a.git:
        from calm_coder.demo.git_baseline import run
        run()
        return
    task = task_from_files(HERE / "task.py", HERE / "test_task.py")
    if a.offline:
        events = [json.loads(l) for l in EVENTS.read_text().splitlines() if l.strip()]
        replay(events, task, a.delay)
    else:
        EVENTS.parent.mkdir(exist_ok=True)
        src, res = asyncio.run(solve(task, n=a.N, seed=a.seed, live=True, max_comps=64, events_out=EVENTS))
        console.print("[green]verified[/green]" if src else f"[red]not verified ({res.unsolvable_reason})[/red]")
        events = [json.loads(l) for l in EVENTS.read_text().splitlines() if l.strip()]
    shuffled(task, events)


if __name__ == "__main__":
    main()
