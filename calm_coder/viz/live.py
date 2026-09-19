"""Live terminal view of the store (§3.16).

Attach to a running solve via `LiveView.on_event`, or replay a logged run:
  python -m calm_coder.viz.live --replay runs/<dir>/events/<task>__calm__s0.jsonl [--delay 0.05]
  python -m calm_coder.viz.live --replay <events.jsonl> --replay-shuffled 20
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from calm_coder.store.defs import Definition, Outcome
from calm_coder.store.derive import slot_evidence, verified
from calm_coder.store.store import Store

STYLE = {"pass": "bold black on green", "fail": "white on red", "error": "white on red",
         "timeout": "white on red", "inconclusive": "black on yellow", None: "dim"}


class LiveView:
    def __init__(self, task, store: Store | None = None, title: str = ""):
        self.task, self.store = task, store or Store()
        self.title = title or task.task_id
        self.dupes = 0
        self.comps: set[str] = set()
        self.ticker: deque[str] = deque(maxlen=8)
        self.live: Live | None = None

    # ---- feed
    def on_event(self, ev: dict) -> None:
        k = ev["kind"]
        if k == "def":
            d: Definition = ev["def"]
            if not ev["new"]:
                self.dupes += 1
                self.ticker.append(f"[dim]dup   {d.kind} {d.hash[:8]} ({d.slot or d.name}) collapsed[/dim]")
            else:
                self.ticker.append(f"def   {d.kind} {d.hash[:8]} → {d.slot or 'helper'}")
        elif k == "outcome":
            if len(ev["comp"].bindings) == len(self.task.slots):
                self.comps.add(ev["comp"].id)
            self.ticker.append(f"test  {ev['test'][:34]:<34} {ev['result']}")
        elif k == "comp":
            self.comps.add(ev["comp"].id)
        self.refresh()

    def refresh(self) -> None:
        if self.live:
            self.live.update(self.render())

    # ---- view
    def _best(self, d: Definition) -> str | None:
        ev = slot_evidence(self.store, self.task, d)
        for r in ("pass", "inconclusive", "fail", "error", "timeout"):
            if ev[r]:
                return r
        return None

    def render(self):
        v = verified(self.store, self.task)
        grid = Table(show_header=False, box=None, padding=(0, 1))
        grid.add_column(style="bold", no_wrap=True)
        width = max((len(self.store.defs_for_slot(s.id)) for s in self.task.slots), default=0)
        for _ in range(max(width, 1)):
            grid.add_column(no_wrap=True)
        vc = next(iter(sorted(v)), None)
        vbind = {}
        if vc:
            o = next(o for o in self.store.outcomes() if o.comp_id == vc)
            vbind = dict(o.bindings)
        for s in self.task.slots:
            cells = []
            for d in sorted(self.store.defs_for_slot(s.id), key=lambda d: d.hash):
                t = Text(f" {d.hash[:8]} ", style=STYLE[self._best(d)])
                if vbind.get(s.id) == d.hash:
                    t.stylize("underline")
                cells.append(t)
            grid.add_row(s.id, *cells)
        root = Text(f"  {self.task.class_name}  ", style="bold black on green" if v else "bold white on grey30")
        root.append(f"  verified: {vc[:12]}" if vc else "  searching…", style="green" if v else "dim")
        n_defs = len(self.store.defs())
        counters = Table.grid(padding=(0, 3))
        counters.add_row(f"defs [bold]{n_defs}[/bold]", "conflicts [bold green]0[/bold green]",
                         f"dupes collapsed [bold]{self.dupes}[/bold]", f"comps tested [bold]{len(self.comps)}[/bold]",
                         f"verified [bold]{len(v)}[/bold]")
        return Panel(Group(root, Text(""), grid, Text(""), counters, Text(""),
                           Text.from_markup("\n".join(self.ticker) or " ")),
                     title=f"CALM store · {self.title}", border_style="green" if v else "blue")

    def __enter__(self):
        self.live = Live(self.render(), refresh_per_second=12, console=Console())
        self.live.__enter__()
        return self

    def __exit__(self, *exc):
        self.live.update(self.render())
        self.live.__exit__(*exc)
        self.live = None


def replay(events: list[dict], task, delay: float) -> None:
    view = LiveView(task, title=f"{task.task_id} (replay)")
    with view:
        for e in events:
            if e["kind"] == "def":
                d = Definition.from_json(e["data"])
                view.on_event({"kind": "def", "new": view.store.add_def(d), "def": d})
            else:
                o = Outcome.from_json(e["data"])
                view.store.add_outcome(o)
                from calm_coder.store.defs import Composition
                view.on_event({"kind": "outcome", "test": o.test_id, "result": o.result,
                               "comp": Composition(o.comp_id, o.bindings)})
            time.sleep(delay)


def main() -> None:
    from calm_coder.bench.classeval import load_rows, task_from_row
    from calm_coder.bench.confluence import derived
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--task", help="task id (default: from the file name)")
    ap.add_argument("--delay", type=float, default=0.04)
    ap.add_argument("--replay-shuffled", type=int, default=0)
    ap.add_argument("--skeleton")
    ap.add_argument("--tests")
    a = ap.parse_args()
    p = Path(a.replay)
    events = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if a.skeleton:
        from calm_coder.task import Task
        task = Task.from_skeleton(task_id=p.stem, skeleton=Path(a.skeleton).read_text(), test_src=Path(a.tests).read_text())
    else:
        tid = a.task or p.name.split("__")[0]
        task = task_from_row(next(r for r in load_rows() if r["task_id"] == tid))
    if a.replay_shuffled:
        c = Console()
        base = derived(Store.from_events(events), task)
        rng = random.Random(0)
        same = 0
        for i in range(a.replay_shuffled):
            ev = list(events)
            rng.shuffle(ev)
            ok = derived(Store.from_events(ev), task) == base
            same += ok
            c.print(f"order {i + 1:>2}: {len(ev)} events shuffled → derived facts "
                    + ("[green]identical[/green]" if ok else "[red]DIFFERENT[/red]"))
        c.print(f"[bold]{same}/{a.replay_shuffled} orders: derived facts identical. Nothing to order.[/bold]")
        return
    replay(events, task, a.delay)


if __name__ == "__main__":
    main()
