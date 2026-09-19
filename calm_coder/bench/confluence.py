"""Confluence on real logs (§6.4, H5): replay every event log in 20 shuffled orders and require
identical derived facts. This one is not allowed to fail.

python -m calm_coder.bench.confluence runs/<dir> [--orders 20]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from calm_coder.store.defs import Composition
from calm_coder.store.derive import complete, done, verified
from calm_coder.store.store import Store


def derived(store: Store, task) -> tuple:
    comps = {o.comp_id: Composition.make(dict(o.bindings)) for o in store.outcomes() if o.bindings}
    return (frozenset(d.hash for d in store.defs()), store.outcomes(), verified(store, task), done(store, task),
            tuple(sorted((cid, complete(store, task, c)) for cid, c in comps.items())))


def check_log(events: list[dict], task, orders: int = 20, seed: int = 0) -> tuple[bool, int]:
    base = derived(Store.from_events(events), task)
    rng = random.Random(seed)
    diffs = 0
    for _ in range(orders):
        ev = list(events)
        rng.shuffle(ev)
        diffs += derived(Store.from_events(ev), task) != base
    return diffs == 0, diffs


def check_run(d: Path, orders: int = 20) -> dict:
    from calm_coder.bench.classeval import load_rows, task_from_row
    rows = {r["task_id"]: r for r in load_rows()}
    files = sorted((d / "events").glob("*.jsonl"))
    out = {"files": len(files), "pass": 0, "fail": 0, "orders": orders, "failures": [], "events_total": 0}
    for f in files:
        task = task_from_row(rows[f.name.split("__")[0]])
        events = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        out["events_total"] += len(events)
        ok, diffs = check_log(events, task, orders)
        out["pass" if ok else "fail"] += 1
        if not ok:
            out["failures"].append({"file": f.name, "diffs": diffs})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--orders", type=int, default=20)
    a = ap.parse_args()
    d = Path(a.run_dir)
    res = check_run(d, a.orders)
    (d / "confluence.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "failures"}), res["failures"][:5])


if __name__ == "__main__":
    main()
