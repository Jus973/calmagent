"""Where do unsolved tasks actually die? (Phase 0 of the v2 plan.)

For every unsolved task of a finished run, replay its event log and report the slots for which no
candidate ever passed, together with the exception types their failures raise. The split that
decides whether targeted repair is worth building is:

  interface / shared-state errors (AttributeError, KeyError, TypeError, NameError, IndexError)
  vs value errors (assertion failures) — the former are what a repair prompt carrying the current
  class can actually fix.

python -m analysis.dead_slots runs/<dir> [--arm calm] [--seed 0] [--N 8]
Writes <dir>/analysis/dead_slots.{jsonl,md}; the run's own logs are never touched.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from calm_coder.bench.classeval import load_rows, task_from_row
from calm_coder.bench.posthoc import _rows
from calm_coder.store.store import Store
from calm_coder.v2 import state


def _kind(failures) -> str:
    """One label per dead slot: what a repair prompt would have to fix there."""
    if not failures:
        return "unattributed"
    return "interface" if any(f.interface_error for f in failures) else "value"


def analyze(task, events: list[dict]) -> dict:
    store = Store.from_events(events)
    comp = state.best_class(store, task)
    report = state.dead_slot_report(store, task, comp)
    failures = {slot: state.slot_failures(store, task, slot, comp) for slot in report}
    return {"dead_slots": list(report), "best_class": comp.id if comp else None,
            "n_slots": len(task.slots), "report": report,
            "exc_types": Counter(f.exc_type for fs in failures.values() for f in fs),
            # Per slot, not per failure record: a slot with twenty failing candidates is still one
            # slot, and a dead slot whose failures did not parse is unattributed, not "value".
            "kind_by_slot": {slot: _kind(fs) for slot, fs in failures.items()},
            "interface_errors": sum(1 for fs in failures.values() if _kind(fs) == "interface"),
            "value_errors": sum(1 for fs in failures.values() if _kind(fs) == "value"),
            "unattributed": sum(1 for fs in failures.values() if _kind(fs) == "unattributed")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--arm", default="calm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=8)
    a = ap.parse_args()
    d = Path(a.run_dir)
    out_dir = d / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {r["task_id"]: r for r in load_rows()}
    results = [r for r in _rows(d / "results.jsonl")
               if r["arm"] == a.arm and r.get("N") == a.N and r["seed"] == a.seed]
    if not results:
        raise SystemExit(f"no results in {d} for arm={a.arm} N={a.N} seed={a.seed}; "
                         f"arms present: {sorted({r['arm'] for r in _rows(d / 'results.jsonl')})}")
    out, totals, skipped = [], Counter(), []
    for r in sorted(results, key=lambda r: r["task_id"]):
        ev = _rows(d / "events" / f"{r['task_id']}__{a.arm}__s{a.seed}.jsonl")
        if not ev or r["task_id"] not in rows:
            skipped.append(r["task_id"])
            continue
        info = analyze(task_from_row(rows[r["task_id"]]), ev)
        info["exc_types"] = dict(info["exc_types"])
        row = {"task_id": r["task_id"], "arm": a.arm, "seed": a.seed, "N": a.N, "solved": r["solved"], **info}
        out.append(row)
        totals["tasks"] += 1
        totals["unsolved"] += not r["solved"]
        if not r["solved"]:
            totals["dead_slots"] += len(info["dead_slots"])
            totals["one_dead_slot"] += len(info["dead_slots"]) == 1
            totals["interface_errors"] += info["interface_errors"]
            totals["value_errors"] += info["value_errors"]
            totals["unattributed"] += info["unattributed"]
    (out_dir / "dead_slots.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    md = ["# Dead slots", "",
          f"{totals['unsolved']} of {totals['tasks']} tasks unsolved at N={a.N} ({a.arm}).",
          f"{totals['dead_slots']} dead slots in them; {totals['one_dead_slot']} unsolved tasks have exactly one.",
          f"Dead slots by kind: {totals['interface_errors']} interface/state, "
          f"{totals['value_errors']} value, {totals['unattributed']} unattributed.",
          f"Skipped (no event log or task row): {', '.join(skipped) or 'none'}.", "",
          "| task | solved | dead slots | of | exception types |", "| --- | --- | --- | --- | --- |"]
    for r in out:
        md.append(f"| {r['task_id']} | {r['solved']} | {', '.join(r['dead_slots']) or '-'} | {r['n_slots']} | "
                  f"{', '.join(f'{k}x{v}' for k, v in sorted(r['exc_types'].items())) or '-'} |")
    (out_dir / "dead_slots.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:6]))


if __name__ == "__main__":
    main()
