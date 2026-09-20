"""What content-addressing saves the verifier, measured on recorded runs.

A test class's result depends only on the test module and the fills that class can reach, both
content hashes (`Scheduler.class_key`). So for a recorded sequence of tested compositions the
saving is exact and needs no subprocess: every (composition, test class) pair the run executed is
one unit of work, and the pairs that share a key are one unit of work between them.

Three scopes, each a strict superset of the one before:

  within-config   one (task, arm, seed, N): what the search itself re-tests, because neighbouring
                  compositions differ in one slot and most test classes cannot reach that slot;
  within-task     all configs of a task pooled, which is the N sweep re-testing nested prefixes;
  cross-task      the whole run against one cache, which also catches classes shared between tasks.

python -m analysis.cache_value runs/<dir> [runs/<dir2> ...] [--out results/cache.md]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from calm_coder.agents.scheduler import Scheduler
from calm_coder.bench.classeval import load_rows, task_from_row
from calm_coder.jsonl import read_jsonl
from calm_coder.store.defs import Composition
from calm_coder.store.store import Store


def _events(d: Path, task_id: str, arm: str, seed: int) -> list[dict] | None:
    f = d / "events" / f"{task_id}__{arm}__s{seed}.jsonl"
    return read_jsonl(f) if f.exists() else None


def keys_for_run(d: Path, rows: dict) -> list[dict]:
    """One record per (task, arm, seed, N): the cache key of every (composition, test class) pair
    the config executed, in the order the run executed them."""
    out = []
    for tr in read_jsonl(d / "traces.jsonl"):
        events = _events(d, tr["task_id"], tr["arm"], int(tr["seed"]))
        if not events or tr["task_id"] not in rows:
            continue
        task = task_from_row(rows[tr["task_id"]])
        store = Store.from_events(events)
        sched = Scheduler(task, store)
        pairs, comps = [], []
        for row in tr["trace"]:
            comp = Composition.make(dict(row["bindings"]))
            keys = [sched.class_key(t, comp) for t in row["results"]]
            pairs += keys
            comps.append({"keys": keys, "wall_ms": row.get("wall_ms", 0)})
        if pairs:
            out.append({"task_id": tr["task_id"], "arm": tr["arm"], "seed": int(tr["seed"]),
                        "N": tr.get("N"), "keys": pairs, "comps": comps,
                        "wall_ms": sum(c["wall_ms"] for c in comps)})
    return out


def _saving(groups: list[list[str]]) -> dict:
    """Executions with and without a cache, over groups that each share one cache."""
    pairs = sum(len(g) for g in groups)
    execs = sum(len(set(g)) for g in groups)
    return {"pairs": pairs, "executions": execs, "saved": pairs - execs,
            "saved_frac": (pairs - execs) / pairs if pairs else 0.0}


def _subprocess_saving(records: list[dict], scope) -> dict:
    """Wall clock, bracketed.

    `run_full` narrows a composition's subprocess to the classes that are not already known, so the
    saving is neither all-or-nothing nor the full class-level rate:

      floor  only compositions whose every class is known are skipped, and nothing else is saved;
      est.   time is proportional to the classes actually executed, so a composition with half its
             classes known costs half. Process startup is not saved, which is why this is an
             estimate and the truth sits under it.
    """
    seen: dict[object, set[str]] = defaultdict(set)
    total = skipped = 0
    wall = wall_floor = wall_prop = 0
    for r in records:
        s = seen[scope(r)]
        for c in r["comps"]:
            total += 1
            wall += c["wall_ms"]
            known = sum(k in s for k in c["keys"])
            if known == len(c["keys"]):
                skipped += 1
                wall_floor += c["wall_ms"]
            wall_prop += c["wall_ms"] * known / len(c["keys"])
            s.update(c["keys"])
    return {"compositions": total, "skipped": skipped,
            "skipped_frac": skipped / total if total else 0.0, "wall_ms": wall,
            "wall_ms_floor": int(wall_floor), "wall_floor_frac": wall_floor / wall if wall else 0.0,
            "wall_ms_est": int(wall_prop), "wall_est_frac": wall_prop / wall if wall else 0.0}


def summarize(records: list[dict]) -> dict:
    by_task: dict[str, list[str]] = defaultdict(list)
    everything: list[str] = []
    for r in records:
        by_task[r["task_id"]] += r["keys"]
        everything += r["keys"]
    cfg = lambda r: (r["task_id"], r["arm"], r["seed"], r["N"])   # noqa: E731
    scopes = {
        "within_config": {**_saving([r["keys"] for r in records]),
                          **_subprocess_saving(records, cfg)},
        "within_task": {**_saving(list(by_task.values())),
                        **_subprocess_saving(records, lambda r: r["task_id"])},
        "cross_task": {**_saving([everything]),
                       **_subprocess_saving(records, lambda r: 0)},
    }
    return {"configs": len(records), "tasks": len(by_task), "scopes": scopes,
            "recorded_test_wall_ms": sum(r["wall_ms"] for r in records)}


def render(m: dict, runs: list[str]) -> str:
    L = ["# What content-addressing saves the verifier", "",
         f"{m['configs']} configs over {m['tasks']} tasks, from {', '.join(runs)}.", "",
         "A test class's result is keyed by the test module and the fills that class can reach, "
         "both content hashes, so two pairs with equal keys are one unit of work between them and "
         "the saving is exact rather than sampled.", "",
         "**Work avoided** — (composition, test class) pairs that need no fresh result:", "",
         "| cache scope | pairs | need executing | avoided |",
         "| --- | --- | --- | --- |"]
    labels = {"within_config": "one solve", "within_task": "one task, all N",
              "cross_task": "the whole run"}
    for k, lab in labels.items():
        s = m["scopes"][k]
        L.append(f"| {lab} | {s['pairs']} | {s['executions']} | {s['saved']} ({s['saved_frac']:.1%}) |")
    L += ["", "**Wall clock** — a composition's subprocess is narrowed to the classes that are not "
              "already known, so the saving is bracketed: the floor counts only compositions that "
              "are skipped whole, the estimate assumes time is proportional to the classes still "
              "executed and so does not credit the process startup a narrowed run still pays.", "",
          "| cache scope | compositions | skipped whole | test wall | floor | estimate |",
          "| --- | --- | --- | --- | --- | --- |"]
    for k, lab in labels.items():
        s = m["scopes"][k]
        L.append(f"| {lab} | {s['compositions']} | {s['skipped']} ({s['skipped_frac']:.1%}) | "
                 f"{s['wall_ms'] / 1000:.0f}s | {s['wall_floor_frac']:.1%} | {s['wall_est_frac']:.1%} |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", default="results/cache.md")
    a = ap.parse_args()
    rows = {r["task_id"]: r for r in load_rows()}
    records = [r for d in a.run_dirs for r in keys_for_run(Path(d), rows)]
    if not records:
        raise SystemExit("no traces with event logs in those runs")
    m = summarize(records)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(m, a.run_dirs))
    out.with_suffix(".json").write_text(json.dumps(m, indent=1))
    print(out.read_text())


if __name__ == "__main__":
    main()
