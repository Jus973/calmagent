"""Zero-token recon: what does pooling already-sampled candidates buy? (Phase 0 of the v2 plan.)

For every task of a finished run, build one store out of

  * the per-method fills the CALM arm sampled (from its event log), and
  * the methods of every whole-class sample arm C produced (decomposed on the way in),

then run the normal search. The three columns that matter are: solved by C alone, solved by CALM
alone, solved by the pool. The gap is the headroom a producer-agnostic store gives us before a
single new token is spent.

python -m analysis.pool_existing runs/<dir> [--seed 0] [--N 8] [--cap 2000] [--par 6]
Writes <dir>/analysis/pool_existing.{jsonl,md}; the run's own logs are never touched.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from calm_coder.agents.scheduler import Scheduler
from calm_coder.bench.classeval import load_rows, task_from_row
from calm_coder.bench.posthoc import _ranked_cands, _rows, exhaustive
from calm_coder.store.derive import verified
from calm_coder.store.store import Store
from calm_coder.v2.decompose import ingest_class_sample


async def pool_task(task, calm_events: list[dict], class_samples: list[dict], *, cap: int, par: int,
                    max_comps: int) -> dict:
    store = Store.from_events([e for e in calm_events if e["kind"] == "def"])
    from_calm = {s.id: len(store.defs_for_slot(s.id)) for s in task.slots}
    seed_comps = []
    for s in sorted(class_samples, key=lambda s: s["sample_idx"]):
        ing = ingest_class_sample(task, store, s["text"], {"from": "c_sample", "sample_idx": s["sample_idx"]})
        if ing.complete and not ing.error:
            seed_comps.append(ing.bindings)
    cands = {s.id: sorted(d.hash for d in store.defs_for_slot(s.id)) for s in task.slots}
    out = {"slots": {s.id: {"calm": from_calm[s.id], "pooled": len(cands[s.id])} for s in task.slots},
           "seed_comps": len(seed_comps), "pooled_solved": False, "pooled_comps": 0,
           "exhaustive": None, "events": store.event_log()}
    if any(not v for v in cands.values()):
        out["unsolvable_reason"] = "no_fill_for_some_slot"
        return out
    sched = Scheduler(task, store, max_comps=max_comps)
    await sched.phase1([h for hs in cands.values() for h in hs])
    res = await sched.search(cands, {}, {})
    out["pooled_solved"] = res.verified_comp is not None
    out["pooled_comps"] = len(res.trace)
    if not out["pooled_solved"]:
        stub_pass = {h for h, r in sched._stub.items() if r == "pass"}
        out["exhaustive"] = await exhaustive(task, store, _ranked_cands(task, store, stub_pass), cap, par)
        out["pooled_solved_exhaustive"] = bool(verified(store, task))
    out["events"] = store.event_log()
    return out


async def main_async(d: Path, seed: int, N: int, cap: int, par: int, max_comps: int) -> None:
    out_dir = d / "analysis"
    (out_dir / "events").mkdir(parents=True, exist_ok=True)
    done = {r["task_id"] for r in _rows(out_dir / "pool_existing.jsonl")}
    rows = {r["task_id"]: r for r in load_rows()}
    results = _rows(d / "results.jsonl")
    calm = {r["task_id"]: r for r in results if r["arm"] == "calm" and r["N"] == N and r["seed"] == seed}
    c = {r["task_id"]: r for r in results if r["arm"] == "c" and r["N"] == N and r["seed"] == seed}
    samples: dict[str, list[dict]] = {}
    for s in _rows(d / "class_samples.jsonl"):
        if s["arm"] == "c" and s["seed"] == seed and s["sample_idx"] < N:
            samples.setdefault(s["task_id"], []).append(s)
    for tid in sorted(set(calm) | set(c)):
        if tid in done or tid not in rows:
            continue
        task = task_from_row(rows[tid])
        r = await pool_task(task, _rows(d / "events" / f"{tid}__calm__s{seed}.jsonl"),
                            samples.get(tid, []), cap=cap, par=par, max_comps=max_comps)
        events = r.pop("events", [])
        (out_dir / "events" / f"{tid}__pool.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        row = {"task_id": tid, "calm_solved": calm.get(tid, {}).get("solved"),
               "c_solved": c.get(tid, {}).get("solved"), **r}
        with open(out_dir / "pool_existing.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(row["task_id"], "c=", row["c_solved"], "calm=", row["calm_solved"], "pooled=", row["pooled_solved"])
    write_md(out_dir)


def write_md(out_dir: Path) -> None:
    rows = _rows(out_dir / "pool_existing.jsonl")
    if not rows:
        return
    n = len(rows)
    got = lambda k: sum(1 for r in rows if r.get(k))            # noqa: E731
    new = [r["task_id"] for r in rows if r["pooled_solved"] and not (r["c_solved"] or r["calm_solved"])]
    lines = ["# Pooling existing samples (zero new tokens)", "",
             f"| arm | solved | of {n} |", "| --- | --- | --- |",
             f"| C | {got('c_solved')} | {n} |",
             f"| CALM | {got('calm_solved')} | {n} |",
             f"| pooled store, same search | {got('pooled_solved')} | {n} |",
             f"| pooled store, exhaustive | {got('pooled_solved') + got('pooled_solved_exhaustive')} | {n} |", "",
             f"Tasks only the pool solves: {', '.join(new) or 'none'}", "",
             "| task | C | CALM | pooled | candidates/slot (CALM -> pooled) |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        cand = ", ".join(f"{s}:{v['calm']}->{v['pooled']}" for s, v in r["slots"].items())
        lines.append(f"| {r['task_id']} | {r['c_solved']} | {r['calm_solved']} | {r['pooled_solved']} | {cand} |")
    (out_dir / "pool_existing.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--par", type=int, default=6)
    ap.add_argument("--max-comps", type=int, default=64)
    a = ap.parse_args()
    asyncio.run(main_async(Path(a.run_dir), a.seed, a.N, a.cap, a.par, a.max_comps))


if __name__ == "__main__":
    main()
